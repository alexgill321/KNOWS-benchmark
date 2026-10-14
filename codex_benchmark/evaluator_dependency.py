"""Proof-backed exclusions for two reviewed evaluator dependency failures.

This does not run a browser, replay a model, alter a grade, or save episode.json.
The caller must own the episode (or hold the drained campaign account lock), and
save the mutated episode only after apply_dependency_review returns successfully.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from .setup_recovery import _read_object, _sync_directory


KIND = "evaluator_dependency"
VERSION = 1
MARKER = "evaluator_dependency_review"
TERMINAL = {"completed", "action_limit", "timeout"}
_MODEL_FIELDS = ("model", "effort", "thread_id", "turn_id", "status", "tool_calls",
                 "final_text", "token_usage", "error")
_INVARIANTS = (*_MODEL_FIELDS, "task_id", "workspace_id", "workspace_url", "prompt_sha256",
               "browsing_history", "run_targets_path", "seed", "repetition")
_SUGGEST = re.compile(r"^\[IMDb\] Suggest API matches for '(.*)' \([^\n]*\) -> (\[.*\])$")
_NO_JSON = re.compile(r"^\[IMDb\] No valid JSON-LD Movie data for '(.*)' \((tt\d+)\)$")
_GOT_DATA = re.compile(r"^\[IMDb\] Got data for .* \((tt\d+)\)")
_COLOR_ERROR = "colorhexa unreachable - could not verify any hex value."
_FAMILY = re.compile(r"knows\.(?:sheets_55_Movie_Recommendation|sheets_45_Personal_WeddingPlanner_weddingcolorpallette)\.\d+")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encoded(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _object(payload: bytes) -> dict:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _steps(evaluation: dict):
    for checkpoint in evaluation.get("detailed_report", {}).get("checkpoints", []):
        for step in checkpoint.get("steps", []):
            yield checkpoint, step


def _title_list_contains(text: str, title: str) -> bool:
    return (text == title or text.startswith(title + ", ") or text.endswith(", " + title)
            or ", " + title + ", " in text)


def _detect(task_id: str, evaluation: dict, log: bytes) -> list[dict]:
    """Only classify explicit external failures, never generic missing output."""
    if evaluation.get("status") != "completed":
        return []
    if re.fullmatch(r"knows\.sheets_45_Personal_WeddingPlanner_weddingcolorpallette\.\d+", task_id):
        for checkpoint, step in _steps(evaluation):
            if (step.get("category") == "execution_error" and step.get("success") is False
                    and step.get("details") == _COLOR_ERROR):
                return [{"dependency": "colorhexa", "failure": "all_hex_lookups_unavailable",
                         "checkpoint": checkpoint.get("name"), "step": step.get("name")}]
        return []
    if not re.fullmatch(r"knows\.sheets_55_Movie_Recommendation\.\d+", task_id):
        return []
    missing = [step["details"].split("Could not retrieve IMDb data: ", 1)[1]
               for _, step in _steps(evaluation)
               if step.get("success") is False
               and "Could not retrieve IMDb data: " in str(step.get("details", ""))]
    if not missing:
        return []
    candidates: dict[str, set[str]] = {}
    failed: dict[str, set[str]] = {}
    succeeded: set[str] = set()
    for line in log.decode("utf-8", errors="replace").splitlines():
        match = _SUGGEST.fullmatch(line)
        if match:
            try:
                ids = ast.literal_eval(match[2])
            except (ValueError, SyntaxError):
                continue
            if isinstance(ids, list) and all(isinstance(i, str) and re.fullmatch(r"tt\d+", i) for i in ids):
                candidates.setdefault(match[1], set()).update(ids)
        match = _NO_JSON.fullmatch(line)
        if match:
            failed.setdefault(match[1], set()).add(match[2])
        match = _GOT_DATA.match(line)
        if match:
            succeeded.add(match[1])
    corroborated = []
    for title, ids in sorted(candidates.items()):
        matched = sorted((ids & failed.get(title, set())) - succeeded)
        if matched and not (ids & succeeded) and any(_title_list_contains(text, title) for text in missing):
            corroborated.append({"title": title, "candidate_ids": matched})
    return ([{"dependency": "imdb", "failure": "matched_candidates_missing_movie_metadata",
              "corroborated_titles": corroborated}] if corroborated else [])


def _require_identity(episode: dict, directory: Path, result: dict, evaluation: dict) -> None:
    if (episode.get("status") not in TERMINAL or episode.get("evaluation_error")
            or episode.get("task_id") != directory.name or episode.get("model") != directory.parent.name):
        raise ValueError("A matching terminal episode without evaluator error is required")
    for field in ("task_id", "model", "effort", "thread_id", "turn_id", "workspace_id"):
        if not isinstance(episode.get(field), str) or not episode[field]:
            raise ValueError(f"Missing episode identity: {field}")
    if any(episode.get(field) != result.get(field) for field in _MODEL_FIELDS):
        raise ValueError("Saved terminal model result does not match the episode")
    if (evaluation.get("status") != "completed" or evaluation.get("judge_errors")
            or any(evaluation.get(field) != episode[field] for field in ("task_id", "workspace_id"))):
        raise ValueError("A completed matching evaluator report without judge failure is required")
    attempt = evaluation.get("evaluation_attempt_id")
    if (not isinstance(attempt, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attempt)
            or evaluation.get("step_journal", {}).get("attempt_id") != attempt):
        raise ValueError("Current evaluation attempt identity is missing or inconsistent")


def _check_attempt(directory: Path, evaluation: dict, report: bytes) -> None:
    attempt = evaluation["evaluation_attempt_id"]
    if (directory / "evaluation-attempts" / f"{attempt}.json").read_bytes() != report:
        raise ValueError("Archived evaluator attempt differs from the current report")
    events = [json.loads(line) for line in (directory / "evaluation-steps.jsonl").read_text().splitlines()
              if line.strip()]
    matching = [event for event in events if event.get("attempt_id") == attempt]
    if (not matching or matching[0].get("event") != "evaluation_started"
            or matching[-1].get("event") != "evaluation_finished"
            or events[-1].get("attempt_id") != attempt
            or any(event.get("task_id") != evaluation["task_id"]
                   or event.get("workspace_id") != evaluation["workspace_id"] for event in matching)):
        raise ValueError("Current attempt journal is incomplete or does not match the report")


def _log_slice(payload: bytes, start: int, directory: Path) -> bytes:
    if (isinstance(start, bool) or not isinstance(start, int) or start < 0 or start > len(payload)
            or (start and payload[start - 1:start] != b"\n")):
        raise ValueError("Invalid evaluator log byte offset")
    selected = payload[start:]
    endings = []
    for line in selected.splitlines():
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(record, dict) and "status" in record and "output" in record:
            if Path(record["output"]).resolve() == (directory / "evaluation.json").resolve():
                endings.append(record)
    if len(endings) != 1 or endings[0].get("status") != "completed":
        raise ValueError("Log range must identify exactly one completed evaluation")
    return selected


def _validity(failures: list[dict]) -> dict:
    names = ", ".join(sorted(failure["dependency"] for failure in failures))
    return {"status": "excluded_infrastructure", "kind": KIND,
            "reason": f"Reviewed evaluator dependency failure ({names}); preserved grade is diagnostic and excluded from model averages."}


def _save(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def _marker(directory: Path, proof_path: Path, proof_bytes: bytes) -> dict:
    return {"kind": KIND, "version": VERSION, "reviewed": True, "model_replayed": False,
            "proof_path": str(proof_path.relative_to(directory)), "proof_sha256": _sha(proof_bytes)}


def apply_dependency_review(episode: dict, episode_path: Path, log_start: int = 0) -> bool:
    """Archive first, then add proof-backed exclusion to the in-memory episode.

    Return True for newly applied metadata, False for no match or an existing
    exclusion. Refuse inconsistent matching evidence; never clear an exclusion.
    Existing valid reviews are idempotent, including before episode.json is saved.
    """
    episode_path = Path(episode_path).resolve()
    directory = episode_path.parent
    if episode.get(MARKER):
        _validate(episode, episode_path)
        return False
    if episode.get("benchmark_validity", {}).get("status") == "excluded_infrastructure":
        return False
    if not _FAMILY.fullmatch(str(episode.get("task_id", ""))):
        return False
    report_path, result_path, log_path = (directory / name for name in
                                         ("evaluation.json", "agent-result.json", "evaluate.log"))
    report = report_path.read_bytes()
    evaluation = _object(report)
    if evaluation.get("status") != "completed":
        return False
    details = [str(step.get("details", "")) for _, step in _steps(evaluation)]
    if not any(detail == _COLOR_ERROR or "Could not retrieve IMDb data: " in detail for detail in details):
        return False
    log = log_path.read_bytes()
    # Select the supplied slice before detecting, so an older failed attempt can
    # never corroborate a current report. Full structural checks follow a match.
    if isinstance(log_start, bool) or not isinstance(log_start, int) or not 0 <= log_start <= len(log):
        raise ValueError("Invalid evaluator log byte offset")
    failures = _detect(str(episode.get("task_id", "")), evaluation, log[log_start:])
    if not failures:
        return False
    selected = _log_slice(log, log_start, directory)
    original = episode_path.read_bytes()
    result_bytes = result_path.read_bytes()
    result = _object(result_bytes)
    saved_episode = _object(original)
    _require_identity(episode, directory, result, evaluation)
    _require_identity(saved_episode, directory, result, evaluation)
    if any(episode.get(field) != saved_episode.get(field) for field in _INVARIANTS):
        raise ValueError("In-memory episode differs from saved model evidence")
    _check_attempt(directory, evaluation, report)
    key = _sha(_encoded({"report": _sha(report), "result": _sha(result_bytes),
                         "log": _sha(selected), "log_start": log_start, "version": VERSION}))
    archive = directory / "evaluator-dependency-reviews" / key
    proof_path = archive / "proof.json"
    files = {"episode.json": original, "review-input.json": _encoded(episode),
             "evaluation.json": report, "agent-result.json": result_bytes, "evaluate.log": selected}
    proof = {"kind": KIND, "version": VERSION, "task_id": episode["task_id"], "model": episode["model"],
             "evaluation_attempt_id": evaluation["evaluation_attempt_id"],
             "log_start": log_start, "log_end": len(log), "failures": failures,
             "files": {name: _sha(payload) for name, payload in files.items()}}
    if archive.exists():
        proof_bytes = proof_path.read_bytes()
        existing = _object(proof_bytes)
        if any(existing.get(field) != value for field, value in proof.items()):
            raise ValueError("Existing dependency archive does not match this evidence")
        if any((archive / name).read_bytes() != payload for name, payload in files.items()):
            raise ValueError("Existing dependency archive was altered")
    else:
        archive.mkdir(parents=True, mode=0o700)
        _sync_directory(archive.parent)
        for name, payload in files.items():
            _save(archive / name, payload)
        proof["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        proof_bytes = _encoded(proof)
        _save(proof_path, proof_bytes)
    if (episode_path.read_bytes() != original or report_path.read_bytes() != report
            or result_path.read_bytes() != result_bytes or log_path.read_bytes() != log):
        raise RuntimeError("Episode evidence changed during dependency review")
    _check_attempt(directory, evaluation, report)
    episode["benchmark_validity"] = _validity(failures)
    episode[MARKER] = _marker(directory, proof_path, proof_bytes)
    return True


def _validate(episode: dict, episode_path: Path) -> None:
    directory = episode_path.parent.resolve()
    marker = episode.get(MARKER, {})
    if (marker.get("kind") != KIND or marker.get("version") != VERSION
            or marker.get("reviewed") is not True or marker.get("model_replayed") is not False):
        raise ValueError("Missing reviewed evaluator dependency proof")
    relative = marker.get("proof_path", "")
    if not isinstance(relative, str) or not re.fullmatch(r"evaluator-dependency-reviews/[0-9a-f]{64}/proof\.json", relative):
        raise ValueError("Invalid dependency archive path")
    proof_path = directory / relative
    if proof_path.resolve().parent.parent != directory / "evaluator-dependency-reviews":
        raise ValueError("Dependency archive must stay within its episode")
    proof_bytes = proof_path.read_bytes()
    if _sha(proof_bytes) != marker.get("proof_sha256"):
        raise ValueError("Dependency proof changed")
    proof = _object(proof_bytes)
    if (proof.get("kind") != KIND or proof.get("version") != VERSION
            or any(proof.get(field) != episode.get(field) for field in ("task_id", "model"))):
        raise ValueError("Dependency proof identity differs")
    expected_files = {"episode.json", "review-input.json", "evaluation.json", "agent-result.json", "evaluate.log"}
    if set(proof.get("files", {})) != expected_files:
        raise ValueError("Incomplete dependency archive")
    archived = {name: (proof_path.parent / name).read_bytes() for name in expected_files}
    if any(_sha(payload) != proof["files"][name] for name, payload in archived.items()):
        raise ValueError("Archived dependency evidence changed")
    original = _object(archived["episode.json"])
    review_input = _object(archived["review-input.json"])
    evaluation = _object(archived["evaluation.json"])
    result = _object(archived["agent-result.json"])
    for source in (episode, original, review_input):
        _require_identity(source, directory, result, evaluation)
    if any(episode.get(field) != source.get(field) for source in (original, review_input) for field in _INVARIANTS):
        raise ValueError("Model output or episode identity changed after dependency review")
    if (original.get("benchmark_validity", {}).get("status") == "excluded_infrastructure"
            or review_input.get("benchmark_validity", {}).get("status") == "excluded_infrastructure"):
        raise ValueError("Dependency review cannot replace another infrastructure exclusion")
    for name in ("evaluation.json", "agent-result.json"):
        if (directory / name).read_bytes() != archived[name]:
            raise ValueError("Live evaluator or model result changed after dependency review")
    if proof.get("evaluation_attempt_id") != evaluation["evaluation_attempt_id"]:
        raise ValueError("Reviewed evaluation attempt changed")
    _check_attempt(directory, evaluation, archived["evaluation.json"])
    log = (directory / "evaluate.log").read_bytes()
    if len(log) != proof.get("log_end"):
        raise ValueError("Evaluator log advanced after dependency review")
    selected = _log_slice(log, proof.get("log_start"), directory)
    if selected != archived["evaluate.log"]:
        raise ValueError("Reviewed evaluator log range changed")
    failures = _detect(episode["task_id"], evaluation, selected)
    if not failures or failures != proof.get("failures") or episode.get("benchmark_validity") != _validity(failures):
        raise ValueError("Dependency exclusion no longer matches its evidence")


def is_reviewed_evaluator_dependency_diagnostic(episode_path: Path) -> bool:
    """Fail closed unless immutable evidence still proves the narrow exclusion."""
    try:
        episode_path = Path(episode_path).resolve()
        _validate(_read_object(episode_path), episode_path)
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        return False
