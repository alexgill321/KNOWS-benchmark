"""Read reviewed Lookbook corrections without replacing any original record."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re

from .evaluation_corrections import _archive_path, _bound, _object, _sha, _validate_correction_entry
from .evaluator_dependency import _check_attempt
from .travel_date_corrections import _pending_original

REASON = "lookbook_uniqueness_download_scheduling"
ARCHIVE_NAME = "lookbook-uniqueness-20261005T185000Z"
PENDING_NAME = "lookbook-evaluation-review.json"
CORRECTIONS_NAME = "lookbook-evaluation-corrections.json"
TARGETS = frozenset((model, f"knows.slides_39_Personal_Lookbook_PaintColors.{n}")
                    for model, numbers in (("gpt-6-luna", (4, 5)), ("gpt-6-astra", (1, 2, 3, 4, 5)))
                    for n in numbers)
WHITE_TARGET = ("gpt-6-astra", "knows.slides_39_Personal_Lookbook_PaintColors.1")


def _entries(value, exact):
    if not isinstance(value, list):
        raise ValueError("Lookbook targets must be a list")
    result = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("Invalid Lookbook target")
        key = entry.get("model"), entry.get("task_id")
        if (key not in TARGETS or key in result or entry.get("reason") != REASON
                or entry.get("model_replayed") is not False):
            raise ValueError("Unapproved or duplicate Lookbook target")
        result[key] = entry
    if exact and set(result) != TARGETS:
        raise ValueError("Lookbook authority must retain the exact reviewed target set")
    return result


def _evidence(item, archive):
    path = _archive_path(item["path"], archive)
    return path, _bound(path, item["sha256"])


def _steps(report):
    result = {}
    for checkpoint in report.get("detailed_report", {}).get("checkpoints", []):
        for step in checkpoint.get("steps", []):
            key = checkpoint["name"], step["step_id"]
            if key in result:
                raise ValueError("Duplicate Lookbook rubric step")
            result[key] = step
    if len(result) != 19:
        raise ValueError("Lookbook evaluator requires all 19 rubric steps")
    return result


def _journal_steps(directory, report, payload):
    _check_attempt(directory, report, payload)
    expected = _steps(report)
    rows = [json.loads(line) for line in (directory / "evaluation-steps.jsonl").read_bytes().splitlines() if line.strip()]
    matching = [r for r in rows if r.get("attempt_id") == report["evaluation_attempt_id"]]
    actual = {}
    for row in matching:
        if row.get("event") == "step_completed":
            step = row.get("step", {})
            key = row.get("checkpoint_name"), step.get("step_id")
            if key in actual:
                raise ValueError("Duplicate Lookbook journal step")
            actual[key] = step
    if (actual != expected or report.get("step_journal", {}).get("completed_steps") != 19
            or sum(r.get("event") == "evaluation_started" for r in matching) != 1
            or sum(r.get("event") == "evaluation_finished" for r in matching) != 1):
        raise ValueError("Lookbook report and exact journal differ")


def _source(entry, original, directory, archive):
    proof = _object(_bound(_archive_path(entry["source_proof_path"], archive), entry["source_proof_sha256"]))
    if (proof.get("kind") != REASON or proof.get("verified") is not True
            or proof.get("unchanged_rubric") is not True or proof.get("model_replayed") is not False
            or any(proof.get(k) != entry[k] for k in ("model", "task_id", "workspace_id", "original_sha256"))):
        raise ValueError("Lookbook source proof identity differs")
    sources = proof.get("source_snapshots")
    required = {"utils.py", *(f"instance_{n}/evaluator.py" for n in range(1, 6))}
    if not isinstance(sources, list) or len(sources) != 6 or {s.get("relative_path") for s in sources} != required:
        raise ValueError("Lookbook proof requires all six frozen sources")
    for item in sources:
        _evidence(item, archive)
    originals = proof.get("original_snapshots")
    names = {*entry["original_sha256"], "evaluation-steps.jsonl", "evaluate.log",
             f"evaluation-attempts/{original['evaluation_attempt_id']}.json"}
    if not isinstance(originals, list) or len(originals) != len(names) or {s.get("relative_path") for s in originals} != names:
        raise ValueError("Lookbook archived original evidence is incomplete")
    for item in originals:
        _, payload = _evidence(item, archive)
        if _bound(directory / item["relative_path"], item["sha256"]) != payload:
            raise ValueError("Lookbook archived original differs from current saved evidence")
        expected = entry["original_sha256"].get(item["relative_path"])
        if expected is not None and expected != item["sha256"]:
            raise ValueError("Lookbook original hash lineage differs")
    raw = proof.get("raw_snapshots")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Lookbook saved defect evidence missing")
    paths = [_evidence(item, archive)[0] for item in raw]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate Lookbook raw evidence")
    review = _object(_bound(_archive_path(proof["independent_review_path"], archive), proof["independent_review_sha256"]))
    expected = dict(reviewed=True, uniqueness_scheduling_defect_verified=True, model_replayed=False,
                    model=entry["model"], task_id=entry["task_id"],
                    original_report_sha256=entry["original_sha256"]["evaluation.json"],
                    evaluation_attempt_id=original["evaluation_attempt_id"])
    if any(review.get(k) != v or (type(v) is bool and type(review.get(k)) is not bool)
           for k, v in expected.items()):
        raise ValueError("Lookbook defect lacks matching independent review")
    _journal_steps(directory, original, (directory / "evaluation.json").read_bytes())
    return proof


def _white_hold(entry, original, archive):
    proof = _object(_bound(_archive_path(entry["retrieval_hold_proof_path"], archive), entry["retrieval_hold_proof_sha256"]))
    expected = dict(kind="lookbook_white_image_retrieval_review", reviewed=True, model_replayed=False,
                    exact_browser_url_loaded=True, evaluator_fetch_failed=True, original_http_cause_proven=False,
                    model=entry["model"], task_id=entry["task_id"], workspace_id=entry["workspace_id"],
                    original_sha256=entry["original_sha256"], evaluation_attempt_id=original["evaluation_attempt_id"])
    if any(proof.get(k) != v or (type(v) is bool and type(proof.get(k)) is not bool)
           for k, v in expected.items()):
        raise ValueError("Lookbook White retrieval proof identity differs")
    url_sha = proof.get("exact_url_sha256")
    if not isinstance(url_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", url_sha):
        raise ValueError("White retrieval exact URL hash missing")
    evidence = proof["evidence"]
    _, actions = _evidence(evidence["browser_actions"], archive)
    _, navigations = _evidence(evidence["browser_navigations"], archive)
    shot, _ = _evidence(evidence["browser_screenshot"], archive)
    _, log = _evidence(evidence["evaluator_log"], archive)
    match = re.fullmatch(r"(\d{5})_(page_\d+)\.png", shot.name)
    action_rows = [json.loads(line) for line in actions.splitlines() if line.strip()]
    action = [row for row in action_rows if match and row.get("action") == int(match[1])]
    if len(action) != 1 or action[0].get("success") is not True:
        raise ValueError("White screenshot lacks a successful original action")
    navs = [json.loads(line) for line in navigations.splitlines() if line.strip()]
    navs = [row for row in navs if row.get("page_id") == match[2]
            and datetime.fromisoformat(row["timestamp"]) <= datetime.fromisoformat(action[0]["timestamp"])]
    if not navs or _sha(navs[-1].get("url", "").encode()) != url_sha:
        raise ValueError("White screenshot navigation does not match exact source")
    typed = [row for row in action_rows if row.get("tool") == "browser_type"
             and row.get("success") is True and row.get("action", 0) > action[0]["action"]]
    if not any(_sha(url.encode()) == url_sha for row in typed
               for url in re.findall(r'https?://[^\s<>\'\"]+', row.get("arguments", {}).get("text", ""))):
        raise ValueError("White exact source URL was not submitted in a successful typed action")
    prefix = "Wayback fallback failed for "
    if not any(prefix in line and _sha(line.split(prefix, 1)[1].split(": ", 1)[0].strip().encode()) == url_sha
               for line in log.decode().splitlines()):
        raise ValueError("White retrieval failure does not match the exact saved URL")
    _, independent_bytes = _evidence(proof["independent_review"], archive)
    independent = _object(independent_bytes)
    case = independent.get("independent_retrieval_evidence", {}).get("white_case", {})
    if (case.get("color") != "White" or case.get("exact_url_sha256") != url_sha
            or case.get("screenshot_visually_reviewed") is not True
            or case.get("screenshot", {}).get("sha256") != evidence["browser_screenshot"]["sha256"]):
        raise ValueError("White retrieval requires the exact independent visual review")
    return proof


def _reviews(entry, report, archive, white=False):
    _bound(_archive_path(entry["report_path"], archive).with_name("evaluate.log"), entry["evaluate_log_sha256"])
    for label in ("report_review", "independent_review"):
        review = _object(_bound(_archive_path(entry[label + "_path"], archive), entry[label + "_sha256"]))
        expected = dict(reviewed=True, scoring_valid=True, execution_errors_reviewed=True,
                        no_auth_or_judge_failure=True, no_unresolved_evaluator_error=True, model_replayed=False,
                        model=entry["model"], task_id=entry["task_id"], workspace_id=entry["workspace_id"],
                        evaluation_attempt_id=report["evaluation_attempt_id"], report_sha256=entry["report_sha256"],
                        journal_sha256=entry["journal_sha256"], evaluate_log_sha256=entry["evaluate_log_sha256"], scored_reward=report["reward"],
                        original_report_sha256=entry["original_sha256"]["evaluation.json"],
                        original_episode_sha256=entry["original_sha256"]["episode.json"])
        if white:
            expected["white_image_retrieval_resolved"] = True
        if label == "report_review":
            expected.update(independent_review_path=entry["independent_review_path"],
                            independent_review_sha256=entry["independent_review_sha256"])
        if (any(review.get(k) != v or (type(v) is bool and type(review.get(k)) is not bool)
                for k, v in expected.items()) or type(review.get("scored_reward")) not in (int, float)):
            raise ValueError("Lookbook correction requires exact root and independent scoring review")
    if (Path(entry["report_review_path"]).resolve() == Path(entry["independent_review_path"]).resolve()
            or entry["report_review_sha256"] == entry["independent_review_sha256"]):
        raise ValueError("Lookbook report reviews must be independently recorded")


def load_lookbook_correction(episode_path: Path, manifest_path: Path | None = None) -> dict | None:
    episode_path = Path(episode_path).resolve()
    directory = episode_path.parent
    key = directory.parent.name, directory.name
    if key not in TARGETS:
        return None
    run = directory.parent.parent
    archive = (run.parent / "recovery" / ARCHIVE_NAME).resolve()
    pending_path = run.parent / PENDING_NAME
    correction_path = (_archive_path(str(manifest_path), archive) if manifest_path is not None else run.parent / CORRECTIONS_NAME)
    if not pending_path.exists():
        if correction_path.exists():
            raise ValueError("Lookbook correction has no original review authority")
        return None
    pending_bytes = pending_path.read_bytes()
    pending = _object(pending_bytes)
    if (type(pending.get("version")) is not int or pending["version"] != 1
            or pending.get("kind") != "pending_lookbook_evaluation_corrections"
            or pending.get("run_dir") != str(run) or pending.get("model_replayed") is not False):
        raise ValueError("Invalid Lookbook pending review authority")
    entries = _entries(pending.get("targets"), True)
    originals = {}
    for target, entry in entries.items():
        originals[target] = _pending_original(run, target, entry)
        _source(entry, originals[target], run / target[0] / target[1], archive)
        if target == WHITE_TARGET:
            _white_hold(entry, originals[target], archive)
        elif any(k.startswith("retrieval_hold_") for k in entry):
            raise ValueError("White retrieval hold cannot target another Lookbook episode")
    if not correction_path.exists():
        raise ValueError("Pending saved-output Lookbook evaluator correction")
    manifest_bytes = correction_path.read_bytes()
    manifest = _object(manifest_bytes)
    if (type(manifest.get("version")) is not int or manifest["version"] != 1
            or manifest.get("kind") != "reviewed_lookbook_evaluation_corrections"
            or manifest.get("run_dir") != str(run) or manifest.get("model_replayed") is not False
            or manifest.get("pending_manifest_sha256") != _sha(pending_bytes)):
        raise ValueError("Lookbook correction replaced its pending authority")
    corrected = _entries(manifest.get("corrections"), False)
    for target, entry in corrected.items():
        fields = ("workspace_id", "original_sha256", "source_proof_path", "source_proof_sha256",
                  "retrieval_hold_proof_path", "retrieval_hold_proof_sha256")
        if any(entry.get(k) != entries[target].get(k) for k in fields):
            raise ValueError("Lookbook correction changed immutable original lineage")
    if key not in corrected:
        raise ValueError("White image retrieval remains under review" if key == WHITE_TARGET
                         else "Pending saved-output Lookbook evaluator correction")
    entry = corrected[key]
    if key == WHITE_TARGET:
        # This separate evidence must be explicitly reviewed before any future regrade.
        # The present recovery intentionally skips this target rather than probing again.
        raise ValueError("White image retrieval remains under review; no access resolution is authorized")
    correction = _validate_correction_entry(episode_path, entry, correction_path, manifest_bytes,
                                          key[0], key[1], REASON, int(key[1].rsplit(".", 1)[1]))
    report = correction["evaluation"]
    actual, original = _steps(report), _steps(originals[key])
    if (report.get("evaluator_invoked") is not True or report.get("evaluator_source", {}).get("repairs")
            or actual.keys() != original.keys()):
        raise ValueError("Lookbook correction requires the same complete official rubric")
    for step_key, step in actual.items():
        if (any(step.get(k) != original[step_key].get(k) for k in ("name", "max_score"))
                or type(step.get("score")) not in (int, float) or not math.isfinite(step["score"])
                or not 0 <= step["score"] <= step["max_score"]):
            raise ValueError("Lookbook rubric or finite score bounds changed")
    unique = [step for step in actual.values() if step["name"] == "Two Images Are Unique"]
    if (len(unique) != 1 or "download failed" in unique[0].get("details", "").lower()
            or unique[0].get("category") in ("execution_error", "dependency_not_evaluated")):
        raise ValueError("Lookbook uniqueness remains unevaluated")
    total = sum(step["max_score"] for step in actual.values())
    score = sum(step["score"] for step in actual.values())
    if total <= 0 or abs(report["reward"] - score / total) > 1e-12:
        raise ValueError("Lookbook corrected reward differs from its rubric scores")
    _journal_steps(Path(entry["report_path"]).parent, report, Path(entry["report_path"]).read_bytes())
    _reviews(entry, report, archive)
    correction["provenance"].update(pending_manifest_path=str(pending_path), pending_manifest_sha256=_sha(pending_bytes),
                                     **{k: entry[k] for k in ("report_review_path", "report_review_sha256",
                                                             "independent_review_path", "independent_review_sha256")})
    return correction
