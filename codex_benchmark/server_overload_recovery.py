"""Reviewed diagnostic grading for a saved terminal serverOverloaded failure.

Call mutating helpers only under the drained campaign account lock. They never
run a model, browser, or evaluator; grade_saved_episode is a separate caller step.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from uuid import UUID, uuid4

from .setup_recovery import _read_object, _save_exclusive, _sync_directory

KIND = "server_overloaded"
ERROR_MESSAGE = "Selected model is at capacity. Please try a different model."
_FILES = {
    "agent-result.json": "agent-result.json",
    "app-server.jsonl": "app-server.jsonl",
    "history.json": "history.json",
    "workspace.json": "workspace.json",
    "prompt.txt": "prompt.txt",
    "browser-actions.jsonl": "browser/actions.jsonl",
}
_GRADING_FIELDS = {"phase", "phase_started_at", "evaluation_error", "evaluation_deferred", "recovery"}
_PROOF_FIELDS = {"kind", "reviewed", "model_replayed", "prepared_at", "attempt_id",
                 "model_turn_status", "original_episode", "original_episode_sha256", "evidence_sha256"}
_REVIEW_FIELDS = {"reviewed_at", "diagnostic_report_sha256", "diagnostic_evaluation_attempt_id",
                  "diagnostic_journal_sha256"}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encoded(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _save_episode(path: Path, episode: dict) -> None:
    temporary = path.with_name(f".server-overload-recovery-{uuid4()}.tmp")
    _save_exclusive(temporary, _encoded(episode))
    temporary.replace(path)
    _sync_directory(path.parent)


def _payloads(directory: Path) -> dict[str, bytes]:
    return {name: (directory / relative).read_bytes() for name, relative in _FILES.items()}


def _validate_failure(episode: dict, payloads: dict[str, bytes]) -> dict:
    result = json.loads(payloads["agent-result.json"])
    error = result.get("error")
    if (episode.get("status") != "infrastructure_error"
            or episode.get("benchmark_validity", {}).get("status") != "excluded_infrastructure"
            or result.get("status") != "failed" or not isinstance(error, dict)
            or error.get("codexErrorInfo") != "serverOverloaded"
            or error.get("message") != ERROR_MESSAGE):
        raise ValueError("An excluded terminal serverOverloaded failure is required")
    expected_error = f"RuntimeError: Codex turn did not complete: {error}"
    if (episode.get("error") != expected_error
            or episode["benchmark_validity"].get("reason") != expected_error):
        raise ValueError("Episode failure differs from its saved serverOverloaded error")
    for key in ("task_id", "model", "effort", "workspace_id", "thread_id", "turn_id", "prompt_sha256"):
        if not isinstance(episode.get(key), str) or not episode[key]:
            raise ValueError("Saved failure identity is incomplete")
    for key in ("model", "effort", "thread_id", "turn_id", "tool_calls", "final_text", "token_usage"):
        if key not in result or episode.get(key) != result[key]:
            raise ValueError("Model checkpoint differs from episode identity or metrics")
    if type(result["tool_calls"]) is not int or result["tool_calls"] < 1:
        raise ValueError("Diagnostic recovery requires an attempted model turn")
    for key in ("events", "duration_seconds"):
        if key in result and episode.get(key) != result[key]:
            raise ValueError("Saved checkpoint metrics differ from the episode")
    records = [json.loads(line) for line in payloads["app-server.jsonl"].decode("utf-8").splitlines() if line.strip()]
    terminals = [r for r in records if r.get("direction") == "run_result"]
    if len(terminals) != 1 or any(r.get("direction") == "run_error" for r in records):
        raise ValueError("Exactly one terminal run_result and no run_error are required")
    terminal = terminals[0]["message"]
    if (not isinstance(terminal, dict) or terminal.get("status") != "failed"
            or any(result.get(key) != value for key, value in terminal.items())
            or set(result) - set(terminal) - {"events", "duration_seconds"}
            or any(key not in terminal for key in ("model", "effort", "thread_id", "turn_id", "status", "error", "tool_calls", "final_text", "token_usage"))):
        raise ValueError("Saved terminal result and agent checkpoint do not match")
    completed = [r for r in records if r.get("direction") == "receive"
                 and r.get("message", {}).get("method") == "turn/completed"]
    if len(completed) != 1:
        raise ValueError("Exactly one failed-turn confirmation is required")
    params = completed[0]["message"]["params"]
    turn = params.get("turn", {})
    if (params.get("threadId") != result["thread_id"] or turn.get("id") != result["turn_id"]
            or turn.get("status") != "failed" or turn.get("error") != error):
        raise ValueError("Failed-turn confirmation does not match the checkpoint")
    history = json.loads(payloads["history.json"])
    if (not isinstance(history, list) or not all(isinstance(url, str) for url in history)
            or history != episode.get("browsing_history")):
        raise ValueError("Saved browser history does not match the episode")
    workspace = json.loads(payloads["workspace.json"])
    if (workspace.get("verified") is not True or episode.get("verified") is not True
            or workspace.get("workspace_id") != episode["workspace_id"]
            or workspace.get("workspace_url") != episode.get("workspace_url")):
        raise ValueError("Verified workspace identity does not match the episode")
    if _sha(payloads["prompt.txt"]) != episode["prompt_sha256"]:
        raise ValueError("Saved prompt differs from the episode")
    actions = [json.loads(line) for line in payloads["browser-actions.jsonl"].decode("utf-8").splitlines() if line.strip()]
    if not actions or len(actions) != episode.get("browser_actions"):
        raise ValueError("Saved browser action count differs from the episode")
    return result


def prepare_server_overload_diagnostic(episode_path: Path, archive_dir: Path) -> dict:
    """Archive exact failure evidence before evaluation, preserving its status."""
    path = Path(episode_path)
    original_bytes = path.read_bytes()
    episode = json.loads(original_bytes)
    if episode.get("recovery"):
        raise ValueError("Recovery metadata already exists; refusing replacement")
    if any(path.parent.glob("evaluation*")) or path.with_name("evaluate.log").exists():
        raise ValueError("Existing evaluator evidence requires separate review")
    if episode.get("model") != path.parent.parent.name or episode.get("task_id") != path.parent.name:
        raise ValueError("Episode directory does not match task and model")
    payloads = _payloads(path.parent)
    _validate_failure(episode, payloads)
    attempt_id = str(uuid4())
    attempt = Path(archive_dir).resolve() / "server-overloaded-attempts" / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    _sync_directory(attempt.parent)
    _save_exclusive(attempt / "episode.json", original_bytes)
    for name, payload in payloads.items():
        _save_exclusive(attempt / name, payload)
    recovery = {"kind": KIND, "reviewed": False, "model_replayed": False,
                "prepared_at": datetime.now(timezone.utc).isoformat(), "attempt_id": attempt_id,
                "model_turn_status": "failed", "original_episode": str(attempt / "episode.json"),
                "original_episode_sha256": _sha(original_bytes),
                "evidence_sha256": {name: _sha(value) for name, value in payloads.items()}}
    proof_bytes = _encoded(recovery)
    _save_exclusive(attempt / "recovery.json", proof_bytes)
    recovery["proof_sha256"] = _sha(proof_bytes)
    if path.read_bytes() != original_bytes or _payloads(path.parent) != payloads:
        raise RuntimeError("Episode or terminal evidence changed while archiving")
    episode["recovery"] = recovery
    _save_episode(path, episode)
    return episode


def _validate_prepared(path: Path) -> dict:
    episode = _read_object(path)
    recovery = episode.get("recovery", {})
    if (recovery.get("kind") != KIND or recovery.get("model_replayed") is not False
            or recovery.get("model_turn_status") != "failed"):
        raise ValueError("Missing explicit serverOverloaded diagnostic recovery")
    attempt_id = str(UUID(recovery["attempt_id"]))
    original_path = Path(recovery["original_episode"])
    attempt = original_path.parent
    if (original_path.name != "episode.json" or attempt.name != attempt_id
            or attempt.parent.name != "server-overloaded-attempts"):
        raise ValueError("Invalid diagnostic archive identity")
    original_bytes = original_path.read_bytes()
    if _sha(original_bytes) != recovery.get("original_episode_sha256"):
        raise ValueError("Archived original episode changed")
    proof_bytes = (attempt / "recovery.json").read_bytes()
    archived_recovery = json.loads(proof_bytes)
    expected_keys = _PROOF_FIELDS | {"proof_sha256"}
    if recovery.get("reviewed") is True:
        expected_keys |= _REVIEW_FIELDS
    if (_sha(proof_bytes) != recovery.get("proof_sha256")
            or not isinstance(archived_recovery, dict) or set(archived_recovery) != _PROOF_FIELDS
            or archived_recovery.get("reviewed") is not False
            or type(recovery.get("reviewed")) is not bool or set(recovery) != expected_keys
            or any(recovery.get(key) != value for key, value in archived_recovery.items() if key != "reviewed")):
        raise ValueError("Recovery metadata differs from its archived proof")
    original = json.loads(original_bytes)
    if ({k: v for k, v in episode.items() if k not in _GRADING_FIELDS}
            != {k: v for k, v in original.items() if k not in _GRADING_FIELDS}):
        raise ValueError("Failure identity, status, exclusion or model evidence changed")
    if episode.get("model") != path.parent.parent.name or episode.get("task_id") != path.parent.name:
        raise ValueError("Episode directory identity changed")
    payloads = _payloads(path.parent)
    hashes = recovery.get("evidence_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_FILES):
        raise ValueError("Incomplete preserved evidence hashes")
    for name, payload in payloads.items():
        if _sha(payload) != hashes[name] or payload != (attempt / name).read_bytes():
            raise ValueError("Saved model checkpoint or diagnostic source evidence changed")
    _validate_failure(original, payloads)
    return episode


def _diagnostic_bytes(path: Path, episode: dict) -> bytes:
    payload = path.with_name("evaluation.json").read_bytes()
    report = json.loads(payload)
    if (episode.get("evaluation_error") or report.get("status") != "completed"
            or report.get("task_id") != episode["task_id"]
            or report.get("workspace_id") != episode["workspace_id"]
            or report.get("judge_errors")
            or type(report.get("reward")) not in (int, float)
            or not math.isfinite(report["reward"]) or not 0 <= report["reward"] <= 1):
        raise ValueError("A completed matching diagnostic evaluation is required")
    UUID(report["evaluation_attempt_id"])
    from .evaluator_dependency import _check_attempt
    _check_attempt(path.parent, report, payload)
    return payload


def mark_server_overload_diagnostic_reviewed(episode_path: Path) -> dict:
    """Bind a completed diagnostic report while retaining infrastructure exclusion."""
    path = Path(episode_path)
    episode = _validate_prepared(path)
    payload = _diagnostic_bytes(path, episode)
    episode["recovery"].update(reviewed=True, reviewed_at=datetime.now(timezone.utc).isoformat(),
                               diagnostic_report_sha256=_sha(payload),
                               diagnostic_evaluation_attempt_id=json.loads(payload)["evaluation_attempt_id"],
                               diagnostic_journal_sha256=_sha(path.with_name("evaluation-steps.jsonl").read_bytes()))
    _save_episode(path, episode)
    return episode


def is_reviewed_server_overload_diagnostic(episode_path: Path) -> bool:
    """Fail closed on modified, forged, mismatched or incomplete diagnostic proof."""
    try:
        path = Path(episode_path)
        episode = _validate_prepared(path)
        payload = _diagnostic_bytes(path, episode)
        recovery = episode["recovery"]
        return (recovery.get("reviewed") is True
                and recovery.get("diagnostic_report_sha256") == _sha(payload)
                and recovery.get("diagnostic_evaluation_attempt_id") == json.loads(payload)["evaluation_attempt_id"]
                and recovery.get("diagnostic_journal_sha256") == _sha(path.with_name("evaluation-steps.jsonl").read_bytes()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
