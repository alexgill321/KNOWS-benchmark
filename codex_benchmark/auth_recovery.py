"""Preserve authentication-interrupted turns for excluded diagnostic grading."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .setup_recovery import _read_object, _save_exclusive, _sync_directory


KIND = "auth_interrupted"
_METRICS = ("thread_id", "turn_id", "tool_calls", "token_usage", "final_text")
_INVARIANTS = ("task_id", "model", "effort", "status", "error", "benchmark_validity",
               "workspace_id", "workspace_url", "browsing_history", "prompt_sha256",
               "run_targets_path", "seed", "repetition")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encoded(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _save_episode(path: Path, episode: dict) -> None:
    temporary = path.with_name(f".auth-recovery-{uuid4()}.tmp")
    _save_exclusive(temporary, _encoded(episode))
    temporary.replace(path)
    _sync_directory(path.parent)


def _require_auth_failure(episode: dict) -> None:
    if (episode.get("status") != "infrastructure_error"
            or episode.get("benchmark_validity", {}).get("status") != "excluded_infrastructure"
            or not str(episode.get("error", "")).startswith("BrowserAuthenticationError:")):
        raise ValueError("An excluded BrowserAuthenticationError episode is required")
    if not all(isinstance(episode.get(key), str) and episode[key]
               for key in ("task_id", "model", "effort", "workspace_id")):
        raise ValueError("Interrupted episode identity and workspace are required")
    history = episode.get("browsing_history")
    if not isinstance(history, list) or not all(isinstance(url, str) for url in history):
        raise ValueError("Interrupted episode browsing history is invalid")


def _interruption_evidence(log: bytes, episode: dict) -> dict:
    records = [json.loads(line) for line in log.decode("utf-8").splitlines() if line.strip()]
    errors = [record for record in records if record.get("direction") == "run_error"]
    if len(errors) != 1 or any(record.get("direction") == "run_result" for record in records):
        raise ValueError("Exactly one run_error and no terminal run_result are required")
    result = errors[0]["message"]
    if (any(result.get(key) != episode.get(key) for key in ("model", "effort"))
            or result.get("status") != "inProgress"
            or not result.get("thread_id") or not result.get("turn_id")):
        raise ValueError("Interrupted model snapshot does not match the episode")
    completions = [record for record in records if record.get("direction") == "receive"
                   and record.get("message", {}).get("method") == "turn/completed"]
    if len(completions) != 1:
        raise ValueError("Exactly one interrupted-turn confirmation is required")
    params = completions[0]["message"]["params"]
    if (params.get("threadId") != result["thread_id"]
            or params.get("turn", {}).get("id") != result["turn_id"]
            or params["turn"].get("status") != "interrupted"):
        raise ValueError("Turn confirmation does not match the interrupted snapshot")
    return {"run_error": errors[0], "turn_completion": completions[0]}


def prepare_interrupted_auth_diagnostic(episode_path: Path, archive_dir: Path) -> dict:
    """Preserve a stopped turn, without changing its infrastructure failure status.

    Caller must hold the account lock after all original workers have stopped.
    No model, browser, Google operation, or evaluator is invoked here.
    """
    episode_path = Path(episode_path)
    directory = episode_path.parent
    original_bytes = episode_path.read_bytes()
    episode = _read_object(episode_path)
    _require_auth_failure(episode)
    if episode.get("recovery"):
        raise ValueError("Episode already has recovery metadata; refusing replacement")
    if (directory / "agent-result.json").exists() or any(directory.glob("evaluation*")) or (directory / "evaluate.log").exists():
        raise ValueError("Existing model result or evaluator evidence requires separate inspection")
    log = (directory / "app-server.jsonl").read_bytes()
    evidence = _interruption_evidence(log, episode)
    saved_history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    if saved_history != episode["browsing_history"]:
        raise ValueError("Saved history differs from the interrupted episode")
    attempt = Path(archive_dir).resolve() / "auth-interrupted-attempts" / str(uuid4())
    attempt.mkdir(parents=True, exist_ok=False)
    _sync_directory(attempt.parent)
    original_path, snapshot_path = attempt / "episode.json", attempt / "interrupted-snapshot.json"
    snapshot_bytes = _encoded(evidence)
    _save_exclusive(original_path, original_bytes)
    _save_exclusive(snapshot_path, snapshot_bytes)
    recovery = {"kind": KIND, "reviewed": False, "model_replayed": False,
                "prepared_at": datetime.now(timezone.utc).isoformat(), "model_turn_status": "interrupted",
                "original_episode": str(original_path), "original_episode_sha256": _sha(original_bytes),
                "interrupted_snapshot": str(snapshot_path), "interrupted_snapshot_sha256": _sha(snapshot_bytes),
                "app_server_sha256": _sha(log)}
    _save_exclusive(attempt / "recovery.json", _encoded(recovery))
    if episode_path.read_bytes() != original_bytes or (directory / "app-server.jsonl").read_bytes() != log:
        raise RuntimeError("Episode or protocol evidence changed while archiving interruption")
    result = evidence["run_error"]["message"]
    episode.update({key: result[key] for key in _METRICS if key in result})
    episode["recovery"] = recovery
    _save_episode(episode_path, episode)
    return episode


def _validate_prepared(episode_path: Path) -> dict:
    episode = _read_object(episode_path)
    _require_auth_failure(episode)
    recovery = episode.get("recovery", {})
    if (recovery.get("kind") != KIND or recovery.get("model_replayed") is not False
            or recovery.get("model_turn_status") != "interrupted"):
        raise ValueError("Missing explicit authentication-interruption recovery")
    original_path = Path(recovery["original_episode"])
    original_bytes = original_path.read_bytes()
    snapshot_path = Path(recovery["interrupted_snapshot"])
    snapshot_bytes = snapshot_path.read_bytes()
    if (original_path.parent != snapshot_path.parent
            or _sha(original_bytes) != recovery.get("original_episode_sha256")
            or _sha(snapshot_bytes) != recovery.get("interrupted_snapshot_sha256")):
        raise ValueError("Archived authentication-interruption evidence has changed")
    archived_recovery = _read_object(original_path.parent / "recovery.json")
    if any(recovery.get(key) != value for key, value in archived_recovery.items() if key != "reviewed"):
        raise ValueError("Recovery proof differs from its archived record")
    original = json.loads(original_bytes)
    _require_auth_failure(original)
    if any(episode.get(key) != original.get(key) for key in _INVARIANTS):
        raise ValueError("Interrupted episode identity, history, or failure was changed")
    directory = episode_path.parent
    if (directory / "agent-result.json").exists():
        raise ValueError("An interrupted snapshot cannot also have a completed model result")
    log = (directory / "app-server.jsonl").read_bytes()
    if _sha(log) != recovery.get("app_server_sha256"):
        raise ValueError("Protocol evidence changed after interruption preservation")
    evidence = _interruption_evidence(log, original)
    if json.loads(snapshot_bytes) != evidence:
        raise ValueError("Archived interrupted snapshot differs from protocol evidence")
    result = evidence["run_error"]["message"]
    if any(episode.get(key) != result.get(key) for key in _METRICS):
        raise ValueError("Saved model metrics differ from the interrupted snapshot")
    if json.loads((directory / "history.json").read_text(encoding="utf-8")) != episode["browsing_history"]:
        raise ValueError("Interrupted browsing history changed after preservation")
    return episode


def _diagnostic_bytes(episode_path: Path, episode: dict) -> bytes:
    payload = episode_path.with_name("evaluation.json").read_bytes()
    report = json.loads(payload)
    if (episode.get("evaluation_error") or report.get("status") != "completed"
            or report.get("task_id") != episode["task_id"]
            or report.get("workspace_id") != episode["workspace_id"]):
        raise ValueError("A completed diagnostic report matching this artifact is required")
    return payload


def mark_auth_diagnostic_reviewed(episode_path: Path) -> dict:
    """Review completed diagnostic grading under the caller's drained account lock."""
    episode_path = Path(episode_path)
    episode = _validate_prepared(episode_path)
    payload = _diagnostic_bytes(episode_path, episode)
    episode["recovery"].update(reviewed=True, reviewed_at=datetime.now(timezone.utc).isoformat(),
                               diagnostic_report_sha256=_sha(payload))
    _save_episode(episode_path, episode)
    return episode


def is_reviewed_auth_diagnostic(episode_path: Path) -> bool:
    """Read-only proof check used by the queue; malformed evidence stays blocked."""
    try:
        episode_path = Path(episode_path)
        episode = _validate_prepared(episode_path)
        recovery = episode["recovery"]
        return (recovery.get("reviewed") is True
                and recovery.get("diagnostic_report_sha256") == _sha(_diagnostic_bytes(episode_path, episode)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
