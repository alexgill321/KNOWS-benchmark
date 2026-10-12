"""Explicit recovery of provisioning failures before any model turn started."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def validate_pre_agent_failure(episode_path: Path) -> dict:
    """Reject recovery unless the saved failure has no model/evaluator evidence."""
    episode_path = Path(episode_path)
    episode = _read_object(episode_path)
    if (episode.get("status") != "infrastructure_error"
            or "Workspace provisioning failed" not in str(episode.get("error", ""))):
        raise ValueError("Recovery requires an infrastructure provisioning failure")
    if not all(isinstance(episode.get(key), str) and episode[key] for key in ("task_id", "model")):
        raise ValueError("Provisioning failure must identify its task and model")
    model_fields = ("thread_id", "turn_id", "prompt_sha256", "final_text", "token_usage",
                    "tool_calls", "browser_actions", "agent_finished_at", "evaluation_error",
                    "evaluation_deferred", "evaluation")
    if any(key in episode for key in model_fields) or episode.get("browsing_history"):
        raise ValueError("Episode contains model or evaluator evidence; replay is forbidden")
    _require_no_agent_artifacts(episode_path.parent)
    return episode


def _require_no_agent_artifacts(directory: Path) -> None:
    for name in ("agent-result.json", "prompt.txt", "history.json", "evaluation.json",
                 "evaluation-steps.jsonl", "evaluate.log"):
        if (directory / name).exists():
            raise ValueError(f"Existing {name} prevents pre-agent recovery")
    log = directory / "app-server.jsonl"
    if log.exists() and (not log.is_file() or log.stat().st_size):
        raise ValueError("Nonempty app-server log prevents pre-agent recovery")
    for name in ("browser", "evaluation-attempts"):
        path = directory / name
        if path.exists() and (not path.is_dir() or any(item.is_file() for item in path.rglob("*"))):
            raise ValueError(f"Existing {name} artifacts prevent pre-agent recovery")


def _verified_workspace(directory: Path) -> dict:
    workspace = _read_object(directory / "workspace.json")
    workspace_id = workspace.get("workspace_id")
    if (workspace.get("verified") is not True
            or not isinstance(workspace_id, str) or not workspace_id
            or not isinstance(workspace.get("workspace_url"), str) or not workspace["workspace_url"]):
        raise ValueError("Recovery requires a verified workspace with its identity and URL")
    partial_path = directory / "partial-workspace.json"
    if partial_path.exists() and _read_object(partial_path).get("workspace_id") != workspace_id:
        raise ValueError("Verified workspace differs from the preserved partial workspace")
    return workspace


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def complete_setup_recovery(episode_path: Path, archive_dir: Path) -> dict:
    """Archive failed setup and allow the first model turn to start normally.

    The caller must hold the campaign account lock and stop/drain the original
    worker first. Workspace provisioning and verification must already be done.
    This function performs no Google, browser, evaluator, or model operations.
    """
    episode_path = Path(episode_path)
    episode = validate_pre_agent_failure(episode_path)
    original = episode_path.read_bytes()
    directory = episode_path.parent
    workspace_id = _verified_workspace(directory)["workspace_id"]
    if episode.get("workspace_id") and episode["workspace_id"] != workspace_id:
        raise ValueError("Verified workspace differs from the failed episode workspace")
    # Recheck after loading verification, before any archival or episode removal.
    if validate_pre_agent_failure(episode_path) != episode or episode_path.read_bytes() != original:
        raise RuntimeError("Episode changed while preparing setup recovery")
    attempt_id = str(uuid4())
    attempt = Path(archive_dir).resolve() / "setup-attempts" / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    _sync_directory(attempt.parent)
    archived_episode = attempt / "episode.json"
    _save_exclusive(archived_episode, original)
    record = {"setup_attempt_id": attempt_id, "timestamp": datetime.now(timezone.utc).isoformat(),
              "task_id": episode["task_id"], "model": episode["model"],
              "workspace_id": workspace_id, "model_replayed": False,
              "archive_path": str(archived_episode), "reason": episode["error"],
              "original_episode_sha256": hashlib.sha256(original).hexdigest()}
    payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    _save_exclusive(attempt / "setup-recovery.json", payload)
    temporary = directory / f".setup-recovery-{attempt_id}.tmp"
    _save_exclusive(temporary, payload)
    temporary.replace(directory / "setup-recovery.json")
    _sync_directory(directory)
    # Leave the failed episode in place if new model evidence appeared meanwhile.
    if validate_pre_agent_failure(episode_path) != episode or episode_path.read_bytes() != original:
        raise RuntimeError("Episode changed while archiving setup recovery")
    episode_path.unlink()
    _sync_directory(directory)
    return record


def reset_pre_agent_claim(db: Path, episode_path: Path) -> dict:
    """Reset one blocked setup claim after recovery, retaining the global pause.

    The caller must hold the account lock with all original workers drained.
    Receipt and archive evidence are checked again inside the queue transaction.
    """
    from .parallel_queue import _transaction

    episode_path = Path(episode_path)
    directory = episode_path.parent
    with _transaction(Path(db)) as connection:
        if connection.execute("SELECT 1 FROM claims WHERE status='active' LIMIT 1").fetchone():
            raise RuntimeError("All active workers must drain before setup claims can be reset")
        if episode_path.exists():
            raise ValueError("Setup recovery must archive and remove the failed episode first")
        _require_no_agent_artifacts(directory)
        receipt = _read_object(directory / "setup-recovery.json")
        if (receipt.get("model_replayed") is not False
                or receipt.get("task_id") != directory.name
                or receipt.get("model") != directory.parent.name
                or not receipt.get("timestamp")):
            raise ValueError("Invalid setup recovery receipt identity")
        archived = Path(receipt.get("archive_path", ""))
        if (archived.name != "episode.json" or archived.parent.name != receipt.get("setup_attempt_id")
                or archived.parent.parent.name != "setup-attempts"):
            raise ValueError("Invalid setup recovery archive location")
        if _read_object(archived.parent / "setup-recovery.json") != receipt:
            raise ValueError("Setup recovery receipt differs from the archived receipt")
        if hashlib.sha256(archived.read_bytes()).hexdigest() != receipt.get("original_episode_sha256"):
            raise ValueError("Archived failed episode does not match the recovery receipt hash")
        original = validate_pre_agent_failure(archived)
        if any(original.get(key) != receipt.get(key) for key in ("model", "task_id")) or original["error"] != receipt.get("reason"):
            raise ValueError("Archived failure identity differs from the recovery receipt")
        if _verified_workspace(directory)["workspace_id"] != receipt.get("workspace_id"):
            raise ValueError("Verified workspace differs from the recovery receipt")
        row = connection.execute("SELECT * FROM claims WHERE model=? AND task_id=?",
                                 (receipt["model"], receipt["task_id"])).fetchone()
        if row is None or row["status"] != "blocked":
            raise RuntimeError("Only the matching blocked setup claim can be reset")
        connection.execute("""UPDATE claims SET status='pending', worker_id=NULL,
            pid=NULL, claimed_at=NULL, error=NULL WHERE model=? AND task_id=?""",
                           (receipt["model"], receipt["task_id"]))
        return dict(connection.execute("SELECT * FROM claims WHERE model=? AND task_id=?",
                                       (receipt["model"], receipt["task_id"])).fetchone())
