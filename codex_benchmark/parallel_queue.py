"""Durable work claims for one account-locked benchmark coordinator."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


TERMINAL = {"completed", "timeout", "action_limit"}


def classify_existing(directory: Path) -> str:
    """Classify saved evidence without replaying or modifying an episode."""
    directory = Path(directory)
    path = directory / "episode.json"
    if not path.exists():
        return "pending"
    try:
        episode = json.loads(path.read_text())
        if episode.get("status") not in TERMINAL:
            if episode.get("status") == "infrastructure_error" and episode.get("recovery", {}).get("kind") == "server_overloaded":
                from .server_overload_recovery import is_reviewed_server_overload_diagnostic
                if is_reviewed_server_overload_diagnostic(path):
                    return "done"
            if episode.get("status") == "infrastructure_error" and episode.get("recovery", {}).get("kind") == "auth_interrupted":
                from .auth_recovery import is_reviewed_auth_diagnostic
                if is_reviewed_auth_diagnostic(path):
                    return "done"
            return "blocked"
        if (episode.get("benchmark_validity", {}).get("status") == "excluded_infrastructure"
                and not episode.get("recovery", {}).get("reviewed")):
            from .evaluator_dependency import is_reviewed_evaluator_dependency_diagnostic
            if not is_reviewed_evaluator_dependency_diagnostic(path):
                return "blocked"
        evaluation_path = directory / "evaluation.json"
        if not evaluation_path.exists():
            return "pending"
        evaluation = json.loads(evaluation_path.read_text())
        return "done" if evaluation.get("status") == "completed" and not episode.get("evaluation_error") else "pending"
    except (OSError, ValueError, AttributeError, TypeError):
        return "blocked"


def _connect(db: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db), timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


@contextmanager
def _transaction(db: Path):
    connection = _connect(db)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _ensure_cancelled_table(connection: sqlite3.Connection) -> None:
    # Kept separate from claims so already-running workers with the original
    # four-status schema observe only work they can still execute.
    connection.execute("""CREATE TABLE IF NOT EXISTS cancelled_claims (
        model TEXT NOT NULL, task_id TEXT NOT NULL, ordinal INTEGER NOT NULL UNIQUE,
        model_rank INTEGER NOT NULL, reason TEXT NOT NULL, authorization TEXT NOT NULL,
        cancelled_at TEXT NOT NULL, original_claim_json TEXT NOT NULL,
        PRIMARY KEY (model, task_id))""")


def _cancelled_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = connection.execute("SELECT * FROM cancelled_claims ORDER BY ordinal").fetchall()
    for row in rows:
        try:
            original = json.loads(row["original_claim_json"])
            stamp = datetime.fromisoformat(row["cancelled_at"])
            valid = (isinstance(original, dict)
                     and {"model", "task_id", "ordinal", "model_rank", "status",
                          "worker_id", "pid", "claimed_at", "error"}.issubset(original)
                     and all(original.get(key) == row[key] for key in
                             ("model", "task_id", "ordinal", "model_rank"))
                     and original.get("status") == "pending"
                     and all(original.get(key) is None for key in
                             ("worker_id", "pid", "claimed_at", "error"))
                     and all(isinstance(row[key], str) and row[key].strip()
                             for key in ("reason", "authorization"))
                     and stamp.tzinfo is not None)
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            raise ValueError("Invalid archived cancellation evidence")
    return rows


def _require_unstarted_directory(directory: Path) -> None:
    """An unstarted task has no files, links, or unexplained saved directories."""
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("Saved task evidence prevents cancellation")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Saved task evidence prevents cancellation")


def cancel_pending_model(db: Path, run_dir: Path, model: str, *, reason: str,
                         authorization: str, expected_count: int | None = None) -> list[dict]:
    """Cancel only never-started pending claims, preserving evidence and live work.

    Explicit user authorization must be supplied by the caller. This transaction
    may run while other models have active claims: BEGIN IMMEDIATE serializes it
    with claim_next, so a target model already claimed is never cancelled. The
    caller must not initialize/reconcile the live queue as part of cancellation.
    No browser, model, process, task artifact, or pause is changed here.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A target model is required")
    if any(not isinstance(value, str) or not value.strip() for value in (reason, authorization)):
        raise ValueError("A reason and explicit user authorization are required")
    if expected_count is not None and (type(expected_count) is not int or expected_count <= 0):
        raise ValueError("Expected cancellation count must be a positive integer")
    db, run_dir = Path(db), Path(run_dir)
    if not db.is_file():
        raise ValueError("Cancellation requires an existing campaign queue")
    if db.resolve().parent != run_dir.resolve():
        raise ValueError("Cancellation evidence must belong to the queue's run directory")
    with _transaction(db) as connection:
        rows = connection.execute("SELECT * FROM claims WHERE model=? ORDER BY ordinal", (model,)).fetchall()
        if any(row["status"] in {"active", "blocked"} for row in rows):
            raise RuntimeError("Active or blocked target claims prevent cancellation")
        pending = [row for row in rows if row["status"] == "pending"]
        if not pending:
            raise ValueError("No unstarted pending claims exist for the target model")
        if expected_count is not None and len(pending) != expected_count:
            raise RuntimeError("Pending target count differs from the reviewed cancellation")
        for row in pending:
            if any(row[key] is not None for key in ("worker_id", "pid", "claimed_at", "error")):
                raise ValueError("Previously owned pending claim prevents cancellation")
            _require_unstarted_directory(run_dir / row["model"] / row["task_id"])
        _ensure_cancelled_table(connection)
        archived = _cancelled_rows(connection)
        live_identities = {(row["model"], row["task_id"]) for row in
                           connection.execute("SELECT model,task_id FROM claims")}
        live_ordinals = {row[0] for row in connection.execute("SELECT ordinal FROM claims")}
        if any((row["model"], row["task_id"]) in live_identities or row["ordinal"] in live_ordinals
               for row in archived):
            raise ValueError("Cancelled and live claims overlap")
        timestamp = datetime.now(timezone.utc).isoformat()
        result = []
        for row in pending:
            record = {key: row[key] for key in ("model", "task_id", "ordinal", "model_rank")}
            record.update(reason=reason, authorization=authorization, cancelled_at=timestamp,
                          original_claim_json=json.dumps(dict(row), sort_keys=True))
            connection.execute("""INSERT INTO cancelled_claims
                (model,task_id,ordinal,model_rank,reason,authorization,cancelled_at,original_claim_json)
                VALUES (:model,:task_id,:ordinal,:model_rank,:reason,:authorization,:cancelled_at,:original_claim_json)""",
                               record)
            deleted = connection.execute("DELETE FROM claims WHERE model=? AND task_id=? AND status='pending'",
                                         (row["model"], row["task_id"])).rowcount
            if deleted != 1:
                raise RuntimeError("Target claim changed during cancellation")
            result.append(record)
        return result


def cancelled_claims_snapshot(db: Path) -> list[dict]:
    """Read archived cancellations without creating or modifying a database."""
    db = Path(db)
    if not db.exists():
        return []
    if not db.is_file():
        raise ValueError("Campaign queue must be a database file")
    connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cancelled_claims'").fetchone():
            return []
        return [dict(row) for row in _cancelled_rows(connection)]
    finally:
        connection.close()


def initialize_queue(db: Path, schedule: list[tuple[str, str]], models: list[str], run_dir: Path) -> None:
    """Reconcile saved artifacts after the previous workers have released the lock.

    Durable done claims never become runnable. Stale active claims need terminal
    episode evidence before they can become evaluation-only pending work.
    """
    db, run_dir = Path(db), Path(run_dir)
    if len(set(models)) != len(models) or not models:
        raise ValueError("models must be a nonempty unique ordered list")
    if len(set(schedule)) != len(schedule) or any(model not in models for _, model in schedule):
        raise ValueError("schedule must contain unique task/model pairs from models")
    ranks = {model: rank for rank, model in enumerate(models)}
    planned = [(model, task_id, ordinal, ranks[model])
               for ordinal, (task_id, model) in enumerate(schedule)]
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(db)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
    finally:
        connection.close()
    with _transaction(db) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS claims (
            model TEXT NOT NULL, task_id TEXT NOT NULL, ordinal INTEGER NOT NULL UNIQUE,
            model_rank INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','active','done','blocked')),
            worker_id TEXT, pid INTEGER, claimed_at TEXT, error TEXT,
            PRIMARY KEY (model, task_id))""")
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        _ensure_cancelled_table(connection)
        previous = connection.execute("SELECT * FROM claims ORDER BY ordinal").fetchall()
        cancelled = _cancelled_rows(connection)
        previous_by_identity = {(row["model"], row["task_id"]): row for row in previous}
        cancelled_by_identity = {(row["model"], row["task_id"]): row for row in cancelled}
        if previous_by_identity.keys() & cancelled_by_identity.keys():
            raise ValueError("Cancelled and live claims overlap")
        combined = sorted([*previous, *cancelled], key=lambda row: row["ordinal"])
        if combined and [(row["model"], row["task_id"], row["ordinal"], row["model_rank"])
                         for row in combined] != planned:
            raise ValueError("Existing queue schedule or model order differs; refusing replacement")
        for row in cancelled:
            _require_unstarted_directory(run_dir / row["model"] / row["task_id"])
        for model, task_id, ordinal, model_rank in planned:
            if (model, task_id) in cancelled_by_identity:
                continue
            existing = previous_by_identity.get((model, task_id))
            if existing and existing["status"] == "done":
                continue
            directory = run_dir / model / task_id
            status = classify_existing(directory)
            if (existing and existing["status"] in {"active", "blocked"}
                    and not (directory / "episode.json").exists()):
                status = "blocked"
            error = ((existing["error"] if existing else None)
                     or "Saved episode or interrupted claim requires inspection") if status == "blocked" else None
            if existing:
                connection.execute("""UPDATE claims SET status=?, error=?,
                    worker_id=CASE WHEN ?='pending' THEN NULL ELSE worker_id END,
                    pid=CASE WHEN ?='pending' THEN NULL ELSE pid END,
                    claimed_at=CASE WHEN ?='pending' THEN NULL ELSE claimed_at END
                    WHERE model=? AND task_id=?""", (status, error, status, status, status, model, task_id))
            else:
                connection.execute("INSERT INTO claims(model,task_id,ordinal,model_rank,status,error) VALUES(?,?,?,?,?,?)",
                                   (model, task_id, ordinal, model_rank, status, error))


def claim_next(db: Path, worker_id: str, pid: int) -> dict | None:
    """Atomically claim one task behind the current model barrier."""
    if not worker_id or not isinstance(pid, int) or pid <= 0:
        raise ValueError("A worker identity and positive PID are required")
    with _transaction(db) as connection:
        if connection.execute("SELECT 1 FROM metadata WHERE key='pause_reason'").fetchone():
            return None
        if connection.execute("SELECT 1 FROM claims WHERE status='blocked' LIMIT 1").fetchone():
            return None
        rank = connection.execute("SELECT MIN(model_rank) FROM claims WHERE status!='done'").fetchone()[0]
        if rank is None:
            return None
        row = connection.execute("SELECT * FROM claims WHERE status='pending' AND model_rank=? ORDER BY ordinal LIMIT 1", (rank,)).fetchone()
        if row is None:
            return None
        connection.execute("UPDATE claims SET status='active', worker_id=?, pid=?, claimed_at=?, error=NULL WHERE model=? AND task_id=?",
                           (worker_id, pid, datetime.now(timezone.utc).isoformat(), row["model"], row["task_id"]))
        return dict(connection.execute("SELECT * FROM claims WHERE model=? AND task_id=?", (row["model"], row["task_id"])).fetchone())


def finish_claim(db: Path, worker_id: str, model: str, task_id: str, status: str, error: str | None = None) -> None:
    if status not in {"done", "blocked"}:
        raise ValueError("A claim can finish only as done or blocked")
    with _transaction(db) as connection:
        row = connection.execute("SELECT status,worker_id FROM claims WHERE model=? AND task_id=?", (model, task_id)).fetchone()
        if row is None or row["status"] != "active" or row["worker_id"] != worker_id:
            raise RuntimeError("Only the owner of an active claim can finish it")
        connection.execute("UPDATE claims SET status=?,error=? WHERE model=? AND task_id=?", (status, error, model, task_id))


def queue_snapshot(db: Path) -> list[dict]:
    connection = _connect(db)
    try:
        return [dict(row) for row in connection.execute("SELECT * FROM claims ORDER BY ordinal")]
    finally:
        connection.close()


def set_queue_paused(db: Path, reason: str | None) -> None:
    if reason is not None and not isinstance(reason, str):
        raise ValueError("Pause reason must be a string or None")
    with _transaction(db) as connection:
        if reason is None:
            connection.execute("DELETE FROM metadata WHERE key='pause_reason'")
        else:
            connection.execute("INSERT INTO metadata(key,value) VALUES('pause_reason',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (reason,))


def queue_pause_reason(db: Path) -> str | None:
    connection = _connect(db)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key='pause_reason'").fetchone()
        return row[0] if row else None
    finally:
        connection.close()


def clear_drained_queue_pause(db: Path, expected_reason: str) -> None:
    """Clear only the reviewed pause, atomically with the drained-queue check."""
    with _transaction(db) as connection:
        row = connection.execute("SELECT value FROM metadata WHERE key='pause_reason'").fetchone()
        if row is None or row[0] != expected_reason:
            raise RuntimeError("Main queue pause changed; refusing to clear another pause")
        if connection.execute("SELECT 1 FROM claims WHERE status IN ('active','blocked') LIMIT 1").fetchone():
            raise RuntimeError("Main campaign has active or unresolved claims")
        connection.execute("DELETE FROM metadata WHERE key='pause_reason'")
