"""Run an authorized retry batch, then resume the preserved main queue."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .config import ROOT, load_config, write_json
from .parallel import coordinate, validate_contract
from .parallel_queue import clear_drained_queue_pause, queue_pause_reason, queue_snapshot
from .knows import discover_tasks
from .runner import utcnow


def _completed_hashes(run_dir: Path) -> dict[str, str]:
    hashes = {}
    for report in run_dir.glob("*/*/evaluation.json"):
        if json.loads(report.read_text()).get("status") != "completed":
            continue
        for filename in ("episode.json", "evaluation.json", "agent-result.json"):
            path = report.parent / filename
            if path.exists():
                hashes[str(path.relative_to(run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _verify_hashes(run_dir: Path, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        path = run_dir / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("An original completed record changed during the retry handoff")


def _require_drained(run_dir: Path, pause_reason: str) -> None:
    if queue_pause_reason(run_dir / "parallel.sqlite") != pause_reason:
        raise RuntimeError("Main queue pause changed; refusing to clear another pause")
    rows = queue_snapshot(run_dir / "parallel.sqlite")
    if any(row["status"] in {"active", "blocked"} for row in rows):
        raise RuntimeError("Main campaign has active or unresolved claims")


def handoff(state_path: Path, main_config: dict, retry_config: dict, lock_fd: int) -> int:
    """Caller owns the exclusive account lock across both two-worker campaigns."""
    os.fstat(lock_fd)
    state = json.loads(state_path.read_text())
    if state.get("phase") != "ready_for_retry" or state.get("max_total_workers") != 2:
        raise RuntimeError("Handoff is not prepared for a two-worker retry")
    parent, retry = Path(state["parent_run_dir"]), Path(state["retry_run_dir"])
    if parent.resolve() == retry.resolve():
        raise RuntimeError("Retry must preserve the original run directory")
    pause_reason = state["pause_reason"]
    _require_drained(parent, pause_reason)
    validate_contract(main_config, parent, discover_tasks(main_config["benchmark_root"]))
    manifest = validate_contract(retry_config, retry, discover_tasks(retry_config["benchmark_root"]))
    if manifest.get("kind") != "authorized_infrastructure_retry":
        raise RuntimeError("Handoff requires an authorized retry manifest")
    if Path(manifest["parent_run_dir"]).resolve() != parent.resolve():
        raise RuntimeError("Retry lineage does not match the main queue in the handoff")
    hashes = _completed_hashes(parent)
    write_json(Path(state["archive"]) / "completed-record-hashes-before-retry.json", hashes)

    def phase(name: str, **details) -> None:
        state.update(phase=name, updated_at=utcnow(), coordinator_pid=os.getpid(), **details)
        write_json(state_path, state)
        print(f"{utcnow()} handoff {name}", flush=True)

    try:
        phase("retry_running", active_run_dir=str(retry), retry_task_ids=manifest["task_ids"])
        code = coordinate(retry_config, retry, 2, lock_fd)
        _verify_hashes(parent, hashes)
        if code != 0:
            phase("retry_needs_attention", retry_exit_code=code)
            return code
        if any(row["status"] != "done" for row in queue_snapshot(retry / "parallel.sqlite")):
            raise RuntimeError("Retry coordinator returned without completing its claims")
        _require_drained(parent, pause_reason)
        phase("retry_completed_resuming_main", retry_exit_code=0, original_hashes_verified=len(hashes))
        # Both retry workers have exited before coordinate returns. The same
        # account lock remains held while worker configs return to the main run.
        clear_drained_queue_pause(parent / "parallel.sqlite", pause_reason)
        phase("main_running", active_run_dir=str(parent), retry_completed_at=utcnow())
        code = coordinate(main_config, parent, 2, lock_fd)
        phase("main_completed" if code == 0 else "main_needs_attention", main_exit_code=code)
        return code
    except Exception as exc:
        phase("handoff_failed", error_type=type(exc).__name__)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--main-config", type=Path, required=True)
    parser.add_argument("--retry-config", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT / ".codex-benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handoff(args.state.resolve(), load_config(args.main_config),
                       load_config(args.retry_config), lock.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
