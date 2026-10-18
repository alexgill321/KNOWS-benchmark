"""Isolated episode workers sharing a durable, model-ordered queue."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .config import ROOT, apply_evaluator_environment, load_config, write_json
from .knows import discover_tasks
from .runner import SMOKE_TASK, check_judge_access, episode_schedule, run_episode, summarize, utcnow
from .worker_state import prepare_worker_config


def validate_contract(config: dict, run_dir: Path, tasks: list) -> dict:
    """Parallelism changes scheduling only; retain the established campaign."""
    from .browser import TOOLS
    manifest = json.loads((run_dir / "manifest.json").read_text())
    from .infrastructure_retry import RETRY_KIND, validate_retry_contract
    if manifest.get("kind") == RETRY_KIND:
        return validate_retry_contract(config, run_dir, tasks)
    from .post_main_retry import RETRY_KIND as POST_MAIN_RETRY_KIND, validate_retry_contract as validate_post_main_retry
    if manifest.get("kind") == POST_MAIN_RETRY_KIND:
        return validate_post_main_retry(config, run_dir, tasks)
    contract = {key: config.get(key) for key in
                ("models", "efforts", "judge_model", "max_actions", "episode_timeout_seconds")}
    contract.update(seed=config.get("seed", 0), benchmark_commit=subprocess.check_output(
        ["git", "-C", config["benchmark_root"], "rev-parse", "HEAD"], text=True).strip(),
        browser_tools_sha256=hashlib.sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest())
    if any(manifest.get(key) != value for key, value in contract.items()):
        raise RuntimeError("Existing campaign contract does not match; refusing changed settings")
    if (set(manifest["task_ids"]) != {task.task_id for task in tasks}
            or manifest.get("repetitions") != 1 or len(tasks) != manifest["tasks_per_model"]):
        raise RuntimeError("Existing campaign task set or repetition count changed")
    for model in config["models"]:
        report = json.loads((run_dir / model / SMOKE_TASK / "evaluation.json").read_text())
        if report.get("status") != "completed" or report.get("has_execution_errors"):
            raise RuntimeError("Every model's existing smoke evaluation must pass before parallel work")
    return manifest


async def preflight_models(config: dict, directory: Path) -> None:
    from .app_server import AppServer
    async with AppServer(log_path=directory / "preflight-app-server.jsonl") as server:
        models = (await server.request("model/list", {"includeHidden": True, "limit": 100}))["data"]
        catalog = {entry["model"]: entry for entry in models}
        for model in config["models"]:
            supported = [entry["reasoningEffort"] for entry in
                         catalog.get(model, {}).get("supportedReasoningEfforts", [])]
            if config["efforts"][model] not in supported:
                raise RuntimeError(f"Required model/effort unavailable: {model}")
        write_json(directory / "model-catalog.json", models)


def snapshot_sources(destination: Path) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source in sorted((ROOT / "codex_benchmark").glob("*.py")):
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, destination / source.name)
    revision = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    write_json(destination / "sha256.json", hashes)
    return revision


async def worker(config: dict, run_dir: Path, worker_id: str, coordinator_pid: int,
                 inherited_lock_fd: int) -> int:
    from .parallel_queue import claim_next, classify_existing, finish_claim, queue_pause_reason, queue_snapshot
    # The inherited account-lock descriptor survives a coordinator failure until
    # this worker exits. Never let an orphaned browser overlap a replacement run.
    os.fstat(inherited_lock_fd)
    if os.getppid() != coordinator_pid:
        raise RuntimeError("Worker must be launched by its coordinator")
    apply_evaluator_environment(config)
    db = run_dir / "parallel.sqlite"
    tasks = {task.task_id: task for task in discover_tasks(config["benchmark_root"])}
    state_path = run_dir / "workers" / worker_id / "state.json"
    with (Path(config["config_path"]).parent / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            # A lost coordinator never licenses new tasks. Finish an already
            # running episode so its result can still be safely reconciled.
            if os.getppid() != coordinator_pid:
                return 3
            claim = claim_next(db, worker_id, os.getpid())
            if claim is None:
                rows = queue_snapshot(db)
                if queue_pause_reason(db) or any(row["status"] == "blocked" for row in rows):
                    return 2
                if all(row["status"] == "done" for row in rows):
                    return 0
                await asyncio.sleep(3)
                continue
            model, task_id = claim["model"], claim["task_id"]
            state = {"worker_id": worker_id, "pid": os.getpid(), "model": model,
                     "task_id": task_id, "started_at": utcnow(), "state": "active"}
            write_json(state_path, state)
            print(f"{utcnow()} {worker_id} starting {model} {task_id}", flush=True)
            error = None
            try:
                await run_episode(config, tasks[task_id], model, run_dir)
                status = classify_existing(run_dir / model / task_id)
                if status != "done":
                    status, error = "blocked", "Episode requires infrastructure/evaluator inspection"
            except Exception as exc:
                status, error = "blocked", type(exc).__name__
            finish_claim(db, worker_id, model, task_id, status, error)
            state.update(state=status, finished_at=utcnow(), error=error)
            write_json(state_path, state)
            print(f"{utcnow()} {worker_id} {status} {model} {task_id}", flush=True)
            if status == "blocked":
                return 2


def coordinate(config: dict, run_dir: Path, count: int, account_lock_fd: int) -> int:
    from .parallel_health import inspect_episode
    from .parallel_queue import (finish_claim, initialize_queue, queue_pause_reason,
                                 queue_snapshot, set_queue_paused)
    apply_evaluator_environment(config)
    tasks = discover_tasks(config["benchmark_root"])
    manifest = validate_contract(config, run_dir, tasks)
    from .infrastructure_retry import RETRY_KIND, select_retry_tasks
    if manifest.get("kind") == RETRY_KIND:
        tasks = select_retry_tasks(config, run_dir, tasks)
    from .post_main_retry import RETRY_KIND as POST_MAIN_RETRY_KIND, select_retry_tasks as select_post_main_tasks
    if manifest.get("kind") == POST_MAIN_RETRY_KIND:
        tasks = select_post_main_tasks(config, run_dir, tasks)
    session_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    session_dir = run_dir / "parallel-sessions" / session_id
    session_dir.mkdir(parents=True)
    check_judge_access(config)
    asyncio.run(preflight_models(config, session_dir))
    db = run_dir / "parallel.sqlite"
    initialize_queue(db, [(task.task_id, model) for task, model in
                         episode_schedule(tasks, config["models"])], config["models"], run_dir)
    if queue_pause_reason(db) or any(row["status"] == "blocked" for row in queue_snapshot(db)):
        raise RuntimeError("Queue has a preserved pause or unresolved episode; inspect before resuming")
    revision = snapshot_sources(session_dir / "harness-source")
    prepared = []
    for number in range(1, count + 1):
        worker_id = f"worker-{number}"
        worker_config = prepare_worker_config(config, ROOT / "auth-data" / "parallel-workers" / worker_id)
        prepared.append((worker_id, worker_config))
    write_json(session_dir / "session.json", {"started_at": utcnow(), "workers": count,
        "model_order": config["models"], "scheduling": "unique atomic task claims, barrier between models",
        "models": manifest["models"], "efforts": manifest["efforts"], "judge_model": manifest["judge_model"],
        "max_actions": manifest["max_actions"], "episode_timeout_seconds": manifest["episode_timeout_seconds"],
        "seed": manifest["seed"], "repetitions": 1, "source_revision": revision,
        "browser_auth_guard": "interrupt and exclude actual Google sign-in redirects; preserve all evidence",
        "authorization": f"User requested {count} parallel instances with progress and sign-in monitoring"})
    (run_dir.parent / "campaign.pid").write_text(str(os.getpid()) + "\n")
    processes = {}
    with ExitStack() as stack:
        for worker_id, worker_config in prepared:
            directory = run_dir / "workers" / worker_id
            directory.mkdir(parents=True, exist_ok=True)
            log = stack.enter_context((directory / "worker.log").open("a"))
            process = subprocess.Popen([sys.executable, "-m", "codex_benchmark.parallel", "--config",
                worker_config["config_path"], "worker", "--run-dir", str(run_dir),
                "--worker-id", worker_id, "--coordinator-pid", str(os.getpid()),
                "--account-lock-fd", str(account_lock_fd)], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, pass_fds=(account_lock_fd,))
            processes[worker_id] = process
            (directory / "worker.pid").write_text(str(process.pid) + "\n")
        print(f"{utcnow()} {count}-worker campaign started; coordinator {os.getpid()}", flush=True)
        last_counts = None
        last_summary_at = 0.0
        summary = None
        while True:
            rows = queue_snapshot(db)
            workers = []
            for worker_id, process in processes.items():
                active = next((row for row in rows if row["status"] == "active"
                               and row["worker_id"] == worker_id), None)
                health = inspect_episode(run_dir / active["model"] / active["task_id"]) if active else None
                exit_code = process.poll()
                if exit_code is not None:
                    # Re-read after exit: the worker may have released its
                    # claim between this loop's snapshot and process.poll().
                    current = queue_snapshot(db)
                    orphan = next((row for row in current if row["status"] == "active"
                                   and row["worker_id"] == worker_id), None)
                    if orphan:
                        finish_claim(db, worker_id, orphan["model"], orphan["task_id"], "blocked",
                                     "Worker exited before releasing its claim; saved evidence needs inspection")
                        set_queue_paused(db, f"{worker_id} exited with an active task")
                    elif any(row["status"] != "done" for row in current) and not queue_pause_reason(db):
                        set_queue_paused(db, f"{worker_id} exited before the campaign finished")
                # A newly claimed saved terminal result gets a brief chance to
                # enter evaluation before its old finalization age is actionable.
                resuming_saved_result = (active and health and
                    health.get("reason") == "terminal_finalization_stalled" and
                    time.time() - datetime.fromisoformat(active["claimed_at"]).timestamp() < 30)
                if health and health.get("blocked") and not resuming_saved_result:
                    set_queue_paused(db, f"{worker_id}: {health.get('blocked_reason') or health.get('reason') or 'episode requires inspection'}")
                workers.append({"worker_id": worker_id, "pid": process.pid, "exit_code": exit_code,
                                "claim": active, "health": health})
            rows = queue_snapshot(db)
            if any(row["status"] == "blocked" for row in rows):
                set_queue_paused(db, queue_pause_reason(db) or "An episode needs infrastructure/evaluator inspection")
            counts = {status: sum(row["status"] == status for row in rows)
                      for status in ("pending", "active", "done", "blocked")}
            if counts != last_counts or time.monotonic() - last_summary_at >= 60:
                summary = summarize(run_dir)
                last_summary_at = time.monotonic()
            write_json(run_dir / "parallel-status.json", {"updated_at": utcnow(),
                "coordinator_pid": os.getpid(), "session_id": session_id, "workers": workers,
                "queue": counts, "paused_reason": queue_pause_reason(db), "models": summary["models"]})
            if counts != last_counts:
                print(f"{utcnow()} queue {json.dumps(counts)}", flush=True)
                last_counts = counts
            if all(process.poll() is not None for process in processes.values()):
                complete = counts["done"] == len(rows)
                print(f"{utcnow()} {'Campaign complete' if complete else 'Campaign needs attention'}", flush=True)
                return 0 if complete else 2
            time.sleep(10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmark.local.json")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    child = commands.add_parser("worker")
    child.add_argument("--run-dir", type=Path, required=True)
    child.add_argument("--worker-id", required=True)
    child.add_argument("--coordinator-pid", type=int, required=True)
    child.add_argument("--account-lock-fd", type=int, required=True)
    args = parser.parse_args()
    config, run_dir = load_config(args.config), args.run_dir.resolve()
    if args.command == "worker":
        return asyncio.run(worker(config, run_dir, args.worker_id, args.coordinator_pid, args.account_lock_fd))
    with (ROOT / ".codex-benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another campaign or surviving worker holds the account lock")
        return coordinate(config, run_dir, args.workers, lock.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
