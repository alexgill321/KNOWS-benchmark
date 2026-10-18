"""Serial, resumable episodes; smoke episodes are part of the full 330."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, apply_evaluator_environment, load_config, write_json
from .evaluation_log import save_evaluation_report
from .knows import build_prompt, discover_tasks

SMOKE_TASK = "knows.docs_1_formal_letter.1"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def recover_saved_episode(episode_path: Path, *, infrastructure_reason: str | None = None) -> dict:
    """Explicitly recover one inspected terminal turn; never run the model again.

    Call only while holding the account lock, after the old runner has stopped.
    The original episode and protocol logs remain available for audit.
    """
    episode_path = Path(episode_path)
    directory = episode_path.parent
    episode = json.loads(episode_path.read_text())
    if episode.get("status") != "running":
        raise RuntimeError("Only an interrupted running episode can be recovered")
    if (directory / "evaluation.json").exists() or (directory / "agent-result.json").exists():
        raise RuntimeError("Existing result/evaluation requires separate inspection; refusing overwrite")
    events = [json.loads(line) for line in (directory / "app-server.jsonl").read_text().splitlines() if line.strip()]
    terminal = [event for event in events if event.get("direction") == "run_result"]
    if len(terminal) != 1:
        raise RuntimeError("Recovery requires exactly one saved terminal model result")
    result = terminal[0]["message"]
    if (result.get("model") != episode["model"] or result.get("effort") != episode["effort"]
            or result.get("status") not in {"completed", "timeout", "action_limit"}
            or not result.get("thread_id") or not result.get("turn_id")):
        raise RuntimeError("Saved result identity or terminal status does not match the episode")
    if result["status"] == "completed":
        confirmed = any(
            event.get("direction") == "receive"
            and event.get("message", {}).get("method") == "turn/completed"
            and event["message"].get("params", {}).get("threadId") == result["thread_id"]
            and event["message"]["params"].get("turn", {}).get("id") == result["turn_id"]
            and event["message"]["params"]["turn"].get("status") == "completed"
            for event in events)
        if not confirmed:
            raise RuntimeError("Missing matching completed-turn confirmation")
    history = json.loads((directory / "history.json").read_text())
    if not isinstance(history, list) or not all(isinstance(url, str) for url in history):
        raise RuntimeError("Saved browsing history is invalid")
    completed_at = datetime.fromtimestamp(terminal[0]["time"], timezone.utc)
    original = directory / "recovery" / uuid.uuid4().hex / "episode.json"
    write_json(original, episode)
    write_json(directory / "agent-result.json", result)
    episode.update(result)
    episode.update(browsing_history=history, agent_finished_at=completed_at.isoformat(),
                   finished_at=completed_at.isoformat(), phase="ready_for_evaluation",
                   phase_started_at=utcnow())
    actions_path = directory / "browser" / "actions.jsonl"
    if actions_path.exists():
        episode["browser_actions"] = sum(bool(line.strip()) for line in actions_path.read_text().splitlines())
    if episode.get("started_at"):
        episode["agent_elapsed_seconds"] = (completed_at - datetime.fromisoformat(episode["started_at"])).total_seconds()
        episode["elapsed_seconds"] = episode["agent_elapsed_seconds"]
    episode["recovery"] = {"recovered_at": utcnow(), "reviewed": True,
                           "original_episode": str(original), "source": "app-server.jsonl:run_result",
                           "model_replayed": False,
                           "finalization_stall_seconds": (datetime.now(timezone.utc) - completed_at).total_seconds()}
    if infrastructure_reason:
        episode["benchmark_validity"] = {"status": "excluded_infrastructure", "reason": infrastructure_reason}
    write_json(episode_path, episode)
    return episode


def persistent_browser_failure(log_path: Path) -> dict | None:
    """Flag a terminal series of observation timeouts, not recovered UI errors."""
    if not log_path.exists():
        return None
    replies = {}
    for line in log_path.read_text().splitlines():
        event = json.loads(line)
        message = event.get("message", {})
        result = message.get("result", {})
        if event.get("direction") == "send" and isinstance(result, dict) and "contentItems" in result:
            replies[message["id"]] = result
    consecutive = 0
    for reply in reversed(list(replies.values())):
        text = "\n".join(item.get("text", "") for item in reply.get("contentItems", []) if isinstance(item, dict))
        if text.strip() == "Browser action budget exhausted.":
            continue  # App-server limit sentinel, not an executed browser observation.
        if "Observation failed: TimeoutError" not in text:
            break
        consecutive += 1
    if consecutive >= 3:
        return {"status": "excluded_infrastructure",
                "reason": f"Episode ended after {consecutive} consecutive browser-observation timeouts without recovery",
                "consecutive_observation_timeouts": consecutive}
    return None


def episode_schedule(tasks: list, models: list[str], smoke_only: bool = False) -> list:
    ordered = sorted(tasks, key=lambda task: (task.task_id != SMOKE_TASK, task.task_id))
    if smoke_only:
        ordered = [task for task in ordered if task.task_id == SMOKE_TASK]
    smoke = [(task, model) for task in ordered if task.task_id == SMOKE_TASK for model in models]
    remaining = [(task, model) for model in models for task in ordered if task.task_id != SMOKE_TASK]
    return smoke + remaining


def summarize(run_dir: Path) -> dict:
    from .evaluation_corrections import evaluation_for_scoring
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = []
    for path in sorted(run_dir.glob("*/*/episode.json")):
        episode = json.loads(path.read_text())
        evaluation_path = path.with_name("evaluation.json")
        evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
        correction = evaluation_for_scoring(path, evaluation)
        scoring_evaluation = correction["evaluation"]
        validity = episode.get("benchmark_validity", {})
        excluded = validity.get("status") == "excluded_infrastructure"
        rows.append({"task_id": episode["task_id"], "model": episode["model"],
                     "effort": episode["effort"], "status": episode["status"],
                     "evaluation_status": evaluation.get("status", "pending"),
                     "score_source": scoring_evaluation.get("score_source"),
                     "reward": None if excluded or correction["error"] else scoring_evaluation.get("reward"),
                     "diagnostic_reward": evaluation.get("reward") if excluded else None,
                     "benchmark_validity": validity,
                     "has_execution_errors": scoring_evaluation.get("has_execution_errors"),
                     "evaluation_correction": correction["provenance"],
                     "evaluation_correction_error": correction["error"],
                     "elapsed_seconds": episode.get("elapsed_seconds"),
                     "tool_calls": episode.get("tool_calls"),
                     "token_usage": episode.get("token_usage"),
                     "workspace_url": episode.get("workspace_url")})
    by_model = {}
    for model in manifest["models"]:
        records = [row for row in rows if row["model"] == model]
        scored = [row for row in records if row["evaluation_status"] == "completed"
                  and row["reward"] is not None]
        by_model[model] = {"attempted": len(records), "scored": len(scored),
                           "evaluated": sum(row["evaluation_status"] == "completed" for row in records),
                           "planned": manifest["tasks_per_model"],
                           "mean_reward": sum(row["reward"] for row in scored) / len(scored) if scored else None,
                           "perfect_tasks": sum(row["reward"] == 1 for row in scored),
                           "evaluator_error_tasks": sum(bool(row["has_execution_errors"]) for row in scored),
                           "evaluation_correction_errors": sum(bool(row["evaluation_correction_error"]) for row in records),
                           "infrastructure_errors": sum(row["status"] == "infrastructure_error" or
                               row["benchmark_validity"].get("status") == "excluded_infrastructure" for row in records)}
    result = {"updated_at": utcnow(), "run_dir": str(run_dir), "models": by_model, "episodes": rows}
    write_json(run_dir / "summary.json", result)
    return result


async def subprocess_logged(command: list[str], log: Path, timeout: float, env: dict | None = None) -> int:
    with log.open("a") as handle:
        process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=env,
            stdout=handle, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        try:
            return await asyncio.wait_for(process.wait(), timeout)
        except BaseException:
            import signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            raise


async def run_episode(config: dict, task, model: str, run_dir: Path, defer_evaluation: bool = False) -> dict:
    from .app_server import AppServer
    from .browser import BrowserSession
    directory = run_dir / model / task.task_id
    directory.mkdir(parents=True, exist_ok=True)
    episode_path = directory / "episode.json"
    if episode_path.exists():
        old = json.loads(episode_path.read_text())
        if old["status"] != "running":
            evaluation_path = directory / "evaluation.json"
            saved_evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
            needs_grading = (old.get("evaluation_deferred") or old.get("evaluation_error")
                             or saved_evaluation.get("status") != "completed")
            if old["status"] in {"completed", "timeout", "action_limit"} and needs_grading and not defer_evaluation:
                await grade_saved_episode(config, old, episode_path)
            return old
        raise RuntimeError(f"Interrupted episode requires inspection, not automatic replay: {episode_path}")
    episode = {"task_id": task.task_id, "model": model, "effort": config["efforts"][model],
               "benchmark_root": config["benchmark_root"], "started_at": utcnow(),
               "status": "running", "repetition": 1, "seed": config.get("seed", 0),
               "seed_scope": "run label only; app-server has no sampling-seed parameter",
               "max_actions": config["max_actions"], "timeout_seconds": config["episode_timeout_seconds"],
               "judge_model": config.get("judge_model"),
               "browsing_history": []}
    write_json(episode_path, episode)
    started = time.monotonic()
    try:
        if task.family in {"sheets_10_paper_sorting", "slides_17_removeimagesaddplaceholders"}:
            targets = directory / "run_targets.json"
            env = dict(os.environ, KNOWS_RUN_TARGETS=str(targets))
            # Every episode gets an immutable independent mapping, never a shared global file.
            command = [sys.executable, "-m", "codex_benchmark.targets", "--config", config["config_path"],
                       "--task", task.task_id, "--output", str(targets)]
            if not targets.exists():
                code = await subprocess_logged(command, directory / "provision-targets.log", 180, env)
                if code:
                    raise RuntimeError("Per-episode Drive targets could not be provisioned; see provision-targets.log")
            episode["run_targets_path"] = str(targets)
        from .knows import initial_workspace_url
        initial_url = initial_workspace_url(task, episode.get("run_targets_path"))
        if initial_url:
            from .knows import workspace_id_from_url
            episode.update(workspace_id=workspace_id_from_url(initial_url), workspace_url=initial_url)
        elif task.family == "slides_17_removeimagesaddplaceholders":
            episode.update(workspace_id=None, workspace_url=None)
        else:
            workspace = directory / "workspace.json"
            if not workspace.exists():
                code = await subprocess_logged([sys.executable, "-m", "codex_benchmark.provision",
                    "--config", config["config_path"], "--task", task.task_id,
                    "--model", model, "--output", str(workspace)], directory / "provision.log", 180)
                if code:
                    raise RuntimeError("Workspace provisioning failed; see provision.log")
            workspace_metadata = json.loads(workspace.read_text())
            if workspace_metadata.get("verified") is not True:
                raise RuntimeError("Existing workspace has not passed evaluator access verification")
            episode.update(workspace_metadata)
        from .post_main_retry import fresh_episode_server
        server_type = fresh_episode_server(config, run_dir, model, task.task_id, episode, AppServer)
        prompt = build_prompt(task, episode["workspace_url"], episode.get("run_targets_path"))
        (directory / "prompt.txt").write_text(prompt)
        episode["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        write_json(episode_path, episode)
        async with BrowserSession(storage_state=config["storage_state"], headless=config["headless"],
                channel=config["channel"], output_dir=directory / "browser", max_actions=config["max_actions"],
                timeout_ms=60000) as browser:
            if episode["workspace_url"]:
                await browser.page.goto(episode["workspace_url"], wait_until="domcontentloaded", timeout=60000)
            browser.assert_authenticated()
            try:
                async with server_type(log_path=directory / "app-server.jsonl") as server:
                    result = await server.run_turn(model=model, effort=episode["effort"], prompt=prompt,
                        dynamic_tools=browser.tools, tool_handler=browser.call,
                        timeout_seconds=config["episode_timeout_seconds"], max_tool_calls=config["max_actions"])
                    # Persist before either app-server or browser teardown can fail.
                    write_json(directory / "agent-result.json", result)
                    episode.update(result)
                    episode.update(agent_finished_at=utcnow(), phase="finalizing", phase_started_at=utcnow())
                    episode["browsing_history"] = list(browser.history)
                    episode["browser_actions"] = browser.action_count
                    write_json(directory / "history.json", episode["browsing_history"])
                    failure = persistent_browser_failure(directory / "app-server.jsonl")
                    if failure:
                        episode["benchmark_validity"] = failure
                    write_json(episode_path, episode)
                if result.get("status") in {"failed", "interrupted", "inProgress"}:
                    raise RuntimeError(f"Codex turn did not complete: {result.get('error') or result.get('status')}")
                if "accounts.google.com" in browser.page.url:
                    raise RuntimeError("Google browser session lost during episode")
            finally:
                episode["browsing_history"] = list(browser.history)
                episode["browser_actions"] = browser.action_count
                write_json(directory / "history.json", episode["browsing_history"])
                write_json(episode_path, episode)
                episode["storage_state_saved"] = await browser.save_storage_state(
                    config["storage_state"], timeout_seconds=config.get("finalization_timeout_seconds", 10))
                episode["browser_cleanup_errors"] = list(browser.cleanup_errors)
                write_json(episode_path, episode)
        episode["agent_elapsed_seconds"] = time.monotonic() - started
        episode["browser_cleanup_errors"] = list(browser.cleanup_errors)
        episode.update(phase="ready_for_evaluation", phase_started_at=utcnow())
        write_json(episode_path, episode)
        if defer_evaluation:
            episode["evaluation_deferred"] = True
            return episode
        await grade_saved_episode(config, episode, episode_path)
    except Exception as exc:
        if episode.get("status") not in {"completed", "timeout", "action_limit"}:
            episode["status"] = "infrastructure_error"
        episode["benchmark_validity"] = {"status": "excluded_infrastructure",
                                         "reason": f"{type(exc).__name__}: {exc}"}
        episode["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        episode["elapsed_seconds"] = time.monotonic() - started
        episode["finished_at"] = utcnow()
        write_json(episode_path, episode)
    return episode


async def grade_saved_episode(config: dict, episode: dict, episode_path: Path) -> None:
    directory = episode_path.parent
    evaluation_log = directory / "evaluate.log"
    log_start = evaluation_log.stat().st_size if evaluation_log.exists() else 0
    episode.update(phase="evaluating", phase_started_at=utcnow())
    write_json(episode_path, episode)
    try:
        code = await subprocess_logged([sys.executable, "-m", "codex_benchmark.evaluate",
            "--episode", str(episode_path), "--output", str(directory / "evaluation.json")],
            directory / "evaluate.log", config["evaluation_timeout_seconds"])
        if code:
            episode["evaluation_error"] = "Evaluator exited with error; see evaluation.json/evaluate.log"
        else:
            episode.pop("evaluation_error", None)
            episode.pop("evaluation_deferred", None)
    except Exception as exc:
        # A judge timeout does not invalidate or replay the model's completed work.
        episode["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        save_evaluation_report(directory / "evaluation.json", {"status": "evaluation_error", "reward": None,
            "error": episode["evaluation_error"], "task_id": episode["task_id"],
            "step_journal_path": str(directory / "evaluation-steps.jsonl")})
    evaluation_file = directory / "evaluation.json"
    if evaluation_file.exists():
        evaluation = json.loads(evaluation_file.read_text())
        if evaluation.get("workspace_id") and ".slides_" in episode["task_id"]:
            episode["workspace_id"] = evaluation["workspace_id"]
            episode["workspace_url"] = f"https://docs.google.com/presentation/d/{evaluation['workspace_id']}/edit"
        # Preserve the original diagnostic grade, but do not score a model on
        # reference data that a verified evaluator dependency could not supply.
        from .evaluator_dependency import apply_dependency_review
        # The strict review also reads episode.json; persist this attempt's
        # bookkeeping before it validates the saved terminal model evidence.
        write_json(episode_path, episode)
        try:
            apply_dependency_review(episode, episode_path, log_start=log_start)
        except Exception as exc:
            # Keep the completed report and model result intact, but withhold
            # scoring until finalization can verify the dependency evidence.
            episode["evaluation_error"] = f"Evaluator finalization failed: {type(exc).__name__}: {exc}"
    episode.update(phase="evaluation_error" if episode.get("evaluation_error") else "finished",
                   phase_started_at=utcnow())
    write_json(episode_path, episode)


async def run(config: dict, run_dir: Path, suite: str, limit_episodes: int | None = None,
              defer_evaluation: bool = False) -> None:
    from .app_server import AppServer
    from .parallel_queue import cancelled_claims_snapshot
    if cancelled_claims_snapshot(run_dir / "parallel.sqlite"):
        raise RuntimeError("This campaign has cancelled task claims; resume with the parallel coordinator to preserve its scope")
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(config["benchmark_root"])
    schedule = episode_schedule(tasks, config["models"], suite == "smoke")
    if limit_episodes:
        schedule = schedule[:limit_episodes]
    if not defer_evaluation:
        await asyncio.to_thread(check_judge_access, config)
    manifest_path = run_dir / "manifest.json"
    contract = {"models": config["models"], "efforts": config["efforts"],
                "judge_model": config.get("judge_model"),
                "max_actions": config["max_actions"], "episode_timeout_seconds": config["episode_timeout_seconds"],
                "seed": config.get("seed", 0), "benchmark_commit": subprocess.check_output(
                    ["git", "-C", config["benchmark_root"], "rev-parse", "HEAD"], text=True).strip()}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(key) != value for key, value in contract.items()):
            raise RuntimeError("Run settings changed; use a new run directory")
    else:
        manifest = {**contract, "created_at": utcnow(), "repetitions": 1, "tasks_per_model": 110,
                    "total_planned": 110 * len(config["models"]), "python": platform.python_version(),
                    "task_ids": [task.task_id for task in tasks], "browser_viewport": [1440, 900],
                    "agent_tools": "UI-only Playwright; screenshots and accessibility tree",
                    "model_cost": "ChatGPT subscription usage; no API dollar-cost estimate",
                    "judge_cost": "Google Gemini credentials (separate from ChatGPT subscription)"}
        write_json(manifest_path, manifest)
    from .browser import TOOLS
    tools_hash = hashlib.sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest()
    if manifest.get("browser_tools_sha256") not in (None, tools_hash):
        raise RuntimeError("Browser tools changed during this campaign; use a new run directory")
    if "browser_tools_sha256" not in manifest:
        manifest["browser_tools_sha256"] = tools_hash
        write_json(run_dir / "browser-tools.json", TOOLS)
        snapshot = run_dir / "harness-source"
        snapshot.mkdir(exist_ok=True)
        import shutil
        for source in (ROOT / "codex_benchmark").glob("*.py"):
            shutil.copy2(source, snapshot / source.name)
        write_json(manifest_path, manifest)
    # Validate all target models before creating any Google files.
    async with AppServer(log_path=run_dir / "preflight-app-server.jsonl") as server:
        models = (await server.request("model/list", {"includeHidden": True, "limit": 100}))["data"]
        catalog = {m["model"]: m for m in models}
        for model in config["models"]:
            if model not in catalog:
                raise RuntimeError(f"Required model unavailable: {model}")
            supported = [entry["reasoningEffort"] for entry in catalog[model]["supportedReasoningEfforts"]]
            if config["efforts"][model] not in supported:
                raise RuntimeError(f"Unsupported effort {config['efforts'][model]} for {model}: {supported}")
        write_json(run_dir / "model-catalog.json", models)
    for index, (task, model) in enumerate(schedule):
        print(f"[{index + 1}/{len(schedule)}] {model} {config['efforts'][model]} {task.task_id}", flush=True)
        episode = await run_episode(config, task, model, run_dir, defer_evaluation)
        summary = summarize(run_dir)
        print(f"  {episode['status']} ({episode.get('elapsed_seconds', 0):.1f}s)", flush=True)
        from .evaluator_dependency import is_reviewed_evaluator_dependency_diagnostic
        invalid_unreviewed = (episode.get("benchmark_validity", {}).get("status") == "excluded_infrastructure"
                              and not episode.get("recovery", {}).get("reviewed")
                              and not is_reviewed_evaluator_dependency_diagnostic(
                                  run_dir / model / task.task_id / "episode.json"))
        if episode["status"] == "infrastructure_error" or episode.get("evaluation_error") or invalid_unreviewed:
            raise RuntimeError(f"Stopped on infrastructure failure; see {run_dir / model / task.task_id}")
        if task.task_id == SMOKE_TASK and not defer_evaluation:
            evaluation_path = run_dir / model / task.task_id / "evaluation.json"
            evaluation = json.loads(evaluation_path.read_text())
            if evaluation.get("status") != "completed" or evaluation.get("has_execution_errors"):
                raise RuntimeError("Smoke evaluator reported execution errors; fix before the full suite")
    print(json.dumps(summarize(run_dir)["models"], indent=2), flush=True)


def check_judge_access(config: dict) -> None:
    """Reject inaccessible or unfunded judges before consuming model episodes."""
    from google import genai
    kwargs = {"http_options": {"timeout": 30000}}
    if os.environ.get("GOOGLE_AI_API_KEY"):
        kwargs["api_key"] = os.environ["GOOGLE_AI_API_KEY"]
    else:
        kwargs.update(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                      location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    client = genai.Client(**kwargs)
    model = config.get("judge_model") or "gemini-2.5-flash"
    try:
        # Match the evaluator's GenerateContent endpoint and default thinking.
        response = client.models.generate_content(model=model, contents="Reply with the word OK.")
        if not response.text:
            raise RuntimeError("Judge returned no text")
    except Exception as exc:
        raise RuntimeError(f"Judge preflight failed ({model}): {exc}") from exc
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmark.local.json")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--suite", choices=["smoke", "all"], default="smoke")
    run_parser.add_argument("--limit-episodes", type=int, help="Run only the first N scheduled episodes; resume in the same directory")
    run_parser.add_argument("--defer-evaluation", action="store_true", help="Smoke only: save agent artifacts while evaluator access is being repaired")
    run_parser.add_argument("--wait-for-lock", action="store_true", help="Queue behind the active serial smoke run")
    report_parser = sub.add_parser("report")
    report_parser.add_argument("run_dir", type=Path)
    sub.add_parser("list-tasks")
    recover_parser = sub.add_parser("recover", help="Recover one inspected terminal model result without replay")
    recover_parser.add_argument("--episode", required=True, type=Path)
    recover_parser.add_argument("--infrastructure-reason", help="Exclude the affected task from benchmark scores")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "report":
        print(json.dumps(summarize(args.run_dir.resolve())["models"], indent=2))
        return 0
    if args.command == "list-tasks":
        for task in discover_tasks(config["benchmark_root"]):
            print(task.task_id)
        return 0
    apply_evaluator_environment(config)
    if args.command == "run" and args.defer_evaluation and args.suite != "smoke":
        parser.error("Deferred evaluation is limited to smoke episodes; full campaigns require a working evaluator")
    with (ROOT / ".codex-benchmark.lock").open("w") as lock:
        try:
            if getattr(args, "wait_for_lock", False):
                print("Waiting for the active benchmark process to release the account lock…", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if getattr(args, "wait_for_lock", False) else fcntl.LOCK_NB))
        except BlockingIOError:
            parser.error("Another runner is active; Google-account episodes must run serially")
        try:
            if args.command == "recover":
                episode = recover_saved_episode(args.episode.resolve(), infrastructure_reason=args.infrastructure_reason)
                print(json.dumps({key: episode.get(key) for key in
                    ("task_id", "model", "status", "benchmark_validity", "recovery")}))
                return 0
            asyncio.run(run(config, args.run_dir.resolve(), args.suite, args.limit_episodes, args.defer_evaluation))
        except (Exception, KeyboardInterrupt) as exc:
            print(f"Stopped: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
