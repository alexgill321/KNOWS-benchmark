"""One explicit three-task Astra capacity retry; preserve the completed main run.

This manifest path is separate from the older nine-task Luna retry protocol.
Preparation reserves each model/task across all authorization plans. Validation
never creates workspaces, changes main claims, or interprets diagnostic scores
as replacement model grades.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from .infrastructure_retry import _digest_json, _json, _roots, _runtime_config, _sha

RETRY_KIND = "authorized_post_main_capacity_retry"
MODEL = "gpt-6-astra"
TASK_IDS = tuple(sorted(("knows.sheets_25_skitourplan.1", "knows.sheets_38_apartment_finder.2",
                         "knows.slides_29_buy_car_pres.1")))
AUTH_KIND = "user_authorized_post_main_capacity_retry"
_SETTINGS = ("judge_model", "max_actions", "episode_timeout_seconds", "seed")
ORIGINAL_FILES = ("episode.json", "evaluation.json", "agent-result.json", "app-server.jsonl",
                  "evaluation-steps.jsonl", "history.json", "workspace.json", "prompt.txt")
LUNA_RETRY_DIR = "luna-infrastructure-rerun-20261004"
LUNA_VALID_TASKS = ("knows.docs_31_education_lesson_plan.4", "knows.sheets_10_paper_sorting.3")


def _bound(path: Path, digest: str) -> dict:
    if _sha(path) != digest:
        raise ValueError("Post-main retry bound evidence changed")
    return _json(path)


def _descriptor(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha(path)}


def _archive_file(descriptor: dict, parent: Path) -> tuple[Path, dict]:
    path = Path(descriptor["path"])
    if not path.is_absolute() or not path.resolve().is_relative_to((parent.parent / "recovery").resolve()):
        raise ValueError("Post-main authority must be bound inside the recovery archive")
    return path.resolve(), _bound(path, descriptor["sha256"])


def _queue_state(parent: Path) -> dict:
    """Read the complete finished queue without opening a writable connection."""
    db = parent / "parallel.sqlite"
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        claims = [dict(r) for r in conn.execute("SELECT * FROM claims ORDER BY ordinal")]
        cancelled = [dict(r) for r in conn.execute("SELECT * FROM cancelled_claims ORDER BY ordinal")]
        metadata = [dict(r) for r in conn.execute("SELECT * FROM metadata ORDER BY key")]
    if (len(claims) != 221 or any(r["status"] != "done" for r in claims)
            or len(cancelled) != 109 or any(r["model"] != "gpt-6.1-sol" for r in cancelled)
            or any(r["key"] == "pause_reason" for r in metadata)):
        raise ValueError("Parent must remain the completed221-claim/109-cancellation campaign")
    return {"claims": claims, "cancelled_claims": cancelled, "metadata": metadata}


def _authorization(descriptor: dict, parent: Path) -> dict:
    _, value = _archive_file(descriptor, parent)
    ids = value.get("task_ids")
    if (value.get("kind") != AUTH_KIND or value.get("model") != MODEL or not isinstance(ids, list)
            or len(ids) != 3 or any(not isinstance(t, str) for t in ids) or set(ids) != set(TASK_IDS) or len(set(ids)) != 3
            or not isinstance(value.get("statement"), str) or not value["statement"].strip()):
        raise ValueError("Explicit authorization must name exactly the three Astra capacity retries")
    try:
        date = datetime.fromisoformat(value["requested_at"].replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError("Timezone missing")
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Authorization requires a timezone-aware request timestamp") from exc
    _, owner = _archive_file({"path": value["owner_authorization_path"],
                              "sha256": value["owner_authorization_sha256"]}, parent)
    if (value["statement"] != owner.get("owner_request") or value["requested_at"] != owner.get("recorded_at")
            or owner.get("max_total_workers") != 2 or owner.get("originals_preserved") is not True
            or value.get("max_total_workers") != 2 or value.get("no_main_resume") is not True):
        raise ValueError("Derived capacity scope must bind the unchanged owner request")
    return value


def _source_validation(descriptor: dict, parent: Path, root: Path) -> dict:
    _, value = _archive_file(descriptor, parent)
    sources = value.get("source_sha256")
    required = {"codex_benchmark/post_main_retry.py", "codex_benchmark/parallel.py",
                "codex_benchmark/infrastructure_retry.py", "codex_benchmark/worker_state.py",
                "codex_benchmark/runner.py", "codex_benchmark/app_server.py", "codex_benchmark/browser.py",
                "codex_benchmark/provision.py", "codex_benchmark/server_overload_recovery.py",
                "codex_benchmark/config.py", "codex_benchmark/parallel_queue.py",
                "tests/test_codex_post_main_retry.py",
                "codex-results/recovery/post-main-completion-20261005T213000Z/launch_capacity_retry.py"}
    if (value.get("kind") != "reviewed_post_main_retry_sources" or value.get("tests_passed") is not True
            or value.get("independent_review_passed") is not True or not isinstance(sources, dict)
            or not required.issubset(sources)):
        raise ValueError("Post-main retry requires reviewed frozen source bindings")
    _, review = _archive_file(value["independent_review"], parent)
    if review.get("approved") is not True or review.get("source_sha256") != sources:
        raise ValueError("Independent retry approval does not bind exact sources")
    for relative, digest in sources.items():
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root) or _sha(path) != digest:
            raise ValueError("Post-main retry source changed")
    return value


def _original(parent: Path, task: str, effort: str) -> dict:
    from .server_overload_recovery import is_reviewed_server_overload_diagnostic
    directory = parent / MODEL / task
    episode = _json(directory / "episode.json")
    if (episode.get("model") != MODEL or episode.get("task_id") != task
            or episode.get("effort") != effort or episode.get("status") != "infrastructure_error"
            or (episode.get("benchmark_validity") or {}).get("status") != "excluded_infrastructure"
            or episode.get("recovery", {}).get("kind") != "server_overloaded"
            or not is_reviewed_server_overload_diagnostic(directory / "episode.json")):
        raise ValueError("Only exact reviewed serverOverloaded original attempts may be retried")
    agent = _json(directory / "agent-result.json")
    if any(not isinstance(episode.get(k), str) or not episode[k] for k in ("workspace_id", "thread_id", "turn_id")):
        raise ValueError("Original workspace/thread/turn identity missing")
    if any(agent.get(k) != episode.get(k) for k in ("model", "effort", "thread_id", "turn_id")):
        raise ValueError("Original agent identity differs")
    return {"model": MODEL, "task_id": task, "workspace_id": episode["workspace_id"],
            "thread_id": episode["thread_id"], "turn_id": episode["turn_id"],
            "review_kind": "server_overloaded", "files_sha256": {n: _sha(directory / n) for n in ORIGINAL_FILES}}


def _old_luna(parent: Path) -> dict:
    retry = parent.parent / LUNA_RETRY_DIR
    reviewed = parent.parent / "luna-infrastructure-rerun-reviewed-summary.json"
    summary = _json(reviewed)
    entries = {e["task_id"]: e for e in summary["per_task"]}
    result = {"reviewed_summary_sha256": _sha(reviewed), "manifest_sha256": _sha(retry / "manifest.json"), "valid_grades": []}
    for task in LUNA_VALID_TASKS:
        entry = entries[task]
        directory = retry / "gpt-6-luna" / task
        if ((entry.get("benchmark_validity") or {}).get("status") != "eligible"
                or type(entry.get("scored_reward")) not in (int, float)
                or entry.get("evaluation_sha256") != _sha(directory / "evaluation.json")):
            raise ValueError("Prior valid Luna retry disposition changed")
        result["valid_grades"].append({"task_id": task, "scored_reward": entry["scored_reward"],
            "evaluation_attempt_id": entry["evaluation_attempt_id"],
            "files_sha256": {n: _sha(directory / n) for n in ("episode.json", "agent-result.json", "evaluation.json")}})
    return result


def _parent(config: dict, parent: Path, all_tasks: list) -> dict:
    from .parallel import validate_contract
    manifest = _json(parent / "manifest.json")
    if manifest.get("kind") in {RETRY_KIND, "authorized_infrastructure_retry"}:
        raise ValueError("Retry-of-retry campaign is not authorized")
    if config.get("models") != [MODEL] or config.get("efforts") != {MODEL: manifest.get("efforts", {}).get(MODEL)}:
        raise ValueError("Post-main retry must use exactly original Astra/effort")
    if (config.get("max_actions") != 150 or config.get("episode_timeout_seconds") != 1800
            or config.get("evaluation_timeout_seconds", 900) != 900):
        raise ValueError("Original150/1800/900 budgets must be retained")
    for key in _SETTINGS:
        if config.get(key, 0 if key == "seed" else None) != manifest.get(key):
            raise ValueError("Post-main retry changed original settings")
    validate_contract({**config, "models": manifest["models"], "efforts": manifest["efforts"]}, parent, all_tasks)
    ids = [t.task_id for t in all_tasks]
    if len(ids) != len(set(ids)) or not set(TASK_IDS).issubset(ids):
        raise ValueError("Retry tasks absent or duplicated in discovery")
    return manifest


def _registry(parent: Path) -> Path:
    path = parent.parent / "infrastructure-retry-reservations.sqlite"
    if path.is_symlink():
        raise ValueError("Retry registry cannot be a symlink")
    return path


def _rows(registry: Path) -> list[list]:
    with sqlite3.connect(registry.as_uri() + "?mode=ro", uri=True) as conn:
        return [list(row) for row in conn.execute("SELECT plan_id,model,task_id,run_dir,manifest_sha256 FROM reservations ORDER BY plan_id,model,task_id")]


def _historical_identities(parent: Path) -> dict:
    result = {k: [] for k in ("workspace_id", "thread_id", "turn_id")}
    for directory in (parent, parent.parent / LUNA_RETRY_DIR):
        for path in sorted(directory.glob("*/*/episode.json")):
            value = _json(path)
            for key in result:
                identity = value.get(key)
                if isinstance(identity, str) and identity:
                    result[key].append(identity)
    return {key: sorted(set(values)) for key, values in result.items()}


def _fresh_outputs(run: Path, manifest: dict) -> None:
    original_ids = {v for values in manifest["historical_identities"].values() for v in values}
    seen = {k: set() for k in ("workspace_id", "thread_id", "turn_id")}
    for episode_path in run.glob("*/*/episode.json"):
        episode = _json(episode_path)
        if (episode_path.parent.parent.name != MODEL or episode_path.parent.name not in TASK_IDS
                or episode.get("model") != MODEL or episode.get("task_id") != episode_path.parent.name):
            raise ValueError("Foreign retry episode in reserved run")
        for key in ("workspace_id", "thread_id", "turn_id"):
            value = episode.get(key)
            if value is not None:
                if not isinstance(value, str) or not value or value in original_ids or value in seen[key]:
                    raise ValueError("Retry reused historical or another retry workspace/model identity")
                seen[key].add(value)
        workspace = episode_path.with_name("workspace.json")
        if workspace.exists() and _json(workspace).get("workspace_id") in original_ids:
            raise ValueError("Retry workspace file reuses original")
    for task in TASK_IDS:
        directory = run / MODEL / task
        if directory.exists() and not (directory / "episode.json").exists() and any(directory.iterdir()):
            raise ValueError("Unowned retry task files predate its episode")


def _validate(config: dict, run: Path, all_tasks: list, manifest: dict, reservations=True) -> dict:
    run, parent = _roots(run, manifest.get("parent_run_dir", ""))
    if (manifest.get("kind") != RETRY_KIND or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or manifest.get("task_ids") != list(TASK_IDS)
            or manifest.get("models") != [MODEL] or manifest.get("repetitions") != 1
            or manifest.get("total_planned") != 3 or manifest.get("tasks_per_model") != 3):
        raise ValueError("Post-main retry manifest scope differs")
    auth = _authorization(manifest["authorization"], parent)
    _source_validation(manifest["source_validation"], parent, Path(config["benchmark_root"]).resolve())
    original = _parent(config, parent, all_tasks)
    if manifest.get("parent_manifest_sha256") != _sha(parent / "manifest.json"):
        raise ValueError("Original campaign manifest changed")
    expected = {"efforts": {MODEL: original["efforts"][MODEL]}, "benchmark_commit": original["benchmark_commit"],
                "browser_tools_sha256": original["browser_tools_sha256"], **{k: original[k] for k in _SETTINGS}}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Post-main manifest settings changed")
    if _digest_json(manifest.get("runtime_config", {})) != _digest_json(_runtime_config(config)):
        raise ValueError("Post-main runtime configuration changed")
    if manifest.get("parent_queue") != _queue_state(parent) or manifest.get("old_luna") != _old_luna(parent):
        raise ValueError("Completed main or valid Luna retry grades changed")
    originals = [_original(parent, task, original["efforts"][MODEL]) for task in TASK_IDS]
    if manifest.get("originals") != originals:
        raise ValueError("Original capacity evidence changed")
    if manifest.get("historical_identities") != _historical_identities(parent):
        raise ValueError("Historical model/workspace identities changed")
    plan = _digest_json({"kind": RETRY_KIND, "parent_run_dir": str(parent),
                         "parent_manifest_sha256": manifest["parent_manifest_sha256"], "authorization": auth})
    registry = _registry(parent)
    if manifest.get("reservation_plan_id") != plan or manifest.get("reservation_registry") != str(registry):
        raise ValueError("Post-main reservation identity changed")
    prior = manifest.get("prior_reservations")
    if (not isinstance(prior, list) or len(prior) != 3
            or any(not isinstance(row, list) or len(row) != 5 or row[1] != "gpt-6-luna" for row in prior)):
        raise ValueError("Existing three Luna reservations must remain preserved")
    if reservations:
        expected_rows = prior + [[plan, MODEL, task, str(run), _digest_json(manifest)] for task in TASK_IDS]
        if _rows(registry) != sorted(expected_rows):
            raise ValueError("Retry reservations missing, changed, overlapping, or foreign")
    _fresh_outputs(run, manifest)
    return manifest


def validate_retry_contract(config: dict, run_dir: Path, all_tasks: list) -> dict:
    return _validate(config, Path(run_dir), all_tasks, _json(Path(run_dir) / "manifest.json"))


def select_retry_tasks(config: dict, run_dir: Path, all_tasks: list) -> list:
    manifest = validate_retry_contract(config, run_dir, all_tasks)
    by_id = {t.task_id: t for t in all_tasks}
    return [by_id[t] for t in manifest["task_ids"]]


def fresh_episode_server(config: dict, run_dir: Path, model: str, task_id: str,
                         episode: dict, server_type: type) -> type:
    """Validate a freshly provisioned workspace and reject an old thread before turn/start.

    Ordinary campaigns return the unmodified server type. The retry-only adapter
    inspects responses; model arguments, messages and tool contracts are unchanged.
    """
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return server_type
    manifest = _json(manifest_path)
    if manifest.get("kind") != RETRY_KIND:
        return server_type
    from .knows import discover_tasks
    manifest = validate_retry_contract(config, run_dir, discover_tasks(config["benchmark_root"]))
    if model != MODEL or task_id not in TASK_IDS or episode.get("model") != model or episode.get("task_id") != task_id:
        raise ValueError("Unreserved retry episode identity")
    old_workspaces = set(manifest["historical_identities"]["workspace_id"])
    old_threads = set(manifest["historical_identities"]["thread_id"])
    for path in Path(run_dir).glob("*/*/episode.json"):
        if path.parent.name != task_id:
            saved = _json(path)
            old_workspaces.add(saved.get("workspace_id"))
            old_threads.add(saved.get("thread_id"))
    workspace = episode.get("workspace_id")
    if not isinstance(workspace, str) or not workspace or workspace in old_workspaces:
        raise ValueError("Retry requires a fresh workspace before its model turn")

    class FreshRetryServer(server_type):
        async def request(self, method, params, *args, **kwargs):
            response = await super().request(method, params, *args, **kwargs)
            if method == "thread/start":
                thread = response.get("thread", {}).get("id")
                if not isinstance(thread, str) or not thread or thread in old_threads:
                    raise ValueError("Retry requires a fresh thread before turn/start")
            return response
    return FreshRetryServer


def prepare_retry_manifest(config: dict, run_dir: Path, parent_run_dir: Path, all_tasks: list,
                           *, authorization_path: Path, validation_path: Path) -> dict:
    run, parent = _roots(run_dir, parent_run_dir)
    if run.exists() and any(run.iterdir()):
        raise ValueError("Post-main retry needs an empty fresh run directory")
    original = _parent(config, parent, all_tasks)
    auth_descriptor, validation_descriptor = _descriptor(authorization_path), _descriptor(validation_path)
    auth = _authorization(auth_descriptor, parent)
    manifest = {"schema_version": 1, "kind": RETRY_KIND, "parent_run_dir": str(parent),
                "parent_manifest_sha256": _sha(parent / "manifest.json"),
                "authorization": auth_descriptor, "source_validation": validation_descriptor,
                "models": [MODEL], "efforts": {MODEL: original["efforts"][MODEL]},
                "task_ids": list(TASK_IDS), "repetitions": 1, "tasks_per_model": 3, "total_planned": 3,
                "benchmark_commit": original["benchmark_commit"], "browser_tools_sha256": original["browser_tools_sha256"],
                **{k: original[k] for k in _SETTINGS}, "runtime_config": _runtime_config(config),
                "originals": [_original(parent, task, original["efforts"][MODEL]) for task in TASK_IDS],
                "parent_queue": _queue_state(parent), "old_luna": _old_luna(parent),
                "historical_identities": _historical_identities(parent)}
    registry = _registry(parent)
    if not registry.exists():
        raise ValueError("Original reservation registry must already exist")
    manifest.update(reservation_registry=str(registry), reservation_plan_id=_digest_json({"kind": RETRY_KIND,
        "parent_run_dir": str(parent), "parent_manifest_sha256": manifest["parent_manifest_sha256"], "authorization": auth}),
        prior_reservations=_rows(registry))
    _validate(config, run, all_tasks, manifest, reservations=False)
    run.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=run)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(manifest, handle, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        with sqlite3.connect(registry, timeout=30, isolation_level=None) as conn:
            conn.execute("PRAGMA synchronous=FULL"); conn.execute("BEGIN IMMEDIATE")
            try:
                current = [list(r) for r in conn.execute("SELECT plan_id,model,task_id,run_dir,manifest_sha256 FROM reservations ORDER BY plan_id,model,task_id")]
                if current != manifest["prior_reservations"]:
                    raise ValueError("Reservation registry changed during preparation")
                for task in TASK_IDS:
                    if conn.execute("SELECT 1 FROM reservations WHERE model=? AND task_id=?", (MODEL, task)).fetchone():
                        raise ValueError("Astra task already reserved under an earlier authorization; no duplicate replay")
                    conn.execute("INSERT INTO reservations VALUES(?,?,?,?,?)",
                                 (manifest["reservation_plan_id"], MODEL, task, str(run), _digest_json(manifest)))
                os.link(temporary, run / "manifest.json")
                conn.commit()
            except BaseException:
                conn.rollback(); raise
    finally:
        temporary.unlink(missing_ok=True)
    return manifest
