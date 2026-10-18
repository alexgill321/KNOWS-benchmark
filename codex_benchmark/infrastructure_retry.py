"""Explicit, immutable lineage for the nine owner-authorized Luna retries.

This module never runs an episode, changes a queue, or replaces original grades.
A retry is a separate campaign with fresh episode directories.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

RETRY_KIND = "authorized_infrastructure_retry"
RETRY_MODEL = "gpt-6-luna"
AUTHORIZED_TASK_IDS = (
    "knows.docs_31_education_lesson_plan.4",
    "knows.docs_5_influential_papers.5",
    "knows.sheets_10_paper_sorting.3",
    "knows.sheets_45_Personal_WeddingPlanner_weddingcolorpallette.5",
    *(f"knows.sheets_55_Movie_Recommendation.{number}" for number in range(1, 6)),
)
_SETTINGS = ("judge_model", "max_actions", "episode_timeout_seconds", "seed")
_RUNTIME_DEFAULTS = {
    "evaluation_timeout_seconds": 900,
    "headless": True,
    "channel": "chrome",
    "finalization_timeout_seconds": 10,
    # BrowserSession currently receives its own fixed defaults from the runner.
    # Pin explicit optional config values too, without making them take effect.
    "browser_timeout_seconds": None,
    "browser_timeout_ms": None,
    "timeout_ms": None,
    "viewport": None,
}


def _runtime_config(config: dict) -> dict:
    """Pin preparation settings absent from the historical parent manifest."""
    return {key: deepcopy(config.get(key, default)) for key, default in _RUNTIME_DEFAULTS.items()}


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read retry evidence: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Retry evidence must be an object: {path.name}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _targets(value) -> list[str]:
    if (not isinstance(value, (list, tuple)) or len(value) != len(AUTHORIZED_TASK_IDS)
            or any(not isinstance(task, str) for task in value)
            or len(set(value)) != len(value) or set(value) != set(AUTHORIZED_TASK_IDS)):
        raise ValueError("Retry targets must be exactly the nine authorized Luna task IDs")
    return sorted(value)


def _scope(value) -> list[str]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(task, str) for task in value)
            or len(set(value)) != len(value) or not set(value).issubset(AUTHORIZED_TASK_IDS)):
        raise ValueError("Retry batch must be a nonempty unique subset of the nine authorized tasks")
    return sorted(value)


def _authorization(value, tasks: list[str]) -> dict:
    if (not isinstance(value, dict)
            or value.get("kind") != "user_authorized_infrastructure_retry"
            or value.get("model") != RETRY_MODEL
            or _targets(value.get("task_ids")) != tasks
            or not isinstance(value.get("statement"), str) or not value["statement"].strip()):
        raise ValueError("Explicit user authorization for these nine Luna retries is required")
    try:
        requested = datetime.fromisoformat(value["requested_at"].replace("Z", "+00:00"))
        if requested.tzinfo is None:
            raise ValueError("Authorization timestamp must include a timezone")
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Authorization requested_at must be a timezone-aware ISO timestamp") from exc
    normalized = deepcopy(value)
    normalized["task_ids"] = tasks
    return normalized


def _digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _reservation_identity(parent_dir: Path, manifest: dict) -> tuple[Path, str]:
    registry = parent_dir.parent / "infrastructure-retry-reservations.sqlite"
    if registry.is_symlink():
        raise ValueError("Retry reservation registry must not be a symbolic link")
    plan = _digest_json({"parent_run_dir": str(parent_dir),
                         "parent_manifest_sha256": manifest["parent_manifest_sha256"],
                         "authorization": manifest["authorization"]})
    return registry, plan


def _verify_reservations(run_dir: Path, parent_dir: Path, manifest: dict) -> None:
    registry, plan = _reservation_identity(parent_dir, manifest)
    if (manifest.get("reservation_registry") != str(registry)
            or manifest.get("reservation_plan_id") != plan):
        raise ValueError("Retry reservation identity differs from the authorized plan")
    try:
        with sqlite3.connect(registry.as_uri() + "?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT model,task_id,run_dir,manifest_sha256 FROM reservations WHERE plan_id=? AND run_dir=? ORDER BY task_id",
                (plan, str(run_dir))).fetchall()
    except sqlite3.Error as exc:
        raise ValueError("Retry reservation registry is unavailable") from exc
    expected = [(RETRY_MODEL, task, str(run_dir), _digest_json(manifest))
                for task in sorted(manifest["task_ids"])]
    if rows != expected:
        raise ValueError("Retry task reservations are missing, changed, or owned by another batch")


def _roots(run_dir: Path, parent_dir: Path) -> tuple[Path, Path]:
    raw = Path(run_dir).absolute()
    if raw.is_symlink():
        raise ValueError("Retry directory must not be a symbolic link")
    run_dir, parent_dir = raw.resolve(), Path(parent_dir).resolve()
    if (run_dir == parent_dir or run_dir.is_relative_to(parent_dir)
            or parent_dir.is_relative_to(run_dir)):
        raise ValueError("Retry and original campaign directories must be separate and non-nested")
    return run_dir, parent_dir


def _parent(config: dict, parent_dir: Path, all_tasks: list) -> dict:
    parent = _json(parent_dir / "manifest.json")
    if parent.get("kind") == RETRY_KIND:
        raise ValueError("A retry campaign cannot authorize another retry campaign")
    if config.get("models") != [RETRY_MODEL] or RETRY_MODEL not in parent.get("models", []):
        raise ValueError("Only Luna may run in this authorized retry campaign")
    if config.get("efforts", {}).get(RETRY_MODEL) != parent.get("efforts", {}).get(RETRY_MODEL):
        raise ValueError("Retry reasoning effort differs from the original campaign")
    for key in _SETTINGS:
        if config.get(key, 0 if key == "seed" else None) != parent.get(key):
            raise ValueError(f"Retry setting changed: {key}")
    # Preserve the established main-campaign contract and its real smoke gate.
    # Import inside the function so parallel.py can delegate its retry branch here.
    from .parallel import validate_contract
    parent_config = {**config, "models": parent["models"], "efforts": parent["efforts"]}
    validate_contract(parent_config, parent_dir, all_tasks)
    return parent


def _original(parent_dir: Path, task_id: str, parent: dict) -> dict:
    directory = parent_dir / RETRY_MODEL / task_id
    episode_path, report_path = directory / "episode.json", directory / "evaluation.json"
    episode, report = _json(episode_path), _json(report_path)
    if (episode.get("task_id") != task_id or episode.get("model") != RETRY_MODEL
            or episode.get("effort") != parent["efforts"][RETRY_MODEL]
            or episode.get("benchmark_validity", {}).get("status") != "excluded_infrastructure"):
        raise ValueError(f"Original task is not the matching excluded Luna episode: {task_id}")
    if (report.get("status") != "completed" or report.get("task_id") != task_id
            or report.get("workspace_id") != episode.get("workspace_id")
            or episode.get("evaluation_error")):
        raise ValueError(f"Original diagnostic evaluation is incomplete or mismatched: {task_id}")
    validity, recovery = episode.get("benchmark_validity", {}), episode.get("recovery", {})
    lineage = {"task_id": task_id, "model": RETRY_MODEL,
               "episode_sha256": _sha(episode_path), "evaluation_sha256": _sha(report_path)}
    if validity.get("kind") == "evaluator_dependency":
        from .evaluator_dependency import is_reviewed_evaluator_dependency_diagnostic
        if not is_reviewed_evaluator_dependency_diagnostic(episode_path):
            raise ValueError(f"Original evaluator dependency proof is invalid: {task_id}")
        lineage["review_kind"] = "evaluator_dependency"
    elif recovery.get("kind") == "auth_interrupted":
        from .auth_recovery import is_reviewed_auth_diagnostic
        if not is_reviewed_auth_diagnostic(episode_path):
            raise ValueError(f"Original authentication recovery proof is invalid: {task_id}")
        lineage["review_kind"] = "auth_interrupted"
    elif task_id == "knows.docs_31_education_lesson_plan.4":
        if (episode.get("status") != "completed" or recovery.get("reviewed") is not True
                or recovery.get("model_replayed") is not False
                or recovery.get("source") != "app-server.jsonl:run_result"):
            raise ValueError("Lesson-plan finalization recovery has not been reviewed")
        original_path = Path(recovery.get("original_episode", "")).resolve()
        if not original_path.is_relative_to((directory / "recovery").resolve()):
            raise ValueError("Lesson-plan recovery archive is outside its episode")
        original = _json(original_path)
        if (original.get("task_id") != task_id or original.get("model") != RETRY_MODEL
                or original.get("workspace_id") != episode.get("workspace_id")):
            raise ValueError("Lesson-plan recovery archive identity differs")
        try:
            terminals = [entry["message"] for line in (directory / "app-server.jsonl").read_text().splitlines()
                         if (entry := json.loads(line)).get("direction") == "run_result"]
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("Lesson-plan saved terminal result is unreadable") from exc
        fields = ("model", "effort", "thread_id", "turn_id", "status", "tool_calls", "final_text")
        if len(terminals) != 1 or any(episode.get(key) != terminals[0].get(key) for key in fields):
            raise ValueError("Lesson-plan saved terminal result differs from recovered episode")
        lineage.update(review_kind="reviewed_finalization_recovery",
                       recovery_original_sha256=_sha(original_path))
    else:
        raise ValueError(f"Original exclusion has no supported reviewed proof: {task_id}")
    for name in ("app-server.jsonl", "agent-result.json"):
        path = directory / name
        if path.exists():
            lineage[name.replace(".", "_").replace("-", "_") + "_sha256"] = _sha(path)
    return lineage


def _validate(config: dict, run_dir: Path, all_tasks: list, manifest: dict,
              *, check_reservations: bool = True) -> dict:
    if manifest.get("kind") != RETRY_KIND or manifest.get("schema_version") != 1:
        raise ValueError("Not a supported infrastructure retry manifest")
    run_dir, parent_dir = _roots(run_dir, manifest.get("parent_run_dir", ""))
    tasks = _scope(manifest.get("task_ids"))
    authorized = _targets(manifest.get("authorized_task_ids"))
    if manifest.get("authorization") != _authorization(manifest.get("authorization"), authorized):
        raise ValueError("Retry authorization must be normalized and immutable")
    parent_path = parent_dir / "manifest.json"
    if manifest.get("parent_manifest_sha256") != _sha(parent_path):
        raise ValueError("Original campaign manifest changed")
    parent = _parent(config, parent_dir, all_tasks)
    ids = [task.task_id for task in all_tasks]
    if len(ids) != len(set(ids)) or not set(authorized).issubset(ids):
        raise ValueError("Authorized retry task is absent or duplicated in task discovery")
    expected = {"models": [RETRY_MODEL], "efforts": {RETRY_MODEL: parent["efforts"][RETRY_MODEL]},
                "repetitions": 1, "tasks_per_model": len(tasks), "total_planned": len(tasks),
                "benchmark_commit": parent["benchmark_commit"],
                "browser_tools_sha256": parent["browser_tools_sha256"],
                **{key: parent[key] for key in _SETTINGS}}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Retry manifest contract changed: {key}")
    if _digest_json(manifest.get("runtime_config", {})) != _digest_json(_runtime_config(config)):
        raise ValueError("Retry runtime configuration changed after preparation")
    originals = [_original(parent_dir, task, parent) for task in authorized]
    if manifest.get("originals") != originals:
        raise ValueError("Original excluded episode, evaluation, or review evidence changed")
    if check_reservations:
        _verify_reservations(run_dir, parent_dir, manifest)
    return manifest


def validate_retry_contract(config: dict, run_dir: Path, all_tasks: list) -> dict:
    """Validate the exact retry scope, original settings, and immutable lineage."""
    return _validate(config, Path(run_dir), all_tasks, _json(Path(run_dir) / "manifest.json"))


def select_retry_tasks(config: dict, run_dir: Path, all_tasks: list) -> list:
    """Return only this batch's reserved task objects in stable task-ID order."""
    manifest = validate_retry_contract(config, run_dir, all_tasks)
    by_id = {task.task_id: task for task in all_tasks}
    return [by_id[task_id] for task_id in sorted(manifest["task_ids"])]


def prepare_retry_manifest(config: dict, run_dir: Path, parent_run_dir: Path, all_tasks: list,
                           *, task_ids: list[str], authorization: dict) -> dict:
    """Atomically create lineage in a fresh separate directory; never overwrite."""
    run_dir, parent_dir = _roots(run_dir, parent_run_dir)
    tasks = _scope(task_ids)
    authorized = sorted(AUTHORIZED_TASK_IDS)
    authorization = _authorization(authorization, authorized)
    parent = _parent(config, parent_dir, all_tasks)
    manifest = {"schema_version": 1, "kind": RETRY_KIND,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "parent_run_dir": str(parent_dir), "parent_manifest_sha256": _sha(parent_dir / "manifest.json"),
                "authorization": authorization, "models": [RETRY_MODEL],
                "efforts": {RETRY_MODEL: parent["efforts"][RETRY_MODEL]},
                "authorized_task_ids": authorized,
                "task_ids": tasks, "tasks_per_model": len(tasks), "total_planned": len(tasks), "repetitions": 1,
                "benchmark_commit": parent["benchmark_commit"],
                "browser_tools_sha256": parent["browser_tools_sha256"],
                **{key: parent[key] for key in _SETTINGS},
                "runtime_config": _runtime_config(config),
                "originals": [_original(parent_dir, task, parent) for task in authorized]}
    registry, plan = _reservation_identity(parent_dir, manifest)
    manifest.update(reservation_registry=str(registry), reservation_plan_id=plan)
    _validate(config, run_dir, all_tasks, manifest, check_reservations=False)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Retry manifest requires an empty new run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=run_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Reserve each task once for this authorization, including across batches.
        # A crash before COMMIT cannot leave a runnable manifest: validation also
        # requires the matching committed reservation rows.
        with sqlite3.connect(registry, timeout=30, isolation_level=None) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("""CREATE TABLE IF NOT EXISTS reservations (
                    plan_id TEXT NOT NULL, model TEXT NOT NULL, task_id TEXT NOT NULL,
                    run_dir TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
                    PRIMARY KEY(plan_id,model,task_id))""")
                for task in tasks:
                    connection.execute("INSERT INTO reservations VALUES(?,?,?,?,?)",
                                       (plan, RETRY_MODEL, task, str(run_dir), _digest_json(manifest)))
                os.link(temporary, run_dir / "manifest.json")
                connection.commit()
            except (sqlite3.IntegrityError, FileExistsError) as exc:
                connection.rollback()
                raise ValueError("Retry task is already reserved or manifest exists; refusing overlapping replay") from exc
            except BaseException:
                connection.rollback()
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return manifest
