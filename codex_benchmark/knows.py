"""Dependency-light KNOWS discovery and the isolated evaluator bridge.

Task selection and prompt building never import an evaluator or expose its
rubric to the agent. Call ``evaluate_task`` in a separate Python process;
upstream evaluators keep global Google clients and mutate temporary assets.
"""
from __future__ import annotations

import importlib.util
import importlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .evaluation_log import EvaluationJournal, save_evaluation_report


TASKS_RELATIVE = Path("src/browsergym/knows/eval/tasks")
PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
SEGMENTS = {"docs": "document", "sheets": "spreadsheets", "slides": "presentation"}
LABELS = {"docs": "Google Doc", "sheets": "Google Sheet", "slides": "Google Slides presentation"}


class MissingRequiredArtifact(ValueError):
    """An accessible per-episode output folder contains no required artifact."""

    def __init__(self, task_id: str, folder_url: str):
        self.folder_url = folder_url
        super().__init__(f"No copied presentation was produced in the verified output folder for {task_id}")


@dataclass(frozen=True)
class Task:
    family: str
    instance: int
    task_id: str
    path: Path
    kind: str

    @property
    def root(self) -> Path:
        return self.path.parents[6]

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "instance": self.instance,
                "task_id": self.task_id, "path": str(self.path), "kind": self.kind}


def discover_tasks(root: str | Path) -> list[Task]:
    """Find task prompts with matching evaluators, without importing either."""
    directory = Path(root).expanduser().resolve() / TASKS_RELATIVE
    if not directory.is_dir():
        raise FileNotFoundError(f"KNOWS tasks not found: {directory}")
    tasks = []
    for prompt in directory.glob("*/instance_*/task.md"):
        instance_dir, family = prompt.parent, prompt.parent.parent.name
        match = re.fullmatch(r"instance_(\d+)", instance_dir.name)
        kind = family.split("_", 1)[0]
        if match and kind in SEGMENTS and (instance_dir / "evaluator.py").is_file():
            instance = int(match.group(1))
            tasks.append(Task(family, instance, f"knows.{family}.{instance}", instance_dir, kind))
    return sorted(tasks, key=lambda task: (task.family, task.instance))


def find_task(root: str | Path, task_id: str) -> Task:
    for task in discover_tasks(root):
        if task.task_id == task_id:
            return task
    raise ValueError(f"Unknown KNOWS task: {task_id}")


def _target_path(task: Task, path: str | Path | None) -> Path:
    return Path(path or os.environ.get("KNOWS_RUN_TARGETS") or task.root / "run_targets.json").expanduser().resolve()


def get_run_target(task: Task, token: str, run_targets_path: str | Path | None = None) -> str:
    """Match upstream token precedence: environment override, then JSON file."""
    key = f"KNOWS_TARGET_{task.family}_INSTANCE_{task.instance}_{token}".upper()
    value = os.environ.get(key, "").strip()
    path = _target_path(task, run_targets_path)
    if not value:
        try:
            config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read run targets at {path}: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"Run targets at {path} must be a JSON object")
        try:
            value = config.get(task.family, {}).get(f"instance_{task.instance}", {}).get(token)
        except AttributeError as exc:
            raise ValueError(f"Malformed run targets for {task.task_id} in {path}") from exc
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Missing {token} for {task.task_id}. Provision fresh targets with: "
            f"python {task.root / TASKS_RELATIVE / 'provision_run_targets.py'} "
            f"--family {task.family} --instance {task.instance}; "
            f"then supply {path}, or set {key}."
        )
    return value


def resolve_task_prompt(task: Task, run_targets_path: str | Path | None = None) -> str:
    prompt = (task.path / "task.md").read_text(encoding="utf-8")
    return PLACEHOLDER.sub(lambda match: get_run_target(task, match.group(1), run_targets_path), prompt)


def initial_workspace_url(task: Task, run_targets_path: str | Path | None = None) -> str | None:
    """Return the pre-provisioned deck for the one task that edits an existing copy."""
    if task.family == "slides_17_removeimagesaddplaceholders" and task.instance == 1:
        url = get_run_target(task, "WORKING_COPY_URL", run_targets_path)
        workspace_id_from_url(url, "slides")
        return url
    return None


def workspace_id_from_url(url: str, kind: str | None = None) -> str:
    parsed = urlparse(url)
    expected = SEGMENTS[kind] if kind else r"(?:document|spreadsheets|presentation)"
    match = re.match(rf"/{expected}/d/([A-Za-z0-9_-]+)(?:/|$)", parsed.path)
    if parsed.scheme != "https" or parsed.hostname != "docs.google.com" or not match:
        raise ValueError(f"Expected a Google {kind or 'Workspace'} file URL; got {url!r}")
    return match.group(1)


def build_prompt(task: Task, workspace_url: str | None, run_targets_path: str | Path | None = None) -> str:
    """Preserve the task text and append the reference harness's workspace suffix."""
    prompt = resolve_task_prompt(task, run_targets_path)
    if task.family == "slides_17_removeimagesaddplaceholders":
        if task.instance != 1:
            return (prompt.rstrip() + "\n\n---\n"
                    "When you are finished, reply with 'DONE' and include the URL of the "
                    "presentation copy you created in the specified output folder.")
        target = initial_workspace_url(task, run_targets_path)
        if workspace_url and workspace_id_from_url(workspace_url, "slides") != workspace_id_from_url(target, "slides"):
            raise ValueError("slides_17 instance 1 must use the provisioned WORKING_COPY_URL as its workspace")
        return (prompt.rstrip() + "\n\n---\n"
                f"Continue this task in the existing Google Slides presentation. File URL: {target}\n"
                "When you are finished, reply with 'DONE' and include this URL in your message.")
    if not workspace_url:
        raise ValueError(f"A workspace URL is required for {task.task_id}")
    workspace_id_from_url(workspace_url, task.kind)
    return (
        prompt.rstrip() + "\n\n---\n"
        + f"A {LABELS[task.kind]} has already been created for you. "
        + "You MUST complete this task inside that file; "
        + "do NOT create a new file and do NOT rename it.\n"
        + f"File URL: {workspace_url}\n"
        + "When you are finished, reply with 'DONE' and include this URL in your message."
    )


def _list_output_presentations(folder_id: str, drive=None) -> list[dict]:
    if drive is None:
        from browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
        drive, _ = initialize_google_services(service_type="slides")
    if drive is None:
        raise RuntimeError("Cannot initialize Google Drive to locate the output presentation")
    # A query against an inaccessible/nonexistent parent can return an empty
    # list. Confirm access to the folder before treating emptiness as agent failure.
    folder = drive.files().get(fileId=folder_id, fields="id,mimeType,trashed",
                               supportsAllDrives=True).execute()
    if folder.get("mimeType") != "application/vnd.google-apps.folder" or folder.get("trashed"):
        raise RuntimeError("The configured output folder is unavailable or is not a folder")
    files, page_token = [], None
    while True:
        response = drive.files().list(
            q=f"'{folder_id}' in parents and trashed = false and mimeType = 'application/vnd.google-apps.presentation'",
            fields="files(id,name),nextPageToken", pageSize=100,
            pageToken=page_token, supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return files


def resolve_workspace_after_episode(task: Task, initial_id: str | None,
                                    run_targets_path: str | Path | None = None,
                                    final_text: str | None = None) -> str:
    """Select the actual edited artifact, rejecting missing or ambiguous copies."""
    if task.family != "slides_17_removeimagesaddplaceholders":
        if not initial_id:
            raise ValueError(f"workspace_id is required for {task.task_id}")
        return initial_id
    if task.instance == 1:
        target = workspace_id_from_url(initial_workspace_url(task, run_targets_path), "slides")
        if initial_id and initial_id != target:
            raise ValueError("workspace_id does not match the provisioned WORKING_COPY_URL")
        return target
    folder_url = get_run_target(task, "OUTPUT_FOLDER_URL", run_targets_path)
    match = re.search(r"/folders/([A-Za-z0-9_-]+)(?:/|[?#]|$)", folder_url)
    if not match or urlparse(folder_url).hostname != "drive.google.com":
        raise ValueError("OUTPUT_FOLDER_URL must be a Google Drive folder URL")
    files = _list_output_presentations(match.group(1))
    if not files:
        raise MissingRequiredArtifact(task.task_id, folder_url)
    if len(files) > 1:
        mentioned = set(re.findall(r"https://docs\.google\.com/presentation/d/([A-Za-z0-9_-]+)(?=[/?#\s)\]>]|$)", final_text or ""))
        selected = {item["id"] for item in files} & mentioned
        if len(selected) == 1:
            return selected.pop()
        raise ValueError(f"Expected exactly one copied presentation in the fresh output folder for {task.task_id}; found {len(files)}")
    return files[0]["id"]


def _prepare_credentials(root: Path) -> None:
    """Fail before module import instead of silently grading without a judge."""
    for variable, filename in (("SERVICE_ACCOUNT_PATH", "service-account.json"),
                               ("TOKEN_PATH", "token.json"),
                               ("CLIENT_SECRETS_PATH", "credentials.json")):
        candidate = root / "auth-data" / filename
        if variable not in os.environ and candidate.is_file():
            os.environ[variable] = str(candidate)
        elif os.environ.get(variable):
            os.environ[variable] = str(Path(os.environ[variable]).expanduser().resolve())
    service_account = Path(os.environ.get("SERVICE_ACCOUNT_PATH", root / "auth-data/service-account.json"))
    oauth_token = Path(os.environ.get("TOKEN_PATH", root / "auth-data/token.json"))
    secret_manager = os.environ.get("GCP_PROJECT_ID") and os.environ.get("DRIVE_SA_SECRET_ID")
    if not service_account.is_file() and not oauth_token.is_file() and not secret_manager:
        raise RuntimeError("Evaluator credentials missing: configure SERVICE_ACCOUNT_PATH or complete Google OAuth setup to create TOKEN_PATH before grading.")
    if not os.environ.get("GOOGLE_AI_API_KEY") and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        raise RuntimeError("Judge credentials missing: set GOOGLE_AI_API_KEY, or GOOGLE_CLOUD_PROJECT with Vertex AI application-default credentials.")


def evaluator_kwargs(task: Task, grade_fn: Any, workspace_id: str,
                     history: list[str], client_doc_id: str | None = None) -> dict:
    parameters = inspect.signature(grade_fn).parameters
    available = {"workspace_doc_id": workspace_id, "browsing_history": history,
                 "browsing_history_list": history, "cached_models": None}
    if "client_doc_id" in parameters:
        if not client_doc_id:
            matches = re.findall(r"https://docs\.google\.com/document/d/([A-Za-z0-9_-]+)",
                                 (task.path / "task.md").read_text(encoding="utf-8"))
            if len(set(matches)) != 1:
                raise ValueError(f"Supply client_doc_id for {task.task_id}; cannot identify one client-list Doc in task.md")
            client_doc_id = matches[0]
        available["client_doc_id"] = client_doc_id
    kwargs = {key: value for key, value in available.items() if key in parameters}
    missing = [name for name, parameter in parameters.items()
               if parameter.default is inspect.Parameter.empty
               and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
               and name not in kwargs]
    if missing:
        raise ValueError(f"Unsupported required evaluator parameters: {', '.join(missing)}")
    return kwargs


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _strict_judges(override: str | None = None, models_module=None):
    """Detect swallowed judge outages and optionally use one recorded judge.

    Upstream returns an error string on an API exception. A checkpoint may
    mistake that for a negative judgement, so retain a separate failure ledger.
    """
    models = models_module or importlib.import_module("browsergym.knows.eval.eval_utils.models")
    original = dict(models._models)
    state = {"judge_model": override, "requested_judge_models": [],
             "actual_judge_models": {}, "judge_errors": []}
    actual_defaults = {"gemma-google-ai": "gemma-3-27b-it",
                       "gemini-2.5-flash-google-ai": "gemini-2.5-flash",
                       "gemini-3-flash-google-ai": "gemini-3-flash-preview"}

    def factory(name, original_factory):
        actual = override or actual_defaults.get(name, name)
        if name not in state["requested_judge_models"]:
            state["requested_judge_models"].append(name)
        state["actual_judge_models"][name] = actual
        try:
            query = models.gemma_google_ai(model_id=override) if override else original_factory()
        except Exception as exc:
            state["judge_errors"].append(f"{actual}: judge initialization failed ({type(exc).__name__})")
            raise

        def strict_query(messages):
            try:
                response = query(messages)
            except Exception as exc:
                state["judge_errors"].append(f"{actual}: judge request failed ({type(exc).__name__})")
                raise
            if response is None or (isinstance(response, str) and
                    (not response.strip() or response.strip().startswith("Error: Google GenAI API"))):
                error = f"{actual}: judge API returned no usable response"
                state["judge_errors"].append(error)
                raise RuntimeError(error)
            return response
        return strict_query

    for name, original_factory in original.items():
        if name.endswith("-google-ai"):
            models._models[name] = lambda name=name, fn=original_factory: factory(name, fn)
    try:
        yield state
    finally:
        models._models.clear()
        models._models.update(original)


def _prepare_evaluator_source(task: Task, source_path: Path) -> dict[str, Any]:
    """Repair one obsolete unused import only in the isolated evaluator copy."""
    original = source_path.read_bytes()
    effective = original
    repairs = []
    old_import = "from browsergym.knows.eval.eval_scripts.test.test_doc_to_images import convert_pdf_to_pngs"
    new_import = "from browsergym.knows.eval.eval_utils.image_utils import convert_pdf_to_pngs"
    if task.family == "docs_5_influential_papers" and task.instance in (1, 2, 3):
        lines = original.decode("utf-8").splitlines(keepends=True)
        matches = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == old_import]
        if len(matches) > 1:
            raise ValueError("Ambiguous obsolete PDF import in evaluator copy")
        if matches:
            index = matches[0]
            lines[index] = new_import + lines[index][len(old_import):]
            effective = "".join(lines).encode("utf-8")
            source_path.write_bytes(effective)
            repairs.append({"id": "docs5-pdf-import-compat-v1", "old_import": old_import,
                            "new_import": new_import})
    return {"original_sha256": hashlib.sha256(original).hexdigest(),
            "effective_sha256": hashlib.sha256(effective).hexdigest(), "repairs": repairs}


def evaluate_task(task: Task, workspace_id: str | None, history: list[str], output_path: str | Path,
                  *, run_targets_path: str | Path | None = None,
                  client_doc_id: str | None = None, judge_model: str | None = None,
                  final_text: str | None = None) -> dict:
    """Evaluate once in a worker process, persisting success or an explicit error.

    The copy protects gold files from evaluator cleanup and permits independent
    workers to grade the same task safely. Only the evaluator process sees gold.
    """
    output = Path(output_path).expanduser().resolve()
    report: dict[str, Any] = {"task_id": task.task_id, "workspace_id": workspace_id, "judge_model": judge_model}
    judge_state = None
    journal = None
    try:
        if workspace_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", workspace_id):
            raise ValueError("workspace_id must be a Google file ID")
        if not isinstance(history, list) or any(not isinstance(url, str) for url in history):
            raise ValueError("browsing_history must be a list of URL strings")
        _prepare_credentials(task.root)
        resolve_task_prompt(task, run_targets_path)
        os.environ["KNOWS_RUN_TARGETS"] = str(_target_path(task, run_targets_path))
        sys.path.insert(0, str(task.root / "src"))
        workspace_id = resolve_workspace_after_episode(task, workspace_id, run_targets_path, final_text)
        report["workspace_id"] = workspace_id
        journal = EvaluationJournal(output.with_name(output.stem + "-steps.jsonl"), task.task_id, workspace_id)
        with _strict_judges(judge_model) as judge_state, journal, tempfile.TemporaryDirectory(prefix="knows-eval-") as temporary:
            isolated = Path(temporary) / task.family / f"instance_{task.instance}"
            shutil.copytree(task.path, isolated)
            report["evaluator_source"] = _prepare_evaluator_source(task, isolated / "evaluator.py")
            spec = importlib.util.spec_from_file_location("_codex_knows_evaluator", isolated / "evaluator.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load evaluator for {task.task_id}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # All shipped evaluators use resolve_task_dir(__file__). Re-root
            # legacy absolute constants as well, for compatible older releases.
            legacy = getattr(module, "TASK_DIR", None)
            if isinstance(legacy, str):
                old_root = Path(legacy).resolve()
                for key, value in list(vars(module).items()):
                    if isinstance(value, str) and value.startswith(str(old_root) + os.sep):
                        suffix = os.sep if value.endswith(os.sep) else ""
                        setattr(module, key, str(isolated / Path(value).relative_to(old_root)) + suffix)
                module.TASK_DIR = str(isolated) + os.sep
            result = module.grade_checkpoints(**evaluator_kwargs(
                task, module.grade_checkpoints, workspace_id, history, client_doc_id))
            score = result.final_score
            if not isinstance(score, dict) or not isinstance(score.get("total"), (int, float)) or score["total"] <= 0:
                raise ValueError(f"Evaluator returned invalid final_score: {score!r}")
            points = score.get("result")
            if not isinstance(points, (int, float)) or not 0 <= points <= score["total"]:
                raise ValueError(f"Evaluator returned invalid score: {score!r}")
            categories = result.get_category_summary()
            detailed = result.get_detailed_report()
            fatal = any(checkpoint.get("name") == "Evaluation Error"
                        for checkpoint in detailed.get("checkpoints", []))
            report.update(status="evaluation_error" if fatal else "completed",
                          reward=None if fatal else points / score["total"],
                          score_source="official_evaluator", evaluator_invoked=True,
                          result=result.to_dict(), detailed_report=detailed,
                          category_summary=categories,
                          has_execution_errors=bool(categories.get("execution_error", {}).get("failed", 0)))
            if fatal:
                report["error"] = "The upstream evaluator returned an Evaluation Error checkpoint; inspect detailed_report."
    except MissingRequiredArtifact as exc:
        report.update(status="completed", reward=0.0, workspace_id=None,
                      score_source="missing_required_artifact", evaluator_invoked=False,
                      details=str(exc), output_folder_url=exc.folder_url,
                      folder_access_verified=True, result=None, category_summary={},
                      has_execution_errors=False, requested_judge_models=[],
                      actual_judge_models={}, judge_errors=[])
    except Exception as exc:
        report.update(status="evaluation_error", reward=None, error=f"{type(exc).__name__}: {exc}")
    if judge_state is not None:
        report.update(judge_state)
        if judge_state["judge_errors"]:
            report.update(status="evaluation_error", reward=None,
                          error="Judge API failed; this episode cannot receive a model score.")
    if journal is not None:
        report["step_journal"] = journal.report
    attempt_id = journal.attempt_id if journal else str(uuid4())
    report["evaluation_attempt_id"] = attempt_id
    save_evaluation_report(output, report)
    return report
