"""Read one reviewed evaluator correction without replacing original records.

This is a specific parser-repair lineage, not a best-score selection mechanism.
The manifest and every bound input/output are local, read-only evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from uuid import UUID

from .evaluator_dependency import _check_attempt

MODEL = "gpt-6-luna"
TASK = "knows.sheets_25_skitourplan.4"
REASON = "skitourplan_hyperlink_extraction"
MANIFEST_NAME = "reviewed-evaluation-corrections.json"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _object(payload: bytes) -> dict:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Correction evidence must be an object")
    return value


def _bound(path: Path, expected: str) -> bytes:
    payload = path.read_bytes()
    if _sha(payload) != expected:
        raise ValueError("Correction evidence hash mismatch")
    return payload


def _archive_path(value: str, archive_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to(archive_root):
        raise ValueError("Correction evidence must be in the recovery archive")
    return path.resolve()


def load_reviewed_evaluation_correction(episode_path: Path) -> dict | None:
    """Return verified report/provenance; raise for a declared invalid correction."""
    episode_path = Path(episode_path).resolve()
    directory = episode_path.parent
    if (directory.parent.name, directory.name) != (MODEL, TASK):
        return None
    run = directory.parent.parent
    manifest_path = run.parent / MANIFEST_NAME
    if not manifest_path.exists():
        return None
    manifest_bytes = manifest_path.read_bytes()
    manifest = _object(manifest_bytes)
    entries = manifest.get("corrections")
    if (manifest.get("version") != 1 or manifest.get("kind") != "reviewed_evaluator_corrections"
            or manifest.get("run_dir") != str(run) or not isinstance(entries, list) or len(entries) != 1):
        raise ValueError("Invalid reviewed correction manifest scope")
    entry = entries[0]
    if (not isinstance(entry, dict) or entry.get("model") != MODEL or entry.get("task_id") != TASK
            or entry.get("reason") != REASON or entry.get("model_replayed") is not False):
        raise ValueError("Unapproved evaluator correction target or reason")
    return _validate_correction_entry(episode_path, entry, manifest_path, manifest_bytes,
                                      MODEL, TASK, REASON, 4)


def _validate_correction_entry(episode_path: Path, entry: dict, manifest_path: Path,
                               manifest_bytes: bytes, model: str, task: str,
                               reason: str, instance: int) -> dict:
    """Validate one entry after its caller has enforced an exact reviewed scope."""
    directory = episode_path.parent
    run = directory.parent.parent
    if (entry.get("model") != model or entry.get("task_id") != task
            or entry.get("reason") != reason or entry.get("model_replayed") is not False):
        raise ValueError("Correction identity differs from reviewed scope")
    instance_source = f"instance_{instance}/evaluator.py"
    hashes = entry.get("original_sha256")
    names = {"episode.json", "evaluation.json", "agent-result.json"}
    if not isinstance(hashes, dict) or set(hashes) != names:
        raise ValueError("Missing immutable original hashes")
    originals = {name: _object(_bound(directory / name, hashes[name])) for name in names}
    episode, original, agent = (originals[name] for name in ("episode.json", "evaluation.json", "agent-result.json"))
    if (episode.get("model") != model or episode.get("task_id") != task
            or episode.get("workspace_id") != entry.get("workspace_id")
            or not isinstance(episode.get("workspace_id"), str) or not episode["workspace_id"]
            or episode.get("status") not in {"completed", "action_limit", "timeout"}
            or episode.get("evaluation_error")
            or episode.get("benchmark_validity", {}).get("status") == "excluded_infrastructure"):
        raise ValueError("Correction requires the matching eligible terminal original")
    fields = ("model", "effort", "thread_id", "turn_id", "status", "tool_calls", "final_text", "token_usage")
    if (any(field not in agent or agent[field] != episode.get(field) for field in fields)
            or any(not isinstance(episode.get(field), str) or not episode[field]
                   for field in ("effort", "thread_id", "turn_id", "judge_model"))):
        raise ValueError("Original agent identity differs from saved episode")
    archive = (run.parent / "recovery").resolve()
    report_path = _archive_path(entry["report_path"], archive)
    if report_path.name != "evaluation.json":
        raise ValueError("Correction report requires a dedicated evaluation.json archive")
    report_bytes = _bound(report_path, entry["report_sha256"])
    report = _object(report_bytes)
    for evaluation in (original, report):
        reward = evaluation.get("reward")
        if (evaluation.get("status") != "completed" or evaluation.get("judge_errors")
                or evaluation.get("task_id") != task or evaluation.get("workspace_id") != episode["workspace_id"]
                or evaluation.get("score_source") != "official_evaluator"
                or evaluation.get("judge_model") != episode.get("judge_model")
                or type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1):
            raise ValueError("Invalid matching completed official evaluator report")
    attempt = report.get("evaluation_attempt_id")
    UUID(attempt)
    if (attempt != entry.get("evaluation_attempt_id") or attempt == original.get("evaluation_attempt_id")
            or report.get("step_journal", {}).get("attempt_id") != attempt):
        raise ValueError("Correction must have its own matching evaluator attempt")
    journal_path = report_path.with_name("evaluation-steps.jsonl")
    journal_bytes = _bound(journal_path, entry["journal_sha256"])
    if report["step_journal"].get("path") != str(journal_path):
        raise ValueError("Correction journal path is not its archived journal")
    _check_attempt(report_path.parent, report, report_bytes)
    records = [json.loads(line) for line in journal_bytes.splitlines() if line.strip()]
    expected_steps = sum(len(c.get("steps", [])) for c in report.get("detailed_report", {}).get("checkpoints", []))
    completed_steps = sum(r.get("event") == "step_completed" and r.get("attempt_id") == attempt for r in records)
    if not expected_steps or completed_steps != expected_steps or report["step_journal"].get("completed_steps") != expected_steps:
        raise ValueError("Correction rubric steps do not match the complete journal")
    proof_path = _archive_path(entry["source_proof_path"], archive)
    proof = _object(_bound(proof_path, entry["source_proof_sha256"]))
    if (proof.get("kind") != reason or proof.get("task_id") != task or proof.get("verified") is not True
            or proof.get("model_replayed") is not False or proof.get("unchanged_rubric") is not True
            or not isinstance(proof.get("source_snapshots"), list) or not proof["source_snapshots"]):
        raise ValueError("Missing reviewed parser repair source proof")
    snapshots = {}
    for source in proof["source_snapshots"]:
        relative = source.get("relative_path")
        if relative in snapshots or relative not in {"utils.py", *(f"instance_{n}/evaluator.py" for n in range(1, 6))}:
            raise ValueError("Unexpected or duplicate parser repair source snapshot")
        _bound(_archive_path(source["path"], archive), source["sha256"])
        snapshots[relative] = source["sha256"]
    source = report.get("evaluator_source", {})
    if (not {"utils.py", instance_source}.issubset(snapshots)
            or source.get("effective_sha256") != snapshots[instance_source]
            or source.get("original_sha256") != snapshots[instance_source]):
        raise ValueError("Correction evaluator source does not match the reviewed repair snapshots")
    return {"evaluation": report, "provenance": {
        "kind": "reviewed_evaluator_correction", "reason": reason, "model_replayed": False,
        "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_bytes),
        "original_reward": original["reward"], "original_evaluation_attempt_id": original.get("evaluation_attempt_id"),
        "original_sha256": hashes, "report_path": str(report_path), "report_sha256": entry["report_sha256"],
        "evaluation_attempt_id": attempt, "journal_sha256": entry["journal_sha256"],
        "source_proof_path": str(proof_path), "source_proof_sha256": entry["source_proof_sha256"]}}


def evaluation_for_scoring(episode_path: Path, original: dict) -> dict:
    """Withhold unresolved finalization and invalid declared corrections."""
    try:
        episode = _object(Path(episode_path).read_bytes())
        if original.get("status") == "completed" and episode.get("evaluation_error"):
            raise ValueError("Completed evaluator report has unresolved episode evaluation_error")
        if original.get("status") == "completed" and episode.get("phase") == "evaluating":
            raise ValueError("Completed evaluator report is awaiting episode finalization")
        from .image_credit_review import load_image_credit_review
        if load_image_credit_review(episode_path) is not None:
            raise ValueError("Image-credit retrieval is under review; saved score is held")
        from .product_source_review import load_product_source_review
        if load_product_source_review(episode_path) is not None:
            raise ValueError("Product source retrieval is under review; saved score is held")
        from .educational_source_corrections import load_educational_source_correction
        from .lookbook_corrections import load_lookbook_correction
        from .travel_date_corrections import load_travel_date_correction
        correction = load_lookbook_correction(episode_path)
        if correction is None:
            correction = load_educational_source_correction(episode_path)
        if correction is None:
            correction = load_travel_date_correction(episode_path)
        if correction is None:
            correction = load_reviewed_evaluation_correction(episode_path)
        return {"evaluation": correction["evaluation"] if correction else original,
                "provenance": correction["provenance"] if correction else None, "error": None}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return {"evaluation": original, "provenance": None,
                "error": f"{type(error).__name__}: {error}"}
