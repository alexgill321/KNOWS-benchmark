"""Seven explicitly reviewed date-parser holds and immutable grade corrections."""
from __future__ import annotations

from pathlib import Path

from .evaluation_corrections import _archive_path, _bound, _object, _sha, _validate_correction_entry

REASON = "travel_date_annotation_parser"
PENDING_NAME = "travel-date-evaluation-review.json"
CORRECTIONS_NAME = "travel-date-evaluation-corrections.json"
TARGETS = frozenset((model, f"knows.sheets_28_personal_travel_planner.{instance}")
                    for model, instances in (("gpt-6-luna", (2, 5)), ("gpt-6-astra", (1, 2, 3, 4, 5)))
                    for instance in instances)


def _entries(entries: object, *, exact: bool) -> dict:
    if not isinstance(entries, list) or (exact and len(entries) != len(TARGETS)):
        raise ValueError("Invalid date-review target list")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid date-review entry")
        key = entry.get("model"), entry.get("task_id")
        if key not in TARGETS or key in result or entry.get("reason") != REASON:
            raise ValueError("Date review has an unapproved or duplicate target")
        result[key] = entry
    if exact and set(result) != TARGETS:
        raise ValueError("Date review must retain all seven original targets")
    return result


def _pending_original(run: Path, key: tuple, entry: dict) -> dict:
    directory = run / key[0] / key[1]
    hashes = entry.get("original_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {"episode.json", "evaluation.json", "agent-result.json"}:
        raise ValueError("Date-review original hashes are incomplete")
    originals = {name: _object(_bound(directory / name, expected)) for name, expected in hashes.items()}
    episode, report, agent = (originals[name] for name in ("episode.json", "evaluation.json", "agent-result.json"))
    if (entry.get("status") != "pending_saved_output_regrade"
            or episode.get("model") != key[0] or episode.get("task_id") != key[1]
            or episode.get("status") not in {"completed", "timeout", "action_limit"}
            or episode.get("benchmark_validity", {}).get("status") == "excluded_infrastructure"
            or episode.get("evaluation_error")
            or report.get("status") != "completed" or report.get("judge_errors")
            or report.get("task_id") != key[1] or report.get("workspace_id") != episode.get("workspace_id")
            or report.get("evaluation_attempt_id") != entry.get("evaluation_attempt_id")
            or report.get("reward") != entry.get("original_reward")
            or any(not isinstance(episode.get(field), str) or not episode[field]
                   for field in ("effort", "thread_id", "turn_id", "workspace_id"))
            or any(field not in agent or agent[field] != episode.get(field) for field in
                   ("model", "effort", "thread_id", "turn_id", "status", "tool_calls", "final_text", "token_usage"))):
        raise ValueError("Date-review original terminal identity/report differs")
    return report


def _review_file(value: str, run: Path, expected_sha256: str) -> dict:
    path = Path(value)
    if (not path.is_absolute() or not path.resolve().is_relative_to((run.parent / "monitor-checks").resolve())):
        raise ValueError("Date review proof must be an archived monitor review")
    return _object(_bound(path, expected_sha256))


def load_travel_date_correction(episode_path: Path, manifest_path: Path | None = None) -> dict | None:
    """Validate official overlay, or a recovery-archive candidate before publication.

    A candidate path never changes the official manifest or a scoring record.
    Call once for each completed candidate entry before atomically publishing it.
    """
    episode_path = Path(episode_path).resolve()
    directory = episode_path.parent
    key = directory.parent.name, directory.name
    if key not in TARGETS:
        return None
    run = directory.parent.parent
    pending_path = run.parent / PENDING_NAME
    correction_path = (_archive_path(str(manifest_path), (run.parent / "recovery").resolve())
                       if manifest_path is not None else run.parent / CORRECTIONS_NAME)
    if not pending_path.exists():
        if correction_path.exists():
            raise ValueError("Date correction is missing its original reviewed scope")
        return None
    pending_bytes = pending_path.read_bytes()
    pending = _object(pending_bytes)
    if (pending.get("version") != 1 or pending.get("kind") != "pending_reviewed_evaluator_corrections"
            or pending.get("run_dir") != str(run) or pending.get("model_replayed") is not False):
        raise ValueError("Invalid pending date-review manifest")
    entries = _entries(pending.get("corrections"), exact=True)
    # Validate the complete scope so dropped/changed targets cannot silently regain credit.
    for target, entry in entries.items():
        _pending_original(run, target, entry)
    review = _review_file(pending["review_proof"], run, pending["review_proof_sha256"])
    scope = _review_file(pending["scope_proof"], run, pending["scope_proof_sha256"])
    if (review.get("classification") != "verified_evaluator_date_parser_limitation"
            or review.get("explicit_date_format_violation_found") is not False):
        raise ValueError("Date-parser defect has no matching independent review")
    affected = scope.get("affected_reports")
    if not isinstance(affected, list) or len(affected) != len(TARGETS):
        raise ValueError("Date-parser scope proof is incomplete")
    indexed = {(item.get("model"), item.get("task_id")): item for item in affected}
    if set(indexed) != TARGETS or any(
            indexed[target].get("report_sha256") != entry["original_sha256"]["evaluation.json"]
            or indexed[target].get("evaluation_attempt_id") != entry["evaluation_attempt_id"]
            for target, entry in entries.items()):
        raise ValueError("Date-parser scope proof differs from held original reports")
    if not correction_path.exists():
        raise ValueError("Pending saved-output date-parser regrade")
    manifest_bytes = correction_path.read_bytes()
    manifest = _object(manifest_bytes)
    if (manifest.get("version") != 1 or manifest.get("kind") != "reviewed_travel_date_evaluator_corrections"
            or manifest.get("run_dir") != str(run) or manifest.get("model_replayed") is not False
            or manifest.get("pending_manifest_sha256") != _sha(pending_bytes)):
        raise ValueError("Date correction does not bind its original pending review")
    corrected = _entries(manifest.get("corrections"), exact=False)
    for target, entry in corrected.items():
        if entry.get("original_sha256") != entries[target]["original_sha256"]:
            raise ValueError("Date correction replaced its pending original lineage")
    if key not in corrected:
        raise ValueError("Pending saved-output date-parser regrade")
    correction = _validate_correction_entry(episode_path, corrected[key], correction_path, manifest_bytes,
                                           key[0], key[1], REASON, int(key[1].rsplit(".", 1)[1]))
    report = correction["evaluation"]
    step_count = sum(len(checkpoint.get("steps", []))
                     for checkpoint in report.get("detailed_report", {}).get("checkpoints", []))
    if report.get("evaluator_invoked") is not True or step_count != 37:
        raise ValueError("Travel correction requires the invoked complete 37-step evaluator")
    correction["provenance"].update(pending_manifest_path=str(pending_path),
                                     pending_manifest_sha256=_sha(pending_bytes))
    return correction
