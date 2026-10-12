"""Three reviewed source-URL parser holds and immutable evaluator corrections."""
from __future__ import annotations

from pathlib import Path

from .evaluation_corrections import _archive_path, _bound, _object, _sha, _validate_correction_entry
from .travel_date_corrections import _pending_original

REASON = "educational_source_url_extraction"
PENDING_NAME = "educational-source-evaluation-review.json"
CORRECTIONS_NAME = "educational-source-evaluation-corrections.json"
ARCHIVE_NAME = "educational-sources-20261005T152000Z"
TARGETS = frozenset((model, f"knows.slides_26_basic_educational_slide_deck.{instance}")
                    for model, instance in (("gpt-6-luna", 5), ("gpt-6-astra", 4), ("gpt-6-astra", 5)))


def _entries(value, exact):
    if not isinstance(value, list):
        raise ValueError("Educational-source targets must be a list")
    result = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("Invalid educational-source target")
        key = entry.get("model"), entry.get("task_id")
        if (key not in TARGETS or key in result or entry.get("reason") != REASON
                or entry.get("model_replayed") is not False):
            raise ValueError("Unapproved or duplicate educational-source target")
        result[key] = entry
    if exact and set(result) != TARGETS:
        raise ValueError("Educational-source review must retain all three exact targets")
    return result


def _reviewed_source(entry, archive):
    proof_path = _archive_path(entry["source_proof_path"], archive)
    proof = _object(_bound(proof_path, entry["source_proof_sha256"]))
    if (proof.get("kind") != REASON or proof.get("verified") is not True
            or proof.get("model_replayed") is not False or proof.get("unchanged_rubric") is not True
            or any(proof.get(field) != entry.get(field) for field in
                   ("model", "task_id", "workspace_id", "original_sha256"))):
        raise ValueError("Educational-source parser proof identity differs")
    sources = proof.get("source_snapshots")
    allowed = {"utils.py", *(f"instance_{n}/evaluator.py" for n in range(1, 6))}
    if (not isinstance(sources, list) or len(sources) != 6
            or {item.get("relative_path") for item in sources} != allowed):
        raise ValueError("Educational-source proof requires utils and all five frozen wrappers")
    for item in sources:
        _bound(_archive_path(item["path"], archive), item["sha256"])
    raw = proof.get("raw_snapshots")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Educational-source proof requires captured raw presentation/source evidence")
    paths = set()
    for item in raw:
        path = _archive_path(item["path"], archive)
        if path in paths:
            raise ValueError("Duplicate educational-source raw evidence")
        paths.add(path)
        _bound(path, item["sha256"])
    return proof


def _report_reviews(entry, report, archive):
    for path_key, hash_key in (("report_review_path", "report_review_sha256"),
                               ("independent_review_path", "independent_review_sha256")):
        path = _archive_path(entry[path_key], archive)
        review = _object(_bound(path, entry[hash_key]))
        if (review.get("reviewed") is not True or review.get("scoring_valid") is not True
                or review.get("model_replayed") is not False or review.get("execution_errors_reviewed") is not True
                or review.get("no_auth_or_judge_failure") is not True
                or review.get("no_unresolved_evaluator_error") is not True
                or review.get("model") != entry["model"] or review.get("task_id") != entry["task_id"]
                or review.get("workspace_id") != entry["workspace_id"]
                or review.get("evaluation_attempt_id") != report["evaluation_attempt_id"]
                or review.get("report_sha256") != entry["report_sha256"]
                or review.get("journal_sha256") != entry["journal_sha256"]
                or review.get("original_report_sha256") != entry["original_sha256"]["evaluation.json"]
                or review.get("original_episode_sha256") != entry["original_sha256"]["episode.json"]
                or type(review.get("scored_reward")) not in (int, float)
                or review["scored_reward"] != report["reward"]):
            raise ValueError("Educational-source correction lacks exact root and independent scoring review")
    if (Path(entry["report_review_path"]).resolve() == Path(entry["independent_review_path"]).resolve()
            or entry["report_review_sha256"] == entry["independent_review_sha256"]):
        raise ValueError("Educational-source scoring reviews must be independently recorded")


def _collector_source(entry, report, archive):
    """Bind a later collector repair without replacing the original URL proof."""
    original = _reviewed_source(entry, archive)
    wrapper = f"instance_{entry['task_id'].rsplit('.', 1)[1]}/evaluator.py"
    old_sha = next(s['sha256'] for s in original['source_snapshots'] if s['relative_path'] == wrapper)
    source = report.get('evaluator_source', {})
    keys = ('collector_recovery_proof_path', 'collector_recovery_proof_sha256')
    if not any(key in entry for key in keys):
        if source.get('original_sha256') != old_sha or source.get('effective_sha256') != old_sha:
            raise ValueError('Educational-source evaluator differs from its frozen source')
        return
    if not all(key in entry for key in keys) or entry['model'] != 'gpt-6-astra':
        raise ValueError('Incomplete or unapproved collector recovery lineage')
    proof = _object(_bound(_archive_path(entry[keys[0]], archive), entry[keys[1]]))
    if (proof.get('kind') != REASON or proof.get('repair_kind') != 'educational_image_match_collection'
            or proof.get('verified') is not True or proof.get('model_replayed') is not False
            or proof.get('unchanged_rubric') is not True
            or proof.get('parent_source_proof_sha256') != entry['source_proof_sha256']
            or any(proof.get(key) != entry[key] for key in ('model', 'task_id', 'workspace_id', 'original_sha256'))
            or proof.get('outer_evaluation_timeout_seconds') != 900
            or proof.get('max_workers') != 5 or proof.get('collection_timeout_seconds') != 240
            or proof.get('retains_completed_results') is not True or proof.get('fails_closed') is not True):
        raise ValueError('Collector repair proof differs from the reviewed recovery')
    sources = proof.get('source_snapshots')
    allowed = {'utils.py', *(f'instance_{n}/evaluator.py' for n in range(1, 6))}
    if (not isinstance(sources, list) or len(sources) != 6
            or {item.get('relative_path') for item in sources} != allowed):
        raise ValueError('Collector proof requires all six frozen sources')
    for item in sources:
        _bound(_archive_path(item['path'], archive), item['sha256'])
    new_sha = next(s['sha256'] for s in sources if s['relative_path'] == wrapper)
    if source.get('original_sha256') != new_sha or source.get('effective_sha256') != new_sha:
        raise ValueError('Collector report does not use the reviewed wrapper')
    review = _object(_bound(_archive_path(proof['independent_review_path'], archive), proof['independent_review_sha256']))
    if (review.get('approved') is not True or review.get('collector_repair_approved') is not True
            or review.get('source_snapshots') != sources):
        raise ValueError('Collector repair lacks exact independent source review')


def load_educational_source_correction(episode_path: Path, manifest_path: Path | None = None) -> dict | None:
    """Validate a reviewed overlay, or fail closed while its exact review is pending."""
    episode_path = Path(episode_path).resolve()
    directory = episode_path.parent
    key = directory.parent.name, directory.name
    if key not in TARGETS:
        return None
    run = directory.parent.parent
    archive = (run.parent / "recovery" / ARCHIVE_NAME).resolve()
    pending_path = run.parent / PENDING_NAME
    correction_path = (_archive_path(str(manifest_path), archive) if manifest_path is not None
                       else run.parent / CORRECTIONS_NAME)
    if not pending_path.exists():
        if correction_path.exists():
            raise ValueError("Educational-source correction has no original pending review authority")
        return None
    pending_bytes = pending_path.read_bytes()
    pending = _object(pending_bytes)
    if (type(pending.get("version")) is not int or pending["version"] != 1
            or pending.get("kind") != "pending_educational_source_evaluation_corrections"
            or pending.get("run_dir") != str(run) or pending.get("model_replayed") is not False):
        raise ValueError("Invalid educational-source pending review authority")
    entries = _entries(pending.get("targets"), exact=True)
    for target, entry in entries.items():
        _pending_original(run, target, entry)
        _reviewed_source(entry, archive)
    if not correction_path.exists():
        raise ValueError("Pending saved-output educational-source evaluator correction")
    manifest_bytes = correction_path.read_bytes()
    manifest = _object(manifest_bytes)
    if (type(manifest.get("version")) is not int or manifest["version"] != 1
            or manifest.get("kind") != "reviewed_educational_source_evaluation_corrections"
            or manifest.get("run_dir") != str(run) or manifest.get("model_replayed") is not False
            or manifest.get("pending_manifest_sha256") != _sha(pending_bytes)):
        raise ValueError("Educational-source correction differs from its pending review authority")
    corrected = _entries(manifest.get("corrections"), exact=False)
    for target, entry in corrected.items():
        original = entries[target]
        if any(entry.get(field) != original.get(field) for field in
               ("workspace_id", "original_sha256", "source_proof_path", "source_proof_sha256")):
            raise ValueError("Educational-source correction replaced its reviewed original lineage")
    if key not in corrected:
        raise ValueError("Pending saved-output educational-source evaluator correction")
    entry = corrected[key]
    bound_report = _object(_bound(_archive_path(entry['report_path'], archive), entry['report_sha256']))
    _collector_source(entry, bound_report, archive)
    effective_entry = entry
    if 'collector_recovery_proof_path' in entry:
        effective_entry = {**entry, 'source_proof_path': entry['collector_recovery_proof_path'],
                           'source_proof_sha256': entry['collector_recovery_proof_sha256']}
    correction = _validate_correction_entry(episode_path, effective_entry, correction_path, manifest_bytes,
                                          key[0], key[1], REASON, int(key[1].rsplit(".", 1)[1]))
    report = correction["evaluation"]
    count = sum(len(checkpoint.get("steps", [])) for checkpoint in report.get("detailed_report", {}).get("checkpoints", []))
    if report.get("evaluator_invoked") is not True or count != 24:
        raise ValueError("Educational-source correction requires the complete official 24-step evaluator")
    _report_reviews(entry, report, archive)
    correction["provenance"].update(pending_manifest_path=str(pending_path), pending_manifest_sha256=_sha(pending_bytes),
                                     report_review_path=entry["report_review_path"], report_review_sha256=entry["report_review_sha256"],
                                     independent_review_path=entry["independent_review_path"], independent_review_sha256=entry["independent_review_sha256"])
    if 'collector_recovery_proof_path' in entry:
        correction['provenance'].update({key: entry[key] for key in
                                        ('collector_recovery_proof_path', 'collector_recovery_proof_sha256')})
        correction['provenance'].update(source_proof_path=entry['source_proof_path'],
                                        source_proof_sha256=entry['source_proof_sha256'])
    return correction
