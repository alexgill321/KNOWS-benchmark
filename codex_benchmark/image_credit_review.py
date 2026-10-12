"""Read two explicitly reviewed image-retrieval holds without changing records."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import re
from uuid import UUID

from .evaluator_dependency import _check_attempt

MANIFEST_NAME = "image-credit-evaluation-review.json"
KIND = "image_credit_retrieval_review"
REASON = "image_credit_retrieval_pending_review"
ARCHIVE_NAME = "image-credit-20261005T142300Z"
TARGETS = {
    ("gpt-6-astra", "knows.slides_17_removeimagesaddplaceholders.4"): "original-episode",
    ("gpt-6-luna", "knows.slides_17_removeimagesaddplaceholders.3"): "original-luna3",
}
ORIGINAL_NAMES = {"episode.json", "evaluation.json", "agent-result.json"}
PROBE_CAVEAT = "This fresh response does not establish the prior evaluator response status, MIME, bytes or cause."


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _object(payload: bytes) -> dict:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Image-credit review evidence must be an object")
    return value


def _bound(path: Path, digest: str) -> bytes:
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid image-credit evidence digest")
    payload = path.read_bytes()
    if _sha(payload) != digest:
        raise ValueError("Image-credit evidence hash mismatch")
    return payload


def _evidence(item: dict, archive: Path, expected_path: Path | None = None) -> tuple[Path, bytes]:
    path = Path(item["path"])
    if (not path.is_absolute() or not path.resolve().is_relative_to(archive)
            or (expected_path is not None and path.resolve() != expected_path.resolve())):
        raise ValueError("Image-credit evidence must remain in its exact recovery archive")
    return path.resolve(), _bound(path, item["sha256"])


def _matches_url(value: object, expected_sha: str) -> bool:
    return isinstance(value, str) and _sha(value.encode("utf-8")) == expected_sha


def load_image_credit_review(episode_path: Path, manifest_path: Path | None = None) -> dict | None:
    """Return verified pending-review provenance; reject invalid declared evidence.

    Only the two fixed identities can be held. An absent authority has no effect;
    a malformed authority fails closed for these identities, never other tasks.
    An optional recovery-archive candidate permits read-only prepublication checks.
    """
    episode_path = Path(episode_path).resolve()
    directory = episode_path.parent
    key = directory.parent.name, directory.name
    if key not in TARGETS:
        return None
    run = directory.parent.parent
    archive = (run.parent / "recovery" / ARCHIVE_NAME).resolve()
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute() or not manifest_path.resolve().is_relative_to(archive):
            raise ValueError("Image-credit candidate authority must remain in its exact recovery archive")
        manifest_path = manifest_path.resolve()
    else:
        manifest_path = run.parent / MANIFEST_NAME
    if not manifest_path.exists() and manifest_path == run.parent / MANIFEST_NAME:
        return None
    manifest_bytes = manifest_path.read_bytes()
    manifest = _object(manifest_bytes)
    entries = manifest.get("targets")
    if (type(manifest.get("version")) is not int or manifest["version"] != 1
            or manifest.get("kind") != KIND or manifest.get("run_dir") != str(run)
            or not isinstance(entries, list) or len(entries) != len(TARGETS)):
        raise ValueError("Invalid image-credit review authority scope")
    indexed = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid image-credit review target")
        identity = entry.get("model"), entry.get("task_id")
        if (identity not in TARGETS or identity in indexed or entry.get("reason") != REASON
                or entry.get("model_replayed") is not False):
            raise ValueError("Unapproved or duplicate image-credit review target")
        indexed[identity] = entry
    if set(indexed) != set(TARGETS):
        raise ValueError("Image-credit review must retain both declared targets")
    entry = indexed[key]
    hashes = entry.get("original_sha256")
    if not isinstance(hashes, dict) or set(hashes) != ORIGINAL_NAMES:
        raise ValueError("Image-credit review requires all three original hashes")
    originals = {name: _object(_bound(directory / name, hashes[name])) for name in ORIGINAL_NAMES}
    episode, report, agent = (originals[name] for name in ("episode.json", "evaluation.json", "agent-result.json"))
    fields = ("model", "effort", "thread_id", "turn_id", "status", "tool_calls", "final_text", "token_usage")
    if (episode.get("model") != key[0] or episode.get("task_id") != key[1]
            or episode.get("workspace_id") != entry.get("workspace_id")
            or any(not isinstance(episode.get(field), str) or not episode[field]
                   for field in ("workspace_id", "effort", "thread_id", "turn_id", "judge_model"))
            or episode.get("status") not in {"completed", "action_limit", "timeout"}
            or episode.get("evaluation_error") or episode.get("phase") == "evaluating"
            or episode.get("benchmark_validity", {}).get("status") == "excluded_infrastructure"
            or any(field not in agent or agent[field] != episode.get(field) for field in fields)):
        raise ValueError("Image-credit review terminal model identity differs")
    reward = report.get("reward")
    attempt = report.get("evaluation_attempt_id")
    if not isinstance(attempt, str):
        raise ValueError("Image-credit review has no evaluator attempt")
    UUID(attempt)
    if (report.get("status") != "completed" or report.get("judge_errors")
            or report.get("evaluator_invoked") is not True or report.get("score_source") != "official_evaluator"
            or report.get("task_id") != key[1] or report.get("workspace_id") != episode["workspace_id"]
            or report.get("judge_model") != episode["judge_model"]
            or attempt != entry.get("evaluation_attempt_id")
            or report.get("step_journal", {}).get("attempt_id") != attempt
            or type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1):
        raise ValueError("Image-credit review official evaluator identity differs")
    _check_attempt(directory, report, (directory / "evaluation.json").read_bytes())
    proof_path, proof_bytes = _evidence({"path": entry["proof_path"], "sha256": entry["proof_sha256"]}, archive)
    proof = _object(proof_bytes)
    if (any(proof.get(field) != entry.get(field) for field in
            ("model", "task_id", "workspace_id", "reason", "evaluation_attempt_id", "original_sha256"))
            or proof.get("kind") != KIND or proof.get("reviewed") is not True
            or proof.get("model_replayed") is not False or proof.get("exact_browser_url_loaded") is not True
            or proof.get("evaluator_fetch_failed") is not True):
        raise ValueError("Image-credit review is missing its matching independent evidence")
    archived_original = archive / TARGETS[key]
    for name, digest in hashes.items():
        _bound(archived_original / name, digest)
    evidence = proof["evidence"]
    _, actions_bytes = _evidence(evidence["browser_actions"], archive, archived_original / "browser/actions.jsonl")
    _, navigations_bytes = _evidence(evidence["browser_navigations"], archive, archived_original / "browser/navigations.jsonl")
    screenshot, _ = _evidence(evidence["browser_screenshot"], archive)
    if screenshot.parent != (archived_original / "browser/screenshots").resolve() or screenshot.suffix != ".png":
        raise ValueError("Image-credit browser screenshot is outside its original episode")
    _, evaluator_log = _evidence(evidence["evaluator_log"], archive, archived_original / "evaluate.log")
    url_sha = proof.get("exact_url_sha256")
    if not isinstance(url_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", url_sha):
        raise ValueError("Image-credit exact URL digest is missing")
    actions = [json.loads(line) for line in actions_bytes.splitlines() if line.strip()]
    navigations = [json.loads(line) for line in navigations_bytes.splitlines() if line.strip()]
    screenshot_id = re.fullmatch(r"(\d{5})_(page_\d+)\.png", screenshot.name)
    matching_actions = [action for action in actions if screenshot_id
                        and action.get("action") == int(screenshot_id[1])]
    if len(matching_actions) != 1 or matching_actions[0].get("success") is not True:
        raise ValueError("Image-credit screenshot has no matching successful browser action")
    action_time = datetime.fromisoformat(matching_actions[0]["timestamp"])
    prior_navigation = [nav for nav in navigations if nav.get("page_id") == screenshot_id[2]
                        and datetime.fromisoformat(nav["timestamp"]) <= action_time]
    if not prior_navigation or not _matches_url(prior_navigation[-1].get("url"), url_sha):
        raise ValueError("Image-credit screenshot page has no matching exact-URL navigation")
    failure_prefix = "URL validation: Failed to download image from "
    if not any(line.strip().startswith(failure_prefix)
               and _matches_url(line.strip()[len(failure_prefix):], url_sha)
               for line in evaluator_log.decode("utf-8").splitlines()):
        raise ValueError("Image-credit evaluator log has no matching exact-URL retrieval failure")
    probe_binding = proof["current_unauthenticated_probe"]
    probe_path, probe_bytes = _evidence(probe_binding, archive)
    probe = _object(probe_bytes)
    response = _evidence({"path": probe_binding["response_path"], "sha256": probe_binding["response_sha256"]}, archive)[1]
    if (probe_binding.get("distinct_from_original_evaluator_request") is not True
            or probe_binding.get("original_http_cause_proven") is not False
            or probe.get("kind") != "single_unauthenticated_exact_submitted_image_probe"
            or probe.get("task_id") != key[1] or probe.get("url_sha256") != url_sha
            or type(probe.get("status_code")) is not int or probe["status_code"] != 403
            or probe.get("cookies_or_credentials_used") is not False
            or probe.get("browser_session_created") is not False or probe.get("model_or_judge_called") is not False
            or probe.get("limitation") != PROBE_CAVEAT
            or probe.get("body_sha256") != _sha(response) or probe.get("body_bytes") != len(response)):
        raise ValueError("Image-credit fresh probe does not prove the bounded unauthenticated 403 observation")
    _, independent_bytes = _evidence(proof["independent_report_review"], archive)
    independent = _object(independent_bytes)
    cases = independent.get("image_credit_review_cases")
    if independent.get("kind") != "independent_new_report_review" or not isinstance(cases, list):
        raise ValueError("Image-credit independent report review is missing")
    matches = [case for case in cases if (case.get("model"), case.get("task_id")) == key]
    if (len(matches) != 1 or matches[0].get("attempt_id") != attempt
            or matches[0].get("report_sha256") != hashes["evaluation.json"]
            or matches[0].get("exact_url_sha256") != url_sha
            or matches[0].get("recommendation") != "hold_pending_retrieval_review_preserve_raw_grade"
            or matches[0].get("screenshot", {}).get("sha256") != evidence["browser_screenshot"]["sha256"]
            or matches[0].get("screenshot", {}).get("visually_reviewed") is not True):
        raise ValueError("Image-credit independent review differs from the exact saved report and image")
    return {"kind": KIND, "status": "under_review", "reason": REASON, "model_replayed": False,
            "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_bytes),
            "model": key[0], "task_id": key[1], "evaluation_attempt_id": attempt,
            "original_sha256": hashes, "proof_path": str(proof_path), "proof_sha256": _sha(proof_bytes),
            "probe_path": str(probe_path), "original_http_cause_proven": False}
