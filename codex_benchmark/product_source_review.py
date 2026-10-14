"""Read one reviewed product-source hold; never modify a run or assign credit."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from .evaluator_dependency import _check_attempt

MANIFEST_NAME = "product-source-evaluation-review.json"
ARCHIVE_NAME = "product-source-20261005T204100Z"
KIND = "product_source_retrieval_review"
REASON = "product_source_retrieval_pending_review"
TARGET = ("gpt-6-astra", "knows.slides_42_personal_none_product_comparison.2")
ATTEMPT = "cf65fc2c-6b73-4709-8940-bc4869e1d6a3"
SUBMITTED_URL = "microsoft.com/en-us/d/surface-pro-11th-edition/8n9t09p96cmj"
EVALUATOR_URL = "https://" + SUBMITTED_URL
BROWSER_URL = "https://www." + SUBMITTED_URL
NAVIGATION_URL = "https://www.microsoft.com/en-us/store/configure/surface-pro-13-inch/8n9t09p96cmj"
ERROR_LINE = (f"Error fetching {EVALUATOR_URL} with Playwright: Page.content: Unable to retrieve "
              "content because the page is navigating and changing the content.")
ORIGINAL_NAMES = {"episode.json", "agent-result.json", "evaluation.json"}
CONTRACT = (("Title Slide Validation", 6), ("Challenge & Goal", 2), ("Evaluation Criteria", 5),
            ("Device Slides", 24), ("Comparison Slide", 13), ("Recommendation Slide", 3),
            ("Overall Structure", 4))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object(data: bytes) -> dict:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Product-source evidence must be an object")
    return value


def _bound(path: Path, digest: str) -> bytes:
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid product-source evidence hash")
    data = path.read_bytes()
    if _sha(data) != digest:
        raise ValueError("Product-source evidence hash mismatch")
    return data


def _evidence(item: dict, archive: Path, expected: Path | None = None) -> tuple[Path, bytes]:
    path = Path(item["path"])
    if (not path.is_absolute() or not path.resolve().is_relative_to(archive)
            or (expected is not None and path.resolve() != expected.resolve())):
        raise ValueError("Product-source evidence is outside its exact archive location")
    return path.resolve(), _bound(path, item["sha256"])


def _records(data: bytes) -> list[dict]:
    return [_object(line) for line in data.splitlines() if line.strip()]


def _report_steps(report: dict, journal: bytes) -> dict:
    """Require the same complete keyed rubric in every saved report representation."""
    checkpoints = report["detailed_report"]["checkpoints"]
    result_checkpoints = report["result"]["checkpoints"]
    if (not isinstance(checkpoints, list) or len(checkpoints) != 7
            or not isinstance(result_checkpoints, list) or len(result_checkpoints) != 7
            or not all(isinstance(c, dict) and isinstance(c.get("steps"), list) for c in checkpoints)
            or not all(isinstance(c, dict) for c in result_checkpoints)
            or [(c.get("name"), len(c["steps"])) for c in checkpoints] != list(CONTRACT)):
        raise ValueError("Product-source report has a different rubric")
    keyed, total = {}, 0
    for index, (checkpoint, (_, count)) in enumerate(zip(checkpoints, CONTRACT), 1):
        points = 0
        for number, step in enumerate(checkpoint["steps"], 1):
            if (type(step.get("step_id")) is not int or step["step_id"] != number
                    or type(step.get("success")) is not bool or type(step.get("score")) not in (int, float)
                    or step["score"] not in (0, 1) or type(step.get("max_score")) not in (int, float)
                    or step["max_score"] != 1):
                raise ValueError("Invalid keyed product-source criterion")
            keyed[index, checkpoint["name"], number] = step
            points += step["score"]
        if checkpoint.get("score") != f"{points:g}/{count}":
            raise ValueError("Product-source checkpoint aggregate differs")
        result_cp = result_checkpoints[index - 1]
        summary = result_cp.get("step_summary", {})
        if (result_cp.get("name") != checkpoint["name"] or result_cp.get("result") != points
                or result_cp.get("total") != count or summary.get("steps") != checkpoint["steps"]
                or summary.get("total_steps") != count
                or summary.get("successful_steps") != sum(s["success"] for s in checkpoint["steps"])
                or summary.get("failed_steps") != sum(not s["success"] for s in checkpoint["steps"])):
            raise ValueError("Product-source result summary differs")
        total += points
    if (len(report["result"]["checkpoints"]) != 7
            or report["result"].get("final_score") != {"result": total, "total": 57}
            or report["detailed_report"].get("final_score") != {"result": total, "total": 57}
            or not math.isclose(report["reward"], total / 57, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Product-source final score differs")
    events = [e for e in _records(journal) if e.get("attempt_id") == ATTEMPT]
    if (not events or [e.get("sequence") for e in events] != list(range(1, len(events) + 1))
            or events[0].get("event") != "evaluation_started"
            or events[-1].get("event") != "evaluation_finished"
            or events[-1].get("completed_steps") != 57 or events[-1].get("checkpoint_count") != 7
            or report["step_journal"].get("completed_steps") != 57
            or report["step_journal"].get("checkpoint_count") != 7):
        raise ValueError("Product-source attempt journal is incomplete")
    completed = [e for e in events if e.get("event") == "step_completed"]
    journal_keys = [(e.get("checkpoint_index"), e.get("checkpoint_name"), e["step"].get("step_id"))
                    for e in completed]
    if (len(journal_keys) != 57 or len(set(journal_keys)) != 57 or set(journal_keys) != set(keyed)
            or any(e["step"] != keyed[k] for e, k in zip(completed, journal_keys))):
        raise ValueError("Product-source keyed journal differs from report")
    snapshots = [e for e in events if e.get("event") == "checkpoint_snapshot"]
    registered = [e for e in events if e.get("event") == "checkpoint_registered"]
    journal_cps = report["step_journal"].get("checkpoints")
    if (len(snapshots) != 7 or len(registered) != 7 or not isinstance(journal_cps, list)
            or len(journal_cps) != 7 or len(events) != 73):
        raise ValueError("Product-source checkpoint journal differs")
    for index, ((name, count), snap, reg, saved) in enumerate(zip(CONTRACT, snapshots, registered, journal_cps), 1):
        cp = snap.get("checkpoint", {})
        if (snap.get("checkpoint_index") != index or reg.get("checkpoint_index") != index
                or reg.get("checkpoint", {}).get("name") != name
                or reg.get("checkpoint", {}).get("total") != count
                or cp.get("name") != name or cp.get("total") != count
                or cp.get("result") != sum(s["score"] for s in checkpoints[index - 1]["steps"])
                or cp.get("steps") != checkpoints[index - 1]["steps"]
                or saved != {"checkpoint_index": index, **cp}):
            raise ValueError("Product-source checkpoint snapshot differs")
    affected = keyed[4, "Device Slides", 24]
    expected = {"name": "Device 3 - Content From Sources", "success": False, "step_id": 24,
                "details": "No source content found", "score": 0, "max_score": 1, "category": "execution_error"}
    if any(affected.get(k) != v for k, v in expected.items()):
        raise ValueError("Product-source hold has no matching failed source criterion")
    return affected


def load_product_source_review(episode_path: Path, manifest_path: Path | None = None) -> dict | None:
    """Verify one pending hold, including archive candidates, without any writes.

    An absent authority has no effect. A malformed declared authority fails closed
    for the sole fixed task; other model/task identities are never affected.
    """
    directory = Path(episode_path).resolve().parent
    if (directory.parent.name, directory.name) != TARGET:
        return None
    run = directory.parent.parent
    archive = (run.parent / "recovery" / ARCHIVE_NAME).resolve()
    official = run.parent / MANIFEST_NAME
    if manifest_path is None:
        manifest_path = official
        if not official.exists():
            return None
    else:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute() or not manifest_path.resolve().is_relative_to(archive):
            raise ValueError("Product-source candidate must remain in its exact archive")
        manifest_path = manifest_path.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = _object(manifest_bytes)
    entries = manifest.get("targets")
    if (type(manifest.get("version")) is not int or manifest["version"] != 1
            or manifest.get("kind") != KIND or manifest.get("run_dir") != str(run)
            or not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict)):
        raise ValueError("Invalid product-source authority scope")
    entry = entries[0]
    if ((entry.get("model"), entry.get("task_id")) != TARGET or entry.get("reason") != REASON
            or entry.get("model_replayed") is not False or entry.get("evaluation_attempt_id") != ATTEMPT):
        raise ValueError("Unapproved product-source task or attempt")
    hashes = entry.get("original_sha256")
    if not isinstance(hashes, dict) or set(hashes) != ORIGINAL_NAMES:
        raise ValueError("Product-source hold requires all three original hashes")
    original = archive / "original-episode"
    payloads = {name: _bound(directory / name, hashes[name]) for name in ORIGINAL_NAMES}
    for name in ORIGINAL_NAMES:
        _bound(original / name, hashes[name])
    episode, report, agent = (_object(payloads[name]) for name in ("episode.json", "evaluation.json", "agent-result.json"))
    model_fields = ("model", "effort", "thread_id", "turn_id", "status", "tool_calls", "final_text", "token_usage", "error")
    if (episode.get("model") != TARGET[0] or episode.get("task_id") != TARGET[1]
            or episode.get("workspace_id") != entry.get("workspace_id")
            or any(not isinstance(episode.get(f), str) or not episode[f]
                   for f in ("workspace_id", "effort", "thread_id", "turn_id", "judge_model"))
            or episode.get("status") != "completed" or episode.get("phase") != "finished"
            or episode.get("verified") is not True or episode.get("tool_calls") != 150
            or episode.get("error") is not None or episode.get("evaluation_error")
            or (episode.get("benchmark_validity") or {}).get("status") == "excluded_infrastructure"
            or any(f not in agent or agent[f] != episode.get(f) for f in model_fields)):
        raise ValueError("Product-source terminal model identity differs")
    reward = report.get("reward")
    if (report.get("status") != "completed" or report.get("judge_errors")
            or report.get("evaluator_invoked") is not True or report.get("score_source") != "official_evaluator"
            or report.get("evaluation_attempt_id") != ATTEMPT
            or report.get("step_journal", {}).get("attempt_id") != ATTEMPT
            or any(report.get(f) != episode[f] for f in ("task_id", "workspace_id", "judge_model"))
            or type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1):
        raise ValueError("Product-source official evaluator identity differs")
    _check_attempt(directory, report, payloads["evaluation.json"])
    proof_path, proof_bytes = _evidence({"path": entry["proof_path"], "sha256": entry["proof_sha256"]}, archive)
    proof = _object(proof_bytes)
    identity_fields = ("model", "task_id", "workspace_id", "reason", "evaluation_attempt_id", "original_sha256")
    if (any(proof.get(f) != entry.get(f) for f in identity_fields) or proof.get("kind") != KIND
            or proof.get("reviewed") is not True or proof.get("model_replayed") is not False
            or proof.get("exact_browser_source_loaded") is not True or proof.get("evaluator_fetch_failed") is not True
            or proof.get("http_status_or_challenge_cause_proven") is not False):
        raise ValueError("Product-source proof identity differs")
    evidence = proof["evidence"]
    paths = {"browser_actions": "browser/actions.jsonl", "browser_navigations": "browser/navigations.jsonl",
             "browser_screenshot": "browser/screenshots/00026_page_4.png", "evaluator_log": "evaluate.log",
             "evaluation_journal": "evaluation-steps.jsonl", "evaluation_attempt": f"evaluation-attempts/{ATTEMPT}.json"}
    evidence_bytes = {}
    for key, relative in paths.items():
        _, data = _evidence(evidence[key], archive, original / relative)
        # The archived proof must still describe the saved live originals.
        if (directory / relative).read_bytes() != data:
            raise ValueError("Product-source live evidence differs from archived original")
        evidence_bytes[key] = data
    affected = _report_steps(report, evidence_bytes["evaluation_journal"])
    actions = _records(evidence_bytes["browser_actions"])
    selected = {}
    for number in (26, 103):
        matches = [a for a in actions if a.get("action") == number]
        if len(matches) != 1 or matches[0].get("success") is not True:
            raise ValueError("Product-source browser action is missing or unsuccessful")
        selected[number] = matches[0]
    visit, submit = selected[26], selected[103]
    if (visit.get("tool") != "browser_navigate" or visit.get("arguments", {}).get("url") != BROWSER_URL
            or submit.get("tool") != "browser_type"
            or SUBMITTED_URL not in submit.get("arguments", {}).get("text", "").splitlines()
            or datetime.fromisoformat(submit["timestamp"]) <= datetime.fromisoformat(visit["timestamp"])):
        raise ValueError("Product-source saved URL submission differs")
    prior = [n for n in _records(evidence_bytes["browser_navigations"]) if n.get("page_id") == "page_4"
             and datetime.fromisoformat(n["timestamp"]) <= datetime.fromisoformat(visit["timestamp"])]
    if not prior or prior[-1].get("url") != NAVIGATION_URL:
        raise ValueError("Product-source screenshot has no matching saved product navigation")
    log = evidence_bytes["evaluator_log"].decode("utf-8")
    prefix = "Downloading web content for Device 3..."
    suffix = "Device 3: No source content found."
    if log.count(ERROR_LINE) != 1 or prefix not in log or suffix not in log or not (log.index(prefix) < log.index(ERROR_LINE) < log.index(suffix)):
        raise ValueError("Product-source log lacks the exact source retrieval failure")
    reviews = []
    for role in ("root", "independent"):
        review_path, review_bytes = _evidence(proof[f"{role}_review"], archive)
        review = _object(review_bytes)
        expected = {"kind": f"{role}_product_source_retrieval_review", "model": TARGET[0], "task_id": TARGET[1],
                    "evaluation_attempt_id": ATTEMPT, "report_sha256": hashes["evaluation.json"],
                    "original_sha256": hashes, "decision": "hold_pending_source_retrieval_review",
                    "scored_reward": None, "raw_reward_preserved": reward, "model_terminal": "completed",
                    "tool_calls": 150, "source_url_submitted": SUBMITTED_URL, "source_url_evaluator": EVALUATOR_URL,
                    "source_url_browser": BROWSER_URL, "browser_action": 26, "submitted_action": 103,
                    "evaluator_error": ERROR_LINE, "full_keyed_journal_steps": 57}
        flags = {"infrastructure_exclusion": False, "all_keyed_steps_match": True,
                 "http_status_or_challenge_cause_proven": False, "new_source_probe": False,
                 "browser_session_created": False, "model_replayed": False, "evaluator_called": False,
                 "unrelated_criteria_unchanged": True}
        if (any(f not in review or review[f] != v for f, v in expected.items())
                or any(review.get(f) is not v for f, v in flags.items())
                or review.get("affected_step") != {"checkpoint_index": 4, "checkpoint_name": "Device Slides", **affected}
                or review.get("screenshot", {}).get("path") != evidence["browser_screenshot"]["path"]
                or review["screenshot"].get("sha256") != evidence["browser_screenshot"]["sha256"]
                or review["screenshot"].get("visually_reviewed") is not True):
            raise ValueError("Product-source exact report or visual review differs")
        reviews.append(review_path)
    if reviews[0] == reviews[1]:
        raise ValueError("Product-source hold requires distinct root and independent reviews")
    return {"kind": KIND, "status": "under_review", "reason": REASON, "model_replayed": False,
            "model": TARGET[0], "task_id": TARGET[1], "evaluation_attempt_id": ATTEMPT,
            "original_sha256": hashes, "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_bytes),
            "proof_path": str(proof_path), "proof_sha256": _sha(proof_bytes),
            "http_status_or_challenge_cause_proven": False, "scored_reward": None,
            "infrastructure_exclusion": False}
