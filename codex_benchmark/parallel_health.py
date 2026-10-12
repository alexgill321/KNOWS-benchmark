"""Read-only episode health from durable progress records, never credentials.

This module does not import the browser, evaluator, or app-server. Its output is
safe for the coordinator status file: identifiers, counts, timestamps, local
screenshot paths, and URL host/path only. It never returns page text or errors.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlsplit


GRACE_SECONDS = 300
TERMINAL_AGENT = {"completed", "timeout", "action_limit", "failed", "interrupted"}
STATUSES = TERMINAL_AGENT | {"running", "inProgress", "infrastructure_error", "pending", "unknown"}
PHASES = {"running", "provisioning", "finalizing", "ready_for_evaluation", "evaluating",
          "evaluation_error", "finished", "deferred", "pending", "unknown"}


def _timestamp(value) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            result = float(value)
        elif isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            result = parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
        else:
            return None
        return result if math.isfinite(result) and result > 0 else None
    except (ValueError, OverflowError):
        return None


def _iso(timestamp: float | None) -> str | None:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp is not None else None


def _identifier(value, fallback: str) -> str:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,180}", value) else fallback


def _file(root: Path, path: Path) -> bool:
    try:
        return not path.is_symlink() and path.resolve().is_relative_to(root) and path.is_file()
    except (OSError, RuntimeError):
        return False


def _mtime(root: Path, path: Path) -> float | None:
    try:
        return path.stat().st_mtime if _file(root, path) else None
    except OSError:
        return None


def _json(root: Path, path: Path) -> dict:
    if not _file(root, path):
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, ValueError):
        return {}


def _records(root: Path, path: Path):
    """Bound polling work and tolerate the trailing line of an active writer."""
    if not _file(root, path):
        return
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 1024 * 1024))
            if size > 1024 * 1024:
                handle.readline()  # Discard a possibly partial leading record.
            for line in handle:
                try:
                    value = json.loads(line)
                except (UnicodeError, ValueError):
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError:
        return


def _google_auth_location(url) -> dict | None:
    """Match the browser guard without importing its browser runtime."""
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower().rstrip(".") != "accounts.google.com":
            return None
        segments = [segment for segment in unquote(parsed.path).split("/") if segment]
        if [segment.lower() for segment in segments] == ["signin", "oauth", "legacy", "consent"]:
            # Match the browser guard: reviewing OAuth permissions is not a
            # sign-in failure. Previously saved sticky auth evidence stays intact.
            return None
        markers = {"signin", "servicelogin", "accountchooser", "interactivelogin", "challenge", "webreauth"}
        if segments and not any(segment.lower() in markers for segment in segments):
            return None
        allowed = markers | {"v1", "v2", "v3", "identifier", "pwd", "password", "totp", "ipp", "dp",
                             "selection", "denied", "rejected", "disabled", "oauth"}
        path = "/" + "/".join(segment if segment.lower() in allowed else "[redacted]" for segment in segments)
        return {"host": "accounts.google.com", "path": path}
    except ValueError:
        return None


def _location(url) -> dict | None:
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not host:
            return None
        # Only allowlisted sign-in path segments are useful for auth diagnosis.
        # Other destinations expose the host, never document IDs or URL secrets.
        return _google_auth_location(url) or {"host": host.lower().rstrip("."), "path": "/"}
    except ValueError:
        return None


def _deadline(episode: dict, manifest: dict, *keys: str, default: int) -> float:
    for data in (episode, manifest):
        for key in keys:
            value = data.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                return float(value)
    return float(default)


def inspect_episode(directory: Path) -> dict:
    """Inspect one episode without modifying files, contacting services or replaying work."""
    directory = Path(directory).expanduser().resolve()
    now = time.time()
    episode_path = directory / "episode.json"
    episode = _json(directory, episode_path)
    evaluation_path = directory / "evaluation.json"
    evaluation = _json(directory, evaluation_path)
    # Only a manifest is read outside the episode. Never follow credential
    # locations or worker configuration paths stored in any of these records.
    run_directory = directory.parent.parent
    manifest = _json(run_directory, run_directory / "manifest.json")
    progress = []

    def record(timestamp):
        stamp = _timestamp(timestamp)
        if stamp is not None and stamp <= now + 60:
            progress.append(min(stamp, now))
        return stamp

    for key in ("started_at", "phase_started_at", "agent_finished_at", "finished_at"):
        record(episode.get(key))
    if not progress:
        record(_mtime(directory, episode_path))
    started = _timestamp(episode.get("started_at")) or (min(progress) if progress else None)
    phase_started = _timestamp(episode.get("phase_started_at"))
    status = episode.get("status") if episode.get("status") in STATUSES else "unknown"
    phase = episode.get("phase") if episode.get("phase") in PHASES else None
    terminal_at = _timestamp(episode.get("agent_finished_at"))
    terminal_status = status if status in TERMINAL_AGENT else None
    latest_turn_started = None
    navigation = []
    latest_selection = None
    action_count = 0
    for action in _records(directory, directory / "browser" / "actions.jsonl"):
        stamp = record(action.get("timestamp"))
        if isinstance(action.get("action"), int):
            action_count = max(action_count, action["action"])
        if action.get("tool") == "browser_select_tab" and action.get("success") is True:
            arguments = action.get("arguments", {})
            if stamp and isinstance(arguments, dict):
                latest_selection = (stamp, arguments.get("page_id"))
    for entry in _records(directory, directory / "browser" / "navigations.jsonl"):
        stamp = record(entry.get("timestamp"))
        location = _location(entry.get("url"))
        if stamp and location:
            navigation.append((stamp, entry.get("page_id"), location, _google_auth_location(entry.get("url")) is not None))
    for event in _records(directory, directory / "app-server.jsonl"):
        message = event.get("message", {})
        if not isinstance(message, dict):
            continue
        method = message.get("method", "")
        direction = event.get("direction")
        if direction in {"run_result", "run_error"} or (isinstance(method, str) and method.startswith(("item/", "turn/", "thread/tokenUsage/"))):
            stamp = record(event.get("time"))
        else:
            stamp = _timestamp(event.get("time"))
        if method in {"turn/start", "turn/started"} and stamp:
            latest_turn_started = max(latest_turn_started or 0, stamp)
        if direction == "run_result" and message.get("status") in TERMINAL_AGENT and stamp:
            terminal_at = max(terminal_at or 0, stamp)
            terminal_status = message["status"]
    if latest_turn_started and terminal_at and latest_turn_started > terminal_at:
        terminal_at = terminal_status = None
    agent_result = _json(directory, directory / "agent-result.json")
    if terminal_at is None and agent_result.get("status") in TERMINAL_AGENT:
        stamp = _mtime(directory, directory / "agent-result.json")
        if stamp and (not latest_turn_started or stamp >= latest_turn_started):
            terminal_at, terminal_status = stamp, agent_result["status"]
            record(stamp)
    if terminal_at is None and terminal_status:
        terminal_at = _timestamp(episode.get("finished_at")) or phase_started

    latest_screenshot = None
    screenshot_at = None
    for path in (directory / "browser" / "screenshots").rglob("*.png"):
        stamp = _mtime(directory, path)
        if stamp is not None and (screenshot_at is None or stamp > screenshot_at):
            latest_screenshot, screenshot_at = path, stamp
    if screenshot_at:
        record(screenshot_at)
        match = re.fullmatch(r"\d+_(page_\d+)\.png", latest_screenshot.name)
        if match and (not latest_selection or screenshot_at > latest_selection[0]):
            latest_selection = (screenshot_at, match[1])

    evaluator_started = None
    evaluator_event = None
    evaluator_event_at = None
    evaluator_attempt = None
    completed_steps = 0
    checkpoint_count = 0
    for entry in _records(directory, directory / "evaluation-steps.jsonl"):
        stamp = record(entry.get("timestamp"))
        if not stamp:
            continue
        attempt = entry.get("attempt_id")
        if attempt != evaluator_attempt:
            evaluator_attempt = attempt
            completed_steps = checkpoint_count = 0
        event = entry.get("event")
        if event == "evaluation_started":
            evaluator_started = stamp
        if event == "step_completed":
            completed_steps += 1
        if isinstance(entry.get("completed_steps"), int):
            completed_steps = max(completed_steps, entry["completed_steps"])
        if isinstance(entry.get("checkpoint_index"), int):
            checkpoint_count = max(checkpoint_count, entry["checkpoint_index"])
        evaluator_event, evaluator_event_at = event, stamp
    evaluation_at = _mtime(directory, evaluation_path)
    if evaluation:
        record(evaluation_at)
    if phase == "evaluating":
        evaluator_started = max(evaluator_started or 0, phase_started or 0) or None
    fresh_evaluation = bool(evaluation) and (not evaluator_started or (evaluation_at or 0) >= evaluator_started)
    evaluation_error = (fresh_evaluation and evaluation.get("status") == "evaluation_error") or evaluator_event == "evaluation_error"
    evaluation_error = evaluation_error or (phase == "evaluation_error") or (bool(episode.get("evaluation_error")) and phase != "evaluating")
    if phase is None:
        if fresh_evaluation and evaluation.get("status") == "completed":
            phase = "finished"
        elif evaluation_error:
            phase = "evaluation_error"
        elif evaluator_started and (not terminal_at or evaluator_started >= terminal_at):
            phase = "evaluating"
        elif episode.get("evaluation_deferred"):
            phase = "deferred"
        elif terminal_at:
            phase = "finalizing"
        else:
            phase = "running" if status in {"running", "inProgress"} else "pending"

    auth_status, auth_location, auth_at = "unknown", None, None
    navigation.sort(key=lambda item: item[0])
    if navigation:
        selected = navigation[-1]
        if latest_selection and latest_selection[0] >= selected[0]:
            page_events = [entry for entry in navigation if entry[1] == latest_selection[1]]
            if page_events:
                selected = page_events[-1]
        auth_at, _, auth_location, signin = selected
        if signin:
            auth_status = "login_required"
        elif auth_location["host"] in {"docs.google.com", "drive.google.com"}:
            auth_status = "no_login_redirect"
    for auth_path in (directory / "auth-health.json", directory / "browser" / "auth-health.json"):
        auth = _json(directory, auth_path)
        if not auth:
            continue
        if isinstance(auth.get("auth_issue"), dict):
            auth = auth["auth_issue"]
        stamp = _timestamp(auth.get("checked_at") or auth.get("timestamp") or auth.get("updated_at")) or _mtime(directory, auth_path)
        if not stamp:
            continue
        value = auth.get("auth_status", auth.get("status"))
        sticky_issue = value in {"authentication_required", "login_required", "reauth_required", "logged_out", "expired"} or auth.get("authenticated") is False
        if not sticky_issue and auth_at and stamp < auth_at:
            continue
        if value in {"healthy", "authenticated", "signed_in", "ok"} or auth.get("authenticated") is True:
            auth_status = "healthy"
        elif sticky_issue:
            auth_status = "login_required"
        else:
            continue
        auth_at = stamp
        auth_location = _location(auth.get("url") or auth.get("current_url"))
        location = auth.get("auth_location", auth)
        if auth_location is None and isinstance(location, dict) and isinstance(location.get("host"), str) and isinstance(location.get("path"), str):
            auth_location = _location("https://" + location["host"] + location["path"])

    latest_progress = max(progress) if progress else None
    progress_age = max(0, now - latest_progress) if latest_progress is not None else None
    model_deadline = _deadline(episode, manifest, "timeout_seconds", "episode_timeout_seconds", default=1800)
    evaluation_deadline = _deadline(episode, manifest, "evaluation_timeout_seconds", default=900)
    stalled, reason = False, None
    if evaluation_error:
        stalled, reason = True, "evaluation_error"
    elif phase == "evaluating" and evaluator_started and now - evaluator_started > evaluation_deadline + GRACE_SECONDS:
        stalled, reason = True, "evaluator_deadline_exceeded"
    elif terminal_at and phase in {"finalizing", "ready_for_evaluation"} and not episode.get("evaluation_deferred") and now - terminal_at > GRACE_SECONDS:
        stalled, reason = True, "terminal_finalization_stalled"
    elif phase in {"running", "provisioning"} and started and now - started > model_deadline + GRACE_SECONDS and (progress_age is None or progress_age > GRACE_SECONDS):
        stalled, reason = True, "model_deadline_exceeded_without_progress"
    blocked_reason = reason
    if auth_status == "login_required":
        blocked_reason = "google_login_required"
    elif status == "infrastructure_error":
        blocked_reason = "infrastructure_error"
    return {
        "task_id": _identifier(episode.get("task_id"), _identifier(directory.name, "unknown")),
        "model": _identifier(episode.get("model"), _identifier(directory.parent.name, "unknown")),
        "status": status, "phase": phase, "latest_progress_at": _iso(latest_progress),
        "progress_age_seconds": round(progress_age, 3) if progress_age is not None else None,
        "latest_screenshot": str(latest_screenshot) if latest_screenshot else None,
        "auth_status": auth_status, "auth_location": auth_location, "auth_checked_at": _iso(auth_at),
        "stalled": stalled, "reason": blocked_reason, "blocked": blocked_reason is not None,
        "blocked_reason": blocked_reason, "browser_actions": action_count,
        "evaluation_steps": completed_steps, "evaluation_checkpoints": checkpoint_count,
        "evaluation_last_event": evaluator_event if evaluator_event in {"evaluation_started", "step_completed", "checkpoint_registered", "checkpoint_snapshot", "evaluation_finished", "evaluation_error"} else None,
    }
