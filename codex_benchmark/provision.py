"""Supervisor-only provisioning. The evaluated model never imports this module."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .browser import _google_auth_location
from .config import apply_evaluator_environment, load_config, write_json


def capture_provision_failure(page, output: Path, error: Exception, *, phase: str = "create_workspace") -> dict:
    """Save bounded UI evidence without credentials, page text, or URL queries."""
    timestamp = datetime.now(timezone.utc).isoformat()
    parsed = urlsplit(page.url)
    auth_location = _google_auth_location(page.url)
    if auth_location:
        location = auth_location
    else:
        safe_segments = {"document", "spreadsheets", "presentation", "create", "edit", "u", "d",
                         "0", "1", "2", "3", "4", "5", "view", "preview", "signin", "login",
                         "error", "denied", "sorry", "sorry_index"}
        location = {"host": (parsed.hostname or "").lower().rstrip("."),
                    "path": "/" + "/".join(segment if segment.lower() in safe_segments else "[redacted]"
                        for segment in unquote(parsed.path).split("/") if segment)}
    evidence = {"status": "failed", "phase": phase, "checked_at": timestamp,
                "error_type": type(error).__name__, "location": location,
                "auth_status": "login_required" if auth_location else "unknown",
                "screenshot_status": "pending"}
    evidence_path = output.with_name("provision-failure.json")
    write_json(evidence_path, evidence)
    if auth_location:
        write_json(output.with_name("auth-health.json"), {
            "status": "authentication_required", "auth_status": "login_required",
            "checked_at": timestamp, "timestamp": timestamp, "auth_location": auth_location,
            **auth_location, "phase": "provisioning"})
    match = re.match(r"/(document|spreadsheets|presentation)/d/([A-Za-z0-9_-]+)", parsed.path)
    if parsed.hostname == "docs.google.com" and match:
        write_json(output.with_name("partial-workspace.json"), {
            "workspace_id": match.group(2),
            "workspace_url": f"https://docs.google.com/{match.group(1)}/d/{match.group(2)}/edit",
            "verified": False})
    screenshot = output.with_name("provision-failure.png")
    try:
        page.screenshot(path=str(screenshot), type="png", timeout=5000)
        evidence.update(screenshot_status="saved", screenshot_path=screenshot.name)
    except Exception as screenshot_error:
        evidence.update(screenshot_status="failed", screenshot_error_type=type(screenshot_error).__name__)
    write_json(evidence_path, evidence)
    return evidence


def retry_new_file_visibility(request, *, sleep=time.sleep):
    """Drive can briefly return 404 for a file just created in the editor."""
    from googleapiclient.errors import HttpError
    for attempt in range(5):
        try:
            return request()
        except HttpError as exc:
            if exc.resp.status != 404 or attempt == 4:
                raise
            sleep(2 ** attempt)


def provision(config: dict, task_id: str, model: str, output: Path) -> dict:
    from playwright.sync_api import sync_playwright
    sys.path.insert(0, str(Path(config["benchmark_root"]) / "src"))
    from browsergym.knows.doc_setup import create_task_workspace
    from google.oauth2.service_account import Credentials
    from google.oauth2.credentials import Credentials as OwnerCredentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    family, instance = task_id.removeprefix("knows.").rsplit(".", 1)
    kind = family.split("_", 1)[0]
    owner_credentials = OwnerCredentials.from_authorized_user_file(config["token_path"])
    if not owner_credentials.valid:
        owner_credentials.refresh(Request())
    owner_drive = build("drive", "v3", credentials=owner_credentials, cache_discovery=False)
    if config.get("google_owner_email"):
        email = owner_drive.about().get(fields="user(emailAddress)").execute()["user"]["emailAddress"]
        if email.casefold() != config["google_owner_email"].casefold():
            raise RuntimeError("Google OAuth owner differs from the configured benchmark account")
    evaluator_email = json.loads(Path(config["service_account_path"]).read_text())["client_email"]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=config["headless"], channel=config["channel"])
        context = browser.new_context(storage_state=config["storage_state"],
                                      viewport={"width": 1440, "height": 900})
        try:
            page = context.new_page()
            phase = "create_workspace"
            try:
                workspace_id, workspace_url = create_task_workspace(
                    page, agent_name=f"codex_{model}_{output.parent.parent.name}",
                    task_name=task_id, instance_id=int(instance), kind=kind, max_attempts=1,
                    share_with_evaluator=False)
                # Preserve identity even if access verification fails: no orphaned untracked files.
                result = {"workspace_id": workspace_id, "workspace_url": workspace_url}
                phase = "record_workspace_identity"
                write_json(output, result)
                phase = "grant_evaluator_access"
                retry_new_file_visibility(lambda: owner_drive.permissions().create(fileId=workspace_id,
                    body={"type": "user", "role": "writer", "emailAddress": evaluator_email},
                    sendNotificationEmail=False, fields="id").execute())
                phase = "verify_evaluator_access"
                credentials = Credentials.from_service_account_file(config["service_account_path"],
                    scopes=["https://www.googleapis.com/auth/drive"])
                drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
                access = retry_new_file_visibility(lambda: drive.files().get(
                    fileId=workspace_id, fields="id,capabilities(canEdit)").execute())
                if not access.get("capabilities", {}).get("canEdit"):
                    raise RuntimeError("Evaluator service account cannot edit the workspace")
                # The owner browser must still be signed in, not just able to open an anonymous page.
                phase = "verify_browser_authentication"
                if "accounts.google.com" in page.url:
                    raise RuntimeError("Google browser authentication expired during provisioning")
                phase = "save_browser_state"
                context.storage_state(path=config["storage_state"])
                result["verified"] = True
                phase = "record_verified_workspace"
                write_json(output, result)
                return result
            except Exception as error:
                # Diagnostics are best effort and must never replace the provisioning failure.
                try:
                    capture_provision_failure(page, output, error, phase=phase)
                except Exception as capture_error:
                    print(f"Provisioning evidence capture failed: {type(capture_error).__name__}", file=sys.stderr)
                raise
        finally:
            browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    apply_evaluator_environment(config)
    try:
        result = provision(config, args.task, args.model, args.output)
        print(json.dumps(result))
        return 0
    except Exception as exc:
        print(f"Provisioning failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
