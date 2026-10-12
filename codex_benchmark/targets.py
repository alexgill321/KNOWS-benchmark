"""Provision private, owner-owned Drive targets for one benchmark episode."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from uuid import uuid4

from .config import apply_evaluator_environment, load_config, write_json
from .knows import Task, find_task

FOLDER_MIME = "application/vnd.google-apps.folder"
SHEETS_10 = "sheets_10_paper_sorting"
SLIDES_17 = "slides_17_removeimagesaddplaceholders"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"


def load_owner_credentials(config: dict):
    """Refresh an existing owner token; never launch browser consent in a run."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token = Path(config.get("token_path") or Path(config["benchmark_root"]) / "auth-data/token.json")
    guidance = (f"Owner Google OAuth at {token} is missing, expired, or revoked. "
                "Reauthorize the owner account explicitly with "
                "python -m browsergym.knows.doc_setup --reauth-oauth, then retry. "
                "Automated benchmark runs never open a consent flow.")
    if not token.is_file():
        raise RuntimeError(guidance)
    try:
        credentials = Credentials.from_authorized_user_file(str(token), [DRIVE_SCOPE])
        if credentials.valid:
            return credentials
        if not credentials.refresh_token:
            raise RuntimeError(guidance)
        credentials.refresh(Request())
        if not credentials.valid:
            raise RuntimeError(guidance)
        temporary = token.with_suffix(token.suffix + ".refresh.tmp")
        temporary.write_text(credentials.to_json(), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(token)
        return credentials
    except Exception as exc:
        raise RuntimeError(guidance) from exc


def _build_clients(config: dict):
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    owner_credentials = load_owner_credentials(config)
    service_path = Path(config["service_account_path"])
    account = json.loads(service_path.read_text(encoding="utf-8"))
    email = account.get("client_email", "")
    if not isinstance(email, str) or not re.fullmatch(r"[^\s@]+@[^\s@]+\.gserviceaccount\.com", email):
        raise ValueError("service_account_path must identify the named evaluator service account")
    evaluator_credentials = Credentials.from_service_account_file(str(service_path), scopes=[DRIVE_SCOPE])
    return (build("drive", "v3", credentials=owner_credentials, cache_discovery=False),
            build("drive", "v3", credentials=evaluator_credentials, cache_discovery=False), email)


def provision_targets(config: dict, task: Task, output: Path) -> dict:
    """Create the upstream folder schema with owner + named evaluator access only.

    A new top-level root prevents inheritance from a previously public folder.
    Every created ID is journaled before sharing or later API calls. On failure,
    only this fresh root is trashed; its IDs and cleanup outcome remain saved.
    """
    if task.family not in (SHEETS_10, SLIDES_17):
        raise ValueError(f"{task.task_id} does not require Drive write targets")
    output = output.expanduser().resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Episode target files already exist at {output.parent}; inspect them before provisioning again")
    metadata = {"task_id": task.task_id, "status": "preflight", "created_files": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sharing": "owner_and_named_evaluator", "output": str(output)}
    write_json(metadata_path, metadata)
    owner = None
    root_id = None

    def persist():
        write_json(metadata_path, metadata)

    def create(name: str, parent_id: str | None = None, source_id: str | None = None) -> str:
        nonlocal root_id
        body = {"name": name}
        if parent_id:
            body["parents"] = [parent_id]
        if source_id:
            response = owner.files().copy(fileId=source_id, body=body, fields="id").execute()
        else:
            body["mimeType"] = FOLDER_MIME
            response = owner.files().create(body=body, fields="id").execute()
        file_id = response["id"]
        if parent_id is None:
            root_id = file_id  # Keep cleanup possible if the following journal write fails.
        metadata["created_files"].append({"id": file_id, "name": name, "parent_id": parent_id,
                                           "source_id": source_id})
        persist()
        return file_id

    try:
        owner, evaluator, email = _build_clients(config)
        owner_email = owner.about().get(fields="user(emailAddress)").execute().get("user", {}).get("emailAddress")
        if not owner_email:
            raise RuntimeError("Could not verify the Google OAuth owner's email before provisioning")
        expected_email = config.get("google_owner_email")
        if expected_email and expected_email.casefold() != owner_email.casefold():
            raise RuntimeError("Owner OAuth account does not match config google_owner_email")
        metadata.update(status="provisioning", owner_email=owner_email, evaluator_email=email)
        persist()
        root_id = create(f"KNOWS Codex episode {uuid4().hex}")
        metadata["root_folder_id"] = root_id
        persist()
        owner.permissions().create(fileId=root_id,
            body={"type": "user", "role": "writer", "emailAddress": email},
            sendNotificationEmail=False, fields="id").execute()
        family_id = create(task.family, root_id)
        instance_id = create(f"instance_{task.instance}", family_id)
        run_id = create("run_0001", instance_id)
        metadata["run_folder_id"] = run_id
        values = {}
        if task.family == SHEETS_10:
            create("pdfs", run_id)
            create("figures", run_id)
            values["OUTPUT_FOLDER_URL"] = f"https://drive.google.com/drive/folders/{run_id}"
        else:
            images_id = create("images", run_id)
            values["IMAGES_FOLDER_URL"] = f"https://drive.google.com/drive/folders/{images_id}"
            if task.instance == 1:
                prompt = (task.path / "task.md").read_text(encoding="utf-8")
                sources = re.findall(r"https://docs\.google\.com/presentation/d/([A-Za-z0-9_-]+)", prompt)
                if len(set(sources)) != 1:
                    raise ValueError("Cannot identify exactly one source presentation in slides_17 instance 1")
                copy_id = create(f"KNOWS {task.family} instance_1 run_0001", run_id, sources[0])
                values["WORKING_COPY_URL"] = f"https://docs.google.com/presentation/d/{copy_id}/edit"
                # Verify the exact presentation that will be handed to the agent.
                capabilities = evaluator.files().get(fileId=copy_id, fields="id,capabilities(canEdit)").execute()
                if not capabilities.get("capabilities", {}).get("canEdit"):
                    raise RuntimeError("Evaluator cannot edit the provisioned working presentation")
                # Copies should inherit only the newly private parent permissions.
                permissions = owner.permissions().list(fileId=copy_id, fields="permissions(type,emailAddress,role)").execute()
                if any(permission.get("type") in {"anyone", "domain"} for permission in permissions.get("permissions", [])):
                    raise RuntimeError("The copied presentation unexpectedly has public/domain permissions")
            else:
                copies_id = create("copies", run_id)
                values["OUTPUT_FOLDER_URL"] = f"https://drive.google.com/drive/folders/{copies_id}"
        capability = evaluator.files().get(fileId=run_id, fields="id,capabilities(canEdit)").execute()
        if not capability.get("capabilities", {}).get("canEdit"):
            raise RuntimeError("Evaluator cannot access the episode's private Drive folders")
        result = {task.family: {f"instance_{task.instance}": values}}
        write_json(output, result)
        metadata.update(status="complete", targets=values)
        persist()
        return result
    except Exception as exc:
        metadata.update(status="error", error=f"{type(exc).__name__}: {exc}")
        if root_id and owner:
            try:
                owner.files().update(fileId=root_id, body={"trashed": True}, fields="id,trashed").execute()
                metadata["cleanup"] = {"status": "root_trashed", "root_folder_id": root_id}
            except Exception as cleanup_error:
                metadata["cleanup"] = {"status": "failed", "root_folder_id": root_id,
                                       "error": f"{type(cleanup_error).__name__}: {cleanup_error}"}
        else:
            metadata["cleanup"] = {"status": "no_created_root"}
        persist()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        apply_evaluator_environment(config)
        task = find_task(config["benchmark_root"], args.task)
        result = provision_targets(config, task, args.output)
        print(json.dumps(result))
        return 0
    except Exception as exc:
        print(f"Private target provisioning failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
