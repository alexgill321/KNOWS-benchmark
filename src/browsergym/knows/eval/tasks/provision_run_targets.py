#!/usr/bin/env python3
"""Create the Drive folders that write-tasks upload into, for one benchmark run.

Two task families ask the agent to write into Google Drive:

* ``sheets_10_paper_sorting`` -- uploads paper PDFs and figure screenshots
* ``slides_17_removeimagesaddplaceholders`` -- saves extracted images, and
  works on a copy of a presentation

A write destination cannot be shared between users, so the prompts carry
``{{PLACEHOLDER}}`` tokens and this script fills them in with folders **your**
Google account owns. Nothing it creates is committed: the ids land in
``run_targets.json`` (git-ignored), which the harness and the evaluators both
read via ``eval_utils/run_targets.py``.

Run it before a benchmark pass::

    # everything both families need
    python src/browsergym/knows/eval/tasks/provision_run_targets.py --all

    # one family, or one instance
    python src/browsergym/knows/eval/tasks/provision_run_targets.py \\
        --family slides_17_removeimagesaddplaceholders
    python src/browsergym/knows/eval/tasks/provision_run_targets.py \\
        --family sheets_10_paper_sorting --instance 2

Each pass creates a fresh ``run_NNNN`` folder so runs never collide and old
artifacts stay inspectable::

    KNOWS-runs/                                  <- in your My Drive
      sheets_10_paper_sorting/
        instance_1/run_0001/
          pdfs/                                  <- agent uploads paper PDFs
          figures/                               <- agent uploads Figure 1 shots
      slides_17_removeimagesaddplaceholders/
        instance_1/run_0001/
          images/                                <- agent saves extracted images
          Working copy of <deck>                 <- instance 1 only
        instance_2/run_0001/
          images/
          copies/                                <- agent puts its copy here

The task's own read-only source assets are untouched; this only ever creates
new folders in your Drive.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[5]
if str(BASE_PATH) not in sys.path:
    sys.path.append(str(BASE_PATH))

from src.browsergym.knows.eval.eval_utils.google_services_helpers import authenticate
from src.browsergym.knows.eval.eval_utils.google_services_utils import (
    initialize_google_services,
)
from src.browsergym.knows.eval.eval_utils.run_targets import (
    REQUIRED_TARGETS,
    config_path,
    required_tokens,
    save_targets,
)
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

TASKS_DIR = Path(__file__).resolve().parent
FOLDER_MIME = "application/vnd.google-apps.folder"
ROOT_FOLDER_NAME = "KNOWS-runs"

SHEETS_10 = "sheets_10_paper_sorting"
SLIDES_17 = "slides_17_removeimagesaddplaceholders"

# Subfolders each family's prompt refers to by name.
SHEETS_10_SUBFOLDERS = ("pdfs", "figures")

SOURCE_DECK_RE = re.compile(r"presentation/d/([A-Za-z0-9_-]{20,})")


def folder_url(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"


def deck_url(file_id: str) -> str:
    return f"https://docs.google.com/presentation/d/{file_id}/edit"


def build_drive(mode: str):
    """Build a Drive client.

    OAuth is preferred: the targets should be owned by a real user. Service
    accounts have no My Drive storage quota, so they can create folders but
    cannot hold the presentation copy slides_17 instance 1 needs.

    Args:
        mode: ``"oauth"``, ``"service"``, or ``"auto"`` (OAuth, then service).

    Returns:
        The Drive v3 client.
    """
    if mode in ("auto", "oauth"):
        try:
            return build("drive", "v3", credentials=authenticate(["DRIVE"]))
        except Exception as exc:  # noqa: BLE001 - fall back to the other path
            if mode == "oauth":
                raise
            print(f"OAuth unavailable ({exc}); trying service account...")

    drive, _ = initialize_google_services(service_type="drive")
    if drive is None:
        raise RuntimeError(
            "No Google Drive credentials. Provide auth-data/credentials.json for "
            "OAuth, or a service account, then re-run."
        )
    return drive


def share_anyone_writer(drive, file_id: str) -> None:
    """Let the agent's browser session write here regardless of its account."""
    drive.permissions().create(
        fileId=file_id, body={"role": "writer", "type": "anyone"}
    ).execute()
    time.sleep(0.2)


def find_or_create_folder(drive, name: str, parent_id: str = None) -> str:
    """Return the id of *name* under *parent_id*, creating it when absent."""
    query = (
        f"name = '{name}' and mimeType = '{FOLDER_MIME}' and trashed = false"
        + (f" and '{parent_id}' in parents" if parent_id else " and 'root' in parents")
    )
    found = drive.files().list(q=query, fields="files(id)").execute().get("files", [])
    if found:
        return found[0]["id"]

    body = {"name": name, "mimeType": FOLDER_MIME}
    if parent_id:
        body["parents"] = [parent_id]
    folder_id = drive.files().create(body=body, fields="id").execute()["id"]
    time.sleep(0.2)
    return folder_id


def next_run_id(drive, parent_id: str) -> int:
    """One past the highest existing ``run_NNNN`` under *parent_id*."""
    query = f"'{parent_id}' in parents and mimeType = '{FOLDER_MIME}' and trashed = false"
    files = drive.files().list(q=query, fields="files(name)").execute().get("files", [])
    runs = [int(m.group(1)) for f in files
            if (m := re.fullmatch(r"run_(\d+)", f.get("name", "")))]
    return max(runs, default=0) + 1


def source_deck_id(instance_dir: Path) -> str:
    """The read-only presentation a slides_17 instance starts from.

    Read from task.md so it stays correct if an instance is re-pointed.
    """
    text = (instance_dir / "task.md").read_text(encoding="utf-8")
    match = SOURCE_DECK_RE.search(text)
    if not match:
        raise RuntimeError(f"No source presentation URL found in {instance_dir}/task.md")
    return match.group(1)


def copy_presentation(drive, source_id: str, name: str, parent_id: str) -> str:
    """Copy the source deck into *parent_id* and make it agent-writable.

    Raises:
        RuntimeError: if the credentials have no Drive storage quota, which is
            always the case for service accounts.
    """
    try:
        copied = drive.files().copy(
            fileId=source_id, body={"name": name, "parents": [parent_id]}, fields="id"
        ).execute()["id"]
    except HttpError as exc:
        if "storageQuota" not in str(exc):
            raise
        raise RuntimeError(
            "These credentials cannot own files in Drive (service accounts have "
            "no storage quota), and this instance needs a copy of the source "
            "presentation. Re-run with --auth oauth so the copy is owned by your "
            "own Google account."
        ) from exc
    share_anyone_writer(drive, copied)
    return copied


def provision_instance(drive, family: str, instance_num: int, root_id: str,
                       run_id: int = None) -> dict:
    """Create everything *family*/instance_*instance_num* needs for one run."""
    instance = f"instance_{instance_num}"
    instance_dir = TASKS_DIR / family / instance
    if not instance_dir.exists():
        raise FileNotFoundError(f"No such instance: {instance_dir}")

    family_id = find_or_create_folder(drive, family, root_id)
    instance_id = find_or_create_folder(drive, instance, family_id)
    run = run_id if run_id is not None else next_run_id(drive, instance_id)
    run_id_drive = find_or_create_folder(drive, f"run_{run:04d}", instance_id)
    share_anyone_writer(drive, run_id_drive)

    tokens = required_tokens(family, instance)
    values = {}

    if family == SHEETS_10:
        # The prompt names `pdfs` and `figures` *inside* the folder it is given,
        # so the token points at the run folder and the subfolders live under it.
        for sub in SHEETS_10_SUBFOLDERS:
            share_anyone_writer(drive, find_or_create_folder(drive, sub, run_id_drive))
        values["OUTPUT_FOLDER_URL"] = folder_url(run_id_drive)

    elif family == SLIDES_17:
        images_id = find_or_create_folder(drive, "images", run_id_drive)
        share_anyone_writer(drive, images_id)
        values["IMAGES_FOLDER_URL"] = folder_url(images_id)

        if "WORKING_COPY_URL" in tokens:
            source = source_deck_id(instance_dir)
            copy_id = copy_presentation(
                drive, source, f"KNOWS {family} {instance} run_{run:04d}", run_id_drive
            )
            values["WORKING_COPY_URL"] = deck_url(copy_id)
        if "OUTPUT_FOLDER_URL" in tokens:
            copies_id = find_or_create_folder(drive, "copies", run_id_drive)
            share_anyone_writer(drive, copies_id)
            values["OUTPUT_FOLDER_URL"] = folder_url(copies_id)

    else:
        raise ValueError(f"{family} does not use write targets")

    missing = [t for t in tokens if t not in values]
    if missing:
        raise RuntimeError(f"{family}/{instance}: failed to provision {missing}")

    save_targets(family, instance, values)
    print(f"  {instance} (run_{run:04d})")
    for token, value in sorted(values.items()):
        print(f"    {token:<20} {value}")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Provision Drive write targets for KNOWS tasks that upload files."
    )
    parser.add_argument("--family", choices=sorted(REQUIRED_TARGETS),
                        help="Only this task family (default: all that need targets)")
    parser.add_argument("--instance", type=int, choices=[1, 2, 3, 4, 5],
                        help="Only this instance number (default: all five)")
    parser.add_argument("--parent_folder_id",
                        help=f"Existing Drive folder to build under "
                             f"(default: a '{ROOT_FOLDER_NAME}' folder in your My Drive)")
    parser.add_argument("--run_id", type=int,
                        help="Force this run number instead of auto-incrementing")
    parser.add_argument("--auth", choices=("auto", "oauth", "service"), default="auto",
                        help="Credentials to use (default: OAuth, then service account)")
    args = parser.parse_args()

    print("Authenticating with Google Drive...")
    drive = build_drive(args.auth)

    root_id = args.parent_folder_id or find_or_create_folder(drive, ROOT_FOLDER_NAME)
    families = [args.family] if args.family else sorted(REQUIRED_TARGETS)
    instances = [args.instance] if args.instance else [1, 2, 3, 4, 5]

    print(f"Building under {folder_url(root_id)}\n")
    failures = []
    for family in families:
        print(family)
        for instance_num in instances:
            # Keep going: one instance failing (usually a credential that cannot
            # own files) should not block provisioning the other nine.
            try:
                provision_instance(drive, family, instance_num, root_id, args.run_id)
            except Exception as exc:  # noqa: BLE001 - reported in the summary
                print(f"  instance_{instance_num}: FAILED - {exc}")
                failures.append((family, instance_num, exc))
        print()

    print(f"Wrote {config_path()}")
    print("This file is git-ignored and holds ids only you can write to. "
          "Re-run this script for each fresh benchmark pass.")

    if failures:
        print(f"\n{len(failures)} instance(s) could not be provisioned:")
        for family, instance_num, exc in failures:
            print(f"  {family}/instance_{instance_num}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
