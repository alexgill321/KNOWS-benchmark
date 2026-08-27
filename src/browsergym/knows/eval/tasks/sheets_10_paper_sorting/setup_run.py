#!/usr/bin/env python3
"""
Setup script for sheets_10_paper_sorting benchmark runs.

Run this before each benchmark pass to create a fresh destination folder
for each instance. The folder is created inside a parent folder with the
structure: instance_X/run_XXXX (unique run ID for traceability).

What this script does:
1. Creates a new Drive folder: <parent>/instance_X/run_XXXX
2. Updates task.md: replaces [DEST_DRIVE_FOLDER_URL_PLACEHOLDER] or the
   previous dest folder URL with the new folder URL
3. Updates evaluator.py: sets DEST_FOLDER_ID to the new folder ID

Usage:
    # Setup all instances
    python setup_run.py

    # Setup a single instance
    python setup_run.py --instance 2

    # Custom parent folder
    python setup_run.py --parent_folder_id "YOUR_FOLDER_ID"

    # Custom run ID (default: auto-incremented)
    python setup_run.py --run_id 42
"""

import os
import sys
import argparse
import re
import time
import json
from pathlib import Path
from datetime import datetime

# Base path setup
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

from src.browsergym.knows.eval.eval_utils.google_services_helpers import authenticate
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from googleapiclient.discovery import build

# Task-level constants
TASK_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PARENT_FOLDER_ID = "1SPi0Lfuow9qyLNoxHiG-TEOg82IS144N"  # sheets_10 runs parent folder

# Run history file (tracks run IDs for auto-increment)
RUN_HISTORY_FILE = TASK_DIR / "run_history.json"

# Placeholder and URL patterns for task.md replacement
DEST_PLACEHOLDER = "[DEST_DRIVE_FOLDER_URL_PLACEHOLDER]"
DEST_URL_PATTERN = re.compile(
    r'https://drive\.google\.com/drive/folders/[a-zA-Z0-9_-]+(?:\?[^\s]*)?'
)

# Instance configs: source folder IDs (these are static, never change)
INSTANCE_CONFIG = {
    1: {
        "source_folder_id": "1Qm2gLrC3PhRqhlAI_WXBjYKqECdPOwBE",
        "keyword": "dark energy",
    },
    # Add new instances here as they are set up:
    # 2: { "source_folder_id": "...", "keyword": "dark energy" },
    # 3: { "source_folder_id": "...", "keyword": "Monte Carlo" },
    # 4: { "source_folder_id": "...", "keyword": "deep learning" },
    # 5: { "source_folder_id": "...", "keyword": "Bayesian" },
}


def load_run_history() -> dict:
    """Load run history from JSON file."""
    if RUN_HISTORY_FILE.exists():
        with open(RUN_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {"last_run_id": 0, "runs": []}


def save_run_history(history: dict):
    """Save run history to JSON file."""
    with open(RUN_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def get_next_run_id(history: dict) -> int:
    """Get the next auto-incremented run ID."""
    return history.get("last_run_id", 0) + 1


def find_or_create_subfolder(drive_service, parent_id: str, folder_name: str) -> str:
    """Find an existing subfolder by name, or create it if it doesn't exist."""
    # Search for existing folder
    query = (
        f"name = '{folder_name}' and "
        f"'{parent_id}' in parents and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        folder_id = files[0]["id"]
        print(f"  Found existing folder: {folder_name} (ID: {folder_id})")
        return folder_id

    # Create new folder
    body = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = drive_service.files().create(body=body, fields="id").execute()
    folder_id = folder["id"]
    print(f"  Created folder: {folder_name} (ID: {folder_id})")
    return folder_id


def create_run_folder(drive_service, parent_folder_id: str, instance_num: int, run_id: int) -> dict:
    """Create the instance_X/run_XXXX folder structure with pdfs/ and figures/ subfolders.

    Returns dict with folder IDs: {'run': ..., 'pdfs': ..., 'figures': ...}
    """
    # Create or find instance_X folder
    instance_folder_name = f"instance_{instance_num}"
    instance_folder_id = find_or_create_subfolder(
        drive_service, parent_folder_id, instance_folder_name
    )
    time.sleep(0.3)

    # Create run_XXXX folder inside instance folder
    run_folder_name = f"run_{run_id:04d}"
    run_folder_id = find_or_create_subfolder(
        drive_service, instance_folder_id, run_folder_name
    )

    # Set permissions so anyone with link can edit
    drive_service.permissions().create(
        fileId=run_folder_id,
        body={"role": "writer", "type": "anyone"},
    ).execute()
    time.sleep(0.3)

    # Create pdfs/ subfolder
    pdfs_folder_id = find_or_create_subfolder(drive_service, run_folder_id, "pdfs")
    drive_service.permissions().create(
        fileId=pdfs_folder_id,
        body={"role": "writer", "type": "anyone"},
    ).execute()
    time.sleep(0.3)

    # Create figures/ subfolder
    figures_folder_id = find_or_create_subfolder(drive_service, run_folder_id, "figures")
    drive_service.permissions().create(
        fileId=figures_folder_id,
        body={"role": "writer", "type": "anyone"},
    ).execute()

    print(f"  Run folder URL: https://drive.google.com/drive/folders/{run_folder_id}")
    print(f"  PDFs folder URL: https://drive.google.com/drive/folders/{pdfs_folder_id}")
    print(f"  Figures folder URL: https://drive.google.com/drive/folders/{figures_folder_id}")

    return {
        'run': run_folder_id,
        'pdfs': pdfs_folder_id,
        'figures': figures_folder_id,
    }


def update_task_md(instance_dir: Path, folders: dict):
    """Replace dest folder URLs in task.md with new pdfs and figures folder URLs."""
    task_md_path = instance_dir / "task.md"
    content = task_md_path.read_text()

    pdfs_url = f"https://drive.google.com/drive/folders/{folders['pdfs']}"
    figures_url = f"https://drive.google.com/drive/folders/{folders['figures']}"

    # Replace the pdfs/figures folder URLs
    pdfs_pattern = re.compile(
        r'(pdfs folder[^:]*:\s*)'
        r'(https://drive\.google\.com/drive/folders/[a-zA-Z0-9_-]+(?:\?[^\s]*)?)',
        re.IGNORECASE
    )
    figures_pattern = re.compile(
        r'(figures folder[^:]*:\s*)'
        r'(https://drive\.google\.com/drive/folders/[a-zA-Z0-9_-]+(?:\?[^\s]*)?)',
        re.IGNORECASE
    )

    updated = content
    pdfs_match = pdfs_pattern.search(updated)
    figures_match = figures_pattern.search(updated)

    if pdfs_match:
        updated = updated[:pdfs_match.start(2)] + pdfs_url + updated[pdfs_match.end(2):]
        # Re-search after replacement since positions shifted
        figures_match = figures_pattern.search(updated)

    if figures_match:
        updated = updated[:figures_match.start(2)] + figures_url + updated[figures_match.end(2):]

    # Fallback: if no specific patterns found, try replacing all non-source Drive URLs
    if not pdfs_match and not figures_match:
        matches = list(DEST_URL_PATTERN.finditer(updated))
        # Skip first URL (source folder), replace remaining
        for i, match in enumerate(reversed(matches[1:]), 1):
            run_url = f"https://drive.google.com/drive/folders/{folders['run']}"
            updated = updated[:match.start()] + run_url + updated[match.end():]
        if len(matches) <= 1:
            print(f"  WARNING: No dest folder URLs found in task.md")
            return

    task_md_path.write_text(updated)
    print(f"  Updated task.md with pdfs and figures folder URLs")


def update_evaluator(instance_dir: Path, new_folder_id: str):
    """Update DEST_FOLDER_ID in evaluator.py."""
    evaluator_path = instance_dir / "evaluator.py"
    if not evaluator_path.exists():
        print(f"  Skipping evaluator.py (not found)")
        return

    content = evaluator_path.read_text()
    updated = re.sub(
        r'DEST_FOLDER_ID = "[^"]*"',
        f'DEST_FOLDER_ID = "{new_folder_id}"',
        content,
    )
    evaluator_path.write_text(updated)
    print(f"  Updated evaluator.py DEST_FOLDER_ID -> {new_folder_id}")


def prepare_instance(drive_service, instance_num: int, run_id: int, parent_folder_id: str) -> dict:
    """Prepare a single instance for a benchmark run.

    Returns dict with folder IDs: {'run': ..., 'pdfs': ..., 'figures': ...}
    """
    instance_dir = TASK_DIR / f"instance_{instance_num}"

    if not instance_dir.exists():
        print(f"  ERROR: {instance_dir} does not exist")
        return {}

    print(f"\n{'='*60}")
    print(f"Preparing instance_{instance_num} (run_{run_id:04d})")
    print(f"{'='*60}")

    # 1. Create run folder with subfolders
    folders = create_run_folder(drive_service, parent_folder_id, instance_num, run_id)
    time.sleep(0.3)

    # 2. Update task.md with pdfs and figures folder URLs
    update_task_md(instance_dir, folders)

    # 3. Update evaluator.py
    update_evaluator(instance_dir, folders['run'])

    return folders


def main():
    parser = argparse.ArgumentParser(
        description="Setup fresh benchmark run for sheets_10_paper_sorting. "
                    "Creates dest folders and updates task.md + evaluator.py."
    )
    parser.add_argument(
        "--instance",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Prepare only this instance number (default: all configured)",
    )
    parser.add_argument(
        "--parent_folder_id",
        default=DEFAULT_PARENT_FOLDER_ID,
        help="Parent Drive folder ID for creating instance_X/run_XXXX structure",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="Run ID number (default: auto-increment from run_history.json)",
    )
    args = parser.parse_args()

    # Determine run ID
    history = load_run_history()
    run_id = args.run_id if args.run_id is not None else get_next_run_id(history)

    # Authenticate (prefer service account, fall back to OAuth)
    print("Authenticating with Google Drive...")
    drive_service, _ = initialize_google_services(service_type="drive")
    if drive_service is None:
        print("Service account auth failed; falling back to OAuth...")
        credentials = authenticate(["DRIVE"])
        drive_service = build("drive", "v3", credentials=credentials)

    # Determine which instances to prepare
    if args.instance:
        instances = [args.instance]
    else:
        # All instances that have a directory
        instances = sorted([
            int(d.name.split("_")[1])
            for d in TASK_DIR.iterdir()
            if d.is_dir() and d.name.startswith("instance_")
        ])

    print(f"\nRun ID: {run_id:04d}")
    print(f"Instances: {instances}")
    print(f"Parent folder: {args.parent_folder_id}")

    results = {}
    for instance_num in instances:
        folders = prepare_instance(
            drive_service, instance_num, run_id, args.parent_folder_id
        )
        if folders:
            results[f"instance_{instance_num}"] = folders

    # Update run history
    history["last_run_id"] = run_id
    history["runs"].append({
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "instances": results,
    })
    save_run_history(history)

    # Summary
    print("\n" + "=" * 60)
    print(f"Run {run_id:04d} setup complete!")
    print("=" * 60)
    for instance_name, folders in results.items():
        print(f"  {instance_name}:")
        print(f"    run:     https://drive.google.com/drive/folders/{folders['run']}")
        print(f"    pdfs:    https://drive.google.com/drive/folders/{folders['pdfs']}")
        print(f"    figures: https://drive.google.com/drive/folders/{folders['figures']}")
    print(f"\nRun history saved to {RUN_HISTORY_FILE}")
    print("You can now run the benchmark.")


if __name__ == "__main__":
    main()