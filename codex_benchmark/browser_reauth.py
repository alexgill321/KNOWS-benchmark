"""Collect independent worker browser sessions through owner-operated Chrome sign-in."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from .config import ROOT, load_config

DOCS_LANDING = "https://docs.google.com/document/u/0/"
DEFAULT_WORKERS = ("worker-1", "worker-2")
ALLOWED_WORKERS = tuple(f"worker-{number}" for number in range(1, 5))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_workers(value: str) -> tuple[str, ...]:
    workers = tuple(part.strip() for part in value.split(","))
    if not workers or len(set(workers)) != len(workers) or any(w not in ALLOWED_WORKERS for w in workers):
        raise ValueError("Workers must be unique names from worker-1 through worker-4")
    return workers


async def owner_is_verified(page, expected_email: str) -> bool:
    """Read only the Google account button label on a Docs landing page."""
    if not expected_email.strip() or page.is_closed():
        return False
    location = urlsplit(page.url)
    if (location.scheme != "https" or location.hostname != "docs.google.com"
            or location.username or location.password
            or not re.fullmatch(r"/document(?:/u/\d+)?/?", location.path)):
        return False
    # Only inspect visible account buttons. Page text and sign-in hyperlinks do
    # not establish authentication, and labels are never retained or logged.
    buttons = page.get_by_role("button", name=re.compile(r"Google Account", re.IGNORECASE))
    email_pattern = re.compile(r"(?<![\w.+@-])" + re.escape(expected_email.strip())
                               + r"(?![\w.+@-])", re.IGNORECASE)
    for index in range(await buttons.count()):
        button = buttons.nth(index)
        if await button.is_visible():
            label = await button.get_attribute("aria-label", timeout=2000)
            if label and email_pattern.search(label):
                return True
    return False


def _private_directory(path: Path) -> Path:
    # Refuse links anywhere below the caller's root before creating credential
    # directories. Existing files are never treated as browser profiles.
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Private browser paths must not contain symbolic links")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _check_regular(path: Path) -> None:
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink != 1)):
        raise ValueError("Browser state and receipts must be independent regular files")


def _atomic_json(path: Path, value: dict) -> None:
    _check_regular(path)
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_verified_state(worker_directory: Path, state: dict, session_id: str) -> Path | None:
    """Atomically replace a verified session, retaining a private old-state copy."""
    if not isinstance(state, dict) or not all(isinstance(state.get(k), list) for k in ("cookies", "origins")):
        raise ValueError("Browser returned invalid storage state")
    _private_directory(worker_directory)
    destination = worker_directory / "browser-state.json"
    _check_regular(destination)
    backup = None
    if destination.exists():
        backup_directory = _private_directory(worker_directory / "browser-state-backups")
        backup = backup_directory / f"{UUID(session_id)}.json"
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(destination.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
    _atomic_json(destination, state)
    return backup


def assert_no_active_claims(results_directories: list[Path]) -> None:
    """Check only live queues; archived databases can retain old active claims."""
    databases = {(directory / "parallel.sqlite").resolve() for directory in results_directories
                 if (directory / "parallel.sqlite").exists()}
    for database in sorted(databases):
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
            if connection.execute("SELECT 1 FROM claims WHERE status='active' LIMIT 1").fetchone():
                raise RuntimeError("Active campaign claims must be drained before browser sign-in")


@contextmanager
def account_lock(root: Path):
    with (root / ".codex-benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another campaign or surviving worker holds the account lock") from None
        yield


async def _wait_for_owner(context, expected_email: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        pages = [page for page in context.pages if not page.is_closed()]
        if not pages:
            raise RuntimeError("The manual sign-in window was closed")
        for page in pages:
            try:
                if await owner_is_verified(page, expected_email):
                    return
            except Exception:
                # A navigation can detach a label while it is being read.
                # Browser errors may contain page data, so never retain them.
                continue
        await asyncio.sleep(min(2, max(0, deadline - time.monotonic())))
    raise TimeoutError("Manual owner sign-in did not finish before its deadline")


async def reauthenticate_workers(config: dict, *, workers=DEFAULT_WORKERS,
                                 timeout_seconds: float = 1200, session_id: str | None = None,
                                 root: Path = ROOT, run_dir: Path | None = None,
                                 playwright_factory=None) -> dict:
    """Keep the account lock until every requested manual window has closed."""
    root = Path(root).resolve()
    workers = parse_workers(",".join(workers))
    if timeout_seconds <= 0:
        raise ValueError("Sign-in timeout must be positive")
    expected_email = config.get("google_owner_email")
    if not isinstance(expected_email, str) or not expected_email.strip():
        raise ValueError("Config must specify google_owner_email for owner verification")
    session_id = str(UUID(session_id)) if session_id else str(uuid4())
    results = root / "codex-results"
    # The global lock excludes every live campaign process. Only the selected
    # live queue is checked for undrained claims; recovery snapshots are inert.
    directories = [Path(run_dir).resolve() if run_dir else results / "smoke"]
    with account_lock(root):
        assert_no_active_claims(directories)
        results.mkdir(parents=True, exist_ok=True)
        receipts = _private_directory(results / "browser-reauth")
        receipt_path = receipts / f"{session_id}.json"
        if receipt_path.exists() or receipt_path.is_symlink():
            raise ValueError("Sign-in session_id must be fresh")
        status_path = results / "browser-reauth-status.json"
        status = {"session_id": session_id, "started_at": _now(), "status": "starting",
                  "workers": {worker: {"status": "pending"} for worker in workers}}

        def publish():
            status["updated_at"] = _now()
            _atomic_json(receipt_path, status)
            _atomic_json(status_path, status)

        publish()
        if playwright_factory is None:
            from playwright.async_api import async_playwright
            playwright_factory = async_playwright
        current_worker = None
        context = None
        try:
            async with playwright_factory() as playwright:
                for current_worker in workers:
                    assert_no_active_claims(directories)
                    worker_directory = _private_directory(root / "auth-data")
                    worker_directory = _private_directory(worker_directory / "parallel-workers")
                    worker_directory = _private_directory(worker_directory / current_worker)
                    profiles = _private_directory(worker_directory / "signin-profiles")
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                    profile = profiles / f"{stamp}-{session_id}"
                    profile.mkdir(mode=0o700, exist_ok=False)
                    item = status["workers"][current_worker]
                    item.update(status="opening", started_at=_now(), profile_path=str(profile))
                    status.update(status="opening", current_worker=current_worker)
                    publish()
                    # Standard headed Chrome, with all Playwright default
                    # automation flags retained. No previous state is loaded.
                    context = await playwright.chromium.launch_persistent_context(
                        str(profile), channel="chrome", headless=False, locale="en-US",
                        viewport={"width": 1440, "height": 900})
                    try:
                        page = context.pages[0] if context.pages else await context.new_page()
                        await page.bring_to_front()
                        try:
                            await page.goto(DOCS_LANDING, wait_until="domcontentloaded", timeout=60000)
                        except Exception as error:
                            # The owner can continue from a slow/redirecting
                            # sign-in page. Record no URL or exception message.
                            item["navigation_error_type"] = type(error).__name__
                        item["status"] = status["status"] = "waiting_for_owner"
                        publish()
                        print(f"{current_worker}: waiting for manual owner sign-in; session {session_id}", flush=True)
                        await _wait_for_owner(context, expected_email, timeout_seconds)
                        state = await asyncio.wait_for(context.storage_state(), timeout=30)
                        if not any([await owner_is_verified(p, expected_email) for p in context.pages]):
                            raise RuntimeError("Owner verification changed before state could be saved")
                        backup = save_verified_state(worker_directory, state, session_id)
                        item.update(status="saved", verified_at=_now(), saved_at=_now())
                        if backup:
                            item["backup_path"] = str(backup)
                        publish()
                        print(f"{current_worker}: verified browser state saved", flush=True)
                    finally:
                        await asyncio.wait_for(context.close(), timeout=15)
                        context = None
            status.update(status="completed", finished_at=_now(), current_worker=None)
            publish()
            return status
        except BaseException as error:
            status.update(status="aborted" if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
                          else "failed", error_type=type(error).__name__, finished_at=_now())
            if current_worker and status["workers"][current_worker]["status"] != "saved":
                status["workers"][current_worker]["status"] = status["status"]
            publish()
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmark.local.json")
    parser.add_argument("--workers", default=",".join(DEFAULT_WORKERS))
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument("--run-dir", type=Path, help="Live campaign directory (default: codex-results/smoke)")
    parser.add_argument("--session-id", help="Fresh UUID used in private backup and status receipts")
    args = parser.parse_args()
    try:
        result = asyncio.run(reauthenticate_workers(load_config(args.config),
            workers=parse_workers(args.workers), timeout_seconds=args.timeout_seconds,
            session_id=args.session_id, run_dir=args.run_dir))
    except KeyboardInterrupt:
        print("Manual browser sign-in aborted; unsaved worker states are unchanged.", flush=True)
        return 130
    except Exception as error:
        # Browser errors can include sensitive page contents. Inspect the safe
        # status receipt for stage/worker details; never print their messages.
        print(f"Manual browser sign-in stopped ({type(error).__name__}); see browser-reauth-status.json.", flush=True)
        return 1
    print(f"Manual browser sign-in completed; session {result['session_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
