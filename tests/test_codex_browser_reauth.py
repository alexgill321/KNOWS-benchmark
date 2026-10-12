"""Manual browser sign-in safety tests; no live Chrome, Google, or credentials."""
import asyncio
from contextlib import closing
import fcntl
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from codex_benchmark.browser_reauth import (
    DEFAULT_WORKERS, DOCS_LANDING, assert_no_active_claims, owner_is_verified,
    parse_workers, reauthenticate_workers, save_verified_state,
)


class FakeButtons:
    def __init__(self, labels):
        self.labels = labels

    async def count(self):
        return len(self.labels)

    def nth(self, index):
        label, visible = self.labels[index]
        return SimpleNamespace(is_visible=AsyncMock(return_value=visible),
                               get_attribute=AsyncMock(return_value=label))


class FakePage:
    def __init__(self, url=DOCS_LANDING, label="Google Account: Owner (owner@example.test)", visible=True):
        self.url = url
        self.label = label
        self.visible = visible
        self.goto = AsyncMock()
        self.bring_to_front = AsyncMock()

    def is_closed(self):
        return False

    def get_by_role(self, role, *, name):
        assert role == "button"
        return FakeButtons([(self.label, self.visible)] if name.search(self.label) else [])


class ManualBrowserReauthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = {"cookies": [], "origins": []}

    async def test_owner_requires_visible_exact_account_button_and_docs_landing(self):
        self.assertTrue(await owner_is_verified(FakePage(), "OWNER@example.test"))
        for page in (
            FakePage(label="Google Account: Other (other@example.test)"),
            FakePage(label="Google Account: Other (notowner@example.test)"),
            FakePage(label="Google Account: Other (owner@example.test.evil)"),
            FakePage(label="Sign in owner@example.test"),
            FakePage(visible=False),
            FakePage(url="https://accounts.google.com/signin"),
            FakePage(url="https://docs.google.com.evil.test/document/u/0/"),
            FakePage(url="http://docs.google.com/document/u/0/"),
            FakePage(url="https://docs.google.com/document/d/file/edit"),
        ):
            with self.subTest(url=page.url, label=page.label):
                self.assertFalse(await owner_is_verified(page, "owner@example.test"))

    def test_private_atomic_state_and_old_backup(self):
        worker = self.root / "worker-1"
        worker.mkdir()
        old = b'{"cookies": [], "origins": [], "old": true}\n'
        destination = worker / "browser-state.json"
        destination.write_bytes(old)
        backup = save_verified_state(worker, self.state, str(uuid4()))
        self.assertEqual(backup.read_bytes(), old)
        self.assertEqual(json.loads(destination.read_text()), self.state)
        for path in (backup, destination):
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(worker.stat().st_mode & 0o777, 0o700)
        before = destination.read_bytes()
        with self.assertRaises(ValueError):
            save_verified_state(worker, {"cookies": "bad"}, str(uuid4()))
        self.assertEqual(destination.read_bytes(), before)

    def test_state_links_and_reused_backups_rejected(self):
        worker = self.root / "worker-1"
        worker.mkdir()
        destination = worker / "browser-state.json"
        destination.symlink_to(self.root / "other.json")
        with self.assertRaises(ValueError):
            save_verified_state(worker, self.state, str(uuid4()))
        destination.unlink()
        session_id = str(uuid4())
        save_verified_state(worker, self.state, session_id)
        save_verified_state(worker, self.state, session_id)
        with self.assertRaises(FileExistsError):
            save_verified_state(worker, self.state, session_id)

    def test_active_claims_checked_read_only_and_default_is_two(self):
        self.assertEqual(DEFAULT_WORKERS, ("worker-1", "worker-2"))
        self.assertEqual(parse_workers("worker-2, worker-1"), ("worker-2", "worker-1"))
        for value in ("", "worker-1,worker-1", "worker-5", "../worker-1"):
            with self.assertRaises(ValueError):
                parse_workers(value)
        db = self.root / "parallel.sqlite"
        with closing(sqlite3.connect(db)) as connection, connection:
            connection.execute("CREATE TABLE claims(status TEXT)")
            connection.execute("INSERT INTO claims VALUES('active')")
        with self.assertRaises(RuntimeError):
            assert_no_active_claims([self.root])
        with closing(sqlite3.connect(db)) as connection, connection:
            self.assertEqual(connection.execute("SELECT status FROM claims").fetchone()[0], "active")
            connection.execute("UPDATE claims SET status='blocked'")
        archived = self.root / "recovery" / "parallel.sqlite"
        archived.parent.mkdir()
        with closing(sqlite3.connect(archived)) as connection, connection:
            connection.execute("CREATE TABLE claims(status TEXT)")
            connection.execute("INSERT INTO claims VALUES('active')")
        assert_no_active_claims([self.root])

    def fake_playwright(self, contexts):
        class Factory:
            async def __aenter__(self):
                return SimpleNamespace(chromium=SimpleNamespace(launch_persistent_context=launch))

            async def __aexit__(self, *args):
                return False

        opened = []

        async def do_launch(profile, **options):
            # Chrome windows cannot overlap, and the global lock is held.
            if opened:
                opened[-1].close.assert_awaited_once()
            with (self.root / ".codex-benchmark.lock").open("a") as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(options["channel"], "chrome")
            self.assertFalse(options["headless"])
            self.assertNotIn("ignore_default_args", options)
            self.assertNotIn("args", options)
            self.assertNotIn("storage_state", options)
            self.assertEqual(Path(profile).stat().st_mode & 0o777, 0o700)
            context = contexts[len(opened)]
            opened.append(context)
            return context

        launch = AsyncMock(side_effect=do_launch)
        return Factory, launch

    async def test_sequence_saves_two_independent_states_and_safe_receipt(self):
        contexts = [SimpleNamespace(pages=[FakePage()], storage_state=AsyncMock(return_value=self.state),
                                    close=AsyncMock()) for _ in DEFAULT_WORKERS]
        factory, launch = self.fake_playwright(contexts)
        result = await reauthenticate_workers({"google_owner_email": "owner@example.test"},
            root=self.root, playwright_factory=factory)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(launch.await_count, 2)
        profiles = [call.args[0] for call in launch.await_args_list]
        self.assertNotEqual(*profiles)
        for context in contexts:
            context.pages[0].goto.assert_awaited_once_with(DOCS_LANDING, wait_until="domcontentloaded", timeout=60000)
            context.close.assert_awaited_once()
        receipt = (self.root / "codex-results/browser-reauth-status.json").read_text()
        self.assertNotIn("owner@example.test", receipt)
        self.assertNotIn("cookies", receipt)
        for worker in DEFAULT_WORKERS:
            saved = self.root / "auth-data/parallel-workers" / worker / "browser-state.json"
            self.assertEqual(json.loads(saved.read_text()), self.state)

    async def test_abort_preserves_unsaved_worker_and_prior_success(self):
        worker = self.root / "auth-data/parallel-workers/worker-2"
        worker.mkdir(parents=True)
        old = worker / "browser-state.json"
        old.write_text("unchanged old state")
        contexts = [SimpleNamespace(pages=[FakePage()], storage_state=AsyncMock(return_value=self.state),
                                    close=AsyncMock()) for _ in DEFAULT_WORKERS]
        factory, _ = self.fake_playwright(contexts)
        with patch("codex_benchmark.browser_reauth._wait_for_owner",
                   AsyncMock(side_effect=[None, asyncio.CancelledError()])):
            with self.assertRaises(asyncio.CancelledError):
                await reauthenticate_workers({"google_owner_email": "owner@example.test"},
                    root=self.root, playwright_factory=factory)
        self.assertEqual(old.read_text(), "unchanged old state")
        status = json.loads((self.root / "codex-results/browser-reauth-status.json").read_text())
        self.assertEqual(status["status"], "aborted")
        self.assertEqual(status["workers"]["worker-1"]["status"], "saved")
        self.assertEqual(status["workers"]["worker-2"]["status"], "aborted")
        contexts[1].storage_state.assert_not_awaited()
        contexts[1].close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
