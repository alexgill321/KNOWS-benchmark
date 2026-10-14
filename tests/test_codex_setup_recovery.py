"""Provisioning-only recovery must never replay a model or erase setup evidence."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from codex_benchmark.parallel_queue import initialize_queue, queue_pause_reason, queue_snapshot, set_queue_paused
from codex_benchmark.setup_recovery import complete_setup_recovery, reset_pre_agent_claim, validate_pre_agent_failure


class SetupRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "gpt-6-luna" / "knows.sheets_10_paper_sorting.1"
        self.directory.mkdir(parents=True)
        self.episode_path = self.directory / "episode.json"
        self.episode = {"task_id": self.directory.name, "model": "gpt-6-luna", "effort": "max",
                        "status": "infrastructure_error", "browsing_history": [],
                        "error": "RuntimeError: Workspace provisioning failed; see provision.log",
                        "benchmark_validity": {"status": "excluded_infrastructure"}}
        self.write("episode.json", self.episode)
        self.workspace = {"verified": True, "workspace_id": "workspace123",
                          "workspace_url": "https://docs.google.com/spreadsheets/d/workspace123/edit"}
        self.write("workspace.json", self.workspace)
        self.write("run_targets.json", {"folder_id": "already-created-folder"})
        (self.directory / "provision.log").write_text("Saved setup diagnostic\n")
        self.archive = self.root / "recovery"

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value, indent=2) + "\n")

    def assert_rejected(self):
        original = self.episode_path.read_bytes()
        with self.assertRaises((ValueError, RuntimeError)):
            complete_setup_recovery(self.episode_path, self.archive)
        self.assertEqual(original, self.episode_path.read_bytes())
        self.assertFalse(self.archive.exists())
        self.assertFalse((self.directory / "setup-recovery.json").exists())

    def test_complete_preserves_workspace_targets_and_all_setup_evidence(self):
        self.write("partial-workspace.json", {**self.workspace, "verified": False})
        (self.directory / "app-server.jsonl").touch()
        (self.directory / "browser").mkdir()
        before = {p.name: p.read_bytes() for p in self.directory.iterdir() if p.is_file()}
        self.assertEqual(validate_pre_agent_failure(self.episode_path), self.episode)
        record = complete_setup_recovery(self.episode_path, self.archive)
        self.assertFalse(self.episode_path.exists())
        self.assertIs(record["model_replayed"], False)
        self.assertEqual(record["workspace_id"], self.workspace["workspace_id"])
        self.assertIn("+00:00", record["timestamp"])
        saved = Path(record["archive_path"])
        self.assertEqual(saved.read_bytes(), before["episode.json"])
        self.assertEqual(saved.parent.parent, self.archive.resolve() / "setup-attempts")
        self.assertEqual(json.loads((saved.parent / "setup-recovery.json").read_text()), record)
        self.assertEqual(json.loads((self.directory / "setup-recovery.json").read_text()), record)
        for name, payload in before.items():
            if name != "episode.json":
                self.assertEqual((self.directory / name).read_bytes(), payload, name)
        self.assertTrue((self.directory / "browser").is_dir())

    def test_rejects_any_model_or_evaluation_artifact_even_if_empty(self):
        for name in ("agent-result.json", "prompt.txt", "history.json", "evaluation.json",
                     "evaluation-steps.jsonl", "evaluate.log", "browser/actions.jsonl",
                     "browser/screenshots/1.png", "evaluation-attempts/previous.json"):
            with self.subTest(name=name):
                path = self.directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                self.assert_rejected()
                path.unlink()

    def test_nonempty_protocol_log_is_rejected(self):
        (self.directory / "app-server.jsonl").write_text("\n")
        self.assert_rejected()

    def test_wrong_failure_and_saved_model_fields_are_rejected(self):
        for changed in ({"status": "running"}, {"status": "completed"},
                        {"error": "Browser navigation failed"}, {"thread_id": "thread"},
                        {"tool_calls": 0}, {"prompt_sha256": "hash"},
                        {"browsing_history": ["https://example.test"]}):
            with self.subTest(changed=changed):
                self.write("episode.json", {**self.episode, **changed})
                self.assert_rejected()

    def test_unverified_or_missing_workspace_identity_is_rejected(self):
        for changed in ({"verified": False}, {"verified": 1}, {"workspace_id": ""},
                        {"workspace_url": None}):
            with self.subTest(changed=changed):
                self.write("workspace.json", {**self.workspace, **changed})
                self.assert_rejected()

    def test_mismatched_partial_or_episode_workspace_is_rejected(self):
        self.write("partial-workspace.json", {"workspace_id": "different", "verified": False})
        self.assert_rejected()
        (self.directory / "partial-workspace.json").unlink()
        self.write("episode.json", {**self.episode, "workspace_id": "different"})
        self.assert_rejected()

    def test_attempt_archives_are_unique_and_earlier_metadata_is_preserved(self):
        first = complete_setup_recovery(self.episode_path, self.archive)
        self.write("episode.json", self.episode)
        second = complete_setup_recovery(self.episode_path, self.archive)
        self.assertNotEqual(first["archive_path"], second["archive_path"])
        for record in (first, second):
            archive = Path(record["archive_path"])
            self.assertEqual(json.loads(archive.read_text()), self.episode)
            self.assertEqual(json.loads((archive.parent / "setup-recovery.json").read_text()), record)

    def prepare_queue_recovery(self):
        db = self.root / "queue.sqlite"
        initialize_queue(db, [(self.episode["task_id"], self.episode["model"])],
                         [self.episode["model"]], self.root)
        set_queue_paused(db, "Inspect provisioning failure")
        with sqlite3.connect(db) as connection:
            connection.execute("UPDATE claims SET worker_id='worker-1',pid=123,claimed_at='earlier'")
        receipt = complete_setup_recovery(self.episode_path, self.archive)
        return db, receipt

    def test_reset_only_blocked_claim_and_preserve_global_pause(self):
        db, receipt = self.prepare_queue_recovery()
        reset = reset_pre_agent_claim(db, self.episode_path)
        self.assertEqual(reset["status"], "pending")
        for key in ("worker_id", "pid", "claimed_at", "error"):
            self.assertIsNone(reset[key], key)
        self.assertEqual(queue_pause_reason(db), "Inspect provisioning failure")
        self.assertEqual(json.loads((self.directory / "setup-recovery.json").read_text()), receipt)

    def test_reset_rejects_done_pending_active_and_any_other_active_claim(self):
        db, _ = self.prepare_queue_recovery()
        for status in ("done", "pending", "active"):
            with self.subTest(status=status):
                with sqlite3.connect(db) as connection:
                    connection.execute("UPDATE claims SET status=?", (status,))
                with self.assertRaises(RuntimeError):
                    reset_pre_agent_claim(db, self.episode_path)
                self.assertEqual(queue_snapshot(db)[0]["status"], status)
        with sqlite3.connect(db) as connection:
            connection.execute("UPDATE claims SET status='blocked'")
            connection.execute("INSERT INTO claims(model,task_id,ordinal,model_rank,status) VALUES(?,?,?,?,?)",
                               (self.episode["model"], "another-task", 1, 0, "active"))
        with self.assertRaises(RuntimeError):
            reset_pre_agent_claim(db, self.episode_path)
        self.assertEqual(queue_snapshot(db)[0]["status"], "blocked")

    def test_reset_rejects_corrupt_receipt_archive_and_changed_workspace(self):
        db, receipt = self.prepare_queue_recovery()
        for changed in ({"model_replayed": True}, {"task_id": "other"},
                        {"original_episode_sha256": "wrong"}):
            with self.subTest(changed=changed):
                self.write("setup-recovery.json", {**receipt, **changed})
                with self.assertRaises(ValueError):
                    reset_pre_agent_claim(db, self.episode_path)
                self.assertEqual(queue_snapshot(db)[0]["status"], "blocked")
        self.write("setup-recovery.json", receipt)
        archive = Path(receipt["archive_path"])
        original = archive.read_bytes()
        archive.write_bytes(original + b" ")
        with self.assertRaises(ValueError):
            reset_pre_agent_claim(db, self.episode_path)
        archive.write_bytes(original)
        self.write("workspace.json", {**self.workspace, "workspace_id": "different"})
        with self.assertRaises(ValueError):
            reset_pre_agent_claim(db, self.episode_path)
        self.write("workspace.json", self.workspace)
        (self.directory / "prompt.txt").touch()
        with self.assertRaises(ValueError):
            reset_pre_agent_claim(db, self.episode_path)
        self.assertEqual(queue_snapshot(db)[0]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
