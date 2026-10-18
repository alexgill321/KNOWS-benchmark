import json
from pathlib import Path
import tempfile
import unittest

from codex_benchmark.auth_recovery import (
    is_reviewed_auth_diagnostic, mark_auth_diagnostic_reviewed, prepare_interrupted_auth_diagnostic,
)
from codex_benchmark.parallel_queue import classify_existing, initialize_queue, queue_snapshot
from codex_benchmark.runner import summarize


class AuthRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "luna" / "knows.docs_5_influential_papers.5"
        self.directory.mkdir(parents=True)
        self.path = self.directory / "episode.json"
        self.episode = {"status": "infrastructure_error", "task_id": self.directory.name,
                        "model": "luna", "effort": "max", "workspace_id": "doc123",
                        "browsing_history": ["https://example.test"],
                        "error": "BrowserAuthenticationError: Google browser authentication requires attention",
                        "benchmark_validity": {"status": "excluded_infrastructure", "reason": "Authentication failed"}}
        self.result = {"model": "luna", "effort": "max", "thread_id": "thread1", "turn_id": "turn1",
                       "status": "inProgress", "tool_calls": 139, "final_text": "",
                       "token_usage": {"total": {"totalTokens": 500}}, "events": [{"preserved": True}]}
        self.records = [
            {"time": 100, "direction": "run_error", "message": self.result},
            {"time": 101, "direction": "receive", "message": {"method": "turn/completed", "params": {
                "threadId": "thread1", "turn": {"id": "turn1", "status": "interrupted"}}}}]
        self.write("episode.json", self.episode)
        self.write("history.json", self.episode["browsing_history"])
        self.write_log()
        self.archive = self.root / "archive"

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value) + "\n")

    def write_log(self):
        (self.directory / "app-server.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.records))

    def prepare(self):
        return prepare_interrupted_auth_diagnostic(self.path, self.archive)

    def report(self, **changed):
        self.write("evaluation.json", {"status": "completed", "task_id": self.episode["task_id"],
                                      "workspace_id": "doc123", "reward": 0.0, **changed})

    def test_preserves_raw_interruption_without_inventing_completed_result(self):
        original = self.path.read_bytes()
        log = (self.directory / "app-server.jsonl").read_bytes()
        episode = self.prepare()
        for key, value in self.episode.items():
            self.assertEqual(episode[key], value, key)
        self.assertEqual(episode["tool_calls"], 139)
        self.assertEqual(episode["token_usage"], self.result["token_usage"])
        recovery = episode["recovery"]
        self.assertFalse(recovery["reviewed"])
        self.assertFalse(recovery["model_replayed"])
        self.assertEqual(Path(recovery["original_episode"]).read_bytes(), original)
        snapshot = json.loads(Path(recovery["interrupted_snapshot"]).read_text())
        self.assertEqual(snapshot["run_error"], self.records[0])
        self.assertEqual(snapshot["turn_completion"], self.records[1])
        self.assertEqual(snapshot["run_error"]["message"]["status"], "inProgress")
        self.assertEqual((self.directory / "app-server.jsonl").read_bytes(), log)
        self.assertFalse((self.directory / "agent-result.json").exists())
        self.assertEqual(classify_existing(self.directory), "blocked")

    def test_rejects_missing_ambiguous_mismatched_and_completed_model_evidence(self):
        original_records = json.loads(json.dumps(self.records))
        cases = [[], self.records + [self.records[0]], self.records[:1],
                 self.records + [{"direction": "run_result", "message": self.result}]]
        for field, value in (("model", "astra"), ("effort", "low"), ("turn_id", "other"),
                             ("status", "completed")):
            changed = json.loads(json.dumps(original_records))
            changed[0]["message"][field] = value
            cases.append(changed)
        changed = json.loads(json.dumps(original_records))
        changed[1]["message"]["params"]["turn"]["status"] = "completed"
        cases.append(changed)
        for records in cases:
            with self.subTest(records=records):
                self.records = records
                self.write_log()
                before = self.path.read_bytes()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertEqual(self.path.read_bytes(), before)
                self.assertFalse(self.archive.exists())

    def test_refuses_existing_agent_result_evaluation_and_non_auth_failure(self):
        for name in ("agent-result.json", "evaluation.json", "evaluation-steps.jsonl"):
            with self.subTest(name=name):
                self.write(name, {})
                with self.assertRaises(ValueError):
                    self.prepare()
                (self.directory / name).unlink()
        self.write("episode.json", {**self.episode, "error": "TimeoutError"})
        with self.assertRaises(ValueError):
            self.prepare()

    def test_completed_diagnostic_review_keeps_failure_and_skips_queue_replay(self):
        episode = self.prepare()
        self.report()
        self.assertEqual(classify_existing(self.directory), "blocked")
        reviewed = mark_auth_diagnostic_reviewed(self.path)
        self.assertEqual(reviewed["status"], "infrastructure_error")
        self.assertEqual(reviewed["benchmark_validity"], episode["benchmark_validity"])
        self.assertTrue(is_reviewed_auth_diagnostic(self.path))
        self.assertEqual(classify_existing(self.directory), "done")
        before = self.path.read_bytes()
        db = self.root / "queue.sqlite"
        initialize_queue(db, [(self.episode["task_id"], "luna")], ["luna"], self.root)
        self.assertEqual(queue_snapshot(db)[0]["status"], "done")
        self.assertEqual(self.path.read_bytes(), before)
        (self.root / "manifest.json").write_text(json.dumps({"models": ["luna"], "tasks_per_model": 110}))
        summary = summarize(self.root)
        self.assertIsNone(summary["episodes"][0]["reward"])
        self.assertEqual(summary["episodes"][0]["diagnostic_reward"], 0.0)
        self.assertEqual(summary["models"]["luna"]["scored"], 0)
        self.assertEqual(summary["models"]["luna"]["evaluated"], 1)

    def test_review_rejects_failed_mismatched_or_flagged_evaluation(self):
        episode = self.prepare()
        for changed in ({"status": "evaluation_error"}, {"task_id": "other"}, {"workspace_id": "other"}):
            self.report(**changed)
            with self.assertRaises(ValueError):
                mark_auth_diagnostic_reviewed(self.path)
            self.assertEqual(classify_existing(self.directory), "blocked")
        self.report()
        self.write("episode.json", {**episode, "evaluation_error": "Judge failed"})
        with self.assertRaises(ValueError):
            mark_auth_diagnostic_reviewed(self.path)

    def test_modified_proof_or_report_and_forged_review_stay_blocked(self):
        episode = self.prepare()
        self.report()
        mark_auth_diagnostic_reviewed(self.path)
        archive = Path(episode["recovery"]["interrupted_snapshot"])
        original = archive.read_bytes()
        archive.write_bytes(original + b" ")
        self.assertFalse(is_reviewed_auth_diagnostic(self.path))
        self.assertEqual(classify_existing(self.directory), "blocked")
        archive.write_bytes(original)
        self.report(reward=0.5)
        self.assertEqual(classify_existing(self.directory), "blocked")
        self.write("episode.json", {**self.episode, "recovery": {
            "kind": "auth_interrupted", "reviewed": True, "model_replayed": False,
            "model_turn_status": "interrupted"}})
        self.assertEqual(classify_existing(self.directory), "blocked")


if __name__ == "__main__":
    unittest.main()
