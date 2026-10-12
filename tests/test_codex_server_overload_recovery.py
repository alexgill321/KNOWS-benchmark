import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from codex_benchmark.server_overload_recovery import (
    ERROR_MESSAGE, is_reviewed_server_overload_diagnostic,
    mark_server_overload_diagnostic_reviewed, prepare_server_overload_diagnostic,
)
from codex_benchmark.parallel_queue import classify_existing, initialize_queue, queue_snapshot, claim_next
from codex_benchmark.runner import grade_saved_episode, summarize


class ServerOverloadRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "astra" / "knows.sheets_25_skitourplan.1"
        self.directory.mkdir(parents=True)
        self.path = self.directory / "episode.json"
        self.archive = self.root / "recovery"
        self.error = {"message": ERROR_MESSAGE, "codexErrorInfo": "serverOverloaded",
                      "additionalDetails": None, "misalignment": None}
        self.result = {"model": "astra", "effort": "max", "thread_id": "thread1", "turn_id": "turn1",
                       "status": "failed", "tool_calls": 2, "final_text": "",
                       "token_usage": {"total": {"totalTokens": 500}}, "error": self.error}
        self.checkpoint = {**self.result, "events": [], "duration_seconds": 10.0}
        failure = f"RuntimeError: Codex turn did not complete: {self.error}"
        self.episode = {**self.checkpoint, "status": "infrastructure_error", "error": failure,
                        "task_id": self.directory.name, "workspace_id": "sheet123",
                        "workspace_url": "https://docs.google.com/spreadsheets/d/sheet123/edit",
                        "verified": True, "browsing_history": ["https://example.test"],
                        "benchmark_validity": {"status": "excluded_infrastructure", "reason": failure},
                        "prompt_sha256": hashlib.sha256(b"task prompt\n").hexdigest(),
                        "browser_actions": 2, "phase": "finalizing", "phase_started_at": "original"}
        self.records = [
            {"time": 100, "direction": "receive", "message": {"method": "turn/completed", "params": {
                "threadId": "thread1", "turn": {"id": "turn1", "status": "failed", "error": self.error}}}},
            {"time": 101, "direction": "run_result", "message": self.result}]
        self.write("episode.json", self.episode)
        self.write("agent-result.json", self.checkpoint)
        self.write("history.json", self.episode["browsing_history"])
        self.write("workspace.json", {k: self.episode[k] for k in ("workspace_id", "workspace_url", "verified")})
        self.path.with_name("prompt.txt").write_bytes(b"task prompt\n")
        (self.directory / "browser").mkdir()
        (self.directory / "browser/actions.jsonl").write_text('{"action":1}\n{"action":2}\n')
        self.write_log()

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value) + "\n")

    def write_log(self):
        (self.directory / "app-server.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.records))

    def prepare(self):
        return prepare_server_overload_diagnostic(self.path, self.archive)

    def report(self, **changed):
        report = {"status": "completed", "task_id": self.episode["task_id"],
                  "workspace_id": "sheet123", "reward": 0.25,
                  "judge_errors": [], "evaluation_attempt_id": str(uuid4()), **changed}
        self.write("evaluation.json", report)
        attempts = self.directory / "evaluation-attempts"
        attempts.mkdir(exist_ok=True)
        (attempts / (report["evaluation_attempt_id"] + ".json")).write_bytes(
            (self.directory / "evaluation.json").read_bytes())
        events = [{"event": event, "attempt_id": report["evaluation_attempt_id"],
                   "task_id": report["task_id"], "workspace_id": report["workspace_id"]}
                  for event in ("evaluation_started", "evaluation_finished")]
        (self.directory / "evaluation-steps.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    def test_archives_exact_evidence_without_changing_failure_or_checkpoint(self):
        originals = {p.relative_to(self.directory): p.read_bytes() for p in self.directory.rglob("*") if p.is_file()}
        episode = self.prepare()
        self.assertEqual({k: v for k, v in episode.items() if k != "recovery"}, self.episode)
        self.assertFalse(episode["recovery"]["reviewed"])
        self.assertFalse(episode["recovery"]["model_replayed"])
        self.assertEqual(Path(episode["recovery"]["original_episode"]).read_bytes(), originals[Path("episode.json")])
        for relative, payload in originals.items():
            if relative.name != "episode.json":
                self.assertEqual((self.directory / relative).read_bytes(), payload)
        self.assertEqual(classify_existing(self.directory), "blocked")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_rejects_non_overload_and_forged_failure_before_archiving(self):
        cases = [
            {**self.episode, "status": "completed"},
            {**self.episode, "error": "BrowserAuthenticationError: signed out"},
            {**self.episode, "benchmark_validity": {"status": "eligible"}},
            {**self.episode, "benchmark_validity": {"status": "excluded_infrastructure", "reason": "different"}},
            {**self.episode, "turn_id": "other"},
            {**self.episode, "tool_calls": 3},
            {**self.episode, "workspace_id": "other"},
            {**self.episode, "browser_actions": 3},
        ]
        for episode in cases:
            with self.subTest(episode=episode):
                self.write("episode.json", episode)
                before = self.path.read_bytes()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertEqual(self.path.read_bytes(), before)
                self.assertFalse(self.archive.exists())

    def test_requires_exact_failed_checkpoint_protocol_and_turn_confirmation(self):
        original = deepcopy(self.records)
        cases = [[], original[:1], original[1:], original + [original[1]],
                 original + [{"direction": "run_error", "message": self.result}]]
        for location, field, value in ((1, "status", "completed"), (1, "turn_id", "other"),
                                       (1, "tool_calls", 99), (1, "error", {**self.error, "message": "other"})):
            changed = deepcopy(original)
            changed[location]["message"][field] = value
            cases.append(changed)
        changed = deepcopy(original)
        changed[0]["message"]["params"]["turn"]["error"]["codexErrorInfo"] = "internalServerError"
        cases.append(changed)
        for records in cases:
            with self.subTest(records=records):
                self.records = records
                self.write_log()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.archive.exists())
        self.records = original
        self.write_log()
        self.write("agent-result.json", {**self.checkpoint, "tool_calls": 3})
        with self.assertRaises(ValueError):
            self.prepare()

    def test_rejects_existing_evaluation_and_changed_prompt_or_history(self):
        for name in ("evaluation.json", "evaluation-steps.jsonl", "evaluate.log"):
            self.write(name, {})
            with self.assertRaises(ValueError):
                self.prepare()
            (self.directory / name).unlink()
        self.write("history.json", ["https://other.test"])
        with self.assertRaises(ValueError):
            self.prepare()
        self.write("history.json", self.episode["browsing_history"])
        (self.directory / "prompt.txt").write_text("changed")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_actual_grade_api_then_review_keeps_excluded_and_reconciles_without_replay(self):
        episode = self.prepare()
        checkpoint = (self.directory / "agent-result.json").read_bytes()
        db = self.root / "parallel.sqlite"
        schedule = [(self.episode["task_id"], "astra")]
        initialize_queue(db, schedule, ["astra"], self.root)
        self.assertEqual(queue_snapshot(db)[0]["status"], "blocked")
        commands = []

        async def evaluate_only(command, log, timeout, env=None):
            commands.append(command)
            self.report()
            return 0

        with patch("codex_benchmark.runner.subprocess_logged", new=evaluate_only):
            asyncio.run(grade_saved_episode({"evaluation_timeout_seconds": 30}, episode, self.path))
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][1:3], ["-m", "codex_benchmark.evaluate"])
        self.assertEqual(classify_existing(self.directory), "blocked")
        report_before = (self.directory / "evaluation.json").read_bytes()
        reviewed = mark_server_overload_diagnostic_reviewed(self.path)
        self.assertEqual(reviewed["status"], "infrastructure_error")
        self.assertEqual(reviewed["benchmark_validity"], self.episode["benchmark_validity"])
        self.assertEqual(reviewed["error"], self.episode["error"])
        self.assertEqual((self.directory / "agent-result.json").read_bytes(), checkpoint)
        self.assertEqual((self.directory / "evaluation.json").read_bytes(), report_before)
        self.assertTrue(is_reviewed_server_overload_diagnostic(self.path))
        self.assertEqual(classify_existing(self.directory), "done")
        initialize_queue(db, schedule, ["astra"], self.root)
        self.assertEqual(queue_snapshot(db)[0]["status"], "done")
        self.assertIsNone(claim_next(db, "worker-1", 123))
        (self.root / "manifest.json").write_text(json.dumps({"models": ["astra"], "tasks_per_model": 1}))
        summary = summarize(self.root)
        self.assertEqual(summary["models"]["astra"]["scored"], 0)
        self.assertEqual(summary["models"]["astra"]["evaluated"], 1)
        self.assertIsNone(summary["episodes"][0]["reward"])
        self.assertEqual(summary["episodes"][0]["diagnostic_reward"], 0.25)

    def test_bad_or_failed_diagnostic_cannot_be_reviewed(self):
        episode = self.prepare()
        for changed in ({"status": "evaluation_error"}, {"task_id": "other"},
                        {"workspace_id": "other"}, {"evaluation_attempt_id": ""},
                        {"judge_errors": ["failed"]}, {"reward": None}, {"reward": float("nan")},
                        {"reward": True}):
            with self.subTest(changed=changed):
                self.report(**changed)
                with self.assertRaises((ValueError, KeyError)):
                    mark_server_overload_diagnostic_reviewed(self.path)
                self.assertEqual(classify_existing(self.directory), "blocked")
        self.report()
        self.write("episode.json", {**episode, "evaluation_error": "failed"})
        with self.assertRaises(ValueError):
            mark_server_overload_diagnostic_reviewed(self.path)

    def test_live_or_archived_checkpoint_or_report_tampering_fails_closed(self):
        episode = self.prepare()
        self.report()
        mark_server_overload_diagnostic_reviewed(self.path)
        attempt = Path(episode["recovery"]["original_episode"]).parent
        paths = [self.directory / "agent-result.json", self.directory / "app-server.jsonl",
                 self.directory / "history.json", self.directory / "prompt.txt",
                 self.directory / "workspace.json", self.directory / "browser/actions.jsonl",
                 self.directory / "evaluation.json", self.directory / "evaluation-steps.jsonl", attempt / "episode.json",
                 attempt / "agent-result.json", attempt / "recovery.json"]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                if path.name == "recovery.json":
                    value = json.loads(original)
                    value["model_replayed"] = True
                    path.write_text(json.dumps(value))
                else:
                    path.write_bytes(original + b" ")
                self.assertFalse(is_reviewed_server_overload_diagnostic(self.path))
                self.assertEqual(classify_existing(self.directory), "blocked")
                path.write_bytes(original)
                self.assertTrue(is_reviewed_server_overload_diagnostic(self.path))

    def test_removed_added_or_byte_changed_proof_fields_fail_closed(self):
        episode = self.prepare()
        self.report()
        reviewed = mark_server_overload_diagnostic_reviewed(self.path)
        proof = Path(episode["recovery"]["original_episode"]).with_name("recovery.json")
        original = proof.read_bytes()
        stripped = json.loads(original)
        stripped.pop("prepared_at")
        for altered in (b"{}", json.dumps(stripped).encode(), original + b" "):
            with self.subTest(altered=altered[:20]):
                proof.write_bytes(altered)
                self.assertFalse(is_reviewed_server_overload_diagnostic(self.path))
                self.assertEqual(classify_existing(self.directory), "blocked")
        proof.write_bytes(original)
        for key in ("prepared_at", "proof_sha256", "reviewed_at"):
            altered = deepcopy(reviewed)
            del altered["recovery"][key]
            self.write("episode.json", altered)
            self.assertFalse(is_reviewed_server_overload_diagnostic(self.path))
        altered = deepcopy(reviewed)
        altered["recovery"]["unexpected"] = True
        self.write("episode.json", altered)
        self.assertFalse(is_reviewed_server_overload_diagnostic(self.path))
        self.write("episode.json", reviewed)
        self.assertTrue(is_reviewed_server_overload_diagnostic(self.path))

    def test_error_status_validity_and_forged_review_remain_blocked(self):
        episode = self.prepare()
        self.report()
        reviewed = mark_server_overload_diagnostic_reviewed(self.path)
        for key, value in (("error", "other"), ("status", "failed"),
                           ("benchmark_validity", {"status": "eligible"}),
                           ("recovery", {"kind": "server_overloaded", "reviewed": True})):
            self.write("episode.json", {**reviewed, key: value})
            self.assertFalse(is_reviewed_server_overload_diagnostic(self.path))
            self.assertEqual(classify_existing(self.directory), "blocked")
        self.write("episode.json", reviewed)
        self.report(evaluation_attempt_id=str(uuid4()))
        self.assertFalse(is_reviewed_server_overload_diagnostic(self.path))
        self.assertEqual(classify_existing(self.directory), "blocked")

    def test_missing_mismatched_or_incomplete_attempt_journal_stays_blocked(self):
        self.prepare()
        self.report()
        report = json.loads((self.directory / "evaluation.json").read_text())
        archived = self.directory / "evaluation-attempts" / (report["evaluation_attempt_id"] + ".json")
        original = archived.read_bytes()
        archived.unlink()
        with self.assertRaises(OSError):
            mark_server_overload_diagnostic_reviewed(self.path)
        archived.write_bytes(original + b" ")
        with self.assertRaises(ValueError):
            mark_server_overload_diagnostic_reviewed(self.path)
        archived.write_bytes(original)
        journal = self.directory / "evaluation-steps.jsonl"
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        cases = [[], events[:1], events + [{**events[0], "attempt_id": str(uuid4())}],
                 [{**e, "workspace_id": "other"} for e in events]]
        for altered in cases:
            with self.subTest(altered=altered):
                journal.write_text("".join(json.dumps(e) + "\n" for e in altered))
                with self.assertRaises(ValueError):
                    mark_server_overload_diagnostic_reviewed(self.path)
                self.assertEqual(classify_existing(self.directory), "blocked")


if __name__ == "__main__":
    unittest.main()
