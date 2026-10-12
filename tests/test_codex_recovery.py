"""Recovery of saved model outcomes never replays a benchmark episode."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from codex_benchmark.runner import persistent_browser_failure, recover_saved_episode, run_episode, summarize


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name)
        self.model = "gpt-6-luna"
        self.task_id = "knows.docs_31_education_lesson_plan.4"
        self.directory = self.run_dir / self.model / self.task_id
        self.directory.mkdir(parents=True)
        self.episode_path = self.directory / "episode.json"
        self.original = {
            "task_id": self.task_id, "model": self.model, "effort": "max",
            "status": "running", "started_at": "2026-10-02T23:16:28+00:00",
            "workspace_id": "workspace123",
            "workspace_url": "https://docs.google.com/document/d/workspace123/edit",
            "prompt_sha256": "saved-prompt-hash", "browsing_history": [],
        }
        self.episode_path.write_text(json.dumps(self.original))
        self.history = ["https://example.test/source?q=full", self.original["workspace_url"]]
        (self.directory / "history.json").write_text(json.dumps(self.history))
        self.result = {
            "model": self.model, "effort": "max", "thread_id": "thread-1",
            "turn_id": "turn-1", "status": "completed", "tool_calls": 24,
            "final_text": "The browser stopped responding; the document is incomplete.",
            "token_usage": {"total": {"totalTokens": 12345, "inputTokens": 12000,
                                       "outputTokens": 345}},
            "error": None,
        }
        self.write_log([self.result])

    def write_log(self, results, *, confirmation=True, confirmation_turn="turn-1"):
        records = []
        if confirmation:
            records.append({"time": 1790984830.69, "direction": "receive", "message": {
                "method": "turn/completed", "params": {"threadId": "thread-1", "turn": {
                    "id": confirmation_turn, "status": "completed", "error": None,
                    "startedAt": 1790983051, "completedAt": 1790984830,
                }}}})
        records.extend({"time": 1790984830.70 + index, "direction": "run_result", "message": result}
                       for index, result in enumerate(results))
        (self.directory / "app-server.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records))

    def assert_recovery_rejected(self):
        before = self.episode_path.read_bytes()
        with self.assertRaises((ValueError, RuntimeError)):
            recover_saved_episode(self.episode_path)
        self.assertEqual(self.episode_path.read_bytes(), before)
        self.assertFalse((self.directory / "agent-result.json").exists())

    def test_recovery_preserves_model_result_history_and_original_episode(self):
        reason = "Browser transport failed during the model turn"
        recovered = recover_saved_episode(self.episode_path, infrastructure_reason=reason)
        saved = json.loads(self.episode_path.read_text())
        agent = json.loads((self.directory / "agent-result.json").read_text())
        self.assertEqual(saved, recovered)
        for key, value in self.result.items():
            self.assertEqual(recovered[key], value, key)
            self.assertEqual(agent[key], value, key)
        self.assertEqual(recovered["browsing_history"], self.history)
        self.assertEqual(recovered["workspace_id"], self.original["workspace_id"])
        self.assertEqual(recovered["prompt_sha256"], self.original["prompt_sha256"])
        self.assertEqual(recovered["benchmark_validity"], {
            "status": "excluded_infrastructure", "reason": reason})
        archives = [json.loads(path.read_text()) for path in self.directory.rglob("*.json")
                    if path != self.episode_path]
        self.assertIn(self.original, archives)
        self.assertFalse((self.directory / "evaluation.json").exists())

    def test_missing_or_ambiguous_model_result_is_rejected(self):
        for results in ([], [self.result, {**self.result, "turn_id": "turn-2"}]):
            with self.subTest(result_count=len(results)):
                self.write_log(results)
                self.assert_recovery_rejected()

    def test_model_effort_and_turn_must_match(self):
        for changed in ({"model": "gpt-6-astra"}, {"effort": "low"}):
            with self.subTest(changed=changed):
                self.write_log([{**self.result, **changed}])
                self.assert_recovery_rejected()
        self.write_log([self.result], confirmation_turn="different-turn")
        self.assert_recovery_rejected()

    def test_completed_result_requires_completed_turn_confirmation(self):
        self.write_log([self.result], confirmation=False)
        self.assert_recovery_rejected()

    def test_unfinished_or_already_finalized_episode_is_not_recovered(self):
        for status in ("inProgress", "failed", "interrupted"):
            with self.subTest(status=status):
                self.write_log([{**self.result, "status": status}])
                self.assert_recovery_rejected()
        self.write_log([self.result])
        self.episode_path.write_text(json.dumps({**self.original, "status": "completed"}))
        self.assert_recovery_rejected()

    def test_existing_result_or_evaluation_is_never_overwritten(self):
        for name in ("agent-result.json", "evaluation.json"):
            with self.subTest(name=name):
                path = self.directory / name
                original = '{"preserved": true}\n'
                path.write_text(original)
                before = self.episode_path.read_bytes()
                with self.assertRaises((ValueError, RuntimeError)):
                    recover_saved_episode(self.episode_path)
                self.assertEqual(path.read_text(), original)
                self.assertEqual(self.episode_path.read_bytes(), before)
                path.unlink()

    def test_timeout_and_action_limit_remain_terminal_partial_runs(self):
        for status in ("timeout", "action_limit"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                path = directory / "episode.json"
                path.write_text(json.dumps(self.original))
                (directory / "history.json").write_text(json.dumps(self.history))
                result = {**self.result, "status": status, "error": "Budget exhausted"}
                (directory / "app-server.jsonl").write_text(json.dumps({
                    "time": 1790984830.70, "direction": "run_result", "message": result}) + "\n")
                recovered = recover_saved_episode(path)
                self.assertEqual(recovered["status"], status)
                self.assertEqual(recovered["error"], result["error"])
                self.assertEqual(recovered["tool_calls"], 24)

    def test_resume_grades_recovered_result_once_without_replaying_model(self):
        recover_saved_episode(self.episode_path)
        config = {"evaluation_timeout_seconds": 1}

        async def successful_grade(*args, **kwargs):
            (self.directory / "evaluation.json").write_text(json.dumps({
                "status": "completed", "reward": .5}))
            return 0

        with patch("codex_benchmark.runner.subprocess_logged", side_effect=successful_grade) as grader, \
             patch("codex_benchmark.browser.BrowserSession") as browser, \
             patch("codex_benchmark.app_server.AppServer") as server:
            task = SimpleNamespace(task_id=self.task_id)
            first = asyncio.run(run_episode(config, task, self.model, self.run_dir))
            evaluation_before = (self.directory / "evaluation.json").read_bytes()
            second = asyncio.run(run_episode(config, task, self.model, self.run_dir))
            self.assertEqual(grader.call_count, 1)
            self.assertIn("codex_benchmark.evaluate", grader.call_args.args[0])
            browser.assert_not_called()
            server.assert_not_called()
        self.assertEqual(first["final_text"], self.result["final_text"])
        self.assertEqual(second["final_text"], self.result["final_text"])
        self.assertEqual((self.directory / "evaluation.json").read_bytes(), evaluation_before)

    def test_resume_retries_saved_evaluation_error_without_episode_error_flag(self):
        recover_saved_episode(self.episode_path)
        self.assertNotIn("evaluation_error", json.loads(self.episode_path.read_text()))
        (self.directory / "evaluation.json").write_text(json.dumps({
            "status": "evaluation_error", "reward": None, "error": "Judge request failed"}))

        async def successful_grade(*args, **kwargs):
            (self.directory / "evaluation.json").write_text(json.dumps({
                "status": "completed", "reward": .5}))
            return 0

        with patch("codex_benchmark.runner.subprocess_logged", side_effect=successful_grade) as grader, \
             patch("codex_benchmark.browser.BrowserSession") as browser, \
             patch("codex_benchmark.app_server.AppServer") as server:
            task = SimpleNamespace(task_id=self.task_id)
            config = {"evaluation_timeout_seconds": 1}
            result = asyncio.run(run_episode(config, task, self.model, self.run_dir))
            asyncio.run(run_episode(config, task, self.model, self.run_dir))
            self.assertEqual(grader.call_count, 1)
            browser.assert_not_called()
            server.assert_not_called()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["final_text"], self.result["final_text"])

    def write_tool_replies(self, replies):
        path = self.directory / "app-server.jsonl"
        path.write_text("".join(json.dumps({
            "time": 1790984830 + index, "direction": "send", "message": {
                "id": f"browser-call-{index}", "result": {
                    "success": success,
                    "contentItems": [{"type": "inputText", "text": text}],
                }}}) + "\n" for index, (success, text) in enumerate(replies)))
        return path

    def test_terminal_observation_timeouts_flag_infrastructure_failure(self):
        path = self.write_tool_replies([
            (False, "Observation failed: TimeoutError: browser unresponsive") for _ in range(3)])
        failure = persistent_browser_failure(path)
        self.assertIsNotNone(failure)
        self.assertEqual(failure["status"], "excluded_infrastructure")
        self.assertEqual(failure["consecutive_observation_timeouts"], 3)

    def test_successful_browser_reply_clears_prior_observation_outage(self):
        path = self.write_tool_replies([
            *[(False, "Observation failed: TimeoutError: browser unresponsive") for _ in range(3)],
            (True, "Page observation: editor is ready"),
        ])
        self.assertIsNone(persistent_browser_failure(path))

    def test_action_budget_reply_does_not_hide_terminal_browser_outage(self):
        path = self.write_tool_replies([
            *[(False, "Observation failed: TimeoutError: browser unresponsive") for _ in range(3)],
            (False, "Browser action budget exhausted."),
        ])
        failure = persistent_browser_failure(path)
        self.assertIsNotNone(failure)
        self.assertEqual(failure["status"], "excluded_infrastructure")
        self.assertEqual(failure["consecutive_observation_timeouts"], 3)

    def test_single_or_recovered_timeout_does_not_exclude_episode(self):
        for recovered in (False, True):
            with self.subTest(recovered=recovered):
                replies = [(False, "Observation failed: TimeoutError: browser unresponsive")]
                if recovered:
                    replies.append((True, "Page observation: editor is ready"))
                path = self.write_tool_replies(replies)
                self.assertIsNone(persistent_browser_failure(path))

    def test_summary_excludes_infrastructure_invalid_diagnostic_zero(self):
        (self.run_dir / "manifest.json").write_text(json.dumps({
            "models": [self.model], "tasks_per_model": 110}))
        recover_saved_episode(self.episode_path, infrastructure_reason="Browser transport failure")
        (self.directory / "evaluation.json").write_text(json.dumps({
            "status": "completed", "reward": 0, "score_source": "official_evaluator"}))
        valid = self.run_dir / self.model / "valid-task"
        valid.mkdir()
        (valid / "episode.json").write_text(json.dumps({
            "task_id": "valid-task", "model": self.model, "effort": "max", "status": "completed"}))
        (valid / "evaluation.json").write_text(json.dumps({"status": "completed", "reward": .5}))
        summary = summarize(self.run_dir)
        row = next(row for row in summary["episodes"] if row["task_id"] == self.task_id)
        self.assertEqual(row["evaluation_status"], "completed")
        self.assertIsNone(row["reward"])
        self.assertEqual(row["diagnostic_reward"], 0)
        model = summary["models"][self.model]
        self.assertEqual(model["evaluated"], 2)
        self.assertEqual(model["scored"], 1)
        self.assertEqual(model["mean_reward"], .5)
        self.assertEqual(model["infrastructure_errors"], 1)


if __name__ == "__main__":
    unittest.main()
