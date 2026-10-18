import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import asyncio
from types import SimpleNamespace
from pathlib import Path

from codex_benchmark.config import MODELS, ROOT
from codex_benchmark.knows import discover_tasks
from codex_benchmark.runner import SMOKE_TASK, episode_schedule, summarize, run_episode, grade_saved_episode


class RunnerTests(unittest.TestCase):
    def test_smoke_is_same_three_episodes_as_start_of_full_suite(self):
        tasks = discover_tasks(ROOT)
        full = episode_schedule(tasks, list(MODELS))
        smoke = episode_schedule(tasks, list(MODELS), smoke_only=True)
        self.assertEqual(len(tasks), 110)
        self.assertEqual(len(full), 330)
        self.assertEqual(len({(task.task_id, model) for task, model in full}), 330)
        self.assertEqual(smoke, full[:3])
        self.assertTrue(all(task.task_id == SMOKE_TASK for task, _ in smoke))

    def test_unscored_infrastructure_failure_is_not_a_zero_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "manifest.json").write_text(json.dumps({"models": ["model"], "tasks_per_model": 110}))
            for task, status, evaluation in [
                ("one", "completed", {"status": "completed", "reward": .5}),
                ("two", "infrastructure_error", {"status": "evaluation_error", "reward": None}),
            ]:
                path = directory / "model" / task
                path.mkdir(parents=True)
                (path / "episode.json").write_text(json.dumps({"task_id": task, "model": "model", "effort": "max", "status": status}))
                (path / "evaluation.json").write_text(json.dumps(evaluation))
            summary = summarize(directory)["models"]["model"]
            self.assertEqual(summary["mean_reward"], .5)
            self.assertEqual(summary["scored"], 1)
            self.assertEqual(summary["attempted"], 2)
            self.assertEqual(summary["infrastructure_errors"], 1)

    def test_judge_timeout_preserves_agent_status_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            episode_path = directory / "model" / "task" / "episode.json"
            episode_path.parent.mkdir(parents=True)
            episode = {"model": "model", "task_id": "task", "effort": "max", "status": "completed"}
            episode_path.write_text(json.dumps(episode))
            prior = {"status": "evaluation_error", "reward": None,
                     "detailed_report": {"steps": [{"name": "completed before retry"}]}}
            (episode_path.parent / "evaluation.json").write_text(json.dumps(prior))
            config = {"evaluation_timeout_seconds": 1}
            with patch("codex_benchmark.runner.subprocess_logged", new=AsyncMock(side_effect=TimeoutError("judge unavailable"))):
                asyncio.run(grade_saved_episode(config, episode, episode_path))
            self.assertEqual(episode["status"], "completed")
            self.assertIn("evaluation_error", episode)
            archived = [json.loads(path.read_text()) for path in
                        (episode_path.parent / "evaluation-attempts").glob("*.json")]
            self.assertEqual(len(archived), 2)
            self.assertIn(prior, archived)
            self.assertTrue(any("step_journal_path" in report for report in archived))

            async def successful_grade(*args, **kwargs):
                (episode_path.parent / "evaluation.json").write_text(json.dumps({"status": "completed", "reward": .5}))
                return 0

            with patch("codex_benchmark.runner.subprocess_logged", side_effect=successful_grade) as grade, \
                 patch("codex_benchmark.browser.BrowserSession") as browser:
                result = asyncio.run(run_episode(config, SimpleNamespace(task_id="task"), "model", directory))
                self.assertEqual(grade.call_count, 1)
                browser.assert_not_called()
            self.assertEqual(result["status"], "completed")
            self.assertNotIn("evaluation_error", result)


if __name__ == "__main__":
    unittest.main()
