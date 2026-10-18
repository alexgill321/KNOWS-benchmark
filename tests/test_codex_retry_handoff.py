import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codex_benchmark.retry_handoff import handoff


class RetryHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.parent, self.retry = root / "main", root / "retry"
        episode = self.parent / "luna" / "task"
        episode.mkdir(parents=True)
        (episode / "episode.json").write_text('{"status":"completed"}')
        self.report = episode / "evaluation.json"
        self.report.write_text('{"status":"completed","reward":0.5}')
        self.state_path = root / "handoff.json"
        self.state_path.write_text(json.dumps({"phase": "ready_for_retry", "max_total_workers": 2,
            "parent_run_dir": str(self.parent), "retry_run_dir": str(self.retry),
            "pause_reason": "authorized drain", "archive": str(root / "archive")}))
        self.lock = (root / "lock").open("a")
        self.addCleanup(self.lock.close)
        self.config = {"benchmark_root": str(root)}

    def invoke(self, coordinate, pause=None):
        with patch("codex_benchmark.retry_handoff.discover_tasks", return_value=[]), \
             patch("codex_benchmark.retry_handoff.validate_contract", return_value={
                 "kind": "authorized_infrastructure_retry", "task_ids": ["task"],
                 "parent_run_dir": str(self.parent)}), \
             patch("codex_benchmark.retry_handoff.queue_snapshot", return_value=[{"status": "done"}]), \
             patch("codex_benchmark.retry_handoff.queue_pause_reason", side_effect=pause,
                   return_value="authorized drain"), \
             patch("codex_benchmark.retry_handoff.clear_drained_queue_pause") as clear, \
             patch("codex_benchmark.retry_handoff.coordinate", side_effect=coordinate) as run:
            result = handoff(self.state_path, self.config, self.config, self.lock.fileno())
        return result, clear, run

    def test_sequential_two_worker_runs_retain_same_account_lock(self):
        result, clear, run = self.invoke([0, 0])
        self.assertEqual(result, 0)
        self.assertEqual([call.args[1] for call in run.call_args_list], [self.retry, self.parent])
        self.assertTrue(all(call.args[2:] == (2, self.lock.fileno()) for call in run.call_args_list))
        clear.assert_called_once_with(self.parent / "parallel.sqlite", "authorized drain")
        self.assertEqual(json.loads(self.state_path.read_text())["phase"], "main_completed")

    def test_failed_retry_does_not_resume_main_or_clear_pause(self):
        result, clear, run = self.invoke([2])
        self.assertEqual(result, 2)
        self.assertEqual(run.call_count, 1)
        clear.assert_not_called()
        self.assertEqual(json.loads(self.state_path.read_text())["phase"], "retry_needs_attention")

    def test_changed_original_record_prevents_main_handoff(self):
        def changed(*args):
            self.report.write_text('{"status":"completed","reward":1}')
            return 0
        with self.assertRaisesRegex(RuntimeError, "original completed record changed"):
            self.invoke(changed)
        self.assertEqual(json.loads(self.state_path.read_text())["phase"], "handoff_failed")

    def test_changed_main_pause_is_not_cleared(self):
        with self.assertRaisesRegex(RuntimeError, "pause changed"):
            self.invoke([0], ["authorized drain", "Google auth requires owner"])


if __name__ == "__main__":
    unittest.main()
