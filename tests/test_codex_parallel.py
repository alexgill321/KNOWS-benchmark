import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from codex_benchmark.parallel import coordinate, worker
from codex_benchmark.parallel_queue import (
    initialize_queue, queue_pause_reason, queue_snapshot, set_queue_paused,
)


class ParallelWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.db = self.run_dir / "parallel.sqlite"
        self.config = {"config_path": str(self.root / "config.json"), "benchmark_root": str(self.root)}
        self.tasks = [SimpleNamespace(task_id="first"), SimpleNamespace(task_id="second")]
        self.models = ["luna"]
        self.lock = (self.root / ".account.lock").open("a")
        self.addCleanup(self.lock.close)

    def save_episode(self, task_id, *, evaluated=False):
        directory = self.run_dir / "luna" / task_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "episode.json").write_text(json.dumps({"status": "completed"}))
        if evaluated:
            (directory / "evaluation.json").write_text(json.dumps({"status": "completed", "reward": .5}))
        return directory

    def initialize(self):
        initialize_queue(self.db, [(task.task_id, "luna") for task in self.tasks], self.models, self.run_dir)

    def run_worker(self, run_episode, *, parent_sequence=None):
        parent = os.getppid()
        with ExitStack() as stack:
            stack.enter_context(patch("codex_benchmark.parallel.apply_evaluator_environment"))
            stack.enter_context(patch("codex_benchmark.parallel.discover_tasks", return_value=self.tasks))
            called = stack.enter_context(patch("codex_benchmark.parallel.run_episode", side_effect=run_episode))
            summary = stack.enter_context(patch("codex_benchmark.parallel.summarize"))
            if parent_sequence:
                stack.enter_context(patch("codex_benchmark.parallel.os.getppid", side_effect=parent_sequence))
            result = asyncio.run(worker(self.config, self.run_dir, "worker-1", parent, self.lock.fileno()))
            summary.assert_not_called()
        return result, called

    def test_saved_terminal_episode_is_evaluated_once_and_completed_episode_untouched(self):
        completed = self.save_episode("first", evaluated=True)
        pending = self.save_episode("second")
        preserved = {path: path.read_bytes() for path in completed.iterdir()}
        pending_before = (pending / "episode.json").read_bytes()
        self.initialize()

        async def grade_saved(config, task, model, run_dir):
            self.assertEqual(task.task_id, "second")
            self.assertEqual((pending / "episode.json").read_bytes(), pending_before)
            self.save_episode("second", evaluated=True)
            return {"status": "completed"}

        result, called = self.run_worker(grade_saved)
        self.assertEqual(result, 0)
        self.assertEqual(called.call_count, 1)
        self.assertEqual({path: path.read_bytes() for path in preserved}, preserved)
        self.assertEqual({row["status"] for row in queue_snapshot(self.db)}, {"done"})
        result, called = self.run_worker(grade_saved)
        self.assertEqual(result, 0)
        called.assert_not_called()

    def test_worker_exception_blocks_queue_without_starting_next_episode(self):
        self.initialize()

        async def failed(*args):
            raise RuntimeError("Private details must not reach queue metadata")

        result, called = self.run_worker(failed)
        self.assertEqual(result, 2)
        self.assertEqual(called.call_count, 1)
        rows = queue_snapshot(self.db)
        self.assertEqual([row["status"] for row in rows], ["blocked", "pending"])
        self.assertEqual(rows[0]["error"], "RuntimeError")

    def test_pause_during_episode_allows_completion_but_stops_new_claims(self):
        self.initialize()

        async def finish_current(config, task, model, run_dir):
            set_queue_paused(self.db, "Another worker needs inspection")
            self.save_episode(task.task_id, evaluated=True)
            return {"status": "completed"}

        result, called = self.run_worker(finish_current)
        self.assertEqual(result, 2)
        self.assertEqual(called.call_count, 1)
        self.assertEqual([row["status"] for row in queue_snapshot(self.db)], ["done", "pending"])

    def test_orphan_worker_does_not_claim_new_work(self):
        self.initialize()
        result, called = self.run_worker(AsyncMock(), parent_sequence=[os.getppid(), -1])
        self.assertEqual(result, 3)
        called.assert_not_called()
        self.assertEqual({row["status"] for row in queue_snapshot(self.db)}, {"pending"})

    def test_coordinator_pauses_when_worker_dies_before_first_claim(self):
        config = {**self.config, "models": self.models, "efforts": {"luna": "max"},
                  "judge_model": "test-judge", "max_actions": 5, "episode_timeout_seconds": 10, "seed": 0}
        manifest = {key: config[key] for key in ("models", "efforts", "judge_model", "max_actions",
                                                 "episode_timeout_seconds", "seed")}
        dead = SimpleNamespace(pid=101, poll=lambda: 1)
        polls = iter([None, None, 0])
        finishing = SimpleNamespace(pid=102, poll=lambda: next(polls, 0))
        with patch("codex_benchmark.parallel.apply_evaluator_environment"), \
             patch("codex_benchmark.parallel.discover_tasks", return_value=self.tasks), \
             patch("codex_benchmark.parallel.validate_contract", return_value=manifest), \
             patch("codex_benchmark.parallel.check_judge_access"), \
             patch("codex_benchmark.parallel.preflight_models", new=AsyncMock()), \
             patch("codex_benchmark.parallel.snapshot_sources", return_value="test-revision"), \
             patch("codex_benchmark.parallel.prepare_worker_config", return_value=self.config), \
             patch("codex_benchmark.parallel.subprocess.Popen", side_effect=[dead, finishing]) as launch, \
             patch("codex_benchmark.parallel.summarize", return_value={"models": {}}), \
             patch("codex_benchmark.parallel.time.sleep"), \
             patch.dict("sys.modules", {"codex_benchmark.parallel_health": SimpleNamespace(inspect_episode=lambda path: None)}):
            result = coordinate(config, self.run_dir, 2, self.lock.fileno())
        self.assertEqual(result, 2)
        self.assertIsNotNone(queue_pause_reason(self.db))
        self.assertEqual({row["status"] for row in queue_snapshot(self.db)}, {"pending"})
        self.assertEqual(launch.call_count, 2)
        self.assertTrue(all(call.kwargs["pass_fds"] == (self.lock.fileno(),) for call in launch.call_args_list))

    def test_retry_coordinator_schedules_only_validated_batch(self):
        config = {**self.config, "models": self.models, "efforts": {"luna": "max"},
                  "judge_model": "judge", "max_actions": 150, "episode_timeout_seconds": 1800, "seed": 0}
        manifest = {**config, "kind": "authorized_infrastructure_retry"}
        dead = SimpleNamespace(pid=101, poll=lambda: 1)
        with patch("codex_benchmark.parallel.apply_evaluator_environment"), \
             patch("codex_benchmark.parallel.discover_tasks", return_value=self.tasks), \
             patch("codex_benchmark.parallel.validate_contract", return_value=manifest), \
             patch("codex_benchmark.infrastructure_retry.select_retry_tasks", return_value=self.tasks[:1]), \
             patch("codex_benchmark.parallel.check_judge_access"), \
             patch("codex_benchmark.parallel.preflight_models", new=AsyncMock()), \
             patch("codex_benchmark.parallel.snapshot_sources", return_value="test"), \
             patch("codex_benchmark.parallel.prepare_worker_config", return_value=self.config), \
             patch("codex_benchmark.parallel.subprocess.Popen", return_value=dead), \
             patch("codex_benchmark.parallel.summarize", return_value={"models": {}}):
            self.assertEqual(coordinate(config, self.run_dir, 2, self.lock.fileno()), 2)
        self.assertEqual([row["task_id"] for row in queue_snapshot(self.db)], ["first"])

if __name__ == "__main__":
    unittest.main()
