from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

from codex_benchmark.parallel_queue import (
    claim_next, classify_existing, finish_claim, initialize_queue,
    queue_pause_reason, queue_snapshot, set_queue_paused, clear_drained_queue_pause,
)


class ParallelQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.db = self.root / "queue.sqlite3"
        self.models = ["luna", "astra", "sol"]
        self.schedule = [(f"task-{i}", model) for model in self.models for i in range(4)]

    def initialize(self):
        initialize_queue(self.db, self.schedule, self.models, self.root)

    def episode(self, model="luna", task="task-0", status="completed", evaluation="completed", **fields):
        directory = self.root / model / task
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "episode.json").write_text(json.dumps({"status": status, **fields}))
        if evaluation is not None:
            (directory / "evaluation.json").write_text(json.dumps({"status": evaluation, "reward": 0}))
        return directory

    def test_concurrent_claims_are_unique_and_respect_model_barriers(self):
        self.initialize()
        for model in self.models:
            with ThreadPoolExecutor(max_workers=12) as pool:
                claims = list(pool.map(lambda i: claim_next(self.db, f"worker-{i}", i + 1), range(12)))
            claimed = [claim for claim in claims if claim]
            self.assertEqual(len(claimed), 4)
            self.assertEqual({claim["model"] for claim in claimed}, {model})
            self.assertEqual(len({claim["task_id"] for claim in claimed}), 4)
            self.assertIsNone(claim_next(self.db, "waiting", 99))
            for claim in claimed[:-1]:
                finish_claim(self.db, claim["worker_id"], model, claim["task_id"], "done")
            self.assertIsNone(claim_next(self.db, "waiting", 99))
            last = claimed[-1]
            finish_claim(self.db, last["worker_id"], model, last["task_id"], "done")
        self.assertIsNone(claim_next(self.db, "waiting", 99))
        self.assertEqual({row["status"] for row in queue_snapshot(self.db)}, {"done"})

    def test_existing_completed_and_reviewed_invalid_results_are_preserved(self):
        first = self.episode()
        second = self.episode(task="task-1", benchmark_validity={"status": "excluded_infrastructure"},
                              recovery={"reviewed": True})
        before = {path: path.read_bytes() for directory in (first, second) for path in directory.iterdir()}
        self.initialize()
        self.assertEqual([row["status"] for row in queue_snapshot(self.db)[:2]], ["done", "done"])
        self.initialize()
        self.assertEqual(claim_next(self.db, "worker", 1)["task_id"], "task-2")
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_running_infrastructure_and_unreviewed_exclusion_block_claims(self):
        for status, fields in (("running", {}), ("infrastructure_error", {}), ("failed", {}),
                               ("completed", {"benchmark_validity": {"status": "excluded_infrastructure"}})):
            with self.subTest(status=status, fields=fields):
                self.episode(status=status, **fields)
                self.initialize()
                self.assertEqual(queue_snapshot(self.db)[0]["status"], "blocked")
                self.assertIsNone(claim_next(self.db, "worker", 1))

    def test_terminal_without_evaluation_is_pending_for_grading_only(self):
        for index, status in enumerate(("completed", "timeout", "action_limit")):
            directory = self.episode(task=f"task-{index}", status=status, evaluation=None)
            before = (directory / "episode.json").read_bytes()
            self.assertEqual(classify_existing(directory), "pending")
            self.assertEqual((directory / "episode.json").read_bytes(), before)
        self.initialize()
        self.assertEqual([row["status"] for row in queue_snapshot(self.db)[:3]], ["pending"] * 3)

    def test_failed_evaluator_without_episode_flag_remains_pending(self):
        directory = self.episode(evaluation="evaluation_error")
        self.assertEqual(classify_existing(directory), "pending")
        self.initialize()
        self.assertEqual(claim_next(self.db, "worker", 1)["task_id"], "task-0")

    def test_only_claim_owner_can_finish_and_blocked_stops_new_claims(self):
        self.initialize()
        claim = claim_next(self.db, "owner", 1)
        with self.assertRaises(RuntimeError):
            finish_claim(self.db, "other", claim["model"], claim["task_id"], "done")
        with self.assertRaises(ValueError):
            finish_claim(self.db, "owner", claim["model"], claim["task_id"], "pending")
        finish_claim(self.db, "owner", claim["model"], claim["task_id"], "blocked", "Browser needs inspection")
        self.assertIsNone(claim_next(self.db, "next", 2))
        self.assertEqual(queue_snapshot(self.db)[0]["error"], "Browser needs inspection")

    def test_stale_active_claim_requires_saved_terminal_evidence(self):
        self.initialize()
        claim_next(self.db, "old-worker", 1)
        self.initialize()
        self.assertEqual(queue_snapshot(self.db)[0]["status"], "blocked")
        self.episode(status="running", evaluation=None)
        self.initialize()
        self.assertEqual(queue_snapshot(self.db)[0]["status"], "blocked")
        self.episode(evaluation=None)
        self.initialize()
        row = queue_snapshot(self.db)[0]
        self.assertEqual(row["status"], "pending")
        self.assertIsNone(row["worker_id"])
        self.assertIsNone(row["pid"])

    def test_pause_prevents_new_claims_but_allows_existing_claim_to_finish(self):
        self.initialize()
        claim = claim_next(self.db, "active", 1)
        set_queue_paused(self.db, "Inspect browser authentication")
        self.assertEqual(queue_pause_reason(self.db), "Inspect browser authentication")
        self.assertIsNone(claim_next(self.db, "new", 2))
        finish_claim(self.db, "active", claim["model"], claim["task_id"], "done")
        self.initialize()
        self.assertEqual(queue_pause_reason(self.db), "Inspect browser authentication")
        self.assertIsNone(claim_next(self.db, "new", 2))
        set_queue_paused(self.db, None)
        self.assertIsNone(queue_pause_reason(self.db))
        self.assertIsNotNone(claim_next(self.db, "new", 2))

    def test_atomic_pause_clear_preserves_new_pause_and_active_claim(self):
        self.initialize()
        claim = claim_next(self.db, "worker", 42)
        set_queue_paused(self.db, "authorized drain")
        with self.assertRaisesRegex(RuntimeError, "active or unresolved"):
            clear_drained_queue_pause(self.db, "authorized drain")
        finish_claim(self.db, "worker", claim["model"], claim["task_id"], "done")
        set_queue_paused(self.db, "new auth problem")
        with self.assertRaisesRegex(RuntimeError, "pause changed"):
            clear_drained_queue_pause(self.db, "authorized drain")
        self.assertEqual(queue_pause_reason(self.db), "new auth problem")
        set_queue_paused(self.db, "authorized drain")
        clear_drained_queue_pause(self.db, "authorized drain")
        self.assertIsNone(queue_pause_reason(self.db))

    def test_changed_schedule_is_rejected_without_mutating_queue(self):
        self.initialize()
        before = queue_snapshot(self.db)
        with self.assertRaises(ValueError):
            initialize_queue(self.db, list(reversed(self.schedule)), self.models, self.root)
        self.assertEqual(queue_snapshot(self.db), before)


if __name__ == "__main__":
    unittest.main()
