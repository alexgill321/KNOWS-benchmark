from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from codex_benchmark.parallel_queue import (
    cancel_pending_model, claim_next, finish_claim, initialize_queue,
    queue_pause_reason, queue_snapshot, set_queue_paused,
)


class QueueCancellationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'parallel.sqlite'
        self.models = ['luna', 'astra', 'sol']
        self.schedule = [(f'task-{i}', m) for m in self.models for i in range(3)]
        initialize_queue(self.db, self.schedule, self.models, self.root)

    def cancel(self, **kwargs):
        return cancel_pending_model(self.db, self.root, 'sol',
            reason='Owner skipped remaining Sol', authorization='Skip Sol', **kwargs)

    def archived(self):
        with sqlite3.connect(self.db) as db:
            return db.execute('SELECT model,task_id FROM cancelled_claims ORDER BY ordinal').fetchall()

    def test_live_other_model_and_completed_sol_are_preserved(self):
        directory = self.root / 'sol/task-0'
        directory.mkdir(parents=True)
        original = json.dumps({'status': 'completed', 'model': 'sol'})
        (directory / 'episode.json').write_text(original)
        (directory / 'evaluation.json').write_text('{"status":"completed","reward":0.5}')
        initialize_queue(self.db, self.schedule, self.models, self.root)
        claim = claim_next(self.db, 'healthy', 100)
        set_queue_paused(self.db, 'existing unrelated pause')
        before = queue_snapshot(self.db)
        self.cancel(expected_count=2)
        self.assertEqual(queue_snapshot(self.db), [r for r in before if not (r['model'] == 'sol' and r['status'] == 'pending')])
        self.assertEqual(queue_pause_reason(self.db), 'existing unrelated pause')
        self.assertEqual((directory / 'episode.json').read_text(), original)
        self.assertEqual(len(self.archived()), 2)
        finish_claim(self.db, 'healthy', claim['model'], claim['task_id'], 'done')

    def test_restart_does_not_resurrect_cancelled_tasks_or_renumber(self):
        self.cancel(expected_count=3)
        before = queue_snapshot(self.db)
        initialize_queue(self.db, self.schedule, self.models, self.root)
        self.assertEqual(queue_snapshot(self.db), before)
        self.assertEqual(len(self.archived()), 3)
        for _ in range(6):
            row = claim_next(self.db, 'worker', 7)
            self.assertNotEqual(row['model'], 'sol')
            finish_claim(self.db, 'worker', row['model'], row['task_id'], 'done')
        self.assertIsNone(claim_next(self.db, 'worker', 7))
        self.assertTrue(all(r['status'] == 'done' for r in queue_snapshot(self.db)))
        initialize_queue(self.db, self.schedule, self.models, self.root)
        self.assertIsNone(claim_next(self.db, 'worker', 7))

    def test_middle_model_cancellation_preserves_later_ordinal_gaps(self):
        cancel_pending_model(self.db, self.root, 'astra', reason='Scope change', authorization='Skip Astra')
        initialize_queue(self.db, self.schedule, self.models, self.root)
        self.assertEqual([r['ordinal'] for r in queue_snapshot(self.db)], [0, 1, 2, 6, 7, 8])

    def test_saved_pending_evidence_rolls_back_entire_batch(self):
        directory = self.root / 'sol/task-2'
        directory.mkdir(parents=True)
        (directory / 'agent-result.json').write_text('{"status":"completed"}')
        before = queue_snapshot(self.db)
        with self.assertRaises((ValueError, RuntimeError)):
            self.cancel()
        self.assertEqual(queue_snapshot(self.db), before)
        self.assertEqual(self.archived(), [])

    def test_active_blocked_and_stale_pending_target_rejected(self):
        for state, worker in [('active', 'owned'), ('blocked', None), ('pending', 'stale')]:
            with self.subTest(state=state):
                with sqlite3.connect(self.db) as db:
                    db.execute('UPDATE claims SET status=?,worker_id=? WHERE model=? AND task_id=?', (state, worker, 'sol', 'task-0'))
                before = queue_snapshot(self.db)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.cancel()
                self.assertEqual(queue_snapshot(self.db), before)
                self.assertEqual(self.archived(), [])

    def test_count_guard_rejects_stale_request(self):
        before = queue_snapshot(self.db)
        with self.assertRaises((ValueError, RuntimeError)):
            self.cancel(expected_count=2)
        self.assertEqual(queue_snapshot(self.db), before)
        self.assertEqual(self.archived(), [])

    def test_blank_authorization_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            cancel_pending_model(self.db, self.root, 'sol', reason='Skip', authorization=' ')
        self.assertEqual(len(queue_snapshot(self.db)), 9)

    def test_changed_original_schedule_rejected_after_cancellation(self):
        self.cancel()
        before = queue_snapshot(self.db)
        with self.assertRaises((ValueError, RuntimeError)):
            initialize_queue(self.db, list(reversed(self.schedule)), self.models, self.root)
        self.assertEqual(queue_snapshot(self.db), before)

    def test_serial_resume_refuses_cancelled_campaign_before_any_setup(self):
        from codex_benchmark.runner import run
        self.cancel()
        with self.assertRaisesRegex(RuntimeError, 'cancelled task claims'):
            asyncio.run(run({}, self.root, 'full'))

    def test_claim_race_either_cancels_or_preserves_active_owner(self):
        for row in queue_snapshot(self.db):
            if row['model'] != 'sol':
                with sqlite3.connect(self.db) as db:
                    db.execute('UPDATE claims SET status=? WHERE model=? AND task_id=?', ('done', row['model'], row['task_id']))
        barrier = threading.Barrier(2)
        def cancel():
            barrier.wait()
            try:
                return self.cancel()
            except (ValueError, RuntimeError):
                return None
        def claim():
            barrier.wait()
            return claim_next(self.db, 'racing', 42)
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1, f2 = pool.submit(cancel), pool.submit(claim)
            cancelled, claimed = f1.result(), f2.result()
        if cancelled is not None:
            self.assertEqual(len(cancelled), 3)
            self.assertIsNone(claimed)
        else:
            self.assertEqual(claimed['worker_id'], 'racing')
            self.assertEqual(self.archived(), [])
            self.assertEqual(len([r for r in queue_snapshot(self.db) if r['model'] == 'sol']), 3)


if __name__ == '__main__':
    unittest.main()
