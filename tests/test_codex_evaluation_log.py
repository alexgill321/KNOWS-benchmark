import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from uuid import uuid4

from codex_benchmark.evaluation_log import EvaluationJournal


ROOT = Path(__file__).resolve().parents[1]
SCORING = ROOT / "src/browsergym/knows/eval/eval_utils/scoring.py"


def load_scoring():
    # Load the real dataclasses without importing BrowserGym or Google clients.
    name = "_journal_scoring_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, SCORING)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class EvaluationJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "evaluation-steps.jsonl"
        self.scoring = load_scoring()

    def events(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_steps_are_durable_before_exit_and_snapshots_preserve_adjustments(self):
        journal = EvaluationJournal(self.path, "task", "document", scoring_module=self.scoring)
        with journal:
            checkpoint = self.scoring.Checkpoint(total=5, result=0, name="Content")
            step = checkpoint.add_step("Match", True, 7, details="Matched exactly", score=2,
                                       max_score=3, execution_time=0.25, category="deterministic")
            recorded = self.events()[-1]
            self.assertEqual(recorded["event"], "step_completed")
            self.assertEqual(recorded["step"], {
                "step_id": 7, "name": "Match", "success": True, "details": "Matched exactly",
                "score": 2, "max_score": 3, "execution_time": 0.25, "category": "deterministic"})
            self.assertIs(step, checkpoint.steps[0])
            self.assertEqual(checkpoint.result, 2)
            step.category = "fuzzy_match"
            checkpoint.result = 1
            checkpoint.execution_time = 1.5
            # Constructors can receive completed steps without calling add_step.
            other = self.scoring.Checkpoint(1, 0, steps=[self.scoring.EvaluationStep("Absent", False, 1)])
        snapshots = [event for event in self.events() if event["event"] == "checkpoint_snapshot"]
        self.assertEqual([event["checkpoint_index"] for event in snapshots], [1, 2])
        self.assertEqual(snapshots[0]["checkpoint"]["result"], 1)
        self.assertEqual(snapshots[0]["checkpoint"]["execution_time"], 1.5)
        self.assertEqual(snapshots[0]["checkpoint"]["steps"][0]["category"], "fuzzy_match")
        self.assertEqual(snapshots[1]["checkpoint"]["steps"][0]["name"], other.steps[0].name)
        self.assertEqual(journal.report["completed_steps"], 2)
        self.assertEqual(journal.report["checkpoint_count"], 2)
        self.assertEqual(self.events()[-1]["event"], "evaluation_finished")

    def test_exception_restores_methods_and_retry_appends_a_distinct_attempt(self):
        original_init = self.scoring.Checkpoint.__post_init__
        original_add = self.scoring.Checkpoint.add_step
        first = EvaluationJournal(self.path, "task", "document", scoring_module=self.scoring)
        with self.assertRaisesRegex(RuntimeError, "judge unavailable"):
            with first:
                checkpoint = self.scoring.Checkpoint(2, 0, name="Content")
                checkpoint.add_step("Done", True, 1)
                raise RuntimeError("judge unavailable")
        self.assertIs(self.scoring.Checkpoint.__post_init__, original_init)
        self.assertIs(self.scoring.Checkpoint.add_step, original_add)
        before = self.path.read_bytes()
        self.assertEqual(self.events()[-1]["event"], "evaluation_error")
        self.assertEqual(first.report["completed_steps"], 1)
        second = EvaluationJournal(self.path, "task", "document", scoring_module=self.scoring)
        with second:
            checkpoint = self.scoring.Checkpoint(1, 0, name="Retry")
            checkpoint.add_step("Retried", False, 1)
        self.assertTrue(self.path.read_bytes().startswith(before))
        self.assertNotEqual(first.attempt_id, second.attempt_id)
        self.assertEqual({event["attempt_id"] for event in self.events()},
                         {first.attempt_id, second.attempt_id})
        self.assertIs(self.scoring.Checkpoint.__post_init__, original_init)
        self.assertIs(self.scoring.Checkpoint.add_step, original_add)

    def test_abrupt_worker_exit_keeps_completed_step(self):
        script = textwrap.dedent("""
            import importlib.util, os, sys
            from codex_benchmark.evaluation_log import EvaluationJournal
            spec = importlib.util.spec_from_file_location('_crash_scoring', sys.argv[1])
            scoring = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = scoring
            spec.loader.exec_module(scoring)
            with EvaluationJournal(sys.argv[2], 'task', 'document', scoring_module=scoring):
                checkpoint = scoring.Checkpoint(1, 0, name='Content')
                checkpoint.add_step('Saved before crash', True, 1, details='durable')
                os._exit(23)
        """)
        result = subprocess.run([sys.executable, "-c", script, str(SCORING), str(self.path)],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 23, result.stderr)
        events = self.events()
        self.assertEqual(events[-1]["event"], "step_completed")
        self.assertEqual(events[-1]["step"]["details"], "durable")
        self.assertEqual(events[-1]["checkpoint_result"], 1)


if __name__ == "__main__":
    unittest.main()
