"""Exercise the real five checkpoint functions without loading API clients."""
import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from codex_benchmark.config import ROOT as REPOSITORY_ROOT


SOURCE_ROOT = Path(__file__).resolve().parents[1]
FAMILY = SOURCE_ROOT / "src/browsergym/knows/eval/tasks/sheets_7_running_analysis"
EXPECTED = {1: 109, 2: 169, 3: 34, 4: 452, 5: 219}
_spec = importlib.util.spec_from_file_location(
    "running_bound_test_scoring",
    REPOSITORY_ROOT / "src/browsergym/knows/eval/eval_utils/scoring.py",
)
_scoring = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _scoring
_spec.loader.exec_module(_scoring)


def checkpoint(instance, matches, *, missing_columns=False, wrong_values=False, uncapped=False):
    """Load only the real checkpoint AST; fixture rows deliberately repeat a date."""
    source = FAMILY / f"instance_{instance}/evaluator.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "grade_checkpoint_2")
    if uncapped:
        for node in ast.walk(function):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "min"
                    and len(node.value.args) == 2 and isinstance(node.value.args[0], ast.Constant)
                    and node.value.args[0].value == 10):
                node.value = node.value.args[1]
    total = EXPECTED[instance]
    if instance == 5:
        gold = pd.DataFrame({"Activity Type": ["Strength Training"] * total,
                             "Date": ["2020-01-01"] * total,
                             "Total Sets": [10] * total, "Total Reps": [100] * total})
        user = pd.DataFrame({"Date": ["2020-01-01"] * matches,
                             "Sets": [11 if wrong_values else 10] * matches,
                             "Reps": [101 if wrong_values else 100] * matches})
        matched = {"Activity Date": "Date", "Total Sets": "Sets", "Total Reps": "Reps"}
    else:
        gold = pd.DataFrame({"Activity Date": ["2020-01-01"] * total,
                             "Distance": [1.60934] * total,
                             "Average Speed": [26.8224 / 10] * total})
        user = pd.DataFrame({"Date": ["2020-01-01"] * matches,
                             "Miles": [2.0 if wrong_values else 1.0] * matches,
                             "Pace": [20.0 if wrong_values else 10.0] * matches})
        matched = {"Activity Date": "Date"}
    unit_match = {} if missing_columns else {"Distance (Miles)": "Miles", "Speed (min/mile)": "Pace"}
    if missing_columns:
        matched = {}
    loader = Mock(return_value=gold)
    if instance == 4:
        loader.side_effect = [gold, gold.iloc[:0]]
    namespace = dict(os=os, pd=pd, math=math, time=time, DEBUG=False, model=None,
                     DATA_DIR="unused", EXPECTED_TOTAL=452, EXPECTED_STRENGTH=219,
                     table_data=SimpleNamespace(df=user), matched_columns=matched,
                     normalize_date=lambda value: str(value), load_gold_run_activities=loader,
                     match_columns=Mock(return_value=unit_match),
                     Checkpoint=_scoring.Checkpoint, StepCategory=_scoring.StepCategory)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(source), "exec"), namespace)
    with patch.object(pd, "read_csv", return_value=gold):
        return namespace["grade_checkpoint_2"]()


class RunningScoreBoundsTests(unittest.TestCase):
    def test_all_five_real_checkpoints_bound_all_three_scores(self):
        for instance, total in EXPECTED.items():
            for matches in (1, total // 2, total, total + 9, total * 2):
                with self.subTest(instance=instance, matches=matches):
                    result = checkpoint(instance, matches)
                    self.assertEqual(result.total, 30)
                    self.assertEqual(len(result.steps), 3)
                    expected_score = min(10, math.floor(matches / total * 10))
                    self.assertEqual(result.result, expected_score * 3)
                    for step in result.steps:
                        self.assertEqual(step.score, expected_score)
                        self.assertEqual(step.max_score, 10)
                        self.assertEqual(step.success, matches == total)
                        self.assertIn(f"{matches}/{total}", step.details)

    def test_observed_over_max_dates_are_reduced_without_changing_counts(self):
        for instance, matches in ((2, 216), (5, 283)):
            with self.subTest(instance=instance):
                old = checkpoint(instance, matches, uncapped=True).steps[0]
                fixed = checkpoint(instance, matches).steps[0]
                self.assertEqual(old.score, 12)
                self.assertEqual(fixed.score, 10)
                self.assertEqual(fixed.details, old.details.replace("12/10 pts", "10/10 pts"))
                self.assertEqual(fixed.success, old.success)
                self.assertFalse(fixed.success)
                self.assertEqual(fixed.category, old.category)

    def test_above_denominator_but_already_bounded_report_is_unchanged(self):
        old = checkpoint(1, 118, uncapped=True).steps[0]
        fixed = checkpoint(1, 118).steps[0]
        self.assertEqual((fixed.score, fixed.details, fixed.success),
                         (old.score, old.details, old.success))
        self.assertEqual(fixed.score, 10)
        self.assertIn("118/109 dates match (108%)", fixed.details)

    def test_missing_required_columns_remain_zero_and_not_evaluated(self):
        for instance in EXPECTED:
            with self.subTest(instance=instance):
                result = checkpoint(instance, 10, missing_columns=True)
                self.assertEqual(result.result, 0)
                for step in result.steps:
                    self.assertFalse(step.success)
                    self.assertEqual((step.score, step.max_score), (0, 10))
                    self.assertEqual(step.category, _scoring.StepCategory.DEPENDENCY_NOT_EVALUATED)

    def test_wrong_numeric_values_keep_zero_matching_scores(self):
        for instance, total in EXPECTED.items():
            with self.subTest(instance=instance):
                result = checkpoint(instance, total, wrong_values=True)
                self.assertEqual([step.score for step in result.steps], [10, 0, 0])
                self.assertEqual([step.success for step in result.steps], [True, False, False])

    def test_each_wrapper_has_exactly_three_local_caps(self):
        for instance in EXPECTED:
            with self.subTest(instance=instance):
                tree = ast.parse((FAMILY / f"instance_{instance}/evaluator.py").read_text())
                function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                and node.name == "grade_checkpoint_2")
                names = {"date_score", "sets_score", "reps_score"} if instance == 5 else {
                    "date_score", "dist_score", "speed_score"}
                assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                               and node.targets[0].id in names]
                self.assertEqual(len(assignments), 3)
                for assignment in assignments:
                    value = assignment.value
                    self.assertIsInstance(value, ast.Call)
                    self.assertEqual(ast.unparse(value.func), "min")
                    self.assertEqual(ast.literal_eval(value.args[0]), 10)
                    self.assertEqual(ast.unparse(value.args[1].func), "math.floor")


if __name__ == "__main__":
    unittest.main()
