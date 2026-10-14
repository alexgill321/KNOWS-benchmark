"""Offline coverage for image-match futures and fatal unresolved comparisons."""
import ast
import concurrent.futures
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Tuple
import unittest
from unittest.mock import Mock, patch

from codex_benchmark.evaluation_log import EvaluationJournal
from codex_benchmark.knows import TASKS_RELATIVE, _strict_judges, discover_tasks, evaluate_task


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / TASKS_RELATIVE / "slides_26_basic_educational_slide_deck"


def load_collector():
    tree = ast.parse((FAMILY / "utils.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in {"ImageSourceMatchError", "collect_image_source_matches"}]
    namespace = {"concurrent": concurrent, "Dict": Dict, "List": List, "Tuple": Tuple, "Any": Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FAMILY / "utils.py"), "exec"), namespace)
    return SimpleNamespace(**namespace)


class EducationalImageMatchTests(unittest.TestCase):
    def setUp(self):
        self.local = load_collector()

    def test_late_success_and_false_results_survive_existing_cleanup_wait(self):
        for result in ((True, "late exact match"), (False, "completed comparison rejected")):
            with self.subTest(result=result):
                started, release = threading.Event(), threading.Event()
                calls = []
                real_executor = concurrent.futures.ThreadPoolExecutor

                class CleanupExecutor(real_executor):
                    def __exit__(self, *args):
                        release.set()
                        return super().__exit__(*args)

                def work():
                    calls.append("once"); started.set()
                    if not release.wait(2):
                        raise AssertionError("Cleanup did not release the in-flight fixture")
                    return result

                def collection_cutoff(futures, timeout):
                    self.assertEqual(timeout, 240)
                    self.assertTrue(started.wait(2))
                    self.assertFalse(next(iter(futures)).done())
                    raise concurrent.futures.TimeoutError()

                with patch.object(concurrent.futures, "ThreadPoolExecutor", CleanupExecutor), \
                        patch.object(concurrent.futures, "as_completed", collection_cutoff):
                    actual = self.local.collect_image_source_matches([{"id": "late", "func": work}])
                self.assertEqual(actual, {"late": result})
                self.assertEqual(calls, ["once"])

    def test_cancelled_queued_task_is_fatal_even_when_running_task_finishes(self):
        started, release = threading.Event(), threading.Event()
        queued = Mock(return_value=(True, "must not run"))
        real_executor = concurrent.futures.ThreadPoolExecutor

        class CleanupExecutor(real_executor):
            def __exit__(self, *args):
                release.set()
                return super().__exit__(*args)

        def running():
            started.set()
            if not release.wait(2): raise AssertionError("Cleanup did not release fixture")
            return True, "completed"

        def cutoff(futures, timeout):
            self.assertTrue(started.wait(2))
            raise concurrent.futures.TimeoutError()

        with patch.object(concurrent.futures, "ThreadPoolExecutor", CleanupExecutor), \
                patch.object(concurrent.futures, "as_completed", cutoff):
            with self.assertRaisesRegex(self.local.ImageSourceMatchError, "queued: cancelled"):
                self.local.collect_image_source_matches(
                    [{"id": "running", "func": running}, {"id": "queued", "func": queued}], max_workers=1)
        queued.assert_not_called()

    def test_complete_false_results_are_valid_and_empty_batch_is_empty(self):
        tasks = [{"id": "reject", "func": lambda: (False, "different image")},
                 {"id": "accept", "func": lambda: (True, "Source image matches (exact)")}]
        self.assertEqual(self.local.collect_image_source_matches(tasks),
                         {"reject": (False, "different image"), "accept": (True, "Source image matches (exact)")})
        self.assertEqual(self.local.collect_image_source_matches([]), {})

    def test_exception_null_and_malformed_results_are_fatal(self):
        for value in (None, False, {}, [], [True, "list"], (True,), (True, None), (1, "not bool"),
                      (False, "detail", "extra")):
            with self.subTest(value=value), self.assertRaises(self.local.ImageSourceMatchError):
                self.local.collect_image_source_matches([{"id": "bad", "func": lambda value=value: value}])
        with self.assertRaisesRegex(self.local.ImageSourceMatchError, "bad: ValueError"):
            self.local.collect_image_source_matches([{"id": "bad", "func": Mock(side_effect=ValueError("transport fixture"))}])

    def test_duplicate_and_missing_ids_rejected_before_any_work(self):
        work = Mock(return_value=(True, "unused"))
        for ids in (("same", "same"), (None,), ("",), (True,)):
            with self.subTest(ids=ids), self.assertRaises(self.local.ImageSourceMatchError):
                self.local.collect_image_source_matches([{"id": value, "func": work} for value in ids])
        work.assert_not_called()

    def test_all_five_wrappers_bind_only_image_batch_with_unchanged_limits(self):
        for instance in range(1, 6):
            tree = ast.parse((FAMILY / f"instance_{instance}/evaluator.py").read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "collect_image_source_matches"]
            with self.subTest(instance=instance):
                self.assertEqual(len(calls), 1)
                self.assertEqual(ast.unparse(calls[0].args[0]), "img_match_tasks")
                self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords},
                                 {"max_workers": 5, "timeout": 240})

    def test_all_five_actual_error_boundaries_reach_null_reward_bridge(self):
        spec = importlib.util.spec_from_file_location("educational_match_scoring", ROOT / TASKS_RELATIVE.parent / "eval_utils/scoring.py")
        scoring = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: scoring}): spec.loader.exec_module(scoring)
        for instance in range(1, 6):
            with self.subTest(instance=instance), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source_path = FAMILY / f"instance_{instance}/evaluator.py"
                tree = ast.parse(source_path.read_text())
                cp = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "grade_checkpoint_4")
                outer = next(node for node in cp.body if isinstance(node, ast.Try))
                inner = next(node for node in outer.body if isinstance(node, ast.Try)
                             and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                                     and call.func.id == "collect_image_source_matches" for call in ast.walk(node)))
                # Execute the real image-step body and both real exception handlers;
                # only preceding image extraction/network work is fixture data.
                bounded = ast.Try(body=[deepcopy(inner)], handlers=deepcopy(outer.handlers), orelse=[], finalbody=[])
                factory = ast.FunctionDef(name="grade_checkpoint_4", args=deepcopy(cp.args),
                                          body=[bounded, ast.Return(value=ast.Name(id="checkpoint", ctx=ast.Load()))],
                                          decorator_list=[])
                entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "grade_checkpoints")
                fixture = ModuleType("_educational_image_match_fixture")
                fixture.ImageSourceMatchError = self.local.ImageSourceMatchError
                fixture.collect_image_source_matches = self.local.collect_image_source_matches
                fixture.match_source_image = Mock(return_value=None)  # actual collector rejects unresolved work
                fixture.time, fixture.re = time, __import__("re")
                fixture.model_id = "gemini-3-flash-google-ai"
                fixture.ensure_model = lambda _: object()
                fixture.find_url_below_image = lambda *args: "https://example.test/image.png"
                fixture.slide_image_data = [{"image": {"path": "saved.png", "bbox": {"width": 1, "height": 1}}, "links_with_pos": []}]
                fixture.temp_dir, fixture.actual_count = folder, 1
                fixture.checkpoint = scoring.Checkpoint(total=50, result=0, name="Visual Elements")
                fixture.step4_category = scoring.StepCategory.LLM_VLM_JUDGEMENT
                fixture.calculate_percentage_score = scoring.calculate_percentage_score
                fixture.join_extras = lambda *args: ""
                fixture.to_deck_positions = lambda items: items
                fixture.fill_failure_steps = Mock()
                fixture.Checkpoint, fixture.Result, fixture.StepCategory = scoring.Checkpoint, scoring.Result, scoring.StepCategory
                fixture.reset_model_state = lambda: None
                fixture.set_cached_model = lambda _: None
                fixture.setup_presentation = lambda _: None
                fixture.grade_checkpoint_1 = lambda: scoring.Checkpoint(total=0, result=0, name="Title Slide")
                fixture.grade_checkpoint_2 = lambda: scoring.Checkpoint(total=0, result=0, name="Content Slides")
                fixture.grade_checkpoint_3 = lambda: scoring.Checkpoint(total=0, result=0, name="Summary Slide")
                for name in ("CP1_STEPS", "CP2_STEPS", "CP3_STEPS", "CP4_STEPS"): setattr(fixture, name, [])
                module = ast.fix_missing_locations(ast.Module(body=[factory, deepcopy(entry)], type_ignores=[]))
                task_dir = root / TASKS_RELATIVE / "slides_match_guard_fixture/instance_1"
                task_dir.mkdir(parents=True)
                (task_dir / "task.md").write_text("Use the saved presentation.\n")
                (task_dir / "evaluator.py").write_text("from _educational_image_match_fixture import *\n" + ast.unparse(module))
                task = discover_tasks(root)[0]
                models = SimpleNamespace(_models={}, gemma_google_ai=Mock())
                output = root / "evaluation.json"
                with patch.dict(sys.modules, {fixture.__name__: fixture}), \
                        patch("codex_benchmark.knows._prepare_credentials"), \
                        patch("codex_benchmark.knows._strict_judges", lambda override=None: _strict_judges(override, models)), \
                        patch("codex_benchmark.knows.EvaluationJournal", lambda *args: EvaluationJournal(*args, scoring_module=scoring)), \
                        patch.dict(os.environ, {}, clear=True):
                    report = evaluate_task(task, "saved_presentation", [], output)
                self.assertEqual(report["status"], "evaluation_error")
                self.assertIsNone(report["reward"])
                self.assertIn("ImageSourceMatchError", report["error"])
                self.assertIsNone(json.loads(output.read_text())["reward"])
                fixture.fill_failure_steps.assert_not_called()
                models.gemma_google_ai.assert_not_called()


if __name__ == "__main__":
    unittest.main()
