from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from codex_benchmark.evaluation_log import EvaluationJournal
from codex_benchmark.knows import TASKS_RELATIVE, Task, _prepare_evaluator_source, evaluate_task


ROOT = Path(__file__).resolve().parents[1]
OLD = "from browsergym.knows.eval.eval_scripts.test.test_doc_to_images import convert_pdf_to_pngs"
NEW = "from browsergym.knows.eval.eval_utils.image_utils import convert_pdf_to_pngs"


class EvaluatorCompatibilityTests(unittest.TestCase):
    def test_three_shipped_evaluators_change_only_obsolete_import_in_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            for instance in (1, 2, 3):
                with self.subTest(instance=instance):
                    task_path = ROOT / TASKS_RELATIVE / "docs_5_influential_papers" / f"instance_{instance}"
                    task = Task("docs_5_influential_papers", instance, f"test.{instance}", task_path, "docs")
                    original_path = task_path / "evaluator.py"
                    original = original_path.read_bytes()
                    copied = Path(temporary) / f"evaluator_{instance}.py"
                    copied.write_bytes(original)
                    audit = _prepare_evaluator_source(task, copied)
                    expected = original.replace(OLD.encode(), NEW.encode(), 1)
                    self.assertNotEqual(original, expected)
                    self.assertEqual(copied.read_bytes(), expected)
                    self.assertEqual(original_path.read_bytes(), original)
                    self.assertEqual(audit["original_sha256"], hashlib.sha256(original).hexdigest())
                    self.assertEqual(audit["effective_sha256"], hashlib.sha256(expected).hexdigest())
                    self.assertEqual(audit["repairs"], [{"id": "docs5-pdf-import-compat-v1",
                                                        "old_import": OLD, "new_import": NEW}])

    def test_repair_is_restricted_to_known_family_and_instances(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "evaluator.py"
            source = (OLD + "\nRUBRIC_POINTS = 17\n").encode()
            for family, instance in (("docs_example", 1), ("docs_5_influential_papers", 4),
                                     ("docs_5_influential_papers", 5)):
                with self.subTest(family=family, instance=instance):
                    copied.write_bytes(source)
                    audit = _prepare_evaluator_source(SimpleNamespace(family=family, instance=instance), copied)
                    self.assertEqual(copied.read_bytes(), source)
                    self.assertEqual(audit["repairs"], [])
                    self.assertEqual(audit["original_sha256"], audit["effective_sha256"])

    def test_already_current_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "evaluator.py"
            source = (NEW + "\nRUBRIC_POINTS = 17\n").encode()
            copied.write_bytes(source)
            audit = _prepare_evaluator_source(SimpleNamespace(family="docs_5_influential_papers", instance=1), copied)
            self.assertEqual(copied.read_bytes(), source)
            self.assertEqual(audit["repairs"], [])
            self.assertEqual(audit["original_sha256"], audit["effective_sha256"])

    def test_evaluation_uses_repaired_copy_and_saves_provenance_on_success_or_error(self):
        spec = importlib.util.spec_from_file_location("_compat_scoring", ROOT / "src/browsergym/knows/eval/eval_utils/scoring.py")
        scoring = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: scoring}):
            spec.loader.exec_module(scoring)
        body = textwrap.dedent('''
            class Result:
                final_score = {"total": 1, "result": 1}
                def to_dict(self): return {"final_score": self.final_score}
                def get_detailed_report(self): return {"checkpoints": []}
                def get_category_summary(self): return {}
            def grade_checkpoints(workspace_doc_id):
                assert convert_pdf_to_pngs() == "official utility"
                return Result()
        ''')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_path = root / TASKS_RELATIVE / "docs_5_influential_papers" / "instance_1"
            task_path.mkdir(parents=True)
            (task_path / "task.md").write_text("Write a paper summary.")
            task = Task("docs_5_influential_papers", 1, "test.1", task_path, "docs")
            for fail in (False, True):
                with self.subTest(fail=fail):
                    source = OLD + "\n" + body
                    if fail:
                        source += '\nraise RuntimeError("later evaluator failure")\n'
                    (task_path / "evaluator.py").write_text(source)
                    output = root / ("failed.json" if fail else "success.json")
                    with patch("codex_benchmark.knows._prepare_credentials"), \
                         patch("codex_benchmark.knows._strict_judges", return_value=nullcontext({"judge_errors": []})), \
                         patch("codex_benchmark.knows.EvaluationJournal", side_effect=lambda *args: EvaluationJournal(*args, scoring_module=scoring)), \
                         patch.dict(os.environ, {}, clear=True), \
                         patch.object(sys, "path", list(sys.path)), \
                         patch.dict(sys.modules, {"browsergym.knows.eval.eval_utils.image_utils": SimpleNamespace(
                             convert_pdf_to_pngs=lambda: "official utility")}):
                        report = evaluate_task(task, "workspace123", [], output)
                    self.assertEqual(report["status"], "evaluation_error" if fail else "completed")
                    self.assertEqual(report["reward"], None if fail else 1)
                    self.assertEqual((task_path / "evaluator.py").read_text(), source)
                    self.assertEqual(report["evaluator_source"]["original_sha256"], hashlib.sha256(source.encode()).hexdigest())
                    self.assertEqual(report["evaluator_source"]["effective_sha256"], hashlib.sha256(source.replace(OLD, NEW, 1).encode()).hexdigest())
                    self.assertEqual(len(report["evaluator_source"]["repairs"]), 1)
                    self.assertEqual(json.loads(output.read_text()), report)


if __name__ == "__main__":
    unittest.main()
