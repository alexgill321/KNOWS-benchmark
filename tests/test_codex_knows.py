import json
import importlib.util
import os
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
import sys
from unittest.mock import MagicMock, patch
from codex_benchmark.evaluation_log import EvaluationJournal

from codex_benchmark.knows import (
    TASKS_RELATIVE, build_prompt, discover_tasks, evaluate_task,
    evaluator_kwargs, initial_workspace_url, resolve_task_prompt,
    resolve_workspace_after_episode, workspace_id_from_url,
    _strict_judges, _list_output_presentations, MissingRequiredArtifact,
)


class KnowsBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.models = SimpleNamespace(_models={}, gemma_google_ai=lambda model_id: lambda messages: "Yes")
        guard_patch = patch("codex_benchmark.knows._strict_judges",
                            lambda override=None: _strict_judges(override, self.models))
        guard_patch.start()
        self.addCleanup(guard_patch.stop)
        scoring_path = Path(__file__).resolve().parents[1] / "src/browsergym/knows/eval/eval_utils/scoring.py"
        spec = importlib.util.spec_from_file_location("test_codex_journal_scoring", scoring_path)
        self.scoring = importlib.util.module_from_spec(spec)
        modules_patch = patch.dict(sys.modules, {"test_codex_journal_scoring": self.scoring})
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        spec.loader.exec_module(self.scoring)
        journal_patch = patch("codex_benchmark.knows.EvaluationJournal", lambda *args:
                              EvaluationJournal(*args, scoring_module=self.scoring))
        journal_patch.start()
        self.addCleanup(journal_patch.stop)

    def task(self, family="docs_example", prompt="Write a letter.\n", evaluator="", instance=1):
        path = self.root / TASKS_RELATIVE / family / f"instance_{instance}"
        path.mkdir(parents=True, exist_ok=True)
        (path / "task.md").write_text(prompt, encoding="utf-8")
        (path / "evaluator.py").write_text(evaluator, encoding="utf-8")
        return next(task for task in discover_tasks(self.root) if task.family == family and task.instance == instance)

    def test_real_catalog_has_all_110_tasks_without_importing_google(self):
        root = Path(__file__).resolve().parents[1]
        tasks = discover_tasks(root)
        self.assertEqual(len(tasks), 110)
        self.assertEqual(len({task.family for task in tasks}), 22)
        self.assertEqual({kind: sum(t.kind == kind for t in tasks)
                          for kind in ("docs", "sheets", "slides")},
                         {"docs": 25, "sheets": 45, "slides": 40})
        self.assertTrue(all(task.root == root for task in tasks))

    def test_prompt_preserves_task_and_validates_file_kind(self):
        task = self.task(prompt="First line.\n\nSecond line.\n")
        url = "https://docs.google.com/document/d/doc123/edit"
        prompt = build_prompt(task, url)
        self.assertTrue(prompt.startswith("First line.\n\nSecond line.\n\n---\n"))
        self.assertIn("do NOT create a new file and do NOT rename it.", prompt)
        self.assertEqual(workspace_id_from_url(url, "docs"), "doc123")
        with self.assertRaises(ValueError):
            build_prompt(task, "https://docs.google.com/spreadsheets/d/sheet123/edit")
        with self.assertRaises(ValueError):
            workspace_id_from_url("https://docs.google.com.evil.test/document/d/abc/edit")

    def test_targets_use_official_nested_mapping_and_environment_precedence(self):
        task = self.task("sheets_10_paper_sorting", "Upload to {{OUTPUT_FOLDER_URL}}.")
        target_file = self.root / "targets.json"
        target_file.write_text(json.dumps({task.family: {"instance_1": {"OUTPUT_FOLDER_URL": "file-url"}}}))
        key = "KNOWS_TARGET_SHEETS_10_PAPER_SORTING_INSTANCE_1_OUTPUT_FOLDER_URL"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_task_prompt(task, target_file), "Upload to file-url.")
        with patch.dict(os.environ, {key: "environment-url"}, clear=True):
            self.assertEqual(resolve_task_prompt(task, self.root / "absent.json"), "Upload to environment-url.")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Provision fresh targets"):
                resolve_task_prompt(task)

    def test_signature_variants_and_client_source_are_supplied(self):
        task = self.task("slides_30_Work_Wikipedia_Photos",
                         "Read https://docs.google.com/document/d/clients_123/edit?usp=sharing.")
        def evaluator(workspace_doc_id, client_doc_id=None, browsing_history_list=None):
            pass
        history = ["https://example.org/?q=full"]
        self.assertEqual(evaluator_kwargs(task, evaluator, "output", history),
                         {"workspace_doc_id": "output", "client_doc_id": "clients_123", "browsing_history_list": history})
        def unsupported(workspace_doc_id, surprise):
            pass
        with self.assertRaisesRegex(ValueError, "surprise"):
            evaluator_kwargs(task, unsupported, "output", history)

    def test_slides17_uses_existing_copy_or_agent_copy_without_conflicting_suffix(self):
        family = "slides_17_removeimagesaddplaceholders"
        task1 = self.task(family, "Edit {{WORKING_COPY_URL}} and save to {{IMAGES_FOLDER_URL}}.")
        task2 = self.task(family, "First make a copy into {{OUTPUT_FOLDER_URL}}.", instance=2)
        working = "https://docs.google.com/presentation/d/working123/edit"
        targets = self.root / "targets.json"
        targets.write_text(json.dumps({family: {
            "instance_1": {"WORKING_COPY_URL": working, "IMAGES_FOLDER_URL": "images"},
            "instance_2": {"OUTPUT_FOLDER_URL": "https://drive.google.com/drive/folders/output123"},
        }}))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(initial_workspace_url(task1, targets), working)
            prompt = build_prompt(task1, working, targets)
            self.assertIn("existing Google Slides", prompt)
            self.assertNotIn("already been created", prompt)
            with self.assertRaisesRegex(ValueError, "WORKING_COPY_URL"):
                build_prompt(task1, "https://docs.google.com/presentation/d/wrong123/edit", targets)
            prompt = build_prompt(task2, None, targets)
            self.assertIn("First make a copy", prompt)
            self.assertNotIn("do NOT create", prompt)
            self.assertEqual(resolve_workspace_after_episode(task1, None, targets), "working123")
            with patch("codex_benchmark.knows._list_output_presentations", return_value=[{"id": "copied123"}]):
                self.assertEqual(resolve_workspace_after_episode(task2, None, targets), "copied123")
            with patch("codex_benchmark.knows._list_output_presentations", return_value=[]):
                with self.assertRaises(MissingRequiredArtifact):
                    resolve_workspace_after_episode(task2, None, targets)
            with patch("codex_benchmark.knows._list_output_presentations", return_value=[{"id": "a"}, {"id": "b"}]):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    resolve_workspace_after_episode(task2, None, targets)
                self.assertEqual(resolve_workspace_after_episode(task2, None, targets,
                    "DONE https://docs.google.com/presentation/d/b/edit"), "b")
                for final_text in ("DONE https://docs.google.com/presentation/d/outside/edit",
                                   "DONE https://docs.google.com/presentation/d/a/edit https://docs.google.com/presentation/d/b/edit"):
                    with self.assertRaisesRegex(ValueError, "exactly one"):
                        resolve_workspace_after_episode(task2, None, targets, final_text)

    def test_missing_slides_copy_is_zero_agent_score_without_fake_checkpoints(self):
        family = "slides_17_removeimagesaddplaceholders"
        task = self.task(family, "Make a copy in {{OUTPUT_FOLDER_URL}}.",
                         evaluator="raise AssertionError('must not load evaluator')", instance=2)
        targets = self.root / "targets.json"
        targets.write_text(json.dumps({family: {"instance_2": {
            "OUTPUT_FOLDER_URL": "https://drive.google.com/drive/folders/output123"}}}))
        with patch("codex_benchmark.knows._prepare_credentials"), patch.dict(os.environ, {}, clear=True):
            with patch("codex_benchmark.knows._list_output_presentations", return_value=[]):
                report = evaluate_task(task, None, [], self.root / "report.json", run_targets_path=targets)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["reward"], 0.0)
            self.assertEqual(report["score_source"], "missing_required_artifact")
            self.assertFalse(report["evaluator_invoked"])
            self.assertIsNone(report["result"])
            self.assertNotIn("final_score", report)
            self.assertNotIn("detailed_report", report)
            with patch("codex_benchmark.knows._list_output_presentations", side_effect=PermissionError("folder inaccessible")):
                failure = evaluate_task(task, None, [], self.root / "failure.json", run_targets_path=targets)
            self.assertEqual(failure["status"], "evaluation_error")
            self.assertIsNone(failure["reward"])

    def test_folder_access_is_verified_before_empty_listing(self):
        drive = MagicMock()
        drive.files.return_value.get.return_value.execute.side_effect = PermissionError("not accessible")
        with self.assertRaises(PermissionError):
            _list_output_presentations("folder123", drive)
        drive.files.return_value.list.assert_not_called()
        drive.files.return_value.get.return_value.execute.side_effect = None
        drive.files.return_value.get.return_value.execute.return_value = {
            "id": "folder123", "mimeType": "application/vnd.google-apps.folder", "trashed": False}
        drive.files.return_value.list.return_value.execute.return_value = {"files": []}
        self.assertEqual(_list_output_presentations("folder123", drive), [])

    def test_evaluator_writes_only_to_isolated_copy_and_preserves_results(self):
        evaluator = textwrap.dedent('''
            from pathlib import Path
            TASK_DIR = str(Path(__file__).parent) + "/"
            class Result:
                final_score = {"total": 4, "result": 3}
                def to_dict(self): return {"final_score": self.final_score}
                def get_detailed_report(self): return {"checkpoints": [{"name": "Content", "steps": []}]}
                def get_category_summary(self): return {"deterministic": {"total": 4, "passed": 3, "failed": 1}}
            def grade_checkpoints(workspace_doc_id, browsing_history_list=None):
                assert workspace_doc_id == "workspace123"
                assert browsing_history_list == ["https://example.org/?q=exact"]
                Path(TASK_DIR, "gold.txt").write_text("mutated")
                return Result()
        ''')
        task = self.task(evaluator=evaluator)
        gold = task.path / "gold.txt"
        gold.write_text("original")
        output = self.root / "report.json"
        with patch("codex_benchmark.knows._prepare_credentials"), patch.dict(os.environ, {}, clear=True):
            report = evaluate_task(task, "workspace123", ["https://example.org/?q=exact"], output)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["reward"], 0.75)
        self.assertEqual(gold.read_text(), "original")
        self.assertEqual(json.loads(output.read_text()), report)

    def test_missing_credentials_produce_explicit_error_without_import(self):
        task = self.task(evaluator="raise AssertionError('should not import')")
        with patch.dict(os.environ, {}, clear=True):
            report = evaluate_task(task, "doc123", [], self.root / "report.json")
        self.assertEqual(report["status"], "evaluation_error")
        self.assertIsNone(report["reward"])
        self.assertIn("Evaluator credentials missing", report["error"])

    def test_all_steps_and_each_grading_attempt_are_retained(self):
        evaluator = textwrap.dedent('''
            from test_codex_journal_scoring import Checkpoint, Result
            def grade_checkpoints(workspace_doc_id):
                cp = Checkpoint(2, 0, name="Letter content")
                cp.add_step("Name", True, 1, details="Exact match", execution_time=.1, category="deterministic")
                cp.add_step("Email", False, 2, details="Missing email", execution_time=.2, category="structural")
                return Result([cp])
        ''')
        task = self.task(evaluator=evaluator)
        output = self.root / "evaluation.json"
        with patch("codex_benchmark.knows._prepare_credentials"), patch.dict(os.environ, {}, clear=True):
            first = evaluate_task(task, "doc123", [], output)
            second = evaluate_task(task, "doc123", [], output)
        for report in (first, second):
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["reward"], .5)
            nested = report["result"]["checkpoints"][0]["step_summary"]["steps"]
            detailed = report["detailed_report"]["checkpoints"][0]["steps"]
            recorded = report["step_journal"]["checkpoints"][0]["steps"]
            self.assertEqual(nested, detailed)
            self.assertEqual(nested, recorded)
            self.assertEqual(len(nested), 2)
            attempt = self.root / "evaluation-attempts" / (report["evaluation_attempt_id"] + ".json")
            self.assertEqual(json.loads(attempt.read_text()), report)
        self.assertNotEqual(first["evaluation_attempt_id"], second["evaluation_attempt_id"])
        self.assertEqual(json.loads(output.read_text()), second)
        events = [json.loads(line) for line in (self.root / "evaluation-steps.jsonl").read_text().splitlines()]
        self.assertEqual(sum(event["event"] == "step_completed" for event in events), 4)

    def test_evaluator_exception_is_not_a_zero_model_score(self):
        task = self.task(evaluator="def grade_checkpoints(workspace_doc_id):\n    raise RuntimeError('judge unavailable')\n")
        with patch("codex_benchmark.knows._prepare_credentials"), patch.dict(os.environ, {}, clear=True):
            report = evaluate_task(task, "doc123", [], self.root / "report.json")
        self.assertEqual(report["status"], "evaluation_error")
        self.assertIsNone(report["reward"])
        self.assertIn("judge unavailable", report["error"])

    def test_judge_override_records_original_and_actual_and_restores_factories(self):
        original = lambda: lambda messages: "Old judge"
        self.models._models["gemini-2.5-flash-google-ai"] = original
        with _strict_judges("gemini-3.8-flash", self.models) as state:
            self.assertEqual(self.models._models["gemini-2.5-flash-google-ai"]()([]), "Yes")
            self.assertEqual(state["requested_judge_models"], ["gemini-2.5-flash-google-ai"])
            self.assertEqual(state["actual_judge_models"], {"gemini-2.5-flash-google-ai": "gemini-3.8-flash"})
        self.assertIs(self.models._models["gemini-2.5-flash-google-ai"], original)

    def test_swallowed_judge_outage_never_becomes_a_zero_agent_score(self):
        self.models._models["gemini-2.5-flash-google-ai"] = lambda: lambda messages: "Error: Google GenAI API unavailable"
        evaluator = textwrap.dedent('''
            import test_codex_judge_models as models
            class Result:
                final_score = {"total": 4, "result": 0}
                def to_dict(self): return {"final_score": self.final_score}
                def get_detailed_report(self): return {"checkpoints": [{"name": "Content", "steps": []}]}
                def get_category_summary(self): return {"llm_vlm_judgement": {"total": 4, "passed": 0, "failed": 4}}
            def grade_checkpoints(workspace_doc_id):
                try:
                    models._models["gemini-2.5-flash-google-ai"]()([])
                except Exception:
                    pass  # Upstream checkpoint helpers sometimes swallow errors.
                return Result()
        ''')
        task = self.task(evaluator=evaluator)
        with patch("codex_benchmark.knows._prepare_credentials"), patch.dict(os.environ, {}, clear=True), patch.dict("sys.modules", {"test_codex_judge_models": self.models}):
            report = evaluate_task(task, "doc123", [], self.root / "report.json")
        self.assertEqual(report["status"], "evaluation_error")
        self.assertIsNone(report["reward"])
        self.assertTrue(report["judge_errors"])
        self.assertEqual(report["result"]["final_score"]["result"], 0)


if __name__ == "__main__":
    unittest.main()
