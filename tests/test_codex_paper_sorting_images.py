"""Offline regression tests: Drive thumbnail HTML must never reach the judge."""
import ast
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Dict, List
import unittest
from unittest.mock import Mock, patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "src/browsergym/knows/eval/tasks/sheets_10_paper_sorting"
spec = importlib.util.spec_from_file_location("paper_sorting_figures_test", FAMILY / "figure_images.py")
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def image_bytes(fmt="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (11, 7), "red").save(output, format=fmt)
    return output.getvalue()


def definitions(path, names, namespace):
    # Evaluator imports initialize Google services; execute only the actual
    # function definitions so every test stays offline and credential-free.
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class FigureDownloadTests(unittest.TestCase):
    def download(self, url, *, content=None, http_response=None, drive_error=None):
        drive = Mock()
        drive.files.return_value.get_media.return_value.execute.return_value = content or image_bytes()
        drive.files.return_value.get_media.return_value.execute.side_effect = drive_error
        public = Mock(return_value=http_response)
        arxiv = Mock(return_value=http_response)
        result = figures.download_figure_image(
            url, drive_service=drive, http_get=public, arxiv_get=arxiv, headers={})
        self.addCleanup(lambda: Path(result).unlink(missing_ok=True))
        return result, drive, public, arxiv

    def test_drive_thumbnail_reads_exact_authenticated_file_not_public_html(self):
        raw = image_bytes()
        path, drive, public, arxiv = self.download(
            "https://drive.google.com/thumbnail?id=fixture_image-ID_123&sz=w1000", content=raw)
        drive.files.return_value.get_media.assert_called_once_with(
            fileId="fixture_image-ID_123", supportsAllDrives=True)
        public.assert_not_called()
        arxiv.assert_not_called()
        self.assertEqual(Path(path).read_bytes(), raw)

    def test_drive_error_never_falls_back_to_public_login_or_exposes_error_body(self):
        drive, public = Mock(), Mock()
        drive.files.return_value.get_media.return_value.execute.side_effect = RuntimeError("secret query body")
        with self.assertRaisesRegex(figures.FigureImageError, "^Google Drive figure retrieval failed$"):
            figures.download_figure_image(
                "https://drive.google.com/thumbnail?id=test", drive_service=drive,
                http_get=public, arxiv_get=public, headers={})
        public.assert_not_called()

    def test_thumbnail_parser_rejects_ambiguity_and_never_uses_drive_for_other_hosts(self):
        for url in ("https://drive.google.com/thumbnail", "https://drive.google.com/thumbnail?id=",
                    "https://drive.google.com/thumbnail?id=a&id=b", "https://drive.google.com/thumbnail?id=../x",
                    "http://drive.google.com/thumbnail?id=a", "https://user@drive.google.com/thumbnail?id=a",
                    "https://drive.google.com:444/thumbnail?id=a", "https://drive.google.com:bad/thumbnail?id=a"):
            with self.subTest(url=url), self.assertRaises(figures.FigureImageError):
                figures.drive_thumbnail_id(url)
        for url in ("https://drive.google.com.evil.invalid/thumbnail?id=a",
                    "https://evil.invalid/thumbnail?id=a", "https://drive.google.com/file/d/a/view"):
            with self.subTest(url=url):
                self.assertIsNone(figures.drive_thumbnail_id(url))

    def test_public_image_normalizes_real_format_instead_of_mislabeling_png(self):
        response = Mock(content=image_bytes("JPEG"))
        path, drive, public, arxiv = self.download("https://example.invalid/figure.jpg", http_response=response)
        drive.files.assert_not_called()
        public.assert_called_once_with("https://example.invalid/figure.jpg", headers={}, timeout=30)
        arxiv.assert_not_called()
        with Image.open(path) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (11, 7))

    def test_arxiv_retains_existing_retry_transport(self):
        response = Mock(content=image_bytes())
        _, drive, public, arxiv = self.download(
            "https://arxiv.org/html/fixture/x1.png", http_response=response)
        arxiv.assert_called_once_with("https://export.arxiv.org/html/fixture/x1.png", headers={}, timeout=30)
        drive.files.assert_not_called()
        public.assert_not_called()

    def test_html_pdf_truncation_and_unavailable_candidate_fail_closed_not_false(self):
        # Undecodable bytes alone cannot tell an invalid submission from an
        # authorization page, external outage, or corrupted transfer.
        for body in (b"<!DOCTYPE html><html>Sign in</html>", b"%PDF-1.7\n", image_bytes()[:35]):
            with self.subTest(body=body[:10]), self.assertRaises(figures.FigureImageError):
                self.download("https://example.invalid/figure", http_response=Mock(content=body))
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("404 private URL")
        with self.assertRaisesRegex(figures.FigureImageError, "^Figure HTTP retrieval failed$"):
            self.download("https://example.invalid/figure", http_response=response)


class FigureEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        (self.base / "gold.png").write_bytes(image_bytes())

    def namespace(self, instance):
        namespace = dict(os=os, time=time, List=List, Dict=Dict, Any=Any, Result=object,
                         TASK_DIR=str(self.base), FigureImageError=figures.FigureImageError,
                         validate_figure_image=figures.validate_figure_image,
                         extract_image_url_from_cell=lambda value: value,
                         download_image_from_url=Mock(), fast_parallel_vlm_calls=Mock())
        definitions(FAMILY / f"instance_{instance}/evaluator.py",
                    {"parallel_validate_figures", "grade_checkpoints"}, namespace)
        return namespace

    def test_every_instance_sends_verified_original_images_and_keeps_judge_verdict(self):
        for instance in range(1, 6):
            for verdict in (True, False):
                with self.subTest(instance=instance, verdict=verdict):
                    ns = self.namespace(instance)
                    candidate = self.base / "download.png"
                    candidate.write_bytes(image_bytes())
                    ns["download_image_from_url"].return_value = str(candidate)
                    def judge(tasks, model, **kwargs):
                        content = tasks[0]["messages"][1]["content"]
                        self.assertEqual([c["image"] for c in content if c["type"] == "image"],
                                         [str(self.base / "gold.png"), str(candidate)])
                        for c in content:
                            if c["type"] == "image":
                                figures.validate_figure_image(c["image"], source="Test image")
                        return {"original_3": verdict}
                    ns["fast_parallel_vlm_calls"].side_effect = judge
                    result = ns["parallel_validate_figures"](
                        [{"id": "original_3", "gold_path": "gold.png", "user_url": "https://example.invalid/figure"}], object())
                    self.assertEqual(result, {"original_3": verdict})
                    self.assertFalse(candidate.exists())

    def test_all_five_real_download_wrappers_use_the_authenticated_helper(self):
        module_name = "browsergym.knows.eval.tasks.sheets_10_paper_sorting.utils"
        local_utils = ModuleType(module_name)
        local_utils._arxiv_request_with_retry = Mock()
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                drive, requests = Mock(), Mock()
                drive.files.return_value.get_media.return_value.execute.return_value = image_bytes()
                ns = dict(DRIVE_SERVICE=drive, requests=requests, ARXIV_HEADERS={},
                          download_figure_image=figures.download_figure_image)
                definitions(FAMILY / f"instance_{instance}/evaluator.py", {"download_image_from_url"}, ns)
                with patch.dict(sys.modules, {module_name: local_utils}):
                    path = ns["download_image_from_url"]("https://drive.google.com/thumbnail?id=fixture-ID&sz=w1000")
                try:
                    figures.validate_figure_image(path, source="Test image")
                    requests.get.assert_not_called()
                    drive.files.return_value.get_media.assert_called_once_with(
                        fileId="fixture-ID", supportsAllDrives=True)
                finally:
                    Path(path).unlink()

    def test_invalid_reference_or_candidate_stops_before_judge_all_instances(self):
        for instance in range(1, 6):
            for role in ("reference", "candidate"):
                with self.subTest(instance=instance, role=role):
                    ns = self.namespace(instance)
                    gold, candidate = self.base / "gold.png", self.base / "candidate.png"
                    gold.write_bytes(image_bytes() if role == "candidate" else b"<html>arXiv</html>")
                    candidate.write_bytes(image_bytes() if role == "reference" else b"<html>Sign in</html>")
                    ns["download_image_from_url"].return_value = str(candidate)
                    with self.assertRaises(figures.FigureImageError):
                        ns["parallel_validate_figures"](
                            [{"id": "figure", "gold_path": "gold.png", "user_url": "https://example.invalid/image"}], object())
                    ns["fast_parallel_vlm_calls"].assert_not_called()
                    self.assertFalse(candidate.exists())

    def test_download_failure_cleans_earlier_downloads_and_propagates(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                ns = self.namespace(instance)
                candidate = self.base / "download.png"
                candidate.write_bytes(image_bytes())
                ns["download_image_from_url"].side_effect = [str(candidate), figures.FigureImageError("unavailable")]
                tasks = [{"id": str(i), "gold_path": "gold.png", "user_url": "https://example.invalid/figure"}
                         for i in range(2)]
                with self.assertRaises(figures.FigureImageError):
                    ns["parallel_validate_figures"](tasks, object())
                self.assertFalse(candidate.exists())
                ns["fast_parallel_vlm_calls"].assert_not_called()

    def test_missing_figures_keep_empty_comparison_result(self):
        for instance in range(1, 6):
            ns = self.namespace(instance)
            self.assertEqual(ns["parallel_validate_figures"]([], object()), {})
            ns["download_image_from_url"].assert_not_called()
            ns["fast_parallel_vlm_calls"].assert_not_called()

    def test_errors_reach_existing_fatal_evaluation_checkpoint_all_instances(self):
        scoring = ROOT / "src/browsergym/knows/eval/eval_utils/scoring.py"
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                ns = self.namespace(instance)
                definitions(scoring, {"StepCategory"}, ns)
                checkpoint = Mock()
                ns.update(setup=Mock(), Checkpoint=Mock(return_value=checkpoint), Result=Mock(),
                          grade_checkpoint_1=Mock(side_effect=figures.FigureImageError("Reference figure invalid")))
                ns["grade_checkpoints"]("saved-workspace", [])
                self.assertEqual(ns["Checkpoint"].call_args.kwargs["name"], "Evaluation Error")
                self.assertEqual(checkpoint.add_step.call_args.kwargs["category"], ns["StepCategory"].EXECUTION_ERROR)
                ns["Result"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
