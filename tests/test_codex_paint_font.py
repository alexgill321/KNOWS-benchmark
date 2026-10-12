"""Offline regressions for inherited font sizes in Paint Colors recommendations."""
import ast
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
import unittest


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "src/browsergym/knows/eval").is_dir())
EVAL = ROOT / "src/browsergym/knows/eval"
SOURCE_ROOT = Path(os.environ.get("KNOWS_PAINT_EVALUATOR_ROOT", str(
    EVAL / "tasks/slides_39_Personal_Lookbook_PaintColors")))


def load_definitions(path, names, namespace):
    # Importing an evaluator initializes Google services and judge models.
    # Compile its actual grading function and pure helpers without those effects.
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing definitions in {path}: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def load_checkpoint(instance):
    namespace = {"__name__": __name__, "dataclass": dataclass, "field": field,
                 "time": time, "re": re, "List": List, "Dict": Dict,
                 "Any": Any, "Optional": Optional}
    load_definitions(EVAL / "eval_utils/scoring.py",
                     {"StepCategory", "EvaluationStep", "Checkpoint"}, namespace)
    load_definitions(EVAL / "eval_utils/slides_utils.py",
                     {"_extract_text_from_text_element", "extract_text_boxes_from_slide",
                      "get_text_style_from_shape", "is_text_big"}, namespace)
    # Bounding boxes are extracted alongside text, but the recommendation
    # checkpoint never uses them. All text/font extraction and scoring are real.
    namespace["get_element_bbox"] = lambda element: {}
    load_definitions(EVAL / "eval_utils/text_utils.py",
                     {"_normalize_text", "keyword_exact_match", "keywords_exact_match"}, namespace)
    load_definitions(SOURCE_ROOT / f"instance_{instance}/evaluator.py",
                     {"grade_checkpoint_4"}, namespace)
    return namespace["grade_checkpoint_4"]


def context(text="Navy is the best choice", font_size=None, placeholder="TITLE", exists=True):
    style = {} if font_size is None else {"fontSize": font_size}
    shape = {"text": {"textElements": [{"textRun": {"content": text, "style": style}}]}}
    if placeholder:
        shape["placeholder"] = {"type": placeholder}
    recommendation = {"pageElements": [{"objectId": "recommendation-title", "shape": shape}]}
    slides = [{}, {}, recommendation]
    return {"presentation_data": {"slides": slides}, "slides": slides,
            "recommendation_slide": recommendation if exists else None,
            "recommendation_slide_idx": 2 if exists else None,
            "color_names": ["Sage Green", "Navy", "Terracotta"]}


class PaintRecommendationFontTests(unittest.TestCase):
    def checkpoints(self, **kwargs):
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                yield load_checkpoint(instance)(context(**kwargs))

    def assert_steps(self, checkpoint, scores):
        self.assertEqual(len(checkpoint.steps), 4)
        self.assertEqual([step.step_id for step in checkpoint.steps], [1, 2, 3, 4])
        self.assertEqual([step.score for step in checkpoint.steps], scores)
        self.assertEqual(checkpoint.result, sum(scores))
        self.assertEqual(checkpoint.total, 10)
        self.assertTrue(all(step.category != "execution_error" for step in checkpoint.steps))

    def test_inherited_title_font_retains_all_earned_points(self):
        # Google Slides omits inherited fontSize; the existing is_text_big
        # helper accepts TITLE/CENTERED_TITLE/SUBTITLE inheritance. Previously
        # detail formatting raised after recording the first 8 earned points.
        for placeholder in ("TITLE", "CENTERED_TITLE", "SUBTITLE"):
            for checkpoint in self.checkpoints(placeholder=placeholder):
                self.assert_steps(checkpoint, [2, 3, 3, 2])
                self.assertEqual(checkpoint.steps[3].details, "Could not determine font size")
                self.assertEqual(checkpoint.steps[3].category, "deterministic")

    def test_unspecified_nonplaceholder_font_keeps_eight_points(self):
        for checkpoint in self.checkpoints(placeholder=None):
            self.assert_steps(checkpoint, [2, 3, 3, 0])
            self.assertFalse(checkpoint.steps[3].success)

    def test_explicit_large_font_keeps_original_threshold_and_detail(self):
        for checkpoint in self.checkpoints(font_size={"magnitude": 18, "unit": "PT"}):
            self.assert_steps(checkpoint, [2, 3, 3, 2])
            self.assertEqual(checkpoint.steps[3].details, "Font size: 18pt")

    def test_explicit_small_font_is_not_promoted_by_title_placeholder(self):
        for checkpoint in self.checkpoints(font_size={"magnitude": 17, "unit": "PT"}):
            self.assert_steps(checkpoint, [2, 3, 3, 0])
            self.assertEqual(checkpoint.steps[3].details, "Font size: 17pt")

    def test_emu_size_uses_existing_conversion(self):
        for magnitude, points in ((18 * 12700, 2), (17 * 12700, 0)):
            for checkpoint in self.checkpoints(font_size={"magnitude": magnitude, "unit": "EMU"}):
                self.assert_steps(checkpoint, [2, 3, 3, points])

    def test_missing_target_phrase_does_not_award_dependent_points(self):
        for checkpoint in self.checkpoints(text="My recommendation"):
            self.assert_steps(checkpoint, [2, 0, 0, 0])
            self.assertEqual(checkpoint.steps[3].category, "dependency_not_evaluated")

    def test_missing_recommendation_still_fails_all_four_checks(self):
        for checkpoint in self.checkpoints(exists=False):
            self.assert_steps(checkpoint, [0, 0, 0, 0])
            self.assertEqual(checkpoint.steps[0].category, "structural")
            self.assertTrue(all(step.category == "dependency_not_evaluated"
                                for step in checkpoint.steps[1:]))


if __name__ == "__main__":
    unittest.main()
