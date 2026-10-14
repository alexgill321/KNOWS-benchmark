"""Offline image-format regression tests for the Wikipedia Photos evaluator."""
import ast
import io
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, List
import unittest
from unittest.mock import Mock

from PIL import Image


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "src/browsergym/knows/eval").is_dir())
EVAL = ROOT / "src/browsergym/knows/eval"
SOURCE = Path(os.environ.get("KNOWS_WIKIPEDIA_UTILS_SOURCE", str(
    EVAL / "tasks/slides_30_Work_Wikipedia_Photos/utils.py")))


def load_definitions(path, names, namespace):
    # The source modules initialize Google services at import time. Compile the
    # actual functions under test without executing service initialization.
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing definitions in {path}: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def make_image(image_format, animated=False):
    first = Image.new("RGBA", (8, 6), (255, 0, 0, 255))
    second = Image.new("RGBA", (8, 6), (0, 0, 255, 255))
    buffer = io.BytesIO()
    options = {"lossless": True} if image_format == "WEBP" else {}
    if image_format == "JPEG":
        first = first.convert("RGB")
    if animated:
        options.update(save_all=True, append_images=[second], duration=100, loop=0)
    first.save(buffer, format=image_format, **options)
    buffer.seek(0)
    image = Image.open(buffer)
    image.load()
    return image


class WikipediaClientImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.namespace = {"os": os, "shutil": shutil, "time": time,
                          "List": List, "Dict": Dict, "SYMMETRY_TOLERANCE": 0.05}
        load_definitions(EVAL / "eval_utils/scoring.py", {"StepCategory"}, self.namespace)
        load_definitions(SOURCE, {"_save_client_image", "evaluate_single_client"}, self.namespace)
        judge_namespace = {"os": os, "parse_response": lambda response: response == "Yes"}
        load_definitions(EVAL / "eval_utils/image_utils.py", {"binary_judge_image"}, judge_namespace)
        self.namespace["binary_judge_image"] = judge_namespace["binary_judge_image"]

    def test_animated_gif_and_webp_use_first_static_frame_and_preserve_position(self):
        for image_format in ("GIF", "WEBP"):
            with self.subTest(format=image_format), make_image(image_format, animated=True) as image:
                image.seek(1)
                path = self.namespace["_save_client_image"](image, self.temp.name, image_format)
                self.assertEqual(image.tell(), 1)
                self.assertEqual(Path(path).suffix, ".png")
                with Image.open(path) as saved:
                    self.assertEqual(saved.format, "PNG")
                    self.assertEqual(saved.n_frames, 1)
                    self.assertEqual(saved.mode, "RGBA")
                    self.assertEqual(saved.getpixel((0, 0)), (255, 0, 0, 255))

    def test_webp_transparency_is_preserved(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (3, 3), (30, 60, 90, 127)).save(buffer, format="WEBP", lossless=True)
        buffer.seek(0)
        with Image.open(buffer) as image:
            path = self.namespace["_save_client_image"](image, self.temp.name, "alpha")
        with Image.open(path) as saved:
            self.assertEqual(saved.getpixel((0, 0)), (30, 60, 90, 127))

    def test_already_supported_formats_keep_native_format(self):
        for image_format in ("PNG", "JPEG", "BMP", "TIFF"):
            with self.subTest(format=image_format), make_image(image_format) as image:
                path = self.namespace["_save_client_image"](image, self.temp.name, "supported")
                self.assertEqual(Path(path).suffix, "." + image_format.lower())
                with Image.open(path) as saved:
                    self.assertEqual(saved.format, image_format)
                    self.assertEqual(saved.size, image.size)

    def test_save_failure_restores_original_animation_frame(self):
        with make_image("GIF", animated=True) as image:
            image.seek(1)
            with self.assertRaises(FileNotFoundError):
                self.namespace["_save_client_image"](image, str(Path(self.temp.name) / "absent"), "failure")
            self.assertEqual(image.tell(), 1)

    def test_real_client_evaluator_and_folder_judge_accept_gif_webp_on_both_sides(self):
        for left_format, right_format in (("PNG", "GIF"), ("WEBP", "GIF"), ("GIF", "WEBP")):
            with self.subTest(left=left_format, right=right_format):
                images = {"left": make_image(left_format, animated=left_format != "PNG"),
                          "right": make_image(right_format, animated=True)}
                for image in images.values():
                    self.addCleanup(image.close)
                elements = [
                    {"contentUrl": "left", "bbox": {"x": 0, "y": 10, "width": 20, "height": 20}},
                    {"contentUrl": "right", "bbox": {"x": 60, "y": 10, "width": 20, "height": 20}},
                ]
                observed = []

                def model(messages):
                    paths = [item["image"] for message in messages for item in message["content"]
                             if item["type"] == "image"]
                    observed.append(paths)
                    # Both the left reference and right candidate must reach the
                    # actual folder loader as static PNGs, not animated bytes.
                    self.assertEqual(len(paths), 2)
                    for path in paths:
                        with Image.open(path) as image:
                            self.assertEqual(image.format, "PNG")
                            self.assertEqual(image.n_frames, 1)
                    return "Yes"

                def gold_image(url, directory, **kwargs):
                    path = Path(directory) / "gold.png"
                    Image.new("RGB", (8, 6), "red").save(path)
                    return str(path)

                matcher = Mock(side_effect=[(True, "exact"), (False, "no_match")])
                self.namespace.update(
                    SLIDES_SERVICE=object(),
                    extract_slide_images=Mock(return_value=elements),
                    get_element_bbox=lambda element: element["bbox"],
                    download_slide_image=lambda url: images[url],
                    fetch_api_with_retry=Mock(return_value={"query": {"pages": {"1": {
                        "original": {"source": "https://example.invalid/gold.png"}}}}}),
                    download_image_with_retry=gold_image,
                    match_image_tiered=matcher,
                )
                steps = self.namespace["evaluate_single_client"](
                    "Brian May", "https://en.wikipedia.org/wiki/Brian_May",
                    {"Brian May": {"slide_idx": 0, "name_on_top": True}},
                    ["Name", "Left", "Right", "Symmetry"], [{}], 100, model,
                    "saved-presentation", self.temp.name,
                )
                self.assertEqual(len(steps), 4)
                self.assertEqual([step["success"] for step in steps], [True] * 4)
                self.assertEqual([step["category"] for step in steps],
                                 ["spatial", "deterministic", "llm_vlm_judgement", "spatial"])
                self.assertEqual(len(observed), 1)
                self.assertEqual(matcher.call_count, 2)
                self.assertFalse(any(Path(self.temp.name).glob("temp_*")))

    def test_missing_client_slide_still_returns_original_failed_criteria(self):
        steps = self.namespace["evaluate_single_client"](
            "Absent Client", "https://en.wikipedia.org/wiki/Absent", {"Absent Client": None},
            ["Name", "Left", "Right", "Symmetry"], [], 100, Mock(), "saved", self.temp.name)
        self.assertEqual(len(steps), 4)
        self.assertTrue(all(not step["success"] for step in steps))
        self.assertTrue(all(step["category"] == "dependency_not_evaluated" for step in steps))


if __name__ == "__main__":
    unittest.main()
