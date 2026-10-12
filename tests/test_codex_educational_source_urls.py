"""Offline regression for educational-deck source addresses, without a judge."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "src/browsergym/knows/eval/tasks/slides_26_basic_educational_slide_deck"
SHARED = ROOT / "src/browsergym/knows/eval/eval_utils/slides_utils.py"
NAMES = {"_normalize_bare_source_urls", "_normalize_text_source_urls",
         "_normalize_slide_source_urls", "_extract_links_from_text_element",
         "extract_slide_links", "extract_slide_links_with_positions"}


def load_helpers():
    spec = importlib.util.spec_from_file_location("educational_shared_slide_utils", SHARED)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    tree = ast.parse((FAMILY / "utils.py").read_text())
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in NAMES)
             or (isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id.startswith("_BARE_URL")
                         for target in node.targets))]
    namespace = {"re": re, "deepcopy": deepcopy, "Dict": Dict, "List": List, "Any": Any,
                 "_shared_extract_slide_links": shared.extract_slide_links,
                 "_shared_extract_slide_links_with_positions": shared.extract_slide_links_with_positions,
                 "_shared_extract_links_from_text_element": shared._extract_links_from_text_element}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FAMILY / "utils.py"), "exec"), namespace)
    return SimpleNamespace(**namespace), shared


def text_element(content, url=None):
    run = {"content": content}
    if url is not None:
        run["style"] = {"link": {"url": url}, "bold": True}
    return {"textElements": [{"startIndex": 0, "textRun": run}]}


class EducationalSourceUrlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local, cls.shared = load_helpers()

    def test_exact_saved_subdomains_reproduce_old_failure_and_keep_complete_address(self):
        for value in ("guides.loc.gov/k-pop/history-timeline", "science.nasa.gov/mercury/facts/"):
            with self.subTest(value=value):
                text = text_element("Source: " + value)
                self.assertEqual(self.shared._extract_links_from_text_element(text), [value.split(".", 1)[1]])
                self.assertEqual(self.local._extract_links_from_text_element(text), ["https://" + value])

    def test_explicit_schemes_native_hyperlinks_and_case_are_preserved(self):
        for value in ("https://science.nasa.gov/Mercury/Facts/?X=Case#Title",
                      "http://guides.loc.gov/k-pop/history-timeline", "HTTPS://Example.COM/Case",
                      "https://example.com/?next=science.nasa.gov/path#guides.loc.gov",
                      "https://example.com/path/science.nasa.gov"):
            text = text_element("Source: " + value)
            self.assertEqual(self.local._extract_links_from_text_element(text),
                             self.shared._extract_links_from_text_element(text))
        text = text_element("NASA reference", "https://science.nasa.gov/Mercury/Facts/")
        original = deepcopy(text)
        self.assertEqual(self.local._extract_links_from_text_element(text),
                         ["https://science.nasa.gov/Mercury/Facts/"])
        self.assertEqual(text, original)

    def test_subdomain_normalization_matches_existing_fallback_convention(self):
        for value in ("kids.kiddle.co/Great_Pyramid_of_Giza",
                      "A.B.example.co.uk/Case", "science.nasa.gov/mercury/facts/"):
            with self.subTest(value=value):
                self.assertEqual(self.local._extract_links_from_text_element(text_element("(" + value + ").")),
                                 ["https://" + value])
        self.assertEqual(self.local._extract_links_from_text_element(text_element("science.nasa.gov.")),
                         ["https://science.nasa.gov"])

    def test_ordinary_bare_hosts_and_www_addresses_are_unchanged(self):
        # Their missing-scheme image-fetch behavior is outside this repair.
        for value in ("genome.gov", "genome.gov/about-genomics/policy-issues/what-is-Genome-Editing",
                      "britannica.com/art/K-pop", "example.com", "www.example.com/Source",
                      "WWW.science.nasa.gov/mercury/facts/"):
            with self.subTest(value=value):
                self.assertEqual(self.local._normalize_bare_source_urls(value), value)
                text = text_element("Source: " + value)
                self.assertEqual(self.local._extract_links_from_text_element(text),
                                 self.shared._extract_links_from_text_element(text))

    def test_invalid_or_non_url_tokens_are_not_normalized(self):
        # Do not broaden the shared extractor's legacy handling of invalid text.
        for value in ("author@science.nasa.gov", "science..nasa.gov/path", "-science.nasa.gov/path",
                      "science-.nasa.gov/path", "science.nasa.gov:bad/path", "localhost/path",
                      "science.nasa.invalid/path", "https://science.nasa.gov/path",
                      "ftp://science.nasa.gov/path", "1.2.3.4/path", "not a URL"):
            with self.subTest(value=value):
                self.assertEqual(self.local._normalize_bare_source_urls(value), value)

    def test_mixed_explicit_url_with_nested_domain_and_separate_bare_citation(self):
        explicit = "https://example.com/?next=science.nasa.gov/path"
        text = explicit + " Source: guides.loc.gov/k-pop/history-timeline"
        self.assertEqual(self.local._normalize_bare_source_urls(text),
                         explicit + " Source: https://guides.loc.gov/k-pop/history-timeline")

    def test_slide_and_table_adapters_preserve_original_metadata_and_positions(self):
        shape = {"objectId": "source", "transform": {"translateX": 11, "translateY": 22,
                 "scaleX": 2, "scaleY": 3}, "size": {"width": {"magnitude": 10},
                 "height": {"magnitude": 20}}, "shape": {"text": text_element("science.nasa.gov/mercury/facts/")}}
        table = {"objectId": "table", "transform": {"translateX": 4, "translateY": 5},
                 "table": {"tableRows": [{"tableCells": [{"text": text_element("guides.loc.gov/k-pop/history-timeline")}]}]}}
        slide = {"objectId": "slide", "pageElements": [shape, table]}
        before = deepcopy(slide)
        urls = self.local.extract_slide_links(slide)
        self.assertEqual(set(urls), {"https://science.nasa.gov/mercury/facts/", "https://guides.loc.gov/k-pop/history-timeline"})
        positioned = self.local.extract_slide_links_with_positions(slide)
        self.assertEqual(positioned[0]["bbox"], {"x": 11, "y": 22, "width": 20, "height": 60})
        self.assertEqual(positioned[1]["bbox"], {"x": 4, "y": 5, "width": 0, "height": 0})
        self.assertEqual(slide, before)

    def test_multiple_urls_and_native_target_with_different_display_address_keep_existing_semantics(self):
        text = text_element("science.nasa.gov/mercury/facts/ and https://example.com/x",
                            "https://actual.example.org/target")
        self.assertEqual(self.local._extract_links_from_text_element(text),
                         ["https://actual.example.org/target", "https://science.nasa.gov/mercury/facts/",
                          "https://example.com/x"])

    def test_all_five_wrappers_route_every_link_reader_to_family_adapters(self):
        routed = {"extract_slide_links", "extract_slide_links_with_positions", "_extract_links_from_text_element"}
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                tree = ast.parse((FAMILY / f"instance_{instance}/evaluator.py").read_text())
                imports = {alias.asname or alias.name: node.module for node in tree.body
                           if isinstance(node, ast.ImportFrom) for alias in node.names}
                for name in routed:
                    self.assertEqual(imports[name], "browsergym.knows.eval.tasks.slides_26_basic_educational_slide_deck.utils")
                    self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                        and node.func.id == name for node in ast.walk(tree)))

    def test_existing_image_matcher_receives_full_url_and_retains_page_fallback(self):
        tree = ast.parse((FAMILY / "utils.py").read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "match_source_image")
        import os
        download, page = Mock(return_value=None), Mock(return_value=["embedded.png"])
        namespace = {"List": List, "Tuple": tuple, "os": os, "_browser_headers": lambda url: {},
                     "download_image_from_url": download, "download_page_images": page,
                     "match_image_tiered": lambda *a, **k: (True, "exact"), "binary_compare_images": Mock(return_value=False)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(FAMILY / "utils.py"), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "saved.png"; saved.write_bytes(b"existing slide image fixture")
            urls = self.local._extract_links_from_text_element(text_element("science.nasa.gov/mercury/facts/"))
            matched, details = namespace["match_source_image"](urls, str(saved), directory, 0, object())
        self.assertTrue(matched)
        self.assertIn("Page: Source image matches", details)
        self.assertEqual(download.call_args_list[0].args[0], "https://science.nasa.gov/mercury/facts/")
        self.assertEqual(page.call_args.args[0], "https://science.nasa.gov/mercury/facts/")


if __name__ == "__main__":
    unittest.main()
