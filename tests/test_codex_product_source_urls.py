"""Offline source-identity regressions for Product Comparison slides."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Dict, List
import unittest


ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "codex_benchmark").is_dir())
SOURCE_ROOT = Path(os.environ.get("CODEX_PRODUCT_SOURCE_ROOT", ROOT))
FAMILY = SOURCE_ROOT / "src/browsergym/knows/eval/tasks/slides_42_personal_none_product_comparison"
SHARED = ROOT / "src/browsergym/knows/eval/eval_utils/slides_utils.py"
NAMES = {"_normalize_bare_source_urls", "_normalize_text_source_urls",
         "_normalize_slide_source_urls", "extract_slide_links", "ensure_scheme"}
SAVED_URLS = (
    "support.apple.com/en-us/122209",
    "psref.lenovo.com/Product/Yoga/Yoga_9_2_in_1_14ILL10",
    "support.microsoft.com/en-us/surface/models/surface-laptop-7th-edition-features",
)


def load_helpers():
    shared_tree = ast.parse(SHARED.read_text())
    shared_nodes = [node for node in shared_tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in {"extract_slide_links", "_extract_links_from_text_element"}]
    namespace = {"re": re, "deepcopy": deepcopy, "Dict": Dict, "List": List, "Any": Any}
    exec(compile(ast.Module(body=shared_nodes, type_ignores=[]), str(SHARED), "exec"), namespace)
    shared = SimpleNamespace(**namespace)
    tree = ast.parse((FAMILY / "utils.py").read_text())
    local_nodes = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in NAMES
                   or isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id.startswith("_BARE_URL")
                           for target in node.targets)]
    namespace = {"re": re, "deepcopy": deepcopy, "Dict": Dict, "List": List, "Any": Any,
                 "_shared_extract_slide_links": shared.extract_slide_links}
    exec(compile(ast.Module(body=local_nodes, type_ignores=[]), str(FAMILY / "utils.py"), "exec"), namespace)
    return SimpleNamespace(**namespace), shared


def text_element(text, native=None):
    run = {"content": text}
    if native is not None:
        run["style"] = {"link": {"url": native}, "bold": True}
    return {"textElements": [{"startIndex": 0, "textRun": run}]}


def slide(text, native=None):
    return {"objectId": "device", "pageElements": [
        {"objectId": "source", "shape": {"text": text_element(text, native)}}]}


class ProductSourceUrlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local, cls.shared = load_helpers()

    def test_exact_saved_urls_reproduce_old_truncation_and_retain_source_identity(self):
        for url in SAVED_URLS:
            with self.subTest(url=url):
                saved = slide("Official source: " + url)
                self.assertEqual(self.shared.extract_slide_links(saved), [url.split(".", 1)[1]])
                self.assertEqual(self.local.extract_slide_links(saved), ["https://" + url])
                self.assertEqual([self.local.ensure_scheme(link)
                                  for link in self.local.extract_slide_links(saved)], ["https://" + url])

    def test_scheme_www_ordinary_host_and_native_links_preserve_existing_results(self):
        for url in ("https://support.apple.com/en-us/122209", "http://psref.lenovo.com/Case",
                    "HTTPS://Example.COM/Case?X=1#Part", "www.support.apple.com/path",
                    "apple.com/en-us/122209", "https://example.com/?next=psref.lenovo.com/X"):
            with self.subTest(url=url):
                self.assertEqual(self.local.extract_slide_links(slide(url)),
                                 self.shared.extract_slide_links(slide(url)))
        native = slide("Official specifications", "http://support.apple.com/en-us/122209")
        self.assertEqual(self.local.extract_slide_links(native),
                         self.shared.extract_slide_links(native))

    def test_multiple_sources_native_target_and_plaintext_display_remain_distinct(self):
        value = slide(SAVED_URLS[0] + " and https://example.org/reference",
                      "https://actual.example.org/other")
        self.assertEqual(set(self.local.extract_slide_links(value)), {
            "https://" + SAVED_URLS[0], "https://example.org/reference",
            "https://actual.example.org/other"})

    def test_shape_and_table_traversal_preserves_raw_input_and_metadata(self):
        value = slide(SAVED_URLS[0])
        value["pageElements"][0]["transform"] = {"translateX": 35, "translateY": 27}
        value["pageElements"].append({"objectId": "table", "table": {"tableRows": [
            {"tableCells": [{"text": text_element(SAVED_URLS[1])},
                            {"text": text_element("Microsoft", "https://" + SAVED_URLS[2])}]}]}})
        before = deepcopy(value)
        self.assertEqual(set(self.local.extract_slide_links(value)),
                         {"https://" + url for url in SAVED_URLS})
        self.assertEqual(value, before)

    def test_bounded_non_url_tokens_remain_unchanged(self):
        for token in ("person@support.apple.com", "support..apple.com/a", "-support.apple.com/a",
                      "support-.apple.com/a", "support.apple.com:8080/a", "support.apple.invalid/a",
                      "ftp://support.apple.com/a", "1.2.3.4/a", "not a URL"):
            with self.subTest(token=token):
                self.assertEqual(self.local._normalize_bare_source_urls(token), token)
                self.assertEqual(self.local.extract_slide_links(slide(token)),
                                 self.shared.extract_slide_links(slide(token)))

    def test_case_paths_punctuation_and_explicit_url_nested_hosts(self):
        for raw, expected in (
            ("(Support.Apple.com/Case?Param=Value#Part).", "https://Support.Apple.com/Case?Param=Value#Part"),
            ("a.b.example.co.uk/path", "https://a.b.example.co.uk/path"),
            ("support.apple.com.", "https://support.apple.com"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(self.local.extract_slide_links(slide(raw)), [expected])
        explicit = "https://example.com/?next=support.apple.com/path"
        self.assertEqual(self.local._normalize_bare_source_urls(explicit + " " + SAVED_URLS[1]),
                         explicit + " https://" + SAVED_URLS[1])

    def test_missing_text_and_duplicate_sources_preserve_shared_behavior(self):
        for value in ({}, {"pageElements": []}, {"pageElements": [{"image": {"contentUrl": "https://example.com/img"}}]}):
            self.assertEqual(self.local.extract_slide_links(value), self.shared.extract_slide_links(value))
        self.assertEqual(self.local.extract_slide_links(slide(SAVED_URLS[0] + " " + SAVED_URLS[0])),
                         ["https://" + SAVED_URLS[0]])

    def test_all_five_real_wrappers_route_only_source_extractor_through_local_adapter(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                tree = ast.parse((FAMILY / f"instance_{instance}/evaluator.py").read_text())
                imports = {alias.asname or alias.name: node.module for node in tree.body
                           if isinstance(node, ast.ImportFrom) for alias in node.names}
                self.assertEqual(imports["extract_slide_links"],
                                 "browsergym.knows.eval.tasks.slides_42_personal_none_product_comparison.utils")
                self.assertEqual(imports["fetch_url_content"], "browsergym.knows.eval.eval_utils.web_utils")
                self.assertEqual(imports["extract_slide_text"], "browsergym.knows.eval.eval_utils.slides_utils")
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "extract_slide_links"]
                self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
