"""Offline regression coverage for recipe text and links inside Docs tables."""
import ast
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import io
import copy
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import unittest
from unittest.mock import Mock, patch


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "src/browsergym/knows/eval").is_dir())
EVAL = ROOT / "src/browsergym/knows/eval"
SOURCE = Path(os.environ.get("KNOWS_RECIPE_UTILS", str(
    EVAL / "tasks/docs_11_personal_recipe_ocr/utils.py")))


def definitions(path, names, namespace):
    # Compile actual pure functions only; importing either full module creates
    # credentialed services and loads unrelated evaluator dependencies.
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def load_utils():
    namespace = {"__name__": __name__, "dataclass": dataclass, "field": field,
                 "re": re, "os": os, "List": List, "Dict": Dict, "Tuple": Tuple,
                 "Any": Any, "Optional": Optional,
                 "REQUIRED_SECTIONS": ["ingredients", "preparation", "tips"],
                 "SECTION_HEADERS": ["ingredients", "preparation", "tips", "ready in", "serves", "calories"]}
    definitions(SOURCE, {"Recipe", "discover_recipes", "extract_recipe_title",
                        "_iter_recipe_paragraphs", "extract_recipe_text_from_doc",
                        "_recipe_paragraph_record", "_annotate_recipe_paragraph_styles",
                        "_styled_recipe_header_starts", "_preceding_recipe_heading_block", "_recipe_heading_blocks",
                        "extract_hyperlinks", "setup_document"}, namespace)
    return namespace


def paragraph(text, url=None, style=None):
    run = {"content": text}
    if url is not None:
        run["textStyle"] = {"link": {"url": url}}
    result = {"paragraph": {"elements": [{"textRun": run}]}}
    if style is not None:
        result["paragraph"]["paragraphStyle"] = {"namedStyleType": style}
    return result


def table(*rows):
    return {"table": {"tableRows": [
        {"tableCells": [{"content": cell} for cell in row]} for row in rows
    ]}}


def service_for(content):
    document = {"body": {"content": content}}
    service = Mock()
    service.documents.return_value.get.return_value.execute.return_value = document
    return service, document


def recipe(number, title, source):
    return [paragraph(f"RECIPE {number:02d}\n"), paragraph(title + "\n"),
            paragraph(source + "\n"), paragraph("Ingredients\n"),
            paragraph(f"Ingredients for {title}\n"), paragraph("Preparation\n"),
            paragraph(f"Prepare {title} carefully\n"), paragraph("Tips\n"),
            paragraph(f"Keep {title} warm\n")]


class RecipeHyperlinksTests(unittest.TestCase):
    def setUp(self):
        self.utils = load_utils()

    def links(self, content, number=None, titles=None):
        service, _ = service_for(content)
        result = self.utils["extract_hyperlinks"]("offline-document", service, number, titles)
        service.documents.return_value.get.assert_called_once_with(documentId="offline-document")
        return result

    def test_top_level_headers_titles_and_links_keep_original_scope(self):
        content = recipe(1, "Pumpkin Soup", "https://example.test/soup") + recipe(
            2, "Apple Pie", "https://example.test/pie")
        for number, expected in ((1, "soup"), (2, "pie")):
            self.assertEqual(self.links(content, number, ["Pumpkin Soup", "Apple Pie"]),
                             [{"url": f"https://example.test/{expected}",
                               "text": f"https://example.test/{expected}"}])
        self.assertEqual(self.links(content, 3, ["Pumpkin Soup", "Apple Pie"]), [])

    def test_embedded_and_plain_urls_in_multiple_table_cells_are_collected(self):
        content = [table(([paragraph("Source\n", "https://example.test/source")],
                          [paragraph("https://example.test/tips\n")]))]
        self.assertEqual(self.links(content), [
            {"url": "https://example.test/source", "text": "Source"},
            {"url": "https://example.test/tips", "text": "https://example.test/tips"}])

    def test_nested_tables_keep_document_order(self):
        content = [paragraph("https://example.test/before\n"),
                   table(([paragraph("https://example.test/outer\n"),
                           table(([paragraph("https://example.test/nested\n")],))],
                          [paragraph("https://example.test/right\n")])),
                   paragraph("https://example.test/after\n")]
        self.assertEqual([link["url"].rsplit("/", 1)[-1] for link in self.links(content)],
                         ["before", "outer", "nested", "right", "after"])

    def test_two_recipes_in_same_table_do_not_leak_links(self):
        first = recipe(1, "Pumpkin Soup", "https://example.test/soup")
        second = recipe(2, "Apple Pie", "https://example.test/pie")
        # Match recipe templates: title/source in left cell, sections in right.
        content = [paragraph("\n"), table((first[:3], first[3:]), (second[:3], second[3:])),
                   paragraph("\n")]
        for number, expected in ((1, "soup"), (2, "pie")):
            result = self.links(content, number, ["Pumpkin Soup", "Apple Pie"])
            self.assertEqual([link["url"] for link in result], [f"https://example.test/{expected}"])

    def test_mixed_top_level_and_nested_recipe_boundaries(self):
        content = recipe(1, "Pumpkin Soup", "https://example.test/soup") + [
            table(([table((recipe(2, "Apple Pie", "https://example.test/pie"),))],))]
        self.assertEqual([link["url"] for link in self.links(content, 2, ["Pumpkin Soup", "Apple Pie"])],
                         ["https://example.test/pie"])

    def test_title_only_table_boundaries_work_without_recipe_headers(self):
        first = recipe(1, "Pumpkin Soup", "https://example.test/soup")[1:]
        second = recipe(2, "Apple Pie", "https://example.test/pie")[1:]
        content = [table((first, second))]
        self.assertEqual([link["url"] for link in self.links(content, 2, ["Pumpkin Soup", "Apple Pie"])],
                         ["https://example.test/pie"])

    def test_empty_recipe_headers_still_do_not_increment_scope(self):
        first = recipe(1, "Pumpkin Soup", "https://example.test/soup")
        second = recipe(2, "Apple Pie", "https://example.test/pie")
        for content in ([paragraph("RECIPE\n")] + first + second,
                        [table(([paragraph("RECIPE\n")] + first, second))]):
            with self.subTest(table="table" in content[0]):
                self.assertEqual([link["url"] for link in self.links(content, 1, ["Pumpkin Soup", "Apple Pie"])],
                                 ["https://example.test/soup"])

    def test_embedded_plain_duplicate_url_is_not_double_counted(self):
        content = [table(([paragraph("https://example.test/source\n", "https://example.test/source")],))]
        self.assertEqual(len(self.links(content)), 1)


class RecipeDocumentTextTests(unittest.TestCase):
    def setup_document(self, content, document_extra=None):
        namespace = load_utils()
        service, document = service_for(content)
        document.update(document_extra or {})
        # Exercise actual setup_document and the existing actual recursive
        # structure extractor, replacing only external PDF/image side effects.
        services = ModuleType("browsergym.knows.eval.eval_utils.google_services_utils")
        pure = {"get_doc_content": lambda doc_id, svc: document, "HttpError": RuntimeError}
        definitions(EVAL / "eval_utils/google_services_utils.py",
                    {"extract_text_from_doc", "extract_structure_from_doc"}, pure)
        services.extract_text_from_doc = pure["extract_text_from_doc"]
        services.extract_structure_from_doc = pure["extract_structure_from_doc"]
        for name in ("download_doc_as_pdf", "extract_images_from_doc", "extract_images_from_doc_with_cropping"):
            setattr(services, name, Mock())
        images = ModuleType("browsergym.knows.eval.eval_utils.image_utils")
        images.convert_pdf_to_pngs = Mock()
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            services.__name__: services, images.__name__: images
        }), redirect_stdout(stdout):
            result = namespace["setup_document"]("offline-document", directory, Mock(), service)
            recipes = namespace["discover_recipes"](result["doc_text"], result["doc_structure"])
        return result, recipes, stdout.getvalue()

    def test_top_level_text_preserves_exact_text_runs_and_blank_paragraphs(self):
        content = [paragraph("\n"), {"paragraph": {"elements": [
            {"textRun": {"content": "Pumpkin "}}, {"textRun": {"content": "Soup\n"}},
            {"inlineObjectElement": {"inlineObjectId": "ignored-image"}}]}}, paragraph("\n")]
        result, _, log = self.setup_document(content)
        self.assertEqual(result["doc_text"], "\nPumpkin Soup\n\n")
        self.assertIn("Total characters extracted: 15", log)

    def test_table_recipe_text_restores_titles_and_separate_text_ranges(self):
        first = recipe(1, "Pumpkin Soup", "https://example.test/soup")
        second = recipe(2, "Apple Pie", "https://example.test/pie")
        # Existing title parsing accepts titles following RECIPE on its line.
        # Keep that rule unchanged; this regression concerns missing table text.
        first[0] = paragraph("RECIPE Pumpkin Soup\n")
        second[0] = paragraph("RECIPE Apple Pie\n")
        content = [paragraph("\n"), table((first[:3], first[3:]), (second[:3], second[3:])), paragraph("\n")]
        result, recipes, log = self.setup_document(content)
        expected = "\n" + "".join(element["paragraph"]["elements"][0]["textRun"]["content"]
                                   for element in first + second) + "\n"
        self.assertEqual(result["doc_text"], expected)
        self.assertIn(f"Total characters extracted: {len(expected)}", log)
        self.assertEqual([item.title for item in recipes], ["Pumpkin Soup", "Apple Pie"])
        self.assertEqual([item.recipe_num for item in recipes], [1, 2])
        self.assertTrue(all(item.sections_found == ["ingredients", "preparation", "tips"] for item in recipes))
        self.assertIn("Pumpkin Soup", recipes[0].text)
        self.assertNotIn("Apple Pie", recipes[0].text)
        self.assertIn("Apple Pie", recipes[1].text)
        self.assertNotIn("Pumpkin Soup", recipes[1].text)
        self.assertEqual(recipes[0].end_index, recipes[1].start_index)
        self.assertNotIn("Could not find", log)

    def test_numbered_recipe_headers_use_the_following_actual_titles(self):
        first = recipe(1, "Pumpkin Soup", "https://example.test/soup")
        second = recipe(2, "Apple Pie", "https://example.test/pie")
        _, top_level, _ = self.setup_document(first + second)
        _, in_table, _ = self.setup_document([table((first, second))])
        self.assertEqual([item.title for item in top_level], ["Pumpkin Soup", "Apple Pie"])
        self.assertEqual([item.title for item in in_table], [item.title for item in top_level])

    def test_nested_table_text_matches_recursive_structure_order(self):
        content = [paragraph("prefix\n"), table(([paragraph("outer\n"),
                    table(([paragraph("nested\n")],))], [paragraph("right\n")])), paragraph("suffix\n")]
        result, _, _ = self.setup_document(content)
        self.assertEqual(result["doc_text"], "prefix\nouter\nnested\nright\nsuffix\n")
        self.assertEqual(result["doc_text"], "".join(item["content"] for item in result["doc_structure"]
                                                    if item["type"] == "text"))


class RecipeTitleMarkerTests(unittest.TestCase):
    def setUp(self):
        self.extract_title = load_utils()["extract_recipe_title"]

    def test_numbered_and_bulleted_markers_are_not_recipe_titles(self):
        for marker in ("RECIPE 01", "RECIPE • 01", "RECIPE •01", "Recipe 2",
                       "RECIPE 01 • THE ORIGINAL", "RECIPE 02  •  QUICK CLASSIC",
                       "RECIPE • 03 • CRUNCHY TOPPING"):
            with self.subTest(marker=marker):
                self.assertEqual(self.extract_title(marker + "\nPumpkin Soup\nIngredients\n"),
                                 "Pumpkin Soup")

    def test_empty_numbered_marker_does_not_invent_a_title(self):
        self.assertEqual(self.extract_title("RECIPE • 01\n\n"), "")

    def test_marker_after_leading_blank_lines_uses_the_following_title(self):
        self.assertEqual(self.extract_title("\n\nRECIPE • 01\nCardamom Cookies\n"),
                         "Cardamom Cookies")

    def test_existing_same_line_and_concatenated_titles_are_preserved(self):
        for text, expected in (("RECIPE Pumpkin Soup\nIngredients", "Pumpkin Soup"),
                               ("RECIPE\x0bPumpkin Soup\nIngredients", "Pumpkin Soup"),
                               ("RECIPESoupe au Potiron\nIngredients", "Soupe au Potiron"),
                               ("RECIPE\nPumpkin Soup\nIngredients", "Pumpkin Soup"),
                               ("RECIPE 7 Layer Dip\nIngredients", "7 Layer Dip")):
            with self.subTest(text=text):
                self.assertEqual(self.extract_title(text), expected)

    def test_author_labels_are_not_reinterpreted_as_numbered_markers(self):
        self.assertEqual(self.extract_title("RECIPE • KING ARTHUR TEST KITCHEN\nIngredients\n"),
                         "• KING ARTHUR TEST KITCHEN")


def styled_recipe(title, source, credit="RECIPE • TEST COLLECTION"):
    heading = [paragraph(title + "\n", style="HEADING_1")]
    if source:
        heading.extend([paragraph(source + "\n", style="NORMAL_TEXT"), paragraph("\n", style="NORMAL_TEXT")])
    if credit:
        heading.append(paragraph(credit + "\n", style="NORMAL_TEXT"))
    return heading + [paragraph("Description of the recipe\n", style="NORMAL_TEXT"),
                      paragraph("Ingredients\n"), paragraph("One cup flour\n"),
                      paragraph("Preparation\n"), paragraph("Mix and bake\n"),
                      paragraph("Tips\n"), paragraph("Serve warm\n")]


class RecipeStyledBoundaryTests(RecipeDocumentTextTests):
    def scoped_urls(self, content, recipes):
        namespace = load_utils()
        titles = [item.title for item in recipes]
        result = []
        for recipe_item in recipes:
            service, _ = service_for(content)
            links = namespace["extract_hyperlinks"]("offline-document", service,
                                                   recipe_item.recipe_num, titles)
            result.append([link["url"] for link in links])
        return result

    def records(self, content):
        namespace = load_utils()
        return [namespace["_recipe_paragraph_record"](element["paragraph"], index)
                for index, element in enumerate(content)]

    def test_repeated_credit_labels_use_distinct_titles_and_source_urls(self):
        content = styled_recipe("Original Cookies", None)
        content += styled_recipe("Oat Cookies", "https://example.test/oat")
        content += styled_recipe("Wheat Cookies", "https://example.test/wheat")
        _, recipes, _ = self.setup_document(content)
        self.assertEqual([item.title for item in recipes], ["Original Cookies", "Oat Cookies", "Wheat Cookies"])
        self.assertEqual(self.scoped_urls(content, recipes),
                         [[], ["https://example.test/oat"], ["https://example.test/wheat"]])
        self.assertNotIn("Oat Cookies", recipes[0].text)
        self.assertNotIn("Wheat Cookies", recipes[1].text)
        self.assertTrue(all(a.end_index == b.start_index for a, b in zip(recipes, recipes[1:])))

    def test_plain_and_uppercase_template_labels_do_not_double_count_styled_title(self):
        for label in ("RECIPE", "RECIPE TERRACOTTA  •  OVEN BAKED FLATBREAD"):
            for style in ("TITLE", "HEADING_1"):
                with self.subTest(label=label, style=style):
                    first = styled_recipe("Original Flatbread", None, credit=label)
                    first[0]["paragraph"]["paragraphStyle"]["namedStyleType"] = style
                    first.append(paragraph("https://example.test/tip\n"))
                    second = styled_recipe("Quick Flatbread", "https://example.test/quick", credit=None)
                    third = styled_recipe("Easy Flatbread", "https://example.test/easy", credit=None)
                    content = first + second + third
                    _, recipes, _ = self.setup_document(content)
                    self.assertEqual([item.title for item in recipes],
                                     ["Original Flatbread", "Quick Flatbread", "Easy Flatbread"])
                    self.assertEqual(recipes[0].structure_indices[0], 0)
                    self.assertEqual(self.scoped_urls(content, recipes),
                                     [["https://example.test/tip"], ["https://example.test/quick"],
                                      ["https://example.test/easy"]])

    def test_template_label_requires_qualified_heading_and_exact_uppercase_shape(self):
        namespace = load_utils()
        heading = paragraph("Original Flatbread\n", style="TITLE")
        label = "RECIPE TERRACOTTA • OVEN BAKED FLATBREAD\n"
        invalid = [
            [paragraph("Original Flatbread\n", style="NORMAL_TEXT"), paragraph(label, style="NORMAL_TEXT")],
            [heading, paragraph(label)],
            [heading, paragraph("RECIPE Pumpkin Soup • Original\n", style="NORMAL_TEXT")],
            [heading, paragraph("RECIPE 01 • ORIGINAL\n", style="NORMAL_TEXT")],
            [heading, paragraph("RECIPE TERRACOTTA • Oven Baked Flatbread\n", style="NORMAL_TEXT")],
            [paragraph("RECIPE\n", style="NORMAL_TEXT")],
            [paragraph("Original Flatbread\n", style="NORMAL_TEXT"), paragraph("RECIPE\n", style="NORMAL_TEXT")],
        ]
        for index, content in enumerate(invalid):
            with self.subTest(case=index):
                self.assertEqual(namespace["_styled_recipe_header_starts"](self.records(content)), {})

    def test_plain_recipe_before_unstyled_title_retains_original_boundary_rule(self):
        first = recipe(1, "Original Flatbread", "https://example.test/original")
        second = recipe(2, "Quick Flatbread", "https://example.test/quick")
        first[0] = paragraph("RECIPE\n", style="NORMAL_TEXT")
        second[0] = paragraph("RECIPE\n", style="NORMAL_TEXT")
        content = first + second
        namespace = load_utils()
        for number, url in ((1, "https://example.test/original"), (2, "https://example.test/quick")):
            service, _ = service_for(content)
            self.assertEqual([link["url"] for link in namespace["extract_hyperlinks"](
                "offline-document", service, number, ["Original Flatbread", "Quick Flatbread"])], [url])

    def test_credit_heading_override_is_bounded_and_requires_exact_style_evidence(self):
        namespace = load_utils()
        heading = paragraph("Oat Cookies\n", style="HEADING_1")
        label = paragraph("RECIPE • TEST COLLECTION\n", style="NORMAL_TEXT")
        blocker = paragraph("not a source URL\n", style="NORMAL_TEXT")
        inline = {"paragraph": {"elements": [{"inlineObjectElement": {"inlineObjectId": "photo"}}]}}
        invalid = [
            [heading, blocker, label],
            [heading, inline, label],
            [heading, paragraph("Ingredients\n"), label],
            [heading, paragraph("https://example.test/one\n"), paragraph("https://example.test/two\n"), label],
            [heading] + [paragraph("\n")] * 7 + [label],
            [paragraph("Oat Cookies\n"), label],
            [heading, paragraph("RECIPE • TEST COLLECTION\n", style="HEADING_1")],
            [heading, paragraph("RECIPE • 01\n", style="NORMAL_TEXT")],
            [heading, paragraph("RECIPE Oat Cookies\n", style="NORMAL_TEXT")],
        ]
        for index, content in enumerate(invalid):
            with self.subTest(case=index):
                self.assertEqual(namespace["_styled_recipe_header_starts"](self.records(content)), {})
        self.assertEqual(namespace["_styled_recipe_header_starts"](self.records([
            paragraph("Older Heading\n", style="HEADING_1"), heading, label])), {2: 1})

    def test_title_style_is_accepted_for_credit_layout(self):
        content = styled_recipe("Original Cookies", None)
        content[0]["paragraph"]["paragraphStyle"]["namedStyleType"] = "TITLE"
        _, recipes, _ = self.setup_document(content)
        self.assertEqual([item.title for item in recipes], ["Original Cookies"])

    def test_no_recipe_marker_prefers_heading_over_description(self):
        content = styled_recipe("Healthy Pizza", "https://example.test/pizza", credit=None)
        content += styled_recipe("Frozen Pizza", "https://example.test/frozen", credit=None)
        _, recipes, _ = self.setup_document(content)
        self.assertEqual([item.title for item in recipes], ["Healthy Pizza", "Frozen Pizza"])
        self.assertEqual(self.scoped_urls(content, recipes),
                         [["https://example.test/pizza"], ["https://example.test/frozen"]])

    def test_consecutive_heading_block_starts_at_first_title(self):
        content = styled_recipe("Pasta Puttanesca", "https://example.test/pasta", credit=None)
        content.insert(1, paragraph("for Dez\n", style="HEADING_1"))
        content += styled_recipe("Bean Pasta", "https://example.test/beans", credit=None)
        _, recipes, _ = self.setup_document(content)
        self.assertEqual([item.title for item in recipes], ["Pasta Puttanesca for Dez", "Bean Pasta"])
        self.assertEqual(self.scoped_urls(content, recipes),
                         [["https://example.test/pasta"], ["https://example.test/beans"]])

    def test_styled_fallback_preserves_unstyled_fallback_and_explicit_markers(self):
        unstyled = styled_recipe("Healthy Pizza", "https://example.test/pizza", credit=None)
        unstyled[0]["paragraph"].pop("paragraphStyle")
        _, recipes, _ = self.setup_document(unstyled)
        self.assertEqual([item.title for item in recipes], ["Description of the recipe"])
        explicit = recipe(1, "Soup", "https://example.test/soup")
        explicit.insert(0, paragraph("Editorial Heading\n", style="HEADING_1"))
        _, recipes, _ = self.setup_document(explicit)
        self.assertEqual(recipes[0].structure_indices[0], 1)

    def test_styled_fallback_is_bounded_and_does_not_cross_prior_section_cluster(self):
        namespace = load_utils()
        content = [paragraph("Too Far\n", style="HEADING_1")]
        content += [paragraph("description\n")] * 33 + [paragraph("Ingredients\n")]
        records = self.records(content)
        self.assertIsNone(namespace["_preceding_recipe_heading_block"](records, len(records) - 1, 0))
        content = [paragraph("Prior Recipe\n", style="HEADING_1"),
                   paragraph("Tips\n"), paragraph("description\n"), paragraph("Ingredients\n")]
        self.assertIsNone(namespace["_preceding_recipe_heading_block"](self.records(content), 3, 2))

    def test_separate_styled_heading_groups_are_not_guessed(self):
        namespace = load_utils()
        content = [paragraph("Editorial Heading\n", style="HEADING_1"),
                   paragraph("description\n"), paragraph("Possible Recipe\n", style="HEADING_1"),
                   paragraph("Ingredients\n")]
        self.assertIsNone(namespace["_preceding_recipe_heading_block"](self.records(content), 3, 0))

    def test_annotation_keeps_duplicate_text_styles_and_inline_images_in_order(self):
        namespace = load_utils()
        content = [paragraph("Repeated\n", style="HEADING_1"), paragraph("Repeated\n", style="NORMAL_TEXT"),
                   {"paragraph": {"paragraphStyle": {"namedStyleType": "NORMAL_TEXT"}, "elements": [
                       {"textRun": {"content": "before"}},
                       {"inlineObjectElement": {"inlineObjectId": "photo"}},
                       {"textRun": {"content": "after\n"}}]}}]
        _, document = service_for(content)
        document["inlineObjects"] = {"photo": {"inlineObjectProperties": {"embeddedObject": {}}}}
        pure = {"get_doc_content": lambda doc_id, service: document}
        definitions(EVAL / "eval_utils/google_services_utils.py", {"extract_structure_from_doc"}, pure)
        with redirect_stdout(io.StringIO()):
            original = pure["extract_structure_from_doc"]("offline-document", Mock())
        saved = copy.deepcopy(original)
        actual = namespace["_annotate_recipe_paragraph_styles"](original, document)
        self.assertEqual([(x["type"], x["content"]) for x in actual], [(x["type"], x["content"]) for x in saved])
        for before, after in zip(saved, actual):
            self.assertTrue(all(after["metadata"][k] == v for k, v in before["metadata"].items()))
        self.assertEqual([x["metadata"].get("recipe_named_style_type") for x in actual],
                         ["HEADING_1", "NORMAL_TEXT", "NORMAL_TEXT", None, "NORMAL_TEXT"])
        self.assertTrue(actual[2]["metadata"]["recipe_boundary_barrier"])
        self.assertTrue(actual[4]["metadata"]["recipe_boundary_barrier"])

    def test_annotation_does_not_guess_on_mismatched_structure(self):
        namespace = load_utils()
        original = [{"type": "text", "content": "different\n", "metadata": {}}]
        saved = copy.deepcopy(original)
        _, document = service_for([paragraph("Heading\n", style="HEADING_1")])
        with redirect_stdout(io.StringIO()):
            result = namespace["_annotate_recipe_paragraph_styles"](original, document)
        self.assertEqual(result, saved)


if __name__ == "__main__":
    unittest.main()
