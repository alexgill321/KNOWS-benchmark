"""Offline regression tests for the verified music-client alias only."""
import ast
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import re
import runpy
import time
import unittest
from typing import Any, Dict, List
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "src/browsergym/knows/eval"
FAMILY = EVAL / "tasks/slides_30_Work_Wikipedia_Photos"
MUSICIAN_URL = "https://en.wikipedia.org/wiki/Taj_Mahal_(musician)"


def definitions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Missing expected production function")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def helpers():
    ns = {"re": re}
    definitions(FAMILY / "utils.py", {
        "_normalize_name_parts", "_initial_matches", "name_exact_match",
        "resolve_client_reference_name", "client_name_matches",
    }, ns)
    return ns


class WikipediaAliasTests(unittest.TestCase):
    def setUp(self):
        self.ns = helpers()
        self.cache = {"taj mahal": MUSICIAN_URL}

    def test_exact_gold_bound_alias_resolves_without_mutating_inputs(self):
        before = copy.deepcopy(self.cache)
        self.assertEqual(self.ns["resolve_client_reference_name"](
            "musician taj mahal", self.cache), "taj mahal")
        self.assertTrue(self.ns["client_name_matches"](
            "Taj Mahal", "musician taj mahal", self.cache))
        self.assertEqual(self.cache, before)

    def test_missing_wrong_or_conflicting_gold_does_not_create_alias(self):
        for cache in ({}, {"taj mahal": "https://en.wikipedia.org/wiki/Taj_Mahal"},
                      {"taj mahal": MUSICIAN_URL + "?other=1"},
                      {"taj mahal": MUSICIAN_URL,
                       "musician taj mahal": "https://en.wikipedia.org/wiki/Other"}):
            with self.subTest(cache=cache):
                self.assertEqual(self.ns["resolve_client_reference_name"](
                    "musician taj mahal", cache), "musician taj mahal")
                self.assertFalse(self.ns["client_name_matches"](
                    "Taj Mahal", "musician taj mahal", cache))

    def test_no_generic_occupation_or_fuzzy_alias_rules(self):
        for client, title in (("scientist Taj Mahal", "Taj Mahal"),
                              ("musician Taj Mahal Jr", "Taj Mahal"),
                              ("musician Taj Mahal", "Taj Henry Mahal"),
                              ("musician Taj Mahal", "Taj Mahal Palace")):
            with self.subTest(client=client, title=title):
                self.assertFalse(self.ns["client_name_matches"](title, client, self.cache))

    def test_existing_name_variants_and_non_alias_references_unchanged(self):
        for first, second in (("John M. Smith", "John Michael Smith"),
                              ("Brian May", "Brian May"),
                              ("John Williams", "Mike Mills"),
                              ("Musician Taj Mahal", "musician taj mahal")):
            with self.subTest(first=first, second=second):
                self.assertEqual(self.ns["client_name_matches"](first, second, self.cache),
                                 self.ns["name_exact_match"](first, second))
        self.assertEqual(self.ns["resolve_client_reference_name"]("brian may", self.cache),
                         "brian may")

    def test_real_five_checkpoint_functions_keep_client_and_step_names(self):
        scoring = runpy.run_path(str(EVAL / "eval_utils/scoring.py"))
        for instance in range(1, 6):
            for usable_gold in (True, False):
                with self.subTest(instance=instance, usable_gold=usable_gold):
                    ns = helpers()
                    cache = self.cache.copy() if usable_gold else {}
                    clients = ["musician taj mahal"]
                    called = []

                    def evaluate(client, wiki_title, mapping, names, *args):
                        called.append((client, wiki_title, copy.deepcopy(mapping)))
                        self.assertEqual(client, "musician taj mahal")
                        self.assertEqual(wiki_title, "Taj_Mahal_(musician)")
                        self.assertEqual(mapping[client], {"slide_idx": 1, "name_on_top": True})
                        return [{"name": f"{client} - {name}", "success": True,
                                 "detail": "Offline downstream test result",
                                 "execution_time": 0, "category": "deterministic"}
                                for name in names]

                    ns.update(time=time, Checkpoint=scoring["Checkpoint"],
                              StepCategory=scoring["StepCategory"], model=object(),
                              gold_clients=clients, wiki_urls_cache=cache,
                              presentation_data={"slides": [{"text": "Title"},
                                                            {"text": "Taj Mahal"}]},
                              presentation_id="saved-workspace", DATA_DIR="unused",
                              get_slide_dimensions=lambda presentation: (100, 60),
                              extract_text_boxes_from_slide=lambda slide: [{"text": slide["text"]}],
                              is_text_in_title_position=lambda slide, text: True,
                              evaluate_single_client=evaluate,
                              parallel_execute=lambda tasks, **kwargs: {
                                  task["id"]: task["func"](*task["args"]) for task in tasks})
                    definitions(FAMILY / f"instance_{instance}/evaluator.py",
                                {"grade_checkpoint_2"}, ns)
                    with redirect_stdout(io.StringIO()):
                        checkpoint = ns["grade_checkpoint_2"]()
                    self.assertEqual(clients, ["musician taj mahal"])
                    self.assertEqual(cache, self.cache if usable_gold else {})
                    self.assertEqual(checkpoint.total, 4)
                    self.assertEqual(len(checkpoint.steps), 4)
                    self.assertTrue(all(step.name.startswith("musician taj mahal - ")
                                        for step in checkpoint.steps))
                    self.assertEqual(len(called), int(usable_gold))
                    self.assertEqual(checkpoint.result, 4 if usable_gold else 0)
                    if not usable_gold:
                        self.assertTrue(all(step.category == "execution_error"
                                            for step in checkpoint.steps))

    def test_all_five_url_checks_resolve_only_the_bound_lookup(self):
        for instance in range(1, 6):
            path = FAMILY / f"instance_{instance}/evaluator.py"
            tree = ast.parse(path.read_text())
            for checkpoint in ("grade_checkpoint_2", "grade_checkpoint_3", "grade_checkpoint_4"):
                function = next(node for node in tree.body
                                if isinstance(node, ast.FunctionDef) and node.name == checkpoint)
                assignments = [node for node in ast.walk(function)
                               if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name) and target.id == "client_url"
                                       for target in node.targets)]
                self.assertEqual(len(assignments), 1)
                expression = compile(ast.Expression(assignments[0].value), str(path), "eval")
                for client in ("musician taj mahal", "brian may"):
                    cache = {**self.cache, "brian may": "https://en.wikipedia.org/wiki/Brian_May"}
                    ns = {**self.ns, "client": client, "wiki_urls_cache": cache}
                    self.assertEqual(eval(expression, ns),
                                     MUSICIAN_URL if client.startswith("musician") else cache[client])


class WikipediaParenthesesTests(unittest.TestCase):
    def setUp(self):
        self.ns = {**helpers(), "Dict": Dict, "List": List, "Any": Any, "urlsplit": urlsplit}
        definitions(EVAL / "eval_utils/slides_utils.py",
                    {"extract_slide_links", "_extract_links_from_text_element"}, self.ns)
        definitions(FAMILY / "utils.py", {"extract_slide_links_with_wikipedia_parentheses"}, self.ns)
        self.extract = self.ns["extract_slide_links_with_wikipedia_parentheses"]

    @staticmethod
    def slide(text):
        return {"pageElements": [{"shape": {"text": {"textElements": [
            {"textRun": {"content": text}}]}}}]}

    def test_full_parenthetical_paths_retained_without_rewriting_wrong_identity(self):
        correct = "https://en.wikipedia.org/wiki/Steve_Miller_(musician)"
        wrong = "https://en.wikipedia.org/wiki/Steve_Smith_(musician)"
        slide = self.slide(correct + "\n" + wrong)
        before = self.ns["extract_slide_links"](slide)
        after = self.extract(slide)
        self.assertTrue(set(before).issubset(after))
        self.assertIn(correct, after)
        self.assertIn(wrong, after)
        self.assertNotIn("https://en.wikipedia.org/wiki/Steve_Smith_(drummer)", after)

    def test_existing_native_links_preserved_exactly(self):
        urls = ["https://en.wikipedia.org/wiki/John_Williams", MUSICIAN_URL,
                "https://example.org/photo_(one)", "https://www.photographer.example/",
                "http://en.wikipedia.org/wiki/Brian_May", "https://example.org/?next=value"]
        slide = {"pageElements": [{"shape": {"text": {"textElements": [
            {"textRun": {"content": "native label", "style": {"link": {"url": url}}}}
            for url in urls]}}}]}
        self.assertEqual(set(self.extract(slide)), set(urls))

    def test_table_text_uses_same_recovery(self):
        slide = {"pageElements": [{"table": {"tableRows": [{"tableCells": [
            {"text": {"textElements": [{"textRun": {"content": MUSICIAN_URL}}]}}]}]}}]}
        self.assertIn(MUSICIAN_URL, self.extract(slide))

    def test_external_wrapping_punctuation_is_not_part_of_recovered_url(self):
        self.assertIn(MUSICIAN_URL, self.extract(self.slide("(" + MUSICIAN_URL + ").")))
        nested = "https://en.wikipedia.org/wiki/Example_(topic_(year))"
        self.assertIn(nested, self.extract(self.slide(nested + ";")))

    def test_unbalanced_and_non_wikipedia_candidates_get_no_extra_links(self):
        for text in ("https://en.wikipedia.org/wiki/Taj_Mahal_(musician",
                     "https://en.wikipedia.org/wiki/Taj_)Mahal_(musician)",
                     "https://en.wikipedia.org.evil.example/wiki/Taj_(musician)",
                     "https://user@en.wikipedia.org/wiki/Taj_(musician)",
                     "https://en.wikipedia.org/notwiki/Taj_(musician)",
                     "https://m.wikipedia.org/wiki/Taj_(musician)",
                     "https://example.org/?next=https://en.wikipedia.org/wiki/Taj_(musician)",
                     "https://en.wikipedia.org/wiki/Brian_May"):
            with self.subTest(text=text):
                slide = self.slide(text)
                self.assertEqual(set(self.extract(slide)), set(self.ns["extract_slide_links"](slide)))

    def test_only_text_run_fields_are_scanned(self):
        slide = {"pageElements": [{"description": MUSICIAN_URL,
                                    "image": {"sourceUrl": MUSICIAN_URL}}]}
        self.assertEqual(self.extract(slide), [])

    def test_actual_five_url_checkpoints_keep_wrong_steve_smith_failed(self):
        scoring = runpy.run_path(str(EVAL / "eval_utils/scoring.py"))
        csv = {line.split(",", 1)[0].strip().lower(): line.split(",", 1)[1].strip()
               for line in (FAMILY / "instance_2/data/gold_wikis.csv").read_text().splitlines()
               if line.strip()}
        clients = [key for key in csv if key != "madonna"]
        clients[-1] = "musician taj mahal"
        submitted = [csv[self.ns["resolve_client_reference_name"](client, csv)] for client in clients]
        submitted[clients.index("steve smith")] = "https://en.wikipedia.org/wiki/Steve_Smith_(musician)"
        last = self.slide("\n".join(submitted))
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                ns = {**self.ns, "time": time, "Checkpoint": scoring["Checkpoint"],
                      "StepCategory": scoring["StepCategory"], "model": object(),
                      "gold_clients": clients, "wiki_urls_cache": csv,
                      "photographer_city": "Springfield, MA",
                      "presentation_data": {"slides": [{}, {}, last]}}
                definitions(FAMILY / f"instance_{instance}/evaluator.py", {"grade_checkpoint_3"}, ns)
                with redirect_stdout(io.StringIO()):
                    result = ns["grade_checkpoint_3"]()
                self.assertEqual(result.steps[0].score, 8)
                self.assertEqual(result.steps[0].max_score, 9)
                self.assertEqual(result.steps[0].details,
                                 "Found 8/9 Wikipedia URLs. Missing: ['steve smith']")
                self.assertEqual(result.total, 11)


if __name__ == "__main__":
    unittest.main()
