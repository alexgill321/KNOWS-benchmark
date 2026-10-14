"""Offline regression coverage for the actual task-local KBB history selector."""
import ast
from pathlib import Path
import re
import unittest
from typing import List, Optional
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/browsergym/knows/eval/tasks/slides_29_buy_car_pres/utils.py"


def selector():
    # Compile the production pure function without loading network/model clients.
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "find_kbb_url_for_car")
    namespace = {"re": re, "List": List, "Optional": Optional,
                 "urlparse": urlparse}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name]


class KBBHistoryHostTests(unittest.TestCase):
    def setUp(self):
        self.find = selector()

    def test_actual_audi_history_search_cannot_beat_real_vehicle_page(self):
        search = ("https://html.duckduckgo.com/html/"
                  "?q=site%3Akbb.com%2Faudi%2Fa3%2F2015%2F+TDI+MSRP+horsepower+consumer+rating")
        vehicle = "https://www.kbb.com/audi/a3/2015/tdi-premium-sedan-4d/"
        review = "https://www.kbb.com/audi/a3/2015/consumer-reviews/"
        self.assertEqual(self.find([search, vehicle, review], "Audi A3 TDI"), vehicle)

    def test_non_kbb_urls_do_not_establish_a_visit(self):
        for url in (
            "https://search.example/?q=kbb.com/audi/a3",
            "https://search.example/kbb.com/audi/a3",
            "https://kbb.com.evil.example/audi/a3",
            "https://notkbb.com/audi/a3",
            "https://kbb.com@evil.example/audi/a3",
            "https://user@www.kbb.com/audi/a3",
            "https://user:pass@www.kbb.com/audi/a3",
            "ftp://www.kbb.com/audi/a3",
            "file://www.kbb.com/audi/a3",
            "//www.kbb.com/audi/a3",
            "https://[invalid/audi/a3/kbb.com",
        ):
            with self.subTest(url=url):
                self.assertIsNone(self.find([url], "Audi A3"))

    def test_http_https_root_and_real_subdomains_allowed(self):
        for url in ("http://kbb.com/audi/a3", "https://www.kbb.com/audi/a3",
                    "https://cars.kbb.com/audi/a3", "HTTPS://WWW.KBB.COM/audi/a3"):
            with self.subTest(url=url):
                self.assertEqual(self.find([url], "Audi A3"), url)

    def test_overview_preference_stays_unchanged(self):
        overview = "https://www.kbb.com/audi/a3/2015/"
        review = overview + "consumer-reviews/"
        self.assertEqual(self.find([review, overview], "Audi A3"), overview)

    def test_stronger_token_match_stays_unchanged(self):
        general = "https://www.kbb.com/audi/a3/2015/"
        trim = general + "tdi-premium-sedan-4d/"
        self.assertEqual(self.find([general, trim], "Audi A3 TDI"), trim)

    def test_equal_score_keeps_first_history_entry(self):
        first = "https://www.kbb.com/audi/a3/2015/longer-trim-path/"
        second = "https://www.kbb.com/audi/a3/2015/"
        self.assertEqual(self.find([first, second], "Audi A3"), first)
        self.assertEqual(self.find([second, first], "Audi A3"), second)

    def test_year_only_overlap_does_not_match(self):
        self.assertIsNone(self.find(["https://www.kbb.com/honda/civic/2015/"],
                                    "2015 Audi A3"))

    def test_no_history_or_name_stays_unmatched(self):
        self.assertIsNone(self.find([], "Audi A3"))
        self.assertIsNone(self.find(["https://www.kbb.com/audi/a3/"], ""))


if __name__ == "__main__":
    unittest.main()
