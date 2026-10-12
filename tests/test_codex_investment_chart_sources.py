"""Offline investment chart range regressions, with no Google or judge calls."""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from typing import List, Optional, Union
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from codex_benchmark.evaluation_log import EvaluationJournal
from codex_benchmark.knows import TASKS_RELATIVE, _strict_judges, discover_tasks, evaluate_task


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / TASKS_RELATIVE / "sheets_6_investmenttracker"
SHARED = ROOT / "src/browsergym/knows/eval/eval_utils"


def definitions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


ns = dict(pd=pd, re=re, Union=Union, Optional=Optional)
definitions(SHARED / "chart_utils.py", {"_parse_numeric_cell"}, ns)
nodes = [n for n in ast.parse((FAMILY / "chart_sources.py").read_text()).body if not isinstance(n, ast.ImportFrom)]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FAMILY / "chart_sources.py"), "exec"), ns)
helpers = SimpleNamespace(**ns)


def scoring_module():
    spec = importlib.util.spec_from_file_location("investment_chart_scoring", SHARED / "scoring.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    return module


def chart(kind="BAR", source_sheet=7, start=15, end=26, headers=1):
    domain = {"sheetId": source_sheet, "startRowIndex": start, "endRowIndex": end,
              "startColumnIndex": 0, "endColumnIndex": 1}
    series = {**domain, "startColumnIndex": 1, "endColumnIndex": 2}
    domain_data, series_data = ({"sourceRange": {"sources": [r]}} for r in (domain, series))
    spec = ({"pieChart": {"domain": domain_data, "series": series_data}}
            if kind == "PIE" else {"basicChart": {"chartType": kind, "headerCount": headers,
                "domains": [{"domain": domain_data}], "series": [{"series": series_data}]}})
    return {"chart_type": kind, "sheet_id": 99, "sheet_name": "Chart host is not source",
            "raw_chart": {"spec": spec}}


def properties(sheet_id=7, title="Tech Watchlist"):
    return {"properties": {"sheetId": sheet_id, "title": title,
                           "gridProperties": {"rowCount": 100, "columnCount": 20}}}


def service(values=None, sheets=None, title="Tech Watchlist"):
    client = Mock()
    client.spreadsheets.return_value.get.return_value.execute.return_value = {"sheets": sheets or [properties(title=title)]}
    quoted = "'" + title.replace("'", "''") + "'"
    client.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = {
        "range": quoted + "!A1:T100", "majorDimension": "ROWS", "values": values if values is not None else []}
    return client


class InvestmentChartSourceTests(unittest.TestCase):
    def test_below_first_table_uses_absolute_chart_cells_with_relative_header(self):
        labels = [f"STOCK{i}" for i in range(10)]
        rows = [["unrelated", "$999"] for _ in range(15)] + [["Ticker", "Weight"]]
        rows += [[label, f"{i + 1}%"] for i, label in enumerate(labels)]
        rows += [["after chart", "987%"]]
        c = chart()
        sources = helpers.fetch_chart_source_data("saved_sheet", service(rows), [c])
        first_df = pd.DataFrame({"Ticker": ["WRONG"] * 10, "Value": [999] * 10})
        self.assertEqual(helpers.extract_chart_domain_data(c, first_df, sources), labels)
        self.assertEqual(helpers.extract_chart_series_data(c, first_df, sources), list(map(float, range(1, 11))))

        # The prior shared extractor still reproduces the observed defect even
        # with the correct full-sheet rows: fitting column bounds choose df.
        prior = dict(ns, Dict=dict, Any=object, List=list)
        definitions(SHARED / "chart_utils.py", {"extract_chart_domain_data"}, prior)
        c["data_range"] = {"domain_range": {"start_row": None}}
        self.assertEqual(prior["extract_chart_domain_data"](c, first_df, rows), [])

    def test_renamed_apostrophe_tab_is_quoted_and_source_id_beats_chart_host(self):
        client = service([["label", "weight"], ["AAPL", "25%"]], title="Owner's Watchlist")
        c = chart(start=0, end=2)
        sources = helpers.fetch_chart_source_data("saved_sheet", client, [c])
        client.spreadsheets.return_value.values.return_value.get.assert_called_once_with(
            spreadsheetId="saved_sheet", range="'Owner''s Watchlist'", majorDimension="ROWS",
            valueRenderOption="FORMATTED_VALUE")
        self.assertEqual(set(sources), {7})
        self.assertEqual(helpers.extract_chart_domain_data(c, None, sources), ["AAPL"])

    def test_pie_line_zero_and_multiple_headers_and_nonzero_columns(self):
        for kind in ("BAR", "COLUMN", "LINE", "PIE"):
            for count in ((1,) if kind == "PIE" else (0, 1, 2)):
                with self.subTest(kind=kind, headers=count):
                    c = chart(kind, start=3, end=6 + count, headers=count)
                    cfg = c["raw_chart"]["spec"]["pieChart" if kind == "PIE" else "basicChart"]
                    domain = cfg["domain"] if kind == "PIE" else cfg["domains"][0]["domain"]
                    series = cfg["series"] if kind == "PIE" else cfg["series"][0]["series"]
                    domain["sourceRange"]["sources"][0].update(startColumnIndex=3, endColumnIndex=4)
                    series["sourceRange"]["sources"][0].update(startColumnIndex=4, endColumnIndex=5)
                    rows = [["ignored"]] * 3 + [["header"]] * count
                    rows += [["wrong", "wrong", "wrong", name, value] for name, value in
                             [("A", "$1,200.50"), ("B", "($5.00)"), ("C", "0%")]]
                    sources = helpers.fetch_chart_source_data("saved_sheet", service(rows), [c])
                    self.assertEqual(helpers.extract_chart_domain_data(c, None, sources), ["A", "B", "C"])
                    self.assertEqual(helpers.extract_chart_series_data(c, None, sources), [1200.5, -5.0, 0.0])

    def test_distinct_domain_and_series_source_tabs_are_not_conflated(self):
        c = chart(start=0, end=3)
        c["raw_chart"]["spec"]["basicChart"]["series"][0]["series"]["sourceRange"]["sources"][0]["sheetId"] = 8
        client = service(sheets=[properties(7, "Names"), properties(8, "Values"),
                                 {"properties": {"sheetId": 99, "title": "Chart object"}}])
        client.spreadsheets.return_value.values.return_value.get.return_value.execute.side_effect = [
            {"range": "Names!A1:T100", "values": [["Stock"], ["A"], ["B"]]},
            {"range": "Values!A1:T100", "values": [["unused", "Weight"], ["unused", "12%"], ["unused", "88%"]]}]
        sources = helpers.fetch_chart_source_data("saved_sheet", client, [c])
        self.assertEqual(helpers.extract_chart_domain_data(c, None, sources), ["A", "B"])
        self.assertEqual(helpers.extract_chart_series_data(c, None, sources), [12.0, 88.0])

    def test_no_expected_chart_preserves_no_chart_path_without_fetch(self):
        client = Mock()
        self.assertEqual(helpers.fetch_chart_source_data("saved_sheet", client, [], chart_type="BAR"), {})
        self.assertEqual(helpers.fetch_chart_source_data("saved_sheet", client, [chart("PIE")], chart_type="BAR"), {})
        client.assert_not_called()
        client.spreadsheets.assert_not_called()
        # A malformed later chart must not replace the first chart selected by the rubric.
        first = chart(start=0, end=2)
        sources = helpers.fetch_chart_source_data("saved_sheet", service([["h", "v"], ["A", "1"]]),
                                                  [first, {"chart_type": "BAR"}], chart_type="BAR")
        self.assertEqual(helpers.extract_chart_domain_data(first, None, sources), ["A"])

    def test_header_only_or_empty_actual_cells_are_not_invented_from_dataframe(self):
        c = chart(start=5, end=6)
        sources = helpers.fetch_chart_source_data("saved_sheet", service([["old", "123"]] * 6), [c])
        df = pd.DataFrame({"A": ["made up"], "B": [999]})
        self.assertEqual(helpers.extract_chart_domain_data(c, df, sources), [])
        self.assertEqual(helpers.extract_chart_series_data(c, df, sources), [])
        sources[7]["values"] = []
        self.assertEqual(helpers.extract_chart_domain_data(c, df, sources), [])

    def test_ambiguous_unknown_missing_or_out_of_grid_sources_fail_closed(self):
        base = chart()
        for replacement in ({}, {"sheetId": 999}, {"sheetId": True},
                            {"sheetId": 7, "startRowIndex": -1},
                            {"sheetId": 7, "endRowIndex": 101},
                            {"sheetId": 7, "startColumnIndex": 20, "endColumnIndex": 21}):
            with self.subTest(replacement=replacement):
                c = deepcopy(base)
                c["raw_chart"]["spec"]["basicChart"]["domains"][0]["domain"]["sourceRange"]["sources"] = [replacement]
                with self.assertRaises(helpers.ChartSourceDataError):
                    helpers.fetch_chart_source_data("saved_sheet", service(), [c])
        c = deepcopy(base)
        sources = c["raw_chart"]["spec"]["basicChart"]["domains"][0]["domain"]["sourceRange"]["sources"]
        sources.append(deepcopy(sources[0]))
        with self.assertRaises(helpers.ChartSourceDataError):
            helpers.fetch_chart_source_data("saved_sheet", service(), [c])
        with self.assertRaises(helpers.ChartSourceDataError):
            helpers.fetch_chart_source_data("saved_sheet", service(sheets=[properties(), properties()]), [base])

    def test_automatic_basic_headers_fail_closed_instead_of_guessing(self):
        for value in (None, -1, True, "1"):
            with self.subTest(value=value):
                c = chart(headers=value)
                if value is None:
                    c["raw_chart"]["spec"]["basicChart"].pop("headerCount")
                with self.assertRaises(helpers.ChartSourceDataError):
                    helpers.fetch_chart_source_data("saved_sheet", service(), [c])

    def test_omitted_zero_source_id_uses_verified_zero_tab_and_keeps_raw_metadata(self):
        for kind in ("BAR", "PIE"):
            with self.subTest(kind=kind):
                c = chart(kind, source_sheet=0, start=0, end=3)
                config = next(iter(c["raw_chart"]["spec"].values()))
                for role in ("domain", "series"):
                    data = config[role] if kind == "PIE" else config["domains" if role == "domain" else "series"][0][role]
                    data["sourceRange"]["sources"][0].pop("sheetId")
                before = deepcopy(c)
                client = service([["h", "v"], ["A", "20%"], ["B", "80%"]],
                                 sheets=[properties(99, "Wrong first and host"), properties(0, "Sheet1")], title="Sheet1")
                sources = helpers.fetch_chart_source_data("saved_sheet", client, [c])
                self.assertEqual(set(sources), {0})
                self.assertEqual(helpers.extract_chart_domain_data(c, None, sources), ["A", "B"])
                self.assertEqual(helpers.extract_chart_series_data(c, None, sources), [20.0, 80.0])
                self.assertEqual(c, before)
                client.spreadsheets.return_value.values.return_value.get.assert_called_once_with(
                    spreadsheetId="saved_sheet", range="'Sheet1'", majorDimension="ROWS", valueRenderOption="FORMATTED_VALUE")
                with self.assertRaises(helpers.ChartSourceDataError):
                    helpers.fetch_chart_source_data("saved_sheet", service(sheets=[properties(99, "Host only")]), [c])

    def test_zero_normalization_never_accepts_explicit_null_bool_string_or_unknown_id(self):
        for value in (None, False, True, "0", -1, 999):
            with self.subTest(value=value):
                c = chart(source_sheet=value)
                with self.assertRaises(helpers.ChartSourceDataError):
                    helpers.fetch_chart_source_data("saved_sheet", service(sheets=[properties(0, "Sheet1")]), [c])

    def test_fetch_error_wrong_tab_shifted_range_or_malformed_values_fail_closed(self):
        for result in ({"range": "Other!A1:T100", "values": []},
                       {"range": "'Tech Watchlist'!A16:B26", "values": []},
                       {"range": "'Tech Watchlist'!A1:T100", "values": "wrong"},
                       {"range": "'Tech Watchlist'!A1:T100", "values": [None]},
                       {"range": "'Tech Watchlist'!A1:T100", "majorDimension": "COLUMNS"}):
            with self.subTest(result=result):
                client = service()
                client.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = result
                with self.assertRaises(helpers.ChartSourceDataError):
                    helpers.fetch_chart_source_data("saved_sheet", client, [chart()])
        for endpoint in ("metadata", "values"):
            client = service()
            request = (client.spreadsheets.return_value.get if endpoint == "metadata"
                       else client.spreadsheets.return_value.values.return_value.get)
            request.return_value.execute.side_effect = RuntimeError("unavailable")
            with self.assertRaises(helpers.ChartSourceDataError):
                helpers.fetch_chart_source_data("saved_sheet", client, [chart()])

    def test_all_five_actual_setups_fetch_chart_selected_by_existing_rubric(self):
        for instance, kind in ((1, "BAR"), (2, "PIE"), (3, "BAR"), (4, "LINE"), (5, "PIE")):
            with self.subTest(instance=instance):
                c = chart(kind, start=0, end=2)
                env = dict(pd=pd, SHEETS_SERVICE=service([["h", "v"], ["A", "2"]]),
                           GOLD_LABELS_SHEET_ID="gold", CHART_TYPE=kind,
                           extract_tables_from_sheet=Mock(return_value=[]),
                           extract_charts_from_sheet=Mock(return_value=[c]),
                           fetch_chart_source_data=helpers.fetch_chart_source_data)
                definitions(FAMILY / f"instance_{instance}/evaluator.py", {"setup"}, env)
                env["setup"]("saved_sheet")
                self.assertEqual(helpers.extract_chart_domain_data(c, None, env["full_sheet_data"]), ["A"])

    def checkpoint_namespace(self, instance):
        scoring = scoring_module()
        kind = {1: "BAR", 2: "PIE", 3: "BAR", 4: "LINE", 5: "PIE"}[instance]
        c = chart(kind)
        labels = [f"STOCK{i}" for i in range(10)]
        rows = [["not chart", "999"]] * 15 + [["Ticker", "Weight"]]
        rows += [[name, f"{i + 1}%"] for i, name in enumerate(labels)]
        sources = helpers.fetch_chart_source_data("saved_sheet", service(rows), [c])
        env = dict(pd=pd, time=time, DEBUG=False, CHART_TYPE=kind, NUM_STOCKS=10, NUM_SHARES=100,
                   Checkpoint=scoring.Checkpoint, StepCategory=scoring.StepCategory,
                   ChartSourceDataError=helpers.ChartSourceDataError,
                   extract_chart_domain_data=helpers.extract_chart_domain_data,
                   extract_chart_series_data=helpers.extract_chart_series_data,
                   validate_chart_categories_match=Mock(return_value=(10, 10, [])),
                   validate_chart_values_match=Mock(return_value=(10, 10, [])),
                   calculate_expected_stock_values=Mock(return_value=list(range(1, 11))),
                   chart_data=[c], full_sheet_data=sources,
                   df=pd.DataFrame({"Ticker": labels, "Value": [999] * 10}),
                   matched_columns={"Ticker Symbol": "Ticker"},
                   gold_to_user_ticker_map={name: name for name in labels})
        definitions(FAMILY / f"instance_{instance}/evaluator.py", {"grade_checkpoint_4"}, env)
        return env, labels

    def test_all_five_chart_checkpoints_pass_exact_source_cells_to_unchanged_validators(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                env, labels = self.checkpoint_namespace(instance)
                env["grade_checkpoint_4"]()
                self.assertEqual(env["validate_chart_categories_match"].call_args.args[0], labels)
                self.assertEqual(env["validate_chart_values_match"].call_args_list[0].args[0], list(map(float, range(1, 11))))

    def test_all_five_chart_checkpoint_catches_rethrow_unverifiable_domain_or_series(self):
        for instance in range(1, 6):
            for role in ("domain", "series"):
                with self.subTest(instance=instance, role=role):
                    env, _ = self.checkpoint_namespace(instance)
                    c = env["chart_data"][0]
                    config = next(iter(c["raw_chart"]["spec"].values()))
                    data = (config[role] if "pieChart" in c["raw_chart"]["spec"] else
                            config["domains" if role == "domain" else "series"][0][role])
                    data["sourceRange"]["sources"][0]["sheetId"] = 999
                    with self.assertRaises(helpers.ChartSourceDataError):
                        env["grade_checkpoint_4"]()

    def test_all_five_actual_entrypoints_persist_null_on_source_fetch_failure(self):
        scoring = scoring_module()
        for instance in range(1, 6):
            with self.subTest(instance=instance), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                fixture = ModuleType("_investment_chart_guard_fixture")
                fixture.ChartSourceDataError = helpers.ChartSourceDataError
                client = service()
                client.spreadsheets.return_value.get.return_value.execute.side_effect = RuntimeError("source unavailable")
                fixture.setup = lambda workspace_doc_id: helpers.fetch_chart_source_data(workspace_doc_id, client, [chart()])
                fixture.Checkpoint, fixture.Result, fixture.StepCategory = scoring.Checkpoint, scoring.Result, scoring.StepCategory
                fixture.time, fixture.List = time, List
                source_path = FAMILY / f"instance_{instance}/evaluator.py"
                source = source_path.read_text()
                node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "grade_checkpoints")
                task_dir = root / TASKS_RELATIVE / "sheets_chart_guard_fixture" / "instance_1"
                task_dir.mkdir(parents=True)
                (task_dir / "task.md").write_text("Use the saved Sheet.\n")
                (task_dir / "evaluator.py").write_text("from _investment_chart_guard_fixture import *\n" + ast.get_source_segment(source, node) + "\n")
                task = discover_tasks(root)[0]
                models = SimpleNamespace(_models={}, gemma_google_ai=Mock())
                output = root / "evaluation.json"
                with patch.dict(sys.modules, {"_investment_chart_guard_fixture": fixture}), \
                        patch("codex_benchmark.knows._prepare_credentials"), \
                        patch("codex_benchmark.knows._strict_judges", lambda override=None: _strict_judges(override, models)), \
                        patch("codex_benchmark.knows.EvaluationJournal", lambda *args: EvaluationJournal(*args, scoring_module=scoring)), \
                        patch.dict(os.environ, {}, clear=True):
                    report = evaluate_task(task, "saved_sheet", [], output)
                self.assertEqual(report["status"], "evaluation_error")
                self.assertIsNone(report["reward"])
                self.assertIn("ChartSourceDataError", report["error"])
                self.assertIsNone(json.loads(output.read_text())["reward"])
                models.gemma_google_ai.assert_not_called()


if __name__ == "__main__":
    unittest.main()
