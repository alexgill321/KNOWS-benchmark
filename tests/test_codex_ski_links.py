"""Offline ski evaluator regression: hyperlink labels are not URL targets."""
import ast
import importlib.util
import os
from pathlib import Path
import re
import sys
import time
import traceback
import unittest
from unittest.mock import Mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / 'src/browsergym/knows/eval/tasks/sheets_25_skitourplan'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


utils = load('ski_link_test_utils', FAMILY / 'utils.py')
scoring = load('ski_link_test_scoring', FAMILY.parents[1] / 'eval_utils/scoring.py')
FORECAST = 'https://utahavalanchecenter.org/forecast/salt-lake/1/15/2025'
RUN = 'https://wbsguide.com/2120.php'


def grid(rows, **offsets):
    return {'sheets': [{'data': [dict(rowData=[{'values': row} for row in rows], **offsets)]}]}


def cell(label, **fields):
    return dict(formattedValue=label, **fields)


class SkiLinkExtractionTests(unittest.TestCase):
    def test_exact_saved_forecast_formula_uses_target_and_existing_validator(self):
        label = 'Salt Lake forecast — Jan 15, 2025'
        raw = grid([[cell(label, userEnteredValue={'formulaValue': f'=HYPERLINK("{FORECAST}","{label}")'})]])
        self.assertFalse(utils.is_valid_uac_forecast_url(label))
        self.assertEqual(utils.get_sheet_link_url(raw, 0, 0), FORECAST)
        self.assertTrue(utils.is_valid_uac_forecast_url(utils.get_sheet_link_url(raw, 0, 0)))

    def test_spill_cell_metadata_and_single_rich_text_link_resolve_target(self):
        for fields in ({'hyperlink': RUN},
                       {'effectiveFormat': {'textFormat': {'link': {'uri': RUN}}}},
                       {'userEnteredFormat': {'textFormat': {'link': {'uri': RUN}}}},
                       {'textFormatRuns': [{'format': {'link': {'uri': RUN}}}]}):
            with self.subTest(fields=fields):
                self.assertEqual(utils.get_sheet_link_url(grid([[cell('Open guide', **fields)]]), 0, 0), RUN)

    def test_literal_vertical_array_maps_each_spill_not_only_first(self):
        urls = [f'https://wbsguide.com/{n}.php' for n in (2120, 2122, 2124, 2162)]
        formula = '={' + ';'.join(f'HYPERLINK("{url}","Open guide")' for url in urls) + '}'
        rows = [[cell('Open guide')] for _ in urls]
        rows[0][0]['userEnteredValue'] = {'formulaValue': formula}
        raw = grid(rows, startRow=3, startColumn=2)
        for i, url in enumerate(urls):
            self.assertEqual(utils.get_sheet_link_url(raw, 3 + i, 2), url)
        self.assertEqual(utils.get_sheet_link_url(raw, 7, 2), '')

    def test_unrelated_formula_or_different_rendered_label_is_not_guessed(self):
        cases = [f'=HYPERLINK(A1,"Open guide")', f'=IF(TRUE,HYPERLINK("{RUN}","Open guide"),"")',
                 f'=HYPERLINK("{RUN}","Different label")',
                 f'={{HYPERLINK("{RUN}","Different label")}}',
                 f'={{HYPERLINK("{RUN}","Open guide"),HYPERLINK("{FORECAST}","Forecast")}}']
        for formula in cases:
            with self.subTest(formula=formula):
                raw = grid([[cell('Open guide', userEnteredValue={'formulaValue': formula})]])
                self.assertEqual(utils.get_sheet_link_url(raw, 0, 0), 'Open guide')

    def test_failed_formula_and_failed_spill_do_not_earn_link_credit(self):
        error = {'errorValue': {'type': 'REF', 'message': 'Array result was not expanded'}}
        direct = grid([[cell('#REF!', userEnteredValue={'formulaValue': f'=HYPERLINK("{RUN}","Open guide")'},
                            effectiveValue=error)]])
        self.assertEqual(utils.get_sheet_link_url(direct, 0, 0), '')
        formula = f'={{HYPERLINK("{RUN}","Open guide");HYPERLINK("{RUN}","Open guide")}}'
        spilled = grid([[cell('#REF!', userEnteredValue={'formulaValue': formula}, effectiveValue=error)],
                        [cell('Open guide')]])
        self.assertEqual(utils.get_sheet_link_url(spilled, 1, 0), 'Open guide')

    def test_raw_urls_wrong_urls_and_missing_links_keep_original_validation(self):
        for url in (RUN, 'https://maps.google.com/not-a-guide', '', 'Open guide'):
            result = utils.get_sheet_link_url(grid([[cell(url)]]), 0, 0)
            self.assertEqual(result, url)
            self.assertEqual(utils.is_valid_wbsguide_url(result), utils.is_valid_wbsguide_url(url))
        raw = grid([[cell('Valid-looking label', hyperlink='https://example.invalid/not-a-forecast')]])
        self.assertFalse(utils.is_valid_uac_forecast_url(utils.get_sheet_link_url(raw, 0, 0)))

    def test_ambiguous_rich_text_links_fail_closed(self):
        raw = grid([[cell('Open guide', textFormatRuns=[{'format': {'link': {'uri': RUN}}},
                                                       {'format': {'link': {'uri': FORECAST}}}])]])
        self.assertEqual(utils.get_sheet_link_url(raw, 0, 0), '')
        self.assertEqual(utils.get_row_link_url(raw, ['Open guide'], 0, 0), '')

    def test_dataframe_mapping_accounts_for_nonzero_header_blanks_and_duplicate_labels(self):
        raw = grid([[cell('Notes')], [cell('Run'), cell('Link')], [],
                    [cell('Run A'), cell('Open guide', hyperlink=RUN)], [],
                    [cell('Run B'), cell('Open guide', hyperlink='https://wbsguide.com/2122.php')]])
        self.assertEqual(utils.get_row_link_url(raw, ['Run A', 'Open guide'], 1, 2), RUN)
        self.assertEqual(utils.get_row_link_url(raw, ['Run B', 'Open guide'], 1, 3), 'https://wbsguide.com/2122.php')

    def test_unmatched_formal_table_offset_never_borrows_other_rows_target(self):
        raw = grid([[cell('Other row'), cell('Open guide', hyperlink=RUN)]], startRow=8, startColumn=3)
        self.assertEqual(utils.get_row_link_url(raw, ['Our row', 'Open guide'], 1, 1), 'Open guide')

    def test_identical_rows_with_different_targets_are_ambiguous_even_at_hint(self):
        raw = grid([[cell('Run'), cell('Open guide', hyperlink=RUN)], [],
                    [cell('Run'), cell('Open guide', hyperlink='https://wbsguide.com/2122.php')]])
        self.assertEqual(utils.get_row_link_url(raw, ['Run', 'Open guide'], 1, 0), 'Open guide')


class SkiLinkCheckpointTests(unittest.TestCase):
    def namespace(self, instance):
        ns = {name: getattr(utils, name) for name in dir(utils) if not name.startswith('__')}
        rows = [cell('Run name'), cell('Run link'), cell('Forecast date'), cell('Forecast link')]
        run = [cell('Fixture run'), cell('Open guide', hyperlink=RUN), cell('01/15/2025'),
               cell('Forecast label', userEnteredValue={'formulaValue': f'=HYPERLINK("{FORECAST}","Forecast label")'})]
        raw = grid([rows, run])
        lookup = Mock(return_value={'starting_location': None, 'slope_aspect': 'N', 'slope_angle': 25,
                                   'typical_vertical': 500, 'expected_danger_rating': 'yellow'})
        ns.update(os=os, re=re, time=time, traceback=traceback, Checkpoint=scoring.Checkpoint,
                  StepCategory=scoring.StepCategory, sheet_raw=raw, detected_header_row=0,
                  df=pd.DataFrame([[v['formattedValue'] for v in run]], columns=[v['formattedValue'] for v in rows]),
                  matched_columns={'Run Name': 0, 'Run Link': 1, 'Forecast Date': 2, 'Forecast Link': 3},
                  BROWSING_HISTORY=[RUN, FORECAST], EXPECTED_RUN_COUNT=1, MAX_SLOPE_ANGLE=30,
                  FORECAST_DATE='01/15/2025', FORECAST_DATE_ALT='1/15/2025', model=None,
                  gold_data={'forecast': {'danger_rose_image_url': 'https://example.invalid/rose.png'}},
                  find_run_by_name_or_url=lookup, get_cell_background_color=lambda *a: {},
                  get_cell_value=lambda sheet, r, c: sheet['sheets'][0]['data'][0]['rowData'][r]['values'][c]['formattedValue'],
                  check_merged_cells=lambda *a: False)
        path = FAMILY / f'instance_{instance}/evaluator.py'
        names = {f'grade_checkpoint_{i}' for i in (3, 4, 5, 6)}
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        return ns

    def test_all_five_actual_checkpoints_use_resolved_links_without_changing_rubric(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance):
                ns = self.namespace(instance)
                for number in (3, 4, 5, 6):
                    checkpoint = ns[f'grade_checkpoint_{number}']()
                    steps = {step.name: step for step in checkpoint.steps}
                    if number == 3:
                        self.assertTrue(steps['Run 1 - Link Valid'].success)
                    if number == 4:
                        self.assertTrue(steps['Run Links Visited'].success)
                    if number == 5:
                        self.assertTrue(steps['Forecast Link Valid'].success)
                        self.assertFalse(steps['Danger Rose Present'].success)
                self.assertTrue(ns['find_run_by_name_or_url'].called)
                self.assertTrue(all(call.args[1] == RUN for call in ns['find_run_by_name_or_url'].call_args_list))

    def test_run_history_still_requires_actual_matching_visit(self):
        for instance in range(1, 6):
            ns = self.namespace(instance)
            ns['BROWSING_HISTORY'] = ['https://wbsguide.com/', FORECAST]
            steps = {step.name: step for step in ns['grade_checkpoint_4']().steps}
            self.assertFalse(steps['Run Links Visited'].success)


if __name__ == '__main__':
    unittest.main()
