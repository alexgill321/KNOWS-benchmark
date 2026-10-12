"""Offline regressions for task-local running chart context; no API/judge calls."""
import ast
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'codex_benchmark').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
FAMILY = Path('src/browsergym/knows/eval/tasks/sheets_7_running_analysis')
STAGED = Path(os.environ.get('KNOWS_RUNNING_CHART_TEST_SOURCE', str(Path(__file__).parent / 'staged-source')))
if not (STAGED / FAMILY / 'chart_context.py').exists():
    STAGED = ROOT
spec = importlib.util.spec_from_file_location('running_chart_context_tested', STAGED / FAMILY / 'chart_context.py')
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)
from browsergym.knows.eval.eval_utils import chart_utils as shared
from browsergym.knows.eval.eval_utils.scoring import Checkpoint, StepCategory


def cells(values):
    return [{'values': [{'formattedValue': str(x)} for x in row]} for row in values]


def tab(sheet_id, title, values, start_row=0, start_col=0):
    return {'properties': {'sheetId': sheet_id, 'title': title,
                           'gridProperties': {'rowCount': 100, 'columnCount': 20}},
            'data': [{'startRow': start_row, 'startColumn': start_col, 'rowData': cells(values)}]}


def chart(host=7, title='Daily miles', source=7, columns=(1, 2), headers=1, start=0, end=4):
    ranges = [{'sheetId': source, 'startRowIndex': start, 'endRowIndex': end,
               'startColumnIndex': col, 'endColumnIndex': col + 1} for col in columns]
    series = [{'series': {'sourceRange': {'sources': [item]}}, 'lineStyle': {'type': 'DOTTED'},
               'color': {'red': 1 if i else 0, 'blue': 0 if i else 1}}
              for i, item in enumerate(ranges)]
    return {'chart_id': 4, 'sheet_id': host, 'sheet_name': title, 'chart_type': 'LINE',
            'title': 'Cumulative distance over time',
            'position': {'type': 'overlay', 'anchor_cell': {'row': 1, 'col': 4}, 'height': 200, 'width': 400},
            'raw_chart': {'position': {'overlayPosition': {'anchorCell': {'sheetId': host, 'rowIndex': 1, 'columnIndex': 4}}}, 'spec': {'basicChart': {'chartType': 'LINE', 'headerCount': headers, 'series': series}}}}


def bind(c=None, extra=None):
    first = tab(0, 'Activities', [['Date', 'Activity', 'Name'], ['date', 'Run', 'wrong'],
                                ['date', 'Run', 'wrong'], ['date', 'Walk', 'wrong']])
    source = tab(7, 'Daily miles', [['Date', 'Walk + run (mi)', 'Female proxy (mi)'],
                                  ['date', 2, 2.456], ['date', 4, 2.456], ['date', 8, 2.456]])
    raw = {'sheets': [first, source] + (extra or [])}
    return context.bind_chart_context([c or chart()], raw, 0)[0], raw


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {x.name for x in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


class RunningChartContextTests(unittest.TestCase):
    def test_actual_failure_shape_reads_other_tab_not_activity_text(self):
        c, raw = bind()
        wrong = raw['sheets'][0]['data'][0]['rowData']
        self.assertEqual(shared.get_series_column_values(c, 0, wrong), [])
        self.assertEqual(context.get_series_column_values(c, 0, wrong), [2, 4, 8])
        self.assertEqual(context.get_series_header_label(c, 1, wrong), 'Female proxy (mi)')
        self.assertEqual(context.get_series_column_values(c, 1, wrong), [2.456] * 3)
        self.assertFalse(context.series_source_is_table(c, 0))

    def test_source_tab_can_differ_from_both_host_and_table(self):
        c, raw = bind(chart(source=9), [tab(9, 'Remote', [['Date', 'Pace (min/mile)', 'Baseline'],
                                                        ['date', '8:30', '9:00'], ['date', '7:15', '9:00']])])
        self.assertEqual(context.get_series_column_values(c, 0, []), [8.5, 7.25])
        self.assertEqual(context.get_series_header_label(c, 0, []), 'Pace (min/mile)')
        self.assertEqual(context.chart_host_rows(c), raw['sheets'][1]['data'][0]['rowData'])

    def test_absolute_grid_offsets_and_header_count_zero_one_two(self):
        for headers in (0, 1, 2):
            with self.subTest(headers=headers):
                rows = [['header', 'header']] * headers + [['date', '$2,500'], ['date', '8:30']]
                raw = {'sheets': [tab(0, 'Activities', []), tab(7, 'Daily miles', rows, 5, 3)]}
                c = chart(columns=(4,), start=5, end=7 + headers, headers=headers)
                c = context.bind_chart_context([c], raw, 0)[0]
                self.assertEqual(context.get_series_column_values(c, 0, []), [2500, 8.5])
                self.assertEqual(context.get_series_header_label(c, 0, []), 'header' if headers else '')

    def test_absent_zero_id_and_bounds_are_bound_to_real_metadata_zero(self):
        c = chart(host=7, source=0, columns=(1,))
        source = c['raw_chart']['spec']['basicChart']['series'][0]['series']['sourceRange']['sources'][0]
        for field in ('sheetId', 'startRowIndex'):
            source.pop(field)
        raw = {'sheets': [tab(7, 'Daily miles', []), tab(0, 'Source zero', [['h', 'v'], ['x', 2]])]}
        before = deepcopy(raw), deepcopy(c)
        result = context.bind_chart_context([c], raw, 0)[0]
        self.assertEqual(context.get_series_column_values(result, 0, []), [2])
        self.assertEqual((raw, c), before)
        raw['sheets'].pop()
        with self.assertRaises(context.ChartContextError):context.bind_chart_context([c], raw, 7)

    def test_null_bool_unknown_multiple_or_out_of_bounds_sources_fail_before_identification(self):
        for replacement in ({'sheetId': None}, {'sheetId': True}, {'sheetId': 99},
                            {'startRowIndex': -1}, {'endRowIndex': 101}, {'startColumnIndex': 21},
                            {'endColumnIndex': 4}):
            with self.subTest(replacement=replacement):
                c = chart();c['raw_chart']['spec']['basicChart']['series'][0]['series']['sourceRange']['sources'][0].update(replacement)
                with self.assertRaises(context.ChartContextError):bind(c)
        c = chart();sources = c['raw_chart']['spec']['basicChart']['series'][0]['series']['sourceRange']['sources'];sources.append(deepcopy(sources[0]))
        with self.assertRaises(context.ChartContextError):bind(c)

    def test_ambiguous_host_or_grid_offsets_fail_closed(self):
        c, raw = bind()
        for change in ('title', 'duplicate', 'overlap', 'null_id', 'missing_zero_id'):
            test = deepcopy(raw)
            if change == 'title':test['sheets'][1]['properties']['title'] = 'Wrong'
            elif change == 'duplicate':test['sheets'].append(deepcopy(test['sheets'][1]))
            elif change == 'null_id':test['sheets'][1]['properties']['sheetId'] = None
            elif change == 'missing_zero_id':test['sheets'][0]['properties'].pop('sheetId')
            else:test['sheets'][1]['data'].append(deepcopy(test['sheets'][1]['data'][0]))
            with self.subTest(change=change), self.assertRaises(context.ChartContextError):context.bind_chart_context([chart()], test, 0)

    def test_existing_identification_code_and_globals_are_not_changed(self):
        c, raw = bind();original = shared.identify_series_by_content.__globals__['get_series_column_values']
        self.assertIs(context._identify.__code__, shared.identify_series_by_content.__code__)
        self.assertEqual(context.identify_series_by_content(c, [], ['female'], (2, 3), True), 1)
        self.assertEqual(context.identify_series_by_content(c, [], [], exclude_indices=[1]), 0)
        self.assertIs(shared.identify_series_by_content.__globals__['get_series_column_values'], original)
        # A foreign tab's coincident column number cannot win CP2 column fallback.
        self.assertIsNone(context.identify_series_by_content(c, [], ['absent'],
            matched_columns={'Speed': 'Coincidence'}, column_name='Speed', df=pd.DataFrame(columns=['Date', 'Coincidence'])))

    def test_same_sheet_overlap_only_and_source_data_is_not_first_table(self):
        c, raw = bind();other = deepcopy(c);other['chart_id'] = 9;other['sheet_id'] = 0
        self.assertTrue(shared.check_chart_overlap(c, 0, 400, [other], 0, 7)[0])
        self.assertFalse(context.check_chart_overlap(c, 0, 400, [other], 0, 7)[0])
        other['sheet_id'] = 7
        self.assertTrue(context.check_chart_overlap(c, 0, 400, [other], 0, 7)[0])
        c['_running_context']['table_sheet_id'] = 7
        self.assertTrue(context.check_chart_overlap(c, 0, 400, [], 0, 7)[0])

    def test_all_five_actual_setup_functions_bind_chart_metadata(self):
        for i in range(1, 6):
            with self.subTest(instance=i):
                c, raw = bind();data = SimpleNamespace(df=pd.DataFrame({'Date': ['date']}), start_row=0, end_row=2, start_col=0, end_col=1)
                env = {'extract_sheet_data': Mock(return_value=(data, raw)), 'extract_charts_from_sheet': Mock(return_value=[chart()]),
                       'SHEETS_SERVICE': Mock(), 'bind_chart_context': context.bind_chart_context}
                definitions(STAGED / FAMILY / f'instance_{i}/evaluator.py', {'setup'}, env)
                env['setup']('saved_workspace')
                self.assertEqual(context.get_series_column_values(env['chart_data'][0], 0, env['rows']), [2, 4, 8])

    def test_all_five_evaluation_bridges_preserve_infrastructure_error(self):
        for i in range(1, 6):
            with self.subTest(instance=i):
                env = {'time': time, 'setup': Mock(side_effect=context.ChartContextError('Unknown source')),
                       'ChartContextError': context.ChartContextError}
                definitions(STAGED / FAMILY / f'instance_{i}/evaluator.py', {'grade_checkpoints'}, env)
                with self.assertRaises(context.ChartContextError):env['grade_checkpoints']('saved_workspace')

    def test_actual_cumulative_checkpoint_reads_numeric_foreign_tab_and_preserves_axis_failures(self):
        c, raw = bind(chart(columns=(1,)));c['title'] = 'Cumulative running distance (miles)'
        env = {'time': time, 'pd': pd, 'Checkpoint': Checkpoint, 'StepCategory': StepCategory,
               'chart_data': [c], 'rows': raw['sheets'][0]['data'][0]['rowData'], 'table_sheet_id': 0,
               'table_data': SimpleNamespace(df=pd.DataFrame({'Date': ['date'], 'Activity': ['Run']}),
                                            start_row=0, end_row=453, start_col=0, end_col=7),
               'matched_columns': {}, 'model': None, 'find_cumulative_chart_by_metadata': Mock(return_value=c),
               'build_keyword_match_prompt': lambda *a: 'unchanged prompt mocked',
               'chart_host_rows': context.chart_host_rows, 'get_chart_type': shared.get_chart_type,
               'get_chart_axis_labels': shared.get_chart_axis_labels,
               'fast_parallel_vlm_calls': lambda tasks,*a,**k: {v['id']: True for v in tasks},
               'keywords_match_robust': lambda texts,keywords,**kw: bool(texts) and any(x.lower() in str(texts).lower() for x in keywords),
               'check_chart_overlap': context.check_chart_overlap,
               'get_series_column_values': context.get_series_column_values}
        definitions(STAGED / FAMILY / 'instance_4/evaluator.py', {'grade_checkpoint_5'}, env)
        result = env['grade_checkpoint_5']()
        steps = {v.name: v for v in result.steps}
        self.assertTrue(steps['Cumulative Data'].success)
        self.assertTrue(steps['No Chart Overlap'].success)
        self.assertFalse(steps['X-Axis Date Label'].success)
        self.assertFalse(steps['Y-Axis Distance Label'].success)

    def test_host_rows_reach_existing_url_search_and_all_eleven_plot_checkpoints(self):
        total = 0
        for i in range(1, 6):
            tree = ast.parse((STAGED / FAMILY / f'instance_{i}/evaluator.py').read_text())
            for function in tree.body:
                if not isinstance(function, ast.FunctionDef):continue
                calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
                if any(n.func.id == 'chart_host_rows' for n in calls):
                    total += 1
                    self.assertTrue(any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'rows' for t in n.targets)
                                        and isinstance(n.value, ast.Call) and getattr(n.value.func, 'id', '') == 'chart_host_rows' for n in function.body))
        self.assertEqual(total, 11)
        c, raw = bind();raw['sheets'][1]['data'][0]['rowData'].append({'values': [{'formattedValue': 'https://example.com/baseline'}]})
        c = context.bind_chart_context([chart()], raw, 0)[0]
        self.assertIn('https://example.com/baseline', str(context.chart_host_rows(c)))
        self.assertNotIn('https://example.com/baseline', str(raw['sheets'][0]['data'][0]['rowData']))


if __name__ == '__main__':
    unittest.main()
