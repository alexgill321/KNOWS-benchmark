"""Offline coverage for required travel evaluator row extraction failures."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from codex_benchmark.evaluation_log import EvaluationJournal
from codex_benchmark.knows import TASKS_RELATIVE, _strict_judges, discover_tasks, evaluate_task


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / TASKS_RELATIVE / 'sheets_28_personal_travel_planner'
TREE = ast.parse((FAMILY / 'utils.py').read_text())
CONSTANTS = {'_PARSED_ROW_KEYS'}
NAMES = {'StructuredRowExtractionError', 'build_task_context'}
NODES = [n for n in TREE.body if
         (isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in NAMES) or
         (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in CONSTANTS for t in n.targets))]


_NORMAL_MATCH = object()


def namespace(parsed_rows, match_result=_NORMAL_MATCH):
    # AST loading deliberately excludes credentials, imports with service
    # initialization, and every network/model implementation.
    ns = dict(os=SimpleNamespace(path=os.path, environ={}), pd=pd, Dict=Dict, List=List, Optional=Optional)
    exec(compile(ast.Module(body=NODES, type_ignores=[]), str(FAMILY / 'utils.py'), 'exec'), ns)
    phase1 = {'match': ({'Destination': 'Destination', 'Time of Day': 'Time'},
                        {0: 'Fixture Museum', 1: 'Fixture Restaurant'}, {}), 'gold': {}}
    if match_result is not _NORMAL_MATCH:
        phase1['match'] = match_result
    phase2 = {'rows': parsed_rows, 'types': ['activity', 'food'], 'alt_types': [], 'urls': {}}
    ns.update(parallel_execute=Mock(side_effect=[phase1, phase2]),
              match_and_extract=Mock(), parse_gold_file=Mock(), extract_review_urls=Mock(return_value={}),
              extract_row_structured_data=Mock(), classify_destinations_activity_food=Mock(),
              fetch_review_url_contents=Mock(), classify_time_of_day=lambda value: value.lower())
    return ns


def context(ns):
    df = pd.DataFrame({'Destination': ['Fixture Museum', 'Fixture Restaurant'], 'Time': ['Morning', 'Lunch']})
    return ns['build_task_context'](df, {}, [], 0, '/offline-task', object())


class TravelRowExtractionTests(unittest.TestCase):
    def test_missing_mandatory_match_stops_before_row_parsing(self):
        for matched in (None, {}, (), ({}, {}), ({}, None, {}), 'bad response'):
            with self.subTest(matched=matched):
                ns = namespace(None, match_result=matched)
                with self.assertRaisesRegex(ns['StructuredRowExtractionError'], 'matching is unavailable'):
                    context(ns)
                self.assertEqual(ns['parallel_execute'].call_count, 1)
                ns['extract_review_urls'].assert_not_called()

    def test_valid_empty_match_is_output_dependent_and_keeps_existing_behavior(self):
        ns = namespace(None)
        rows = [{key: None for key in ns['_PARSED_ROW_KEYS']} for _ in range(2)]
        ns = namespace(rows, match_result=({}, {}, {}))
        result = context(ns)
        self.assertEqual(result['matched_columns'], {})
        self.assertEqual(result['dest_names'], {})
        self.assertEqual(len(result['row_data']), 2)
        self.assertEqual(ns['parallel_execute'].call_count, 2)

    def test_timeout_missing_partial_and_malformed_rows_fail_before_scoring(self):
        for rows in (None, [], [{}], [{}, {}], [None, None], {}, 'bad response'):
            with self.subTest(rows=rows):
                ns = namespace(rows)
                with self.assertRaisesRegex(ns['StructuredRowExtractionError'], 'evaluation cannot be scored'):
                    context(ns)
                self.assertEqual(ns['parallel_execute'].call_count, 2)

    def test_full_length_rows_missing_duration_key_fail_explicitly(self):
        ns = namespace(None)
        rows = [{key: None for key in ns['_PARSED_ROW_KEYS']} for _ in range(2)]
        del rows[1]['dur_min']
        ns = namespace(rows)
        with self.assertRaises(ns['StructuredRowExtractionError']):
            context(ns)

    def test_complete_rows_preserve_explicit_nulls_numeric_values_and_types(self):
        ns = namespace(None)
        rows = [{key: None for key in ns['_PARSED_ROW_KEYS']} for _ in range(2)]
        rows[0].update(dur_min=90, travel_min=15, cost_amount=20, transport_mode='walking')
        ns = namespace(rows)
        result = context(ns)
        for i, row in enumerate(result['row_data']):
            for key in ns['_PARSED_ROW_KEYS']:
                self.assertEqual(row[key], rows[i][key])
        self.assertEqual([r['row_type'] for r in result['row_data']], ['activity', 'food'])
        self.assertIsNone(result['row_data'][1]['dur_min'])
        self.assertEqual(ns['parallel_execute'].call_count, 2)

    def test_saved_output_retry_uses_bounded_120_and_180_second_phase_budgets(self):
        failed = namespace(None)
        with self.assertRaises(failed['StructuredRowExtractionError']):
            context(failed)
        rows = [{key: None for key in failed['_PARSED_ROW_KEYS']} for _ in range(2)]
        rows[0]['dur_min'], rows[1]['dur_min'] = 60, 45
        recovered = namespace(rows)
        self.assertEqual([r['dur_min'] for r in context(recovered)['row_data']], [60, 45])
        for ns in (failed, recovered):
            self.assertEqual([call.kwargs['timeout'] for call in ns['parallel_execute'].call_args_list], [120, 180])
            self.assertEqual([call.kwargs['max_workers'] for call in ns['parallel_execute'].call_args_list], [2, 4])

    def test_all_five_real_entrypoints_produce_null_bridge_reward_not_model_zero(self):
        scoring_path = ROOT / 'src/browsergym/knows/eval/eval_utils/scoring.py'
        spec = importlib.util.spec_from_file_location('travel_row_test_scoring', scoring_path)
        scoring = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'travel_row_test_scoring': scoring}):
            spec.loader.exec_module(scoring)
        for instance, phase in [(i, phase) for i in range(1, 6) for phase in ('rows', 'match')]:
            with self.subTest(instance=instance, phase=phase), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                ns = namespace(None, match_result=None) if phase == 'match' else namespace(None)
                fixture = ModuleType('_travel_row_guard_fixture')
                fixture.StructuredRowExtractionError = ns['StructuredRowExtractionError']
                fixture.setup = lambda workspace_doc_id: context(ns)
                fixture.Checkpoint, fixture.Result, fixture.StepCategory = scoring.Checkpoint, scoring.Result, scoring.StepCategory
                fixture.time, fixture.List = time, List
                source_path = FAMILY / f'instance_{instance}/evaluator.py'
                source = source_path.read_text()
                node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'grade_checkpoints')
                evaluator_source = 'from _travel_row_guard_fixture import *\n' + ast.get_source_segment(source, node) + '\n'
                task_dir = root / TASKS_RELATIVE / 'sheets_row_guard_fixture' / 'instance_1'
                task_dir.mkdir(parents=True)
                (task_dir / 'task.md').write_text('Use the saved Sheet.\n')
                (task_dir / 'evaluator.py').write_text(evaluator_source)
                task = discover_tasks(root)[0]
                models = SimpleNamespace(_models={}, gemma_google_ai=Mock())
                output = root / 'evaluation.json'
                with patch.dict(sys.modules, {'_travel_row_guard_fixture': fixture}), \
                        patch('codex_benchmark.knows._prepare_credentials'), \
                        patch('codex_benchmark.knows._strict_judges', lambda override=None: _strict_judges(override, models)), \
                        patch('codex_benchmark.knows.EvaluationJournal', lambda *args: EvaluationJournal(*args, scoring_module=scoring)), \
                        patch.dict(os.environ, {}, clear=True):
                    report = evaluate_task(task, 'saved_sheet', [], output)
                self.assertEqual(report['status'], 'evaluation_error')
                self.assertIsNone(report['reward'])
                self.assertIn('StructuredRowExtractionError', report['error'])
                self.assertEqual(json.loads(output.read_text())['status'], 'evaluation_error')
                self.assertIsNone(json.loads(output.read_text())['reward'])
                models.gemma_google_ai.assert_not_called()


if __name__ == '__main__':
    unittest.main()
