"""Offline uniqueness independence and fail-closed evaluator regressions."""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from codex_benchmark.evaluation_log import EvaluationJournal
from codex_benchmark.knows import TASKS_RELATIVE, _strict_judges, discover_tasks, evaluate_task

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'codex_benchmark').is_dir())
FAMILY = Path('src/browsergym/knows/eval/tasks/slides_39_Personal_Lookbook_PaintColors')
DEFAULT_SOURCE = Path(__file__).parent / 'staged-source'
SOURCE = Path(os.environ.get('KNOWS_LOOKBOOK_TEST_SOURCE', str(DEFAULT_SOURCE if DEFAULT_SOURCE.exists() else ROOT)))


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


spec = importlib.util.spec_from_file_location('lookbook_uniqueness_scoring', ROOT / TASKS_RELATIVE.parent / 'eval_utils/scoring.py')
scoring = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scoring
spec.loader.exec_module(scoring)
HELPERS = {}
definitions(SOURCE / FAMILY / 'utils.py',
            {'ImageUniquenessError', 'require_unique_image_pair', 'require_unique_match_result'}, HELPERS)
HELPERS['os'] = os


class LookbookUniquenessTests(unittest.TestCase):
    def fixture(self, instance, folder, *, image_count=2, alt_count=0, duplicate=False,
                missing_download=False, malformed_download=False, match_result='normal'):
        images = [{'objectId': f'object-{i}', 'contentUrl': f'content-{i}', 'position': i}
                  for i in range(image_count)]
        sources = [{'objectId': item['objectId'], 'source_urls': ['source'] if i < alt_count else []}
                   for i, item in enumerate(images)]
        ctx = {'presentation_data': {'slides': [{}]}, 'color_slides': [{'index': 1, 'color': 'Sage', 'slide': {}}],
               'workspace_doc_id': 'saved_workspace', 'slide_width_emu': 100, 'slide_height_emu': 100}
        queued, matched = [], []

        def download(tasks, **kwargs):
            self.assertEqual(kwargs, {'max_workers': 5, 'use_rate_limit': False})
            queued.extend(deepcopy([{k: v for k, v in task.items() if k != 'func'} for task in tasks]))
            results = {}
            for task in tasks:
                tid = task['id']; path = Path(folder) / (tid + '.png')
                if tid == 'object-1_content' and missing_download:
                    results[tid] = None
                    continue
                if tid == 'object-1_content' and malformed_download:
                    path.write_text('upstream error page, not an image')
                else:
                    Image.new('RGB', (12, 12), 'red' if duplicate or tid != 'object-1_content' else 'blue').save(path)
                results[tid] = str(path)
            return results

        def compare(tasks, **kwargs):
            self.assertEqual(kwargs, {'max_workers': 5})
            matched.extend(deepcopy(tasks))
            if tasks and isinstance(tasks[0]['id'], int) and match_result != 'normal':
                return {} if match_result == 'missing' else {1: match_result}
            # Saved local images supply complete comparisons. Production still
            # calls the unchanged exact/perceptual matcher with the same limits.
            return {task['id']: (Path(task['candidate_path']).read_bytes() == Path(task['gold_path']).read_bytes(),
                                 'exact' if Path(task['candidate_path']).read_bytes() == Path(task['gold_path']).read_bytes() else None)
                    for task in tasks}

        env = dict(HELPERS, os=os, shutil=shutil, time=time, model=object(), model_id='unchanged-judge',
                   DATA_DIR=str(folder), SLIDES_SERVICE=object(), Checkpoint=scoring.Checkpoint,
                   Result=scoring.Result, StepCategory=scoring.StepCategory,
                   calculate_percentage_score=scoring.calculate_percentage_score,
                   check_browsing_history=lambda *a: False,
                   extract_slide_images=lambda *a: images,
                   extract_image_source_urls=lambda *a: sources,
                   get_image_position=lambda image, *a: 'bottom_left' if image['position'] == 0 else 'bottom_right',
                   parallel_execute=lambda tasks, **kwargs: {1: (True, image_count, image_count)},
                   evaluate_image_relevance_vlm=Mock(side_effect=AssertionError('No judge calls')),
                   download_image_from_url=Mock(side_effect=AssertionError('No network calls')),
                   download_alt_image=Mock(side_effect=AssertionError('No network calls')),
                   browser_headers=lambda _: {}, parallel_download=download, parallel_image_match=compare)
        definitions(SOURCE / FAMILY / f'instance_{instance}/evaluator.py', {'grade_checkpoint_3', 'grade_checkpoints'}, env)
        return env, ctx, queued, matched

    def test_all_five_no_alt_distinct_content_earns_only_independent_uniqueness(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance), tempfile.TemporaryDirectory() as folder:
                env, ctx, queued, matched = self.fixture(instance, folder)
                checkpoint = env['grade_checkpoint_3'](ctx, [])
                steps = {s.step_id: s for s in checkpoint.steps}
                self.assertEqual(set(steps), set(range(1, 8)))
                self.assertEqual((steps[5].score, steps[6].score, steps[7].score), (0, 0, 10))
                self.assertEqual([t['id'] for t in queued], ['object-0_content', 'object-1_content'])
                self.assertEqual([t['id'] for t in matched], [1])
                self.assertEqual(steps[7].category, 'fuzzy_match')
                self.assertFalse((Path(folder) / 'temp_images_source_check').exists())
                env['download_image_from_url'].assert_not_called()
                env['evaluate_image_relevance_vlm'].assert_not_called()

    def test_all_five_duplicate_without_alt_still_fails_uniqueness(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance), tempfile.TemporaryDirectory() as folder:
                env, ctx, _, _ = self.fixture(instance, folder, duplicate=True)
                steps = {s.step_id: s for s in env['grade_checkpoint_3'](ctx).steps}
                self.assertEqual((steps[5].score, steps[6].score, steps[7].score), (0, 0, 0))
                self.assertIn('duplicate', steps[7].details)

    def test_existing_alt_matching_and_download_dedup_remain_unchanged(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance), tempfile.TemporaryDirectory() as folder:
                env, ctx, queued, _ = self.fixture(instance, folder, alt_count=2, duplicate=True)
                steps = {s.step_id: s for s in env['grade_checkpoint_3'](ctx).steps}
                self.assertEqual((steps[5].score, steps[6].score, steps[7].score), (10, 10, 0))
                self.assertEqual([t['id'] for t in queued], ['object-0_content', 'object-1_content', 'object-0_alt_0'])
                self.assertIn('2/2 matched', steps[6].details)

    def test_wrong_image_count_keeps_output_failure_without_comparison(self):
        for count in (0, 1, 3):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as folder:
                env, ctx, _, matched = self.fixture(1, folder, image_count=count)
                steps = {s.step_id: s for s in env['grade_checkpoint_3'](ctx).steps}
                self.assertEqual(steps[7].score, 0)
                self.assertIn(f'only {count} image(s)', steps[7].details)
                self.assertEqual(matched, [])

    def test_missing_and_unreadable_content_never_becomes_zero_or_unique(self):
        for instance in range(1, 6):
            for field in ('missing_download', 'malformed_download'):
                with self.subTest(instance=instance, field=field), tempfile.TemporaryDirectory() as folder:
                    env, ctx, _, _ = self.fixture(instance, folder, **{field: True})
                    with self.assertRaises(HELPERS['ImageUniquenessError']): env['grade_checkpoint_3'](ctx)
                    self.assertFalse((Path(folder) / 'temp_images_source_check').exists())

    def test_missing_or_malformed_match_is_fatal_but_explicit_false_is_valid(self):
        for result in ('missing', None, False, (1, None), (True, None), (False,), (False, 'unexpected')):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as folder:
                env, ctx, _, _ = self.fixture(1, folder, match_result=result)
                with self.assertRaises(HELPERS['ImageUniquenessError']): env['grade_checkpoint_3'](ctx)
        for missing in (None, [], False):
            with self.assertRaises(HELPERS['ImageUniquenessError']):
                HELPERS['require_unique_match_result'](missing, 1)
            with self.assertRaises(HELPERS['ImageUniquenessError']):
                HELPERS['require_unique_image_pair']([{}, {}], missing)
        self.assertEqual(HELPERS['require_unique_match_result']({1: (False, None)}, 1), (False, None))
        self.assertEqual(HELPERS['require_unique_match_result']({1: (True, 'perceptual_hash')}, 1), (True, 'perceptual_hash'))

    def test_missing_duplicate_ids_and_nonfile_downloads_are_fatal(self):
        helper = HELPERS['require_unique_image_pair']
        for images, downloaded in (([{}, {}], {}), ([{'objectId': 'same'}]*2, {}),
                                   ([{'objectId': 'a'}, {'objectId': 'b'}], {'a_content': True, 'b_content': True})):
            with self.subTest(images=images), self.assertRaises(HELPERS['ImageUniquenessError']):
                helper(images, downloaded)

    def test_all_five_real_factory_propagates_to_null_reward_bridge(self):
        for instance in range(1, 6):
            with self.subTest(instance=instance), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); env, ctx, _, _ = self.fixture(instance, folder, missing_download=True)
                fixture = ModuleType('_lookbook_uniqueness_fixture')
                for key, value in env.items():
                    if not key.startswith('__'): setattr(fixture, key, value)
                fixture.SLIDES_SERVICE = Mock()
                fixture.SLIDES_SERVICE.presentations.return_value.get.return_value.execute.return_value = {'slides': [{}]}
                fixture.setup = lambda *a: ctx
                for n in (1, 2, 4):
                    setattr(fixture, f'grade_checkpoint_{n}', lambda *a: scoring.Checkpoint(total=0, result=0, name='Fixture'))
                # Compile unchanged actual checkpoint + factory bodies into the
                # fixture module, with only external I/O supplied by mocks.
                tree = ast.parse((SOURCE / FAMILY / f'instance_{instance}/evaluator.py').read_text())
                nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'grade_checkpoint_3', 'grade_checkpoints'}]
                path = root / TASKS_RELATIVE / 'slides_lookbook_guard_fixture/instance_1'
                path.mkdir(parents=True); (path / 'task.md').write_text('Use the saved presentation.\n')
                (path / 'evaluator.py').write_text('from _lookbook_uniqueness_fixture import *\n' + ast.unparse(ast.Module(body=nodes, type_ignores=[])))
                models = SimpleNamespace(_models={}, gemma_google_ai=Mock())
                with patch.dict(sys.modules, {fixture.__name__: fixture}), \
                        patch('codex_benchmark.knows._prepare_credentials'), \
                        patch('codex_benchmark.knows._strict_judges', lambda override=None: _strict_judges(override, models)), \
                        patch('codex_benchmark.knows.EvaluationJournal', lambda *a: EvaluationJournal(*a, scoring_module=scoring)), \
                        patch.dict(os.environ, {}, clear=True):
                    output = root / 'evaluation.json'
                    report = evaluate_task(discover_tasks(root)[0], 'saved_workspace', [], output)
                self.assertEqual(report['status'], 'evaluation_error')
                self.assertIsNone(report['reward'])
                self.assertIn('ImageUniquenessError', report['error'])
                self.assertIsNone(json.loads(output.read_text())['reward'])
                models.gemma_google_ai.assert_not_called()


if __name__ == '__main__': unittest.main()
