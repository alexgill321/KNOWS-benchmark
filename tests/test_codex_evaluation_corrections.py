from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from codex_benchmark.evaluation_corrections import (
    MODEL, TASK, REASON, MANIFEST_NAME, evaluation_for_scoring,
    load_reviewed_evaluation_correction,
)
from codex_benchmark.runner import summarize


class EvaluationCorrectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run = self.root / 'smoke'
        self.directory = self.run / MODEL / TASK
        self.directory.mkdir(parents=True)
        self.path = self.directory / 'episode.json'
        self.archive = self.root / 'recovery' / 'repair' / 'luna4'
        self.archive.mkdir(parents=True)
        self.manifest_path = self.root / MANIFEST_NAME
        self.agent = {'model': MODEL, 'effort': 'max', 'thread_id': 'thread1', 'turn_id': 'turn1',
                      'status': 'completed', 'tool_calls': 149, 'final_text': 'done', 'token_usage': {}}
        self.episode = {**self.agent, 'task_id': TASK, 'workspace_id': 'sheet1',
                        'judge_model': 'gemini-3.8-flash'}
        self.original = {'task_id': TASK, 'workspace_id': 'sheet1', 'judge_model': 'gemini-3.8-flash',
                         'status': 'completed', 'reward': .5, 'judge_errors': [],
                         'score_source': 'official_evaluator', 'evaluation_attempt_id': str(uuid4()),
                         'has_execution_errors': True}
        self.write(self.path, self.episode)
        self.write(self.directory / 'agent-result.json', self.agent)
        self.write(self.directory / 'evaluation.json', self.original)
        self.write(self.run / 'manifest.json', {'models': [MODEL], 'tasks_per_model': 110})
        self.source = self.archive / 'evaluator.py'
        self.source.write_text('# corrected hyperlink extraction\n')
        self.utils = self.archive / 'utils.py'
        self.utils.write_text('# corrected link helper\n')
        attempt = str(uuid4())
        self.journal = self.archive / 'evaluation-steps.jsonl'
        records = [{'event': event, 'attempt_id': attempt, 'task_id': TASK, 'workspace_id': 'sheet1'}
                   for event in ['evaluation_started', 'step_completed', 'evaluation_finished']]
        self.journal.write_text(''.join(json.dumps(r) + '\n' for r in records))
        self.report = {**self.original, 'reward': .625, 'has_execution_errors': False,
                       'evaluation_attempt_id': attempt,
                       'evaluator_source': {'effective_sha256': self.sha(self.source), 'original_sha256': self.sha(self.source)},
                       'step_journal': {'attempt_id': attempt, 'path': str(self.journal), 'completed_steps': 1},
                       'detailed_report': {'checkpoints': [{'steps': [{'name': 'link', 'success': True}]}]}}
        self.report_path = self.archive / 'evaluation.json'
        self.write_report(self.report)
        self.proof_path = self.archive / 'source-proof.json'
        self.proof = {'kind': REASON, 'task_id': TASK, 'verified': True, 'model_replayed': False,
                      'unchanged_rubric': True,
                      'source_snapshots': [{'path': str(self.source), 'sha256': self.sha(self.source), 'relative_path': 'instance_4/evaluator.py'},
                                           {'path': str(self.utils), 'sha256': self.sha(self.utils), 'relative_path': 'utils.py'}]}
        self.write(self.proof_path, self.proof)
        self.entry = {'model': MODEL, 'task_id': TASK, 'workspace_id': 'sheet1', 'reason': REASON,
                      'model_replayed': False,
                      'original_sha256': {n: self.sha(self.directory / n) for n in ['episode.json', 'evaluation.json', 'agent-result.json']},
                      'report_path': str(self.report_path), 'report_sha256': self.sha(self.report_path),
                      'evaluation_attempt_id': attempt, 'journal_sha256': self.sha(self.journal),
                      'source_proof_path': str(self.proof_path), 'source_proof_sha256': self.sha(self.proof_path)}
        self.manifest = {'version': 1, 'kind': 'reviewed_evaluator_corrections', 'run_dir': str(self.run),
                         'corrections': [self.entry]}
        self.write(self.manifest_path, self.manifest)

    def write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj) + '\n')

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_report(self, report):
        self.write(self.report_path, report)
        self.write(self.archive / 'evaluation-attempts' / (report['evaluation_attempt_id'] + '.json'), report)

    def assert_blocked(self):
        value = evaluation_for_scoring(self.path, self.original)
        self.assertIsNotNone(value['error'])
        self.assertIsNone(value['provenance'])
        self.assertEqual(value['evaluation'], self.original)

    def test_verified_correction_changes_summary_only_and_preserves_provenance(self):
        before = {p: p.read_bytes() for p in self.directory.iterdir()}
        verified = load_reviewed_evaluation_correction(self.path)
        self.assertEqual(verified['evaluation']['reward'], .625)
        self.assertEqual(verified['provenance']['original_reward'], .5)
        summary = summarize(self.run)
        self.assertEqual(summary['models'][MODEL]['mean_reward'], .625)
        self.assertEqual(summary['models'][MODEL]['scored'], 1)
        self.assertEqual(summary['models'][MODEL]['evaluator_error_tasks'], 0)
        self.assertEqual(summary['episodes'][0]['evaluation_correction']['original_reward'], .5)
        self.assertEqual({p: p.read_bytes() for p in self.directory.iterdir()}, before)

    def test_lower_correction_is_used_without_best_of_selection(self):
        self.report['reward'] = .25
        self.write_report(self.report)
        self.entry['report_sha256'] = self.sha(self.report_path)
        self.write(self.manifest_path, self.manifest)
        self.assertEqual(summarize(self.run)['models'][MODEL]['mean_reward'], .25)

    def test_no_manifest_or_unrelated_target_keeps_original(self):
        self.manifest_path.unlink()
        self.assertEqual(evaluation_for_scoring(self.path, self.original),
                         {'evaluation': self.original, 'provenance': None, 'error': None})
        self.write(self.manifest_path, self.manifest)
        self.assertIsNone(load_reviewed_evaluation_correction(self.run / MODEL / 'knows.sheets_25_skitourplan.3' / 'episode.json'))

    def test_completed_report_is_held_until_persisted_finalization_error_is_cleared(self):
        self.manifest_path.unlink()
        original_bytes = (self.directory / 'evaluation.json').read_bytes()
        self.write(self.path, {**self.episode, 'evaluation_error': 'Evaluator finalization failed'})
        held = evaluation_for_scoring(self.path, self.original)
        self.assertIn('unresolved episode evaluation_error', held['error'])
        self.assertEqual(held['evaluation'], self.original)
        self.assertIsNone(held['provenance'])
        self.assertEqual(summarize(self.run)['models'][MODEL]['scored'], 0)
        self.write(self.path, self.episode)
        self.assertEqual(evaluation_for_scoring(self.path, self.original),
                         {'evaluation': self.original, 'provenance': None, 'error': None})
        self.assertEqual(summarize(self.run)['models'][MODEL]['scored'], 1)
        self.assertEqual((self.directory / 'evaluation.json').read_bytes(), original_bytes)

    def test_cleared_finalization_error_restores_exact_verified_correction_provenance(self):
        verified = evaluation_for_scoring(self.path, self.original)
        episode_bytes = self.path.read_bytes()
        self.write(self.path, {**self.episode, 'evaluation_error': 'Evaluator finalization failed'})
        self.assertIn('unresolved episode evaluation_error', evaluation_for_scoring(self.path, self.original)['error'])
        self.path.write_bytes(episode_bytes)
        self.assertEqual(evaluation_for_scoring(self.path, self.original), verified)

    def test_completed_report_stays_held_during_pre_review_bookkeeping_write(self):
        self.manifest_path.unlink()
        self.write(self.path, {**self.episode, 'phase': 'evaluating'})
        held = evaluation_for_scoring(self.path, self.original)
        self.assertIn('awaiting episode finalization', held['error'])
        self.assertEqual(held['evaluation'], self.original)
        self.assertIsNone(held['provenance'])
        self.assertEqual(summarize(self.run)['models'][MODEL]['scored'], 0)
        self.write(self.path, {**self.episode, 'phase': 'finished'})
        self.assertIsNone(evaluation_for_scoring(self.path, self.original)['error'])
        self.assertEqual(summarize(self.run)['models'][MODEL]['scored'], 1)

    def test_tampered_original_report_agent_journal_source_or_proof_fails_closed(self):
        paths = [self.path, self.directory / 'agent-result.json', self.directory / 'evaluation.json',
                 self.report_path, self.journal, self.source, self.proof_path,
                 self.archive / 'evaluation-attempts' / (self.report['evaluation_attempt_id'] + '.json')]
        for path in paths:
            with self.subTest(path=path):
                old = path.read_bytes()
                path.write_bytes(old + b' ')
                self.assert_blocked()
                path.write_bytes(old)
        self.path.write_bytes(self.path.read_bytes() + b' ')
        summary = summarize(self.run)
        self.assertEqual(summary['models'][MODEL]['scored'], 0)
        self.assertIsNone(summary['models'][MODEL]['mean_reward'])
        self.assertEqual(summary['models'][MODEL]['evaluation_correction_errors'], 1)

    def test_wrong_scope_duplicate_entry_replayed_model_or_missing_hash_rejected(self):
        variants = []
        for field, value in [('model', 'gpt-6-astra'), ('task_id', TASK[:-1] + '3'),
                             ('workspace_id', 'other'), ('model_replayed', True), ('reason', 'best_score')]:
            item = deepcopy(self.manifest)
            item['corrections'][0][field] = value
            variants.append(item)
        item = deepcopy(self.manifest); item['corrections'].append(deepcopy(self.entry)); variants.append(item)
        item = deepcopy(self.manifest); item['corrections'][0]['original_sha256'].pop('agent-result.json'); variants.append(item)
        item = deepcopy(self.manifest); item['run_dir'] = str(self.root / 'other'); variants.append(item)
        for item in variants:
            with self.subTest(item=item):
                self.write(self.manifest_path, item)
                self.assert_blocked()

    def test_changed_identity_failed_report_boolean_score_same_attempt_or_wrong_source_rejected(self):
        for field, value in [('status', 'evaluation_error'), ('workspace_id', 'other'),
                             ('task_id', 'other'), ('judge_model', 'other'), ('judge_errors', ['bad']),
                             ('reward', True), ('reward', float('nan')),
                             ('score_source', 'invented'), ('evaluation_attempt_id', self.original['evaluation_attempt_id']),
                             ('evaluator_source', {'effective_sha256': '0' * 64})]:
            with self.subTest(field=field):
                report = {**self.report, field: value}
                self.write_report(report)
                self.entry['report_sha256'] = self.sha(self.report_path)
                self.write(self.manifest_path, self.manifest)
                self.assert_blocked()

    def test_incomplete_or_wrong_journal_with_updated_hash_still_rejected(self):
        records = [json.loads(s) for s in self.journal.read_text().splitlines()]
        for changed in (records[:2], records[1:], records[:1] + records[2:],
                        [{**x, 'workspace_id': 'other'} for x in records],
                        records + [{**records[0], 'attempt_id': str(uuid4())}]):
            with self.subTest(changed=changed):
                self.journal.write_text(''.join(json.dumps(r) + '\n' for r in changed))
                self.entry['journal_sha256'] = self.sha(self.journal)
                self.write(self.manifest_path, self.manifest)
                self.assert_blocked()

    def test_invalid_source_proof_and_archive_escape_rejected(self):
        for field, value in [('verified', False), ('unchanged_rubric', False), ('model_replayed', True),
                             ('kind', 'other'), ('task_id', 'other'), ('source_snapshots', [])]:
            proof = {**self.proof, field: value}
            self.write(self.proof_path, proof)
            self.entry['source_proof_sha256'] = self.sha(self.proof_path)
            self.write(self.manifest_path, self.manifest)
            self.assert_blocked()
        self.entry['report_path'] = str(self.directory / 'evaluation.json')
        self.write(self.manifest_path, self.manifest)
        self.assert_blocked()

    def test_unrelated_or_incomplete_source_snapshot_proof_rejected(self):
        for snapshots in (self.proof['source_snapshots'][:1], self.proof['source_snapshots'][1:],
                          [{**x, 'relative_path': 'unrelated.py'} for x in self.proof['source_snapshots']],
                          self.proof['source_snapshots'] + self.proof['source_snapshots'][:1]):
            self.write(self.proof_path, {**self.proof, 'source_snapshots': snapshots})
            self.entry['source_proof_sha256'] = self.sha(self.proof_path)
            self.write(self.manifest_path, self.manifest)
            self.assert_blocked()

    def test_unrelated_output_criterion_error_does_not_invalidate_completed_correction(self):
        self.report['has_execution_errors'] = True
        self.write_report(self.report)
        self.entry['report_sha256'] = self.sha(self.report_path)
        self.write(self.manifest_path, self.manifest)
        verified = load_reviewed_evaluation_correction(self.path)
        self.assertTrue(verified['evaluation']['has_execution_errors'])
        self.assertEqual(summarize(self.run)['models'][MODEL]['scored'], 1)

    def test_monitor_grades_use_verified_correction_and_invalid_proof_withholds(self):
        script = Path(__file__).resolve().parents[1] / 'codex-results/monitor-checks/20261004T111240Z/audit_campaigns.py'
        spec = importlib.util.spec_from_file_location('correction_monitor_test', script)
        audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
        value = audit.audit(self.run, {}, 0, {})
        self.assertEqual(value['grades'][0]['reward'], .625)
        self.assertEqual(value['reviewed_evaluation_corrections'][0]['original_reward'], .5)
        self.assertEqual(value['eligible'], 1)
        self.proof_path.write_text('{}')
        value = audit.audit(self.run, {}, 0, {})
        self.assertIsNone(value['grades'][0]['reward'])
        self.assertEqual(value['eligible'], 0)
        self.assertEqual(value['under_review_count'], 1)
        self.assertEqual(len(value['evaluation_correction_errors']), 1)


if __name__ == '__main__':
    unittest.main()
