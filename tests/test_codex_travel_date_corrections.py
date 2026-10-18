from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from codex_benchmark.evaluation_corrections import evaluation_for_scoring
from codex_benchmark.travel_date_corrections import (
    TARGETS, REASON, PENDING_NAME, CORRECTIONS_NAME, load_travel_date_correction,
)
from codex_benchmark.runner import summarize


class TravelDateCorrectionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.run = self.root / 'smoke'
        self.archive = self.root / 'recovery' / 'travel-dates'
        self.pending_path = self.root / PENDING_NAME
        self.corrected_path = self.root / CORRECTIONS_NAME
        self.entries = []
        self.originals = {}
        self.episodes = {}
        for model, task in sorted(TARGETS):
            directory = self.run / model / task
            agent = {'model': model, 'effort': 'max', 'thread_id': 'thread-' + model + task,
                     'turn_id': 'turn-' + model + task, 'status': 'completed', 'tool_calls': 100,
                     'final_text': 'done', 'token_usage': {}}
            episode = {**agent, 'task_id': task, 'workspace_id': model + task,
                       'judge_model': 'gemini-3.8-flash'}
            report = {'task_id': task, 'workspace_id': episode['workspace_id'], 'status': 'completed',
                      'reward': .5, 'judge_errors': [], 'score_source': 'official_evaluator',
                      'judge_model': 'gemini-3.8-flash', 'evaluation_attempt_id': str(uuid4())}
            for name, value in [('episode.json', episode), ('evaluation.json', report), ('agent-result.json', agent)]:
                self.write(directory / name, value)
            self.entries.append({'model': model, 'task_id': task, 'reason': REASON,
                                 'status': 'pending_saved_output_regrade', 'original_reward': .5,
                                 'evaluation_attempt_id': report['evaluation_attempt_id'],
                                 'original_sha256': {n: self.sha(directory / n)
                                     for n in ['episode.json', 'evaluation.json', 'agent-result.json']}})
            self.originals[model, task] = report
            self.episodes[model, task] = episode
        self.write(self.run / 'manifest.json', {'models': ['gpt-6-luna', 'gpt-6-astra'], 'tasks_per_model': 110})
        review = self.root / 'monitor-checks' / 'review' / 'review.json'
        scope = review.with_name('scope.json')
        self.write(review, {'classification': 'verified_evaluator_date_parser_limitation',
                            'explicit_date_format_violation_found': False})
        self.write(scope, {'affected_reports': [{'model': e['model'], 'task_id': e['task_id'],
                            'report_sha256': e['original_sha256']['evaluation.json'],
                            'evaluation_attempt_id': e['evaluation_attempt_id']} for e in self.entries]})
        self.pending = {'version': 1, 'kind': 'pending_reviewed_evaluator_corrections',
                        'run_dir': str(self.run), 'model_replayed': False,
                        'review_proof': str(review), 'scope_proof': str(scope),
                        'review_proof_sha256': self.sha(review), 'scope_proof_sha256': self.sha(scope),
                        'corrections': self.entries}
        self.write(self.pending_path, self.pending)
        self.corrected = {'version': 1, 'kind': 'reviewed_travel_date_evaluator_corrections',
                          'run_dir': str(self.run), 'model_replayed': False,
                          'pending_manifest_sha256': self.sha(self.pending_path), 'corrections': []}
        self.key = sorted(TARGETS)[0]

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n')

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def path(self, key=None):
        model, task = key or self.key
        return self.run / model / task / 'episode.json'

    def correction(self, key=None, reward=.625):
        key = key or self.key
        model, task = key
        directory = self.archive / model / task
        source = directory / 'evaluator.py'
        utils = directory / 'utils.py'
        directory.mkdir(parents=True)
        source.write_text('# unchanged instance evaluator\n')
        utils.write_text('# corrected annotated date parsing\n')
        attempt = str(uuid4())
        journal = directory / 'evaluation-steps.jsonl'
        events = [{'event': name, 'attempt_id': attempt, 'task_id': task,
                   'workspace_id': self.episodes[key]['workspace_id']} for name in
                  ['evaluation_started'] + ['step_completed'] * 37 + ['evaluation_finished']]
        journal.write_text(''.join(json.dumps(x) + '\n' for x in events))
        report = {**self.originals[key], 'reward': reward, 'evaluation_attempt_id': attempt, 'evaluator_invoked': True,
                  'evaluator_source': {'original_sha256': self.sha(source), 'effective_sha256': self.sha(source)},
                  'step_journal': {'attempt_id': attempt, 'path': str(journal), 'completed_steps': 37},
                  'detailed_report': {'checkpoints': [{'steps': [{'name': f'rubric step {n}', 'success': True}
                                                               for n in range(37)]}]} }
        report_path = directory / 'evaluation.json'
        self.write(report_path, report)
        self.write(directory / 'evaluation-attempts' / (attempt + '.json'), report)
        proof_path = directory / 'source-proof.json'
        self.write(proof_path, {'kind': REASON, 'task_id': task, 'verified': True,
            'model_replayed': False, 'unchanged_rubric': True, 'source_snapshots': [
                {'relative_path': 'utils.py', 'path': str(utils), 'sha256': self.sha(utils)},
                {'relative_path': f'instance_{task.rsplit(".", 1)[1]}/evaluator.py',
                 'path': str(source), 'sha256': self.sha(source)}]})
        original = next(e for e in self.entries if (e['model'], e['task_id']) == key)
        entry = {'model': model, 'task_id': task, 'reason': REASON, 'model_replayed': False,
                 'workspace_id': self.episodes[key]['workspace_id'],
                 'original_sha256': original['original_sha256'],
                 'report_path': str(report_path), 'report_sha256': self.sha(report_path),
                 'evaluation_attempt_id': attempt, 'journal_sha256': self.sha(journal),
                 'source_proof_path': str(proof_path), 'source_proof_sha256': self.sha(proof_path)}
        self.corrected['corrections'].append(entry)
        self.write(self.corrected_path, self.corrected)
        return entry

    def test_exact_seven_pending_grades_held_without_changing_originals(self):
        before = {p: p.read_bytes() for p in self.run.glob('*/*/*.json')}
        for key in TARGETS:
            value = evaluation_for_scoring(self.path(key), self.originals[key])
            self.assertIn('Pending saved-output', value['error'])
        summary = summarize(self.run)
        self.assertEqual(sum(m['evaluated'] for m in summary['models'].values()), 7)
        self.assertEqual(sum(m['scored'] for m in summary['models'].values()), 0)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertIsNone(load_travel_date_correction(self.run / 'gpt-6-luna' /
                           'knows.sheets_28_personal_travel_planner.1' / 'episode.json'))

    def test_completed_subset_releases_only_verified_target_even_if_lower_grade(self):
        self.correction(reward=.25)
        value = evaluation_for_scoring(self.path(), self.originals[self.key])
        self.assertIsNone(value['error'])
        self.assertEqual(value['evaluation']['reward'], .25)
        self.assertEqual(value['provenance']['original_reward'], .5)
        self.assertEqual(value['provenance']['pending_manifest_sha256'], self.sha(self.pending_path))
        summary = summarize(self.run)
        self.assertEqual(sum(m['scored'] for m in summary['models'].values()), 1)
        for key in TARGETS - {self.key}:
            self.assertIsNotNone(evaluation_for_scoring(self.path(key), self.originals[key])['error'])

    def test_candidate_validates_before_publication_without_releasing_official_hold(self):
        self.correction()
        candidate = self.archive / 'candidate-manifest.json'
        candidate.write_bytes(self.corrected_path.read_bytes())
        self.corrected_path.unlink()
        value = load_travel_date_correction(self.path(), manifest_path=candidate)
        self.assertEqual(value['evaluation']['reward'], .625)
        self.assertEqual(value['provenance']['manifest_path'], str(candidate))
        self.assertIn('Pending saved-output', evaluation_for_scoring(self.path(), self.originals[self.key])['error'])
        with self.assertRaises(ValueError):
            load_travel_date_correction(self.path(), manifest_path=self.root / 'unreviewed.json')

    def test_all_seven_corrected_and_existing_originals_preserved(self):
        before = {p: p.read_bytes() for p in self.run.glob('*/*/*.json')}
        for key in TARGETS:
            self.correction(key)
        for key in TARGETS:
            value = load_travel_date_correction(self.path(key))
            self.assertEqual(value['evaluation']['reward'], .625)
        summary = summarize(self.run)
        self.assertEqual(sum(m['scored'] for m in summary['models'].values()), 7)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_missing_wrong_scope_duplicate_or_tampered_pending_fails_closed(self):
        self.correction()
        cases = []
        for field, value in [('kind', 'other'), ('model_replayed', True), ('run_dir', str(self.root))]:
            item = deepcopy(self.pending); item[field] = value; cases.append(item)
        item = deepcopy(self.pending); item['corrections'].pop(); cases.append(item)
        item = deepcopy(self.pending); item['corrections'][0] = item['corrections'][1]; cases.append(item)
        item = deepcopy(self.pending); item['corrections'][0]['task_id'] = 'knows.sheets_28_personal_travel_planner.9'; cases.append(item)
        for item in cases:
            with self.subTest(item=item):
                self.write(self.pending_path, item)
                self.assertIsNotNone(evaluation_for_scoring(self.path(), self.originals[self.key])['error'])
        self.pending_path.unlink()
        self.assertIsNotNone(evaluation_for_scoring(self.path(), self.originals[self.key])['error'])

    def test_tampered_original_even_other_held_target_fails_closed(self):
        self.correction()
        other = next(iter(TARGETS - {self.key}))
        for filename in ['episode.json', 'evaluation.json', 'agent-result.json']:
            path = self.path(other).with_name(filename)
            old = path.read_bytes(); path.write_bytes(old + b' ')
            self.assertIsNotNone(evaluation_for_scoring(self.path(), self.originals[self.key])['error'])
            path.write_bytes(old)

    def test_completed_manifest_cannot_replace_scope_lineage_or_attempt(self):
        self.correction()
        cases = []
        for field, value in [('pending_manifest_sha256', '0' * 64), ('model_replayed', True), ('kind', 'other')]:
            item = deepcopy(self.corrected); item[field] = value; cases.append(item)
        item = deepcopy(self.corrected); item['corrections'].append(item['corrections'][0]); cases.append(item)
        for field, value in [('original_sha256', {}), ('task_id', 'knows.sheets_28_personal_travel_planner.9'),
                             ('evaluation_attempt_id', self.originals[self.key]['evaluation_attempt_id'])]:
            item = deepcopy(self.corrected); item['corrections'][0][field] = value; cases.append(item)
        for item in cases:
            self.write(self.corrected_path, item)
            self.assertIsNotNone(evaluation_for_scoring(self.path(), self.originals[self.key])['error'])

    def test_incomplete_journal_and_unreviewed_source_fail_closed_with_updated_hash(self):
        entry = self.correction()
        journal = Path(entry['report_path']).with_name('evaluation-steps.jsonl')
        old = journal.read_bytes()
        journal.write_bytes(b'\n'.join(old.splitlines()[:-1]) + b'\n')
        entry['journal_sha256'] = self.sha(journal)
        self.write(self.corrected_path, self.corrected)
        self.assertIsNotNone(evaluation_for_scoring(self.path(), self.originals[self.key])['error'])

        journal.write_bytes(old); entry['journal_sha256'] = self.sha(journal)
        proof = Path(entry['source_proof_path']); obj = json.loads(proof.read_text()); obj['verified'] = False
        self.write(proof, obj); entry['source_proof_sha256'] = self.sha(proof)
        self.write(self.corrected_path, self.corrected)
        self.assertIsNotNone(evaluation_for_scoring(self.path(), self.originals[self.key])['error'])

    def test_self_consistent_truncated_rubric_cannot_be_published_as_full_correction(self):
        entry = self.correction()
        report_path = Path(entry['report_path'])
        report = json.loads(report_path.read_text())
        report['detailed_report']['checkpoints'][0]['steps'] = report['detailed_report']['checkpoints'][0]['steps'][:1]
        report['step_journal']['completed_steps'] = 1
        self.write(report_path, report)
        self.write(report_path.parent / 'evaluation-attempts' / (report['evaluation_attempt_id'] + '.json'), report)
        journal = report_path.with_name('evaluation-steps.jsonl')
        events = journal.read_text().splitlines()
        journal.write_text('\n'.join(events[:2] + events[-1:]) + '\n')
        entry['report_sha256'] = self.sha(report_path)
        entry['journal_sha256'] = self.sha(journal)
        self.write(self.corrected_path, self.corrected)
        value = evaluation_for_scoring(self.path(), self.originals[self.key])
        self.assertIn('complete 37-step evaluator', value['error'])

    def test_invoked_evaluator_required_even_with_complete_report_and_journal(self):
        entry = self.correction()
        report_path = Path(entry['report_path'])
        report = json.loads(report_path.read_text())
        for value in (False, None, 1):
            report['evaluator_invoked'] = value
            self.write(report_path, report)
            self.write(report_path.parent / 'evaluation-attempts' / (report['evaluation_attempt_id'] + '.json'), report)
            entry['report_sha256'] = self.sha(report_path)
            self.write(self.corrected_path, self.corrected)
            self.assertIn('invoked complete 37-step', evaluation_for_scoring(self.path(), self.originals[self.key])['error'])

    def test_scope_proof_must_match_all_seven_original_attempts(self):
        self.correction()
        proof = Path(self.pending['scope_proof']); obj = json.loads(proof.read_text())
        obj['affected_reports'][0]['evaluation_attempt_id'] = str(uuid4())
        self.write(proof, obj)
        self.assertIsNotNone(evaluation_for_scoring(self.path(), self.originals[self.key])['error'])

    def test_monitor_uses_holds_and_completed_correction_without_wrapper(self):
        script = Path(__file__).resolve().parents[1] / 'codex-results/monitor-checks/20261004T111240Z/audit_campaigns.py'
        spec = importlib.util.spec_from_file_location('date_monitor_test', script)
        audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
        value = audit.audit(self.run, {}, 0, {})
        self.assertEqual(value['evaluated'], 7)
        self.assertEqual(value['eligible'], 0)
        self.assertEqual(value['under_review_count'], 7)
        self.correction()
        value = audit.audit(self.run, {}, 0, {})
        self.assertEqual(value['eligible'], 1)
        self.assertEqual(value['under_review_count'], 6)
        self.assertEqual(value['reviewed_evaluation_corrections'][0]['reason'], REASON)


if __name__ == '__main__':
    unittest.main()
