from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from codex_benchmark.lookbook_corrections import (
    ARCHIVE_NAME, CORRECTIONS_NAME, PENDING_NAME, REASON, TARGETS, WHITE_TARGET,
    load_lookbook_correction, _reviews,
)
from codex_benchmark.evaluation_corrections import evaluation_for_scoring


class LookbookCorrectionTests(unittest.TestCase):
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n')

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def report(self, directory, episode, reward, source):
        attempt = str(uuid4())
        steps = [{'name': f'criterion{i}', 'step_id': i, 'max_score': 1, 'score': reward,
                  'success': False, 'category': 'fuzzy_match', 'details': 'model output comparison'} for i in range(19)]
        steps[-1]['name'] = 'Two Images Are Unique'
        result = dict(status='completed', task_id=episode['task_id'], workspace_id=episode['workspace_id'],
                      judge_model=episode['judge_model'], reward=reward, evaluation_attempt_id=attempt,
                      evaluator_invoked=True, score_source='official_evaluator', judge_errors=[], has_execution_errors=False,
                      evaluator_source={'original_sha256': source, 'effective_sha256': source, 'repairs': []},
                      step_journal={'attempt_id': attempt, 'path': str(directory/'evaluation-steps.jsonl'), 'completed_steps': 19},
                      detailed_report={'checkpoints': [{'name': 'rubric', 'steps': steps}]})
        self.persist_report(directory, result)
        return result

    def persist_report(self, directory, report):
        identity = {k: report[k] for k in ['task_id', 'workspace_id']}
        identity['attempt_id'] = report['evaluation_attempt_id']
        rows = [dict(identity, event='evaluation_started')]
        rows += [dict(identity, event='step_completed', checkpoint_name=cp['name'], step=step)
                 for cp in report['detailed_report']['checkpoints'] for step in cp['steps']]
        rows += [dict(identity, event='evaluation_finished', completed_steps=len(rows)-1)]
        self.write(directory/'evaluation.json', report)
        self.write(directory/'evaluation-attempts'/f"{report['evaluation_attempt_id']}.json", report)
        (directory/'evaluation-steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve(); self.run=self.root/'smoke'
        self.archive=self.root/'recovery'/ARCHIVE_NAME
        self.pending_path=self.root/PENDING_NAME; self.manifest_path=self.root/CORRECTIONS_NAME
        sources=[]
        for relative in ['utils.py', *(f'instance_{n}/evaluator.py' for n in range(1,6))]:
            path=self.archive/'source'/relative; path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen '+relative)
            sources.append(dict(relative_path=relative,path=str(path),sha256=self.sha(path)))
        self.fixtures=[]
        for model,task in sorted(TARGETS):
            key=model,task; directory=self.run/model/task; archived=self.archive/'originals'/model/task
            output=self.archive/'corrections'/model/task
            agent=dict(model=model,effort='max',status='completed',thread_id=model+task,turn_id='turn',tool_calls=150,final_text='saved',token_usage={})
            episode=dict(agent,task_id=task,workspace_id=model+task,judge_model='same-judge')
            self.write(directory/'episode.json',episode);self.write(directory/'agent-result.json',agent)
            source=next(s['sha256'] for s in sources if s['relative_path']==f'instance_{task[-1]}/evaluator.py')
            original=self.report(directory,episode,.4,source)
            url='https://images.example.test/white.jpg'
            (directory/'evaluate.log').write_text('Wayback fallback failed for '+url+': Expecting value: line 1 column 1 (char 0)\n' if key==WHITE_TARGET else 'completed evaluation\n')
            hashes={name:self.sha(directory/name) for name in ['episode.json','evaluation.json','agent-result.json']}
            originals=[]
            for p in directory.rglob('*'):
                if p.is_file():
                    relative=str(p.relative_to(directory));dest=archived/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
                    originals.append(dict(relative_path=relative,path=str(dest),sha256=self.sha(dest)))
            defect_review=archived/'independent-defect.json'
            self.write(defect_review,dict(reviewed=True,uniqueness_scheduling_defect_verified=True,model_replayed=False,
                model=model,task_id=task,original_report_sha256=hashes['evaluation.json'],evaluation_attempt_id=original['evaluation_attempt_id']))
            raw=archived/'saved-defect-evidence.json';self.write(raw,{'verified_downloaded_pair_omitted':True})
            source_proof=archived/'source-proof.json'
            proof=dict(kind=REASON,verified=True,unchanged_rubric=True,model_replayed=False,model=model,task_id=task,
                workspace_id=episode['workspace_id'],original_sha256=hashes,source_snapshots=sources,original_snapshots=originals,
                raw_snapshots=[dict(path=str(raw),sha256=self.sha(raw))],independent_review_path=str(defect_review),independent_review_sha256=self.sha(defect_review))
            self.write(source_proof,proof)
            pending=dict(model=model,task_id=task,workspace_id=episode['workspace_id'],reason=REASON,model_replayed=False,
                original_sha256=hashes,evaluation_attempt_id=original['evaluation_attempt_id'],original_reward=.4,
                status='pending_saved_output_regrade',source_proof_path=str(source_proof),source_proof_sha256=self.sha(source_proof))
            if key==WHITE_TARGET:
                self.white_proof(pending,original,archived,url)
            corrected=self.report(output,episode,.3,source);(output/'evaluate.log').write_text('corrected evaluation completed\n')
            entry=dict(pending,evaluation_attempt_id=corrected['evaluation_attempt_id'],report_path=str(output/'evaluation.json'),
                report_sha256=self.sha(output/'evaluation.json'),journal_sha256=self.sha(output/'evaluation-steps.jsonl'),evaluate_log_sha256=self.sha(output/'evaluate.log'))
            fixture=dict(key=key,path=directory/'episode.json',directory=directory,archived=archived,original=original,pending=pending,
                proof=proof,proof_path=source_proof,entry=entry,report=corrected,output=output)
            self.reviews(fixture);self.fixtures.append(fixture)
        self.pending=dict(version=1,kind='pending_lookbook_evaluation_corrections',run_dir=str(self.run),model_replayed=False,targets=[f['pending']for f in self.fixtures])
        self.write(self.pending_path,self.pending)
        self.manifest=dict(version=1,kind='reviewed_lookbook_evaluation_corrections',run_dir=str(self.run),model_replayed=False,
            pending_manifest_sha256=self.sha(self.pending_path),corrections=[f['entry']for f in self.fixtures if f['key']!=WHITE_TARGET])
        self.write(self.manifest_path,self.manifest)

    def white_proof(self,entry,original,archived,url):
        timestamp='2026-10-01T00:00:00+00:00';url_sha=hashlib.sha256(url.encode()).hexdigest()
        actions=archived/'browser/actions.jsonl';nav=archived/'browser/navigations.jsonl'
        self.write(actions,{'action':42,'success':True,'timestamp':timestamp})
        with actions.open('a') as f:f.write(json.dumps({'action':49,'tool':'browser_type','success':True,'timestamp':timestamp,
            'arguments':{'text':'White garage. Source: '+url}})+'\n')
        self.write(nav,{'page_id':'page_2','url':url,'timestamp':timestamp})
        shot=archived/'browser/screenshots/00042_page_2.png';shot.parent.mkdir(parents=True);shot.write_bytes(b'bound screenshot')
        independent=archived/'white-independent.json'
        self.write(independent,{'independent_retrieval_evidence':{'white_case':{'color':'White','exact_url_sha256':url_sha,
            'screenshot_visually_reviewed':True,'screenshot':{'sha256':self.sha(shot)}}}})
        proof=dict(kind='lookbook_white_image_retrieval_review',reviewed=True,model_replayed=False,exact_browser_url_loaded=True,
            evaluator_fetch_failed=True,original_http_cause_proven=False,model=entry['model'],task_id=entry['task_id'],
            workspace_id=entry['workspace_id'],original_sha256=entry['original_sha256'],evaluation_attempt_id=original['evaluation_attempt_id'],
            exact_url_sha256=url_sha,evidence={name:{'path':str(path),'sha256':self.sha(path)} for name,path in
                [('browser_actions',actions),('browser_navigations',nav),('browser_screenshot',shot),('evaluator_log',archived/'evaluate.log')]},
            independent_review={'path':str(independent),'sha256':self.sha(independent)})
        path=archived/'white-hold.json';self.write(path,proof)
        entry.update(retrieval_hold_proof_path=str(path),retrieval_hold_proof_sha256=self.sha(path))

    def reviews(self,fixture):
        e=fixture['entry'];r=fixture['report'];output=fixture['output']
        review=dict(reviewed=True,scoring_valid=True,model_replayed=False,execution_errors_reviewed=True,no_auth_or_judge_failure=True,
            no_unresolved_evaluator_error=True,model=e['model'],task_id=e['task_id'],workspace_id=e['workspace_id'],
            evaluation_attempt_id=r['evaluation_attempt_id'],report_sha256=e['report_sha256'],journal_sha256=e['journal_sha256'],
            evaluate_log_sha256=e['evaluate_log_sha256'],original_report_sha256=e['original_sha256']['evaluation.json'],
            original_episode_sha256=e['original_sha256']['episode.json'],scored_reward=r['reward'])
        for label in ['independent_review','report_review']:
            path=output/(label+'.json');value=dict(review,reviewer=label)
            if label=='report_review':value.update(independent_review_path=e['independent_review_path'],independent_review_sha256=e['independent_review_sha256'])
            self.write(path,value);e[label+'_path']=str(path);e[label+'_sha256']=self.sha(path)

    def held(self,f):self.assertIsNotNone(evaluation_for_scoring(f['path'],f['original'])['error'])
    def valid(self):return next(f for f in self.fixtures if f['key']!=WHITE_TARGET)
    def rebind(self,f):
        self.persist_report(f['output'],f['report']);e=f['entry'];e['report_sha256']=self.sha(f['output']/'evaluation.json');e['journal_sha256']=self.sha(f['output']/'evaluation-steps.jsonl');self.reviews(f);self.write(self.manifest_path,self.manifest)

    def test_lower_corrected_scores_release_and_originals_never_change(self):
        before={p:p.read_bytes()for f in self.fixtures for p in f['directory'].rglob('*')if p.is_file()}
        for f in self.fixtures:
            if f['key']==WHITE_TARGET:self.held(f)
            else:self.assertEqual(load_lookbook_correction(f['path'])['evaluation']['reward'],.3)
        self.assertEqual(before,{p:p.read_bytes()for p in before})

    def test_absent_or_invalid_authority_only_affects_exact_target_scope(self):
        self.manifest_path.unlink();self.pending_path.unlink();self.assertIsNone(load_lookbook_correction(self.valid()['path']))
        self.pending_path.write_text('invalid')
        for f in self.fixtures:self.held(f)
        for key in [('gpt-6-luna','knows.slides_39_Personal_Lookbook_PaintColors.1'),('gpt-6-luna','knows.slides_39_Personal_Lookbook_PaintColors.3')]:
            path=self.run/key[0]/key[1]/'episode.json';self.write(path,{'model':key[0],'task_id':key[1]})
            self.assertIsNone(evaluation_for_scoring(path,{'status':'completed','reward':.9})['error'])

    def test_pending_and_partial_publication_never_drop_holds(self):
        self.manifest_path.unlink()
        for f in self.fixtures:self.held(f)
        f=self.valid();self.manifest['corrections']=[f['entry']];self.write(self.manifest_path,self.manifest)
        for other in self.fixtures:
            if other is not f:self.held(other)
        self.assertIsNotNone(load_lookbook_correction(f['path']))

    def test_duplicate_missing_and_changed_authority_scope_fail_closed(self):
        for targets in [self.pending['targets'][:-1],[self.pending['targets'][0]]*len(TARGETS)]:
            self.write(self.pending_path,{**self.pending,'targets':targets})
            for f in self.fixtures:self.held(f)

    def test_original_and_archived_evidence_source_and_review_tamper_rejected(self):
        f=self.valid();paths=[f['directory']/'episode.json',f['directory']/'evaluation-steps.jsonl',f['archived']/'evaluation.json',
            Path(f['proof']['source_snapshots'][0]['path']),Path(f['proof']['independent_review_path']),f['output']/'evaluate.log',
            Path(f['entry']['independent_review_path'])]
        for p in paths:
            old=p.read_bytes();p.write_bytes(old+b' ')
            with self.subTest(path=str(p)):self.held(f)
            p.write_bytes(old)

    def test_rubric_name_id_max_and_finite_score_cannot_change(self):
        f=self.valid();old=deepcopy(f['report'])
        for field,value in [('name','new'),('step_id',999),('max_score',2),('score',float('nan'))]:
            f['report']=deepcopy(old);f['report']['detailed_report']['checkpoints'][0]['steps'][0][field]=value;self.rebind(f);self.held(f)

    def test_wrong_attempt_judge_source_and_reward_cannot_be_released(self):
        f=self.valid();old=deepcopy(f['report'])
        for field,value in [('evaluation_attempt_id',f['original']['evaluation_attempt_id']),('judge_model','other'),('evaluator_invoked',False),('reward',.8)]:
            f['report']=deepcopy(old);f['report'][field]=value;self.rebind(f);self.held(f)

    def test_keyed_journal_does_not_accept_duplicate_steps(self):
        f=self.valid();p=f['output']/'evaluation-steps.jsonl';rows=p.read_text().splitlines();rows[2]=rows[1];p.write_text('\n'.join(rows)+'\n')
        f['entry']['journal_sha256']=self.sha(p);self.reviews(f);self.write(self.manifest_path,self.manifest);self.held(f)

    def test_both_report_reviews_and_root_binding_required(self):
        f=self.valid()
        for label in ['report_review','independent_review']:
            p=Path(f['entry'][label+'_path']);old=json.loads(p.read_text())
            for field in ['scoring_valid','execution_errors_reviewed','no_auth_or_judge_failure','no_unresolved_evaluator_error','evaluate_log_sha256']:
                bad=dict(old);bad.pop(field);self.write(p,bad);f['entry'][label+'_sha256']=self.sha(p);self.write(self.manifest_path,self.manifest);self.held(f)
            self.write(p,old);f['entry'][label+'_sha256']=self.sha(p)

    def test_white_uniqueness_correction_cannot_release_independent_retrieval_hold(self):
        f=next(f for f in self.fixtures if f['key']==WHITE_TARGET)
        self.manifest['corrections'].append(f['entry']);self.write(self.manifest_path,self.manifest)
        with self.assertRaisesRegex(ValueError,'White image retrieval'):load_lookbook_correction(f['path'])
        self.held(f)

    def test_report_review_flags_require_booleans(self):
        f=self.valid()
        for field,value in [('reviewed',1),('model_replayed',0)]:
            self.reviews(f)
            p=Path(f['entry']['report_review_path']);review=json.loads(p.read_text())
            review[field]=value;self.write(p,review);f['entry']['report_review_sha256']=self.sha(p)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'exact root and independent'):
                _reviews(f['entry'],f['report'],self.archive)

    def test_unevaluated_uniqueness_cannot_release_but_output_duplicate_can(self):
        f=self.valid();original=deepcopy(f['report'])
        for changes in [{'details':'White: download failed'},{'category':'execution_error'},
                        {'category':'dependency_not_evaluated'}]:
            f['report']=deepcopy(original)
            f['report']['detailed_report']['checkpoints'][0]['steps'][-1].update(changes)
            self.rebind(f)
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'uniqueness remains unevaluated'):
                load_lookbook_correction(f['path'])
        f['report']=deepcopy(original)
        f['report']['detailed_report']['checkpoints'][0]['steps'][-1]['details']='White: duplicate'
        self.rebind(f);self.assertIsNotNone(load_lookbook_correction(f['path']))

    def test_white_exact_url_visual_review_and_original_cause_cannot_be_fabricated(self):
        f=next(f for f in self.fixtures if f['key']==WHITE_TARGET);p=Path(f['pending']['retrieval_hold_proof_path']);original=json.loads(p.read_text())
        for field,value in [('exact_url_sha256','0'*64),('original_http_cause_proven',True),('exact_browser_url_loaded',False)]:
            self.write(p,{**original,field:value});f['pending']['retrieval_hold_proof_sha256']=self.sha(p);self.write(self.pending_path,self.pending)
            with self.assertRaises(ValueError):load_lookbook_correction(self.valid()['path'])

    def test_candidate_cannot_escape_archive_or_replace_original_lineage(self):
        f=self.valid()
        with self.assertRaises(ValueError):load_lookbook_correction(f['path'],self.root/'candidate.json')
        f['entry']['source_proof_sha256']='0'*64;self.write(self.manifest_path,self.manifest);self.held(f)

    def test_white_requires_exact_successful_submitted_url_not_just_navigation(self):
        f=next(f for f in self.fixtures if f['key']==WHITE_TARGET);p=Path(f['pending']['retrieval_hold_proof_path']);proof=json.loads(p.read_text())
        action=Path(proof['evidence']['browser_actions']['path']);rows=[json.loads(line) for line in action.read_text().splitlines()]
        for replacement in [{**rows[1],'success':False},{**rows[1],'arguments':{'text':'https://images.example.test/different.jpg'}}]:
            action.write_text(json.dumps(rows[0])+'\n'+json.dumps(replacement)+'\n')
            proof['evidence']['browser_actions']['sha256']=self.sha(action);self.write(p,proof)
            f['pending']['retrieval_hold_proof_sha256']=self.sha(p);self.write(self.pending_path,self.pending)
            with self.assertRaisesRegex(ValueError,'not submitted'):load_lookbook_correction(self.valid()['path'])

if __name__=='__main__':unittest.main()
