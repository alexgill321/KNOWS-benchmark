"""Offline tests: separate capacity scope, immutable originals, no duplicate replay."""
from copy import deepcopy
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Exercise the exact staged candidates before installation; normal discovery
# after installation needs no environment override.
if os.environ.get('CODEX_POST_MAIN_STAGE'):
    import codex_benchmark
    codex_benchmark.__path__.insert(0, str(Path(os.environ['CODEX_POST_MAIN_STAGE'])/'codex_benchmark'))
from codex_benchmark import post_main_retry as retry
from codex_benchmark.browser import TOOLS
from codex_benchmark.config import MODELS
from codex_benchmark.runner import SMOKE_TASK
from codex_benchmark.parallel import validate_contract

ROOT=Path(__file__).resolve()
while not (ROOT/'benchmark.local.json').exists() and ROOT!=ROOT.parent:
    ROOT=ROOT.parent
LAUNCHER=ROOT/'codex-results/recovery/post-main-completion-20261005T213000Z/launch_capacity_retry.py'


class PostMainRetryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();self.parent=self.root/'codex-results/main'
        self.run=self.root/'codex-results/astra-retry';self.archive=self.root/'codex-results/recovery/new'
        self.archive.mkdir(parents=True);self.parent.mkdir()
        ids=list(retry.TASK_IDS)+[SMOKE_TASK]+[f'knows.fixture.{n}'for n in range(106)]
        self.tasks=[SimpleNamespace(task_id=t)for t in ids]
        self.parent_manifest={'models':list(MODELS),'efforts':{m:'max'for m in MODELS},'judge_model':'gemini-3.8-flash',
            'max_actions':150,'episode_timeout_seconds':1800,'seed':0,'benchmark_commit':'saved',
            'browser_tools_sha256':hashlib.sha256(json.dumps(TOOLS,sort_keys=True).encode()).hexdigest(),
            'repetitions':1,'tasks_per_model':110,'total_planned':330,'task_ids':ids}
        self.write(self.parent/'manifest.json',self.parent_manifest)
        self.config={k:deepcopy(self.parent_manifest[k])for k in ('judge_model','max_actions','episode_timeout_seconds','seed')}
        self.config.update(models=[retry.MODEL],efforts={retry.MODEL:'max'},benchmark_root=str(self.root),
                           evaluation_timeout_seconds=900,headless=True,channel='chrome')
        for model in MODELS:self.write(self.parent/model/SMOKE_TASK/'evaluation.json',{'status':'completed','has_execution_errors':False})
        self.write(self.parent/'gpt-6-luna'/'knows.other.1'/'episode.json',{'workspace_id':'other-main-workspace','thread_id':'other-main-thread','turn_id':'other-main-turn'})
        for i,task in enumerate(retry.TASK_IDS):
            d=self.parent/retry.MODEL/task
            e={'task_id':task,'model':retry.MODEL,'effort':'max','status':'infrastructure_error','workspace_id':f'old-w{i}',
               'thread_id':f'old-th{i}','turn_id':f'old-t{i}','benchmark_validity':{'status':'excluded_infrastructure'},'recovery':{'kind':'server_overloaded'}}
            for name in retry.ORIGINAL_FILES:self.write(d/name,e if name in ('episode.json','agent-result.json')else {'fixture':name})
        with sqlite3.connect(self.parent/'parallel.sqlite')as c:
            c.executescript('CREATE TABLE claims(ordinal INTEGER,model TEXT,task_id TEXT,status TEXT);CREATE TABLE cancelled_claims(ordinal INTEGER,model TEXT,task_id TEXT,status TEXT);CREATE TABLE metadata(key TEXT,value TEXT);')
            c.executemany('INSERT INTO claims VALUES(?,?,?,?)',[(i,retry.MODEL,f't{i}','done')for i in range(221)])
            c.executemany('INSERT INTO cancelled_claims VALUES(?,?,?,?)',[(i,'gpt-6.1-sol',f'c{i}','cancelled')for i in range(109)])
        luna=self.parent.parent/retry.LUNA_RETRY_DIR
        self.write(luna/'manifest.json',{'old':'manifest'})
        entries=[]
        for i,task in enumerate(retry.LUNA_VALID_TASKS):
            d=luna/'gpt-6-luna'/task
            for name in ('episode.json','agent-result.json','evaluation.json'):self.write(d/name,{'preserved':name,'reward':i/2})
            entries.append({'task_id':task,'benchmark_validity':{'status':'eligible'},'scored_reward':i/2,
                'evaluation_attempt_id':f'old-attempt{i}','evaluation_sha256':retry._sha(d/'evaluation.json')})
        self.write(self.parent.parent/'luna-infrastructure-rerun-reviewed-summary.json',{'per_task':entries})
        with sqlite3.connect(retry._registry(self.parent))as c:
            c.execute('CREATE TABLE reservations(plan_id TEXT,model TEXT,task_id TEXT,run_dir TEXT,manifest_sha256 TEXT,PRIMARY KEY(plan_id,model,task_id))')
            c.executemany('INSERT INTO reservations VALUES(?,?,?,?,?)', [('old','gpt-6-luna',f'luna{i}',str(luna),'old-digest')for i in range(3)])
        owner=self.archive/'owner.json'
        self.write(owner,{'owner_request':'Retry infrastructure failures after completion.','recorded_at':'2026-10-05T21:40:58+00:00','max_total_workers':2,'originals_preserved':True})
        self.auth=self.archive/'capacity-batch-authorization.json'
        self.write(self.auth,{'kind':retry.AUTH_KIND,'model':retry.MODEL,'task_ids':list(retry.TASK_IDS),
            'statement':'Retry infrastructure failures after completion.','requested_at':'2026-10-05T21:40:58+00:00',
            'max_total_workers':2,'no_main_resume':True,'owner_authorization_path':str(owner),'owner_authorization_sha256':retry._sha(owner)})
        self.source=self.archive/'capacity-source-validation.json'
        sources={}
        for name in ('post_main_retry','parallel','infrastructure_retry','worker_state','runner','app_server','browser','provision','server_overload_recovery','config','parallel_queue'):
            p=self.root/f'codex_benchmark/{name}.py';p.parent.mkdir(exist_ok=True);p.write_text(f'# source {name}\n');sources[str(p.relative_to(self.root))]=retry._sha(p)
        for rel in ('tests/test_codex_post_main_retry.py','codex-results/recovery/post-main-completion-20261005T213000Z/launch_capacity_retry.py'):
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# frozen fixture\n');sources[rel]=retry._sha(p)
        review=self.archive/'independent-review.json';self.write(review,{'approved':True,'source_sha256':sources})
        self.write(self.source,{'kind':'reviewed_post_main_retry_sources','tests_passed':True,'independent_review_passed':True,
                              'source_sha256':sources,'independent_review':retry._descriptor(review)})
        self.git=patch('codex_benchmark.parallel.subprocess.check_output',return_value='saved\n').start()
        self.diagnostic=patch('codex_benchmark.server_overload_recovery.is_reviewed_server_overload_diagnostic',return_value=True).start()
        self.addCleanup(patch.stopall)
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
    def read(self,path):return json.loads(path.read_text())
    def prepare(self):return retry.prepare_retry_manifest(self.config,self.run,self.parent,self.tasks,authorization_path=self.auth,validation_path=self.source)
    def validate(self):return retry.validate_retry_contract(self.config,self.run,self.tasks)
    def parent_bytes(self):return {str(p):p.read_bytes()for p in self.parent.rglob('*')if p.is_file()}
    def episode(self,task,**extra):
        value={'model':retry.MODEL,'task_id':task,'status':'running',**extra}
        p=self.run/retry.MODEL/task/'episode.json';self.write(p,value);return value

    def test_exact_three_scope_dispatch_and_originals_preserved(self):
        before=self.parent_bytes();old=retry._rows(retry._registry(self.parent))
        m=self.prepare()
        self.assertEqual(m['task_ids'],list(retry.TASK_IDS));self.assertEqual(m['total_planned'],3)
        self.assertEqual(self.validate(),m);self.assertEqual(validate_contract(self.config,self.run,self.tasks),m)
        self.assertEqual([x.task_id for x in retry.select_retry_tasks(self.config,self.run,self.tasks)],list(retry.TASK_IDS))
        self.assertEqual(before,self.parent_bytes());self.assertEqual([x.name for x in self.run.iterdir()],['manifest.json'])
        self.assertEqual([r for r in retry._rows(retry._registry(self.parent))if r[1]=='gpt-6-luna'],old)

    def test_authorization_exact_subset_and_owner_chain(self):
        original=self.read(self.auth)
        for change in ({'model':'gpt-6-luna'},{'task_ids':list(retry.TASK_IDS)+['knows.other.1']},
                       {'task_ids':[retry.TASK_IDS[0]]*3},{'task_ids':[{},None,0]}, {'statement':'different approval'},
                       {'requested_at':'2026-10-06T21:40:58+00:00'},{'owner_authorization_sha256':'bad'},
                       {'no_main_resume':False}):
            self.write(self.auth,{**original,**change})
            with self.subTest(change=change),self.assertRaises((ValueError,KeyError)):self.prepare()
            self.assertFalse(self.run.exists())
        self.write(self.auth,original)
        owner=Path(original['owner_authorization_path']);owner.write_text(owner.read_text()+' ')
        with self.assertRaises(ValueError):self.prepare()

    def test_original_capacity_proof_and_identity_required(self):
        self.diagnostic.return_value=False
        with self.assertRaises(ValueError):self.prepare()
        self.diagnostic.return_value=True
        p=self.parent/retry.MODEL/retry.TASK_IDS[0]/'episode.json';v=self.read(p)
        for change in ({'status':'completed'},{'recovery':{'kind':'auth_interrupted'}},{'benchmark_validity':{}},{'effort':'high'}):
            self.write(p,{**v,**change})
            with self.subTest(change=change),self.assertRaises(ValueError):self.prepare()
        self.assertFalse(self.run.exists())

    def test_settings_and_normal_worker_runtime_compatibility(self):
        self.prepare()
        worker={**self.config,'storage_state':'/private/worker-1/browser-state.json','token_path':'/private/worker-1/owner-token.json',
                'config_path':'/private/worker-1/config.json','worker_id':'worker-1','output_dir':str(self.run)}
        self.assertEqual(self.validate(),retry.validate_retry_contract(worker,self.run,self.tasks))
        for change in ({'max_actions':149},{'episode_timeout_seconds':1801},{'evaluation_timeout_seconds':901},
                       {'models':['gpt-6-luna']},{'efforts':{retry.MODEL:'ultra'}},{'judge_model':'other'},
                       {'headless':False},{'channel':'chromium'},{'seed':1}):
            with self.subTest(change=change),self.assertRaises((ValueError,RuntimeError)):
                retry.validate_retry_contract({**self.config,**change},self.run,self.tasks)

    def test_source_review_and_live_source_hashes_fail_closed(self):
        value=self.read(self.source)
        for change in ({'tests_passed':False},{'independent_review_passed':1},{'source_sha256':{}},
                       {'independent_review':{'path':str(self.archive/'absent'),'sha256':'bad'}}):
            self.write(self.source,{**value,**change})
            with self.subTest(change=change),self.assertRaises((ValueError,OSError)):self.prepare()
        self.write(self.source,value);self.prepare()
        p=self.root/'codex_benchmark/runner.py';p.write_text('tamper')
        with self.assertRaisesRegex(ValueError,'source changed'):self.validate()

    def test_queue_cancel_and_valid_luna_grade_tampering_rejected(self):
        self.prepare()
        for statement,inverse in (("UPDATE claims SET status='pending' WHERE ordinal=0","UPDATE claims SET status='done' WHERE ordinal=0"),
                                  ("INSERT INTO metadata VALUES('pause_reason','paused')","DELETE FROM metadata"),
                                  ("UPDATE cancelled_claims SET model='gpt-6-astra' WHERE ordinal=0","UPDATE cancelled_claims SET model='gpt-6.1-sol' WHERE ordinal=0")):
            with sqlite3.connect(self.parent/'parallel.sqlite')as c:c.execute(statement)
            with self.subTest(statement=statement),self.assertRaises(ValueError):self.validate()
            with sqlite3.connect(self.parent/'parallel.sqlite')as c:c.execute(inverse)
        p=self.parent.parent/retry.LUNA_RETRY_DIR/'gpt-6-luna'/retry.LUNA_VALID_TASKS[0]/'evaluation.json';p.write_text('{}')
        with self.assertRaises(ValueError):self.validate()

    def test_reservation_copy_manifest_and_changed_authorization_cannot_replay(self):
        self.prepare();before=retry._rows(retry._registry(self.parent));oldpath=self.run
        self.run=self.root/'codex-results/second'
        auth=self.read(self.auth);owner=Path(auth['owner_authorization_path']);v=self.read(owner)
        v['owner_request']='Same task, differently worded owner authorization';self.write(owner,v)
        auth.update(statement=v['owner_request'],owner_authorization_sha256=retry._sha(owner));self.write(self.auth,auth)
        with self.assertRaises(ValueError):self.prepare()
        self.assertEqual(retry._rows(retry._registry(self.parent)),before)
        self.assertFalse((self.run/'manifest.json').exists())
        # A copied old manifest cannot authorize a new output path either.
        self.run.mkdir(exist_ok=True);(self.run/'manifest.json').write_bytes((oldpath/'manifest.json').read_bytes())
        with self.assertRaises(ValueError):self.validate()

    def test_atomic_transaction_detects_race_and_rolls_back(self):
        real=retry._validate
        def race(*args,**kwargs):
            result=real(*args,**kwargs)
            with sqlite3.connect(retry._registry(self.parent))as c:
                c.execute('INSERT INTO reservations VALUES(?,?,?,?,?)',('racing',retry.MODEL,retry.TASK_IDS[0],'other','hash'))
            return result
        with patch.object(retry,'_validate',side_effect=race),self.assertRaisesRegex(ValueError,'changed during preparation'):self.prepare()
        self.assertFalse((self.run/'manifest.json').exists())
        self.assertEqual(len(retry._rows(retry._registry(self.parent))),4)

    def test_manifest_install_failure_rolls_back_all_new_reservations(self):
        before=retry._rows(retry._registry(self.parent))
        with patch.object(retry.os,'link',side_effect=OSError('install failed')),self.assertRaises(OSError):self.prepare()
        self.assertEqual(retry._rows(retry._registry(self.parent)),before);self.assertFalse((self.run/'manifest.json').exists())
        self.prepare();self.validate()

    def test_wrong_or_reused_workspace_thread_and_cross_retry_duplicates(self):
        self.prepare();task=retry.TASK_IDS[0]
        for values in ({'workspace_id':'old-w0'},{'thread_id':'other-main-thread'}, {'turn_id':'old-t1'},
                       {'workspace_id':[]},{'workspace_id':''}):
            self.episode(task,**values)
            with self.subTest(values=values),self.assertRaises(ValueError):self.validate()
        self.episode(task,workspace_id='fresh1',thread_id='newthread1')
        self.episode(retry.TASK_IDS[1],workspace_id='fresh1',thread_id='newthread2')
        with self.assertRaises(ValueError):self.validate()
        self.episode(retry.TASK_IDS[1],workspace_id='fresh2',thread_id='newthread1')
        with self.assertRaises(ValueError):self.validate()

    def test_fresh_server_rejects_old_thread_before_any_turn_request(self):
        self.prepare();task=retry.TASK_IDS[0]
        episode=self.episode(task,workspace_id='fresh-w')
        class Server:
            def __init__(self,thread):self.thread=thread;self.calls=[]
            async def request(self,method,params,*args,**kwargs):
                self.calls.append((method,params));return {'thread':{'id':self.thread}}
        with patch('codex_benchmark.knows.discover_tasks',return_value=self.tasks):
            cls=retry.fresh_episode_server(self.config,self.run,retry.MODEL,task,episode,Server)
        async def attempt(thread):
            server=cls(thread)
            try:
                await server.request('thread/start',{'ephemeral':True})
                await server.request('turn/start',{'prompt':'unchanged'})
            except ValueError:pass
            return server.calls
        self.assertEqual(asyncio.run(attempt('old-th1')),[('thread/start',{'ephemeral':True})])
        self.assertEqual(len(asyncio.run(attempt('new-thread'))),2)
        episode['workspace_id']='other-main-workspace'
        with patch('codex_benchmark.knows.discover_tasks',return_value=self.tasks),self.assertRaises(ValueError):
            retry.fresh_episode_server(self.config,self.run,retry.MODEL,task,episode,Server)

    def test_ordinary_no_manifest_or_main_keeps_exact_server_type(self):
        class Server:pass
        self.assertIs(retry.fresh_episode_server({},self.root/'ordinary','any','any',{},Server),Server)
        self.assertIs(retry.fresh_episode_server({},self.parent,'any','any',{},Server),Server)

    def test_paths_and_existing_output_refused(self):
        original=self.run
        for target in (self.parent,self.parent/'nested',self.parent.parent):
            self.run=target
            with self.subTest(target=target),self.assertRaises(ValueError):self.prepare()
        self.run=original;self.run.mkdir();(self.run/'existing').write_text('do not overwrite')
        with self.assertRaises(ValueError):self.prepare()

    def test_existing_terminal_queue_claim_not_replayed(self):
        from codex_benchmark.parallel_queue import initialize_queue,claim_next
        self.prepare();done=retry.TASK_IDS[0]
        self.episode(done,status='completed',workspace_id='fresh',thread_id='fresh-thread',turn_id='fresh-turn')
        self.write(self.run/retry.MODEL/done/'evaluation.json',{'status':'completed','reward':0.0})
        tasks=retry.select_retry_tasks(self.config,self.run,self.tasks)
        initialize_queue(self.run/'parallel.sqlite',[(t.task_id,retry.MODEL)for t in tasks],[retry.MODEL],self.run)
        claimed=claim_next(self.run/'parallel.sqlite','worker-1',123)
        self.assertNotEqual(claimed['task_id'],done)

    def test_prepare_then_execute_once_uses_only_new_run_and_same_locked_fd(self):
        spec=importlib.util.spec_from_file_location('capacity_launch_fixture',LAUNCHER);launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
        for name,value in [('ROOT',self.root),('ARCHIVE',self.archive),('PARENT',self.parent),('RUN',self.run),('STATUS',self.root/'status.json')]:setattr(launch,name,value)
        for n in (1,2):self.write(self.root/f'auth-data/parallel-workers/worker-{n}/browser-state.json',{'cookies':[],'origins':[]})
        self.write(launch.STATUS,{'phase':'preparing'})
        calls=[]
        def coordinate(config,run,count,fd):
            import fcntl
            self.assertEqual(run,self.run);self.assertEqual(count,2);self.assertEqual(config,self.config)
            with(self.root/'.codex-benchmark.lock').open('a')as other:
                with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertGreaterEqual(fd,0);calls.append(run);return 0
        with patch.object(launch,'load_config',return_value=self.config),patch.object(launch,'discover_tasks',return_value=self.tasks),\
             patch.object(launch,'verify_stopped'),patch.object(launch,'snapshot',return_value={'hashes':{}}),\
             patch.object(launch,'verify_preservation'),patch.object(launch,'coordinate',side_effect=coordinate):
            self.assertEqual(launch.execute(self.root/'config',execute_batch=False),0);self.assertEqual(calls,[])
            self.assertEqual(launch.execute(self.root/'config',execute_batch=True),0);self.assertEqual(calls,[self.run])
            with self.assertRaisesRegex(RuntimeError,'already attempted'):launch.execute(self.root/'config',execute_batch=True)
        self.assertFalse(launch.read(launch.STATUS)['main_resumed']);self.assertEqual(len(retry._rows(retry._registry(self.parent))),6)

    def test_scoring_guard_rejects_malformed_declared_hold_and_overlay_loss(self):
        spec=importlib.util.spec_from_file_location('capacity_scoring_fixture',LAUNCHER);launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
        launch.PARENT=self.parent
        # Exactly221 original files, with fixed six intentional hold identities.
        for p in list(self.parent.glob('*/*/episode.json')):p.unlink()
        ids=list(launch.EXPECTED_HOLDS)+[(retry.MODEL,f'knows.fixture.{i}')for i in range(215)]
        for model,task in ids:self.write(self.parent/model/task/'episode.json',{});self.write(self.parent/model/task/'evaluation.json',{})
        def score(path,original):
            key=(path.parent.parent.name,path.parent.name)
            return {'evaluation':original,'error':launch.EXPECTED_HOLDS.get(key),
                    'provenance':{'valid':True}if key in ids[6:22]else None}
        with patch('codex_benchmark.evaluation_corrections.evaluation_for_scoring',side_effect=score):
            self.assertEqual(len(launch.scoring_state()),221)
        def broken(path,original):
            result=score(path,original)
            if result['error']:result['error']='ValueError: authority hash mismatch'
            return result
        with patch('codex_benchmark.evaluation_corrections.evaluation_for_scoring',side_effect=broken),self.assertRaises(RuntimeError):launch.scoring_state()

if __name__=='__main__':unittest.main()
