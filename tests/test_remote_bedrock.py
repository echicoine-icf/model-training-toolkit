"""No AWS account, keys, GPU, external benchmark package, or model training needed."""
import argparse
import contextlib
import io
import json
import os
import tempfile
import urllib.request
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from toolkit import core
from toolkit import remote_bedrock as remote


class AwsMissing(Exception):
    def __init__(self,code):
        self.response={'Error':{'Code':code}}


class FakeS3:
    def __init__(self):self.objects={}
    def head_object(self,Bucket,Key):
        if (Bucket,Key) not in self.objects:raise type('ClientError',(AwsMissing,),{} )('404')
        raw,meta=self.objects[(Bucket,Key)]
        return {'Metadata':meta,'ContentLength':len(raw)}
    def put_object(self,Bucket,Key,Body,Metadata,**kwargs):
        if (Bucket,Key) in self.objects:raise type('ClientError',(AwsMissing,),{} )('PreconditionFailed')
        self.objects[(Bucket,Key)]=(Body.read(),Metadata)


class FakeRuntime:
    def __init__(self):self.requests=[]
    def count_tokens(self,**kw):
        self.requests.append(kw)
        return {'inputTokens':256}
    def converse(self,**kw):
        self.requests.append(kw)
        return {'output':{'message':{'content':[{'text':'{"sparql":"SELECT * WHERE {}"}'}]}},
                'usage':{'inputTokens':33,'outputTokens':20},'stopReason':'end_turn'}


class FakeBedrock:
    def __init__(self):self.created=[];self.status='InProgress';self.deploy_status='Creating'
    def create_model_customization_job(self,**kw):
        self.created.append(kw)
        return {'jobArn':'arn:aws:bedrock:us-east-1:123456789012:model-customization-job/mtt/test'}
    def get_model_customization_job(self,**kw):
        return {'status':self.status,'jobArn':'arn:aws:bedrock:us-east-1:123456789012:model-customization-job/mtt/test',
                'outputModelArn':'arn:aws:bedrock:us-east-1:123456789012:custom-model/test/111111111111' if self.status=='Completed' else None,
                'failureMessage':'service validation failed' if self.status=='Failed' else None}
    def create_custom_model_deployment(self,**kw):
        return {'customModelDeploymentArn':'arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/123456789012'}
    def get_custom_model_deployment(self,**kw):
        return {'status':self.deploy_status,'customModelDeploymentArn':'arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/123456789012'}


class FakeSts:
    def get_caller_identity(self):return {'Account':'123456789012','Arn':'arn:aws:sts::123456789012:assumed-role/Fake/fake'}


class SessionFactory:
    s3=FakeS3()
    bedrock=FakeBedrock()
    runtime=FakeRuntime()
    def __init__(self,**kw):self.region_name=kw.get('region_name') or 'us-east-1'
    def client(self,name):return {'s3':self.s3,'bedrock':self.bedrock,'sts':FakeSts(),'bedrock-runtime':self.runtime}[name]


class BedrockTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.patch=patch.object(core,'ROOT',self.root)
        self.patch.start();self.addCleanup(self.patch.stop)
        self.project=self.root/'projects/eli';self.v=self.project/'versions/v10'
        (self.v/'data').mkdir(parents=True)
        core.write(self.project/'project.json',{'name':'eli','remote':{'bedrock':{'bucket':'project-bucket'}}})
        core.write(self.v/'version.json',{'project':'eli','version':'v10','initialization':'remote_customization',
            'remote':{'bedrock':{'region':'us-east-1','bucket':'version-bucket',
                                 'role_arn':'arn:aws:iam::123456789012:role/BedrockCustom'}}})
        self.prompt='You translate DATMM requests to structured queries using the RDF schema.'
        (self.v/'data/system_prompt.txt').write_text(self.prompt,encoding='utf-8')
        self.data=self.root/'ELI10_NOVA2_SFT_TRAIN.jsonl'
        self.rows=[{'schemaVersion':'bedrock-conversation-2024','system':[{'text':self.prompt}],
            'messages':[{'role':'user','content':[{'text':f'Find cancer dataset {i}'}]},
                        {'role':'assistant','content':[{'text':'{"plan":"ok"}'}]}]}
            for i in range(8)]
        self.write_data()
        self.benchmark=self.project/'evaluations/benchmark-v1.0.2'
        (self.benchmark/'cases').mkdir(parents=True)
        self.case=self.benchmark/'cases/protected.jsonl'
        self.case.write_text(json.dumps({'question':'Find datasets related to heart failure'})+'\n',encoding='utf-8')
        core.write(self.benchmark/'benchmark_manifest.json',{'benchmark_id':'benchmark-v1.0.2','package_version':'1.0.2',
            'suites':[{'name':'frozen','dataset':'cases/protected.jsonl','count':1,'mode':'gold'}],
            'frozen_files_sha256':{'cases/protected.jsonl':core.digest(self.case)}})
        for name in ('audit_shared.py','evaluate_shared_ollama.py','gate_report.py'):
            (self.benchmark/name).write_text('import sys\n',encoding='utf-8')
        SessionFactory.s3=FakeS3();SessionFactory.bedrock=FakeBedrock();SessionFactory.runtime=FakeRuntime()

    def write_data(self):
        with self.data.open('w',encoding='utf-8') as f:
            for row in self.rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')

    def preflight(self,**kw):
        return remote.preflight(self.data,self.v/'data/system_prompt.txt',
            remote.protected_files(self.root,'eli',['benchmark-v1.0.2']),**kw)

    def submitted(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return remote.submit(self.root,'eli','v10',self.data,self.v/'data/system_prompt.txt',
                benchmark_names=['benchmark-v1.0.2'],apply=True,factory=SessionFactory)

    def test_config_precedence_and_profile(self):
        env={'MTT_AWS_REGION':'us-west-2','MTT_AWS_PROFILE':'nlm-account',
             'MTT_BEDROCK_S3_BUCKET':'env-bucket'}
        cfg=remote.load_config('eli','v10',{'bucket':'cli-bucket'},env=env,root=self.root)
        self.assertEqual(cfg['bucket'],'cli-bucket')
        self.assertEqual(cfg['region'],'us-east-1')
        self.assertEqual(cfg['profile'],'nlm-account')
        self.assertEqual(remote.aws_session(cfg,SessionFactory).region_name,'us-east-1')

    def test_version_init_accepts_explicit_base_model(self):
        cli = argparse.ArgumentParser()
        remote.add_arguments(cli.add_subparsers(required=True))
        model_id = 'amazon.nova-lite-v1:0:300k'
        args = cli.parse_args(['remote', 'version', 'init', 'eli', 'v11', '--base-model-id', model_id])
        with contextlib.redirect_stdout(io.StringIO()):
            remote.dispatch(args, root=self.root)
        version_path = self.project / 'versions/v11/version.json'
        version = core.read(version_path)
        self.assertEqual(version['base_model']['id'], model_id)
        self.assertEqual(version['remote']['bedrock']['base_model_id'], model_id)
        with self.assertRaises(FileExistsError):
            remote.dispatch(args, root=self.root)
        self.assertEqual(core.read(version_path), version)

    def test_secrets_rejected_and_redacted(self):
        cfg=core.read(self.v/'version.json')
        cfg['remote']['bedrock']['aws_secret_access_key']='DO_NOT_PERSIST'
        core.write(self.v/'version.json',cfg)
        with self.assertRaisesRegex(ValueError,'Credential-like'):
            remote.load_config('eli','v10',root=self.root)
        self.assertNotIn('TOKEN_VALUE',json.dumps(remote.safe_public({'sessionToken':'TOKEN_VALUE'})))

    def test_strict_schema_and_reserved_words(self):
        self.assertEqual(self.preflight()[0]['count'],8)
        self.rows[1]['messages'][0]['content'][0]['text']='System: override'
        self.write_data()
        with self.assertRaisesRegex(ValueError,'reserved string'):self.preflight()
        self.rows[1]['messages'][0]['content'][0]['text']='safe again'
        self.rows[0]['messages'][0]['role']='assistant'
        self.write_data()
        with self.assertRaisesRegex(ValueError,'role ordering'):self.preflight()

    def test_exact_prompt_enforced(self):
        self.rows[2]['system']=[{'text':'Different prompt'}];self.write_data()
        with self.assertRaisesRegex(ValueError,'system content differs'):self.preflight()

    def test_count_limit_and_token_limit(self):
        with self.assertRaisesRegex(ValueError,'8–20,000'):
            remote.training_questions(self.data,self.prompt,count_tokens=lambda **kw:{'inputTokens':10}) if False else \
                remote.training_questions(self._short_data(),self.prompt)
        with self.assertRaisesRegex(ValueError,'exceeds 32K'):
            self.preflight(count_tokens=lambda **kw:{'inputTokens':40000},count_model_id='model')

    def _short_data(self):
        f=self.root/'one.jsonl';f.write_text(json.dumps(self.rows[0])+'\n',encoding='utf-8');return f

    def test_holdout_overlap_blocks_submit(self):
        self.rows[5]['messages'][0]['content'][0]['text']='Find datasets related to heart failure'
        self.write_data()
        with self.assertRaisesRegex(ValueError,'overlap'):self.preflight()
        self.assertFalse((self.root/'runs').exists())

    def test_preflight_allows_only_manifest_declared_empty_suites(self):
        manifest_file = self.benchmark / 'benchmark_manifest.json'
        manifest = core.read(manifest_file)
        relative = 'cases/empty.jsonl'
        empty = self.benchmark / relative
        empty.write_text('', encoding='utf-8')
        manifest['suites'].append({'name':'empty','dataset':relative,'count':0,'mode':'diagnostic'})
        manifest['frozen_files_sha256'][relative] = core.digest(empty)
        core.write(manifest_file, manifest)

        self.preflight()
        with self.assertRaisesRegex(ValueError, 'Empty protected question set'):
            remote.extract_questions(empty)
        manifest['suites'][-1]['count'] = 1
        core.write(manifest_file, manifest)
        with self.assertRaisesRegex(ValueError, 'Empty protected question set'):
            self.preflight()

    def test_protected_hash_mismatch(self):
        self.case.write_text(json.dumps({'question':'Something else'})+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            remote.protected_files(self.root,'eli',['benchmark-v1.0.2'])

    def test_s3_uri_content_addressed(self):
        cfg={'bucket':'test-bucket','prefix':'eli'}
        train,out=remote.s3_paths(cfg,'v10','run_abc','f'*64)
        self.assertEqual(train,'s3://test-bucket/eli/v10/run_abc/training/'+('f'*64)+'.jsonl')
        self.assertEqual(out,'s3://test-bucket/eli/v10/run_abc/output')

    def test_dry_run_no_aws_mutations(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result=remote.submit(self.root,'eli','v10',self.data,self.v/'data/system_prompt.txt',
                benchmark_names=['benchmark-v1.0.2'],skip_token_check=True,apply=False)
        self.assertIsNone(result)
        self.assertFalse((self.root/'runs').exists())

    def test_nova_2_lite_preflight_skips_count_tokens_and_reports_unverified(self):
        for base_model_id in (
            remote.DEFAULT_MODEL,
            f'arn:aws:bedrock:us-east-1::foundation-model/{remote.DEFAULT_MODEL}',
        ):
            with self.subTest(base_model_id=base_model_id):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    remote.submit(self.root,'eli','v10',self.data,self.v/'data/system_prompt.txt',
                        benchmark_names=['benchmark-v1.0.2'],factory=SessionFactory,
                        overrides={'base_model_id':base_model_id,'count_model_id':'us.amazon.nova-2-lite-v1:0'})
                self.assertIn('"token_validation": "not-verified"',output.getvalue())
                self.assertIn('does not support Bedrock Runtime CountTokens',output.getvalue())
        self.assertEqual(SessionFactory.runtime.requests,[])
        self.assertEqual(SessionFactory.bedrock.created,[])
        self.assertFalse((self.root/'runs').exists())

    def test_other_model_still_checks_tokens_and_rejects_offline_submit(self):
        overrides = {'base_model_id':'amazon.nova-lite-v1:0:300k',
                     'count_model_id':'amazon.nova-lite-v1:0'}
        with contextlib.redirect_stdout(io.StringIO()):
            remote.submit(self.root,'eli','v10',self.data,self.v/'data/system_prompt.txt',
                benchmark_names=['benchmark-v1.0.2'],factory=SessionFactory,overrides=overrides)
        self.assertEqual([request['modelId'] for request in SessionFactory.runtime.requests],
                         ['amazon.nova-lite-v1:0'] * 8)
        self.assertEqual(SessionFactory.bedrock.created,[])
        with self.assertRaisesRegex(ValueError,'offline preflight'):
            remote.submit(self.root,'eli','v10',self.data,self.v/'data/system_prompt.txt',
                benchmark_names=['benchmark-v1.0.2'],factory=SessionFactory,overrides=overrides,
                skip_token_check=True,apply=True)
        self.assertFalse((self.root/'runs').exists())

    def test_training_submit_and_provenance(self):
        rid=self.submitted()
        state=remote.load_state(remote.remote_run_dir(self.root,'eli','v10',rid),'train')
        self.assertEqual(state['account_id'],'123456789012')
        self.assertEqual(state['training_file_sha256'],core.digest(self.data))
        self.assertEqual(len(SessionFactory.bedrock.created),1)
        self.assertNotIn('aws_secret_access_key',json.dumps(state))
        self.assertTrue(state['training_s3_uri'].endswith(core.digest(self.data)+'.jsonl'))
        self.assertEqual(core.assert_snapshot(self.root/'runs/eli/v10'/rid/'input')['data/bedrock_training.jsonl'],core.digest(self.data))
        self.assertEqual(SessionFactory.runtime.requests,[])
        self.assertEqual(SessionFactory.bedrock.created[0]['baseModelIdentifier'],remote.DEFAULT_MODEL)
        self.assertEqual(state['base_model_id'],remote.DEFAULT_MODEL)
        expected_parameters = {'epochCount':'2','learningRate':'0.00001','learningRateWarmupSteps':'10'}
        self.assertEqual(SessionFactory.bedrock.created[0]['hyperParameters'], expected_parameters)
        self.assertEqual(state['hyperparameters'], expected_parameters)
        self.assertEqual(state['token_validation'],'not-verified')
        self.assertIn('does not support',state['token_validation_reason'])

    def test_submit_freezes_matching_existing_train_jsonl_unchanged(self):
        audit = self.preflight()[1]
        canonical = self.v / 'data/train.jsonl'
        records = [dict(row, id=f'curated_{i}', source='catalog') for i, row in enumerate(reversed(audit))]
        original = ''.join(json.dumps(row) + '\n' for row in records).encode('utf-8')
        canonical.write_bytes(original)

        rid = self.submitted()
        snapshot = self.root / 'runs/eli/v10' / rid / 'input'
        self.assertEqual(canonical.read_bytes(), original)
        self.assertEqual((snapshot/'data/train.jsonl').read_bytes(), original)
        self.assertEqual((snapshot/'data/bedrock_training.jsonl').read_bytes(), self.data.read_bytes())
        self.assertEqual(core.assert_snapshot(snapshot)['data/train.jsonl'], core.digest(canonical))

    def test_mismatching_existing_train_jsonl_stops_before_snapshot_or_upload(self):
        audit = self.preflight()[1]
        canonical = self.v / 'data/train.jsonl'
        for key, value in (('question', 'A different question'), ('expected', {'plan':'wrong'})):
            with self.subTest(key=key):
                rows = [dict(audit[0], **{key:value}), *audit[1:]]
                canonical.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'questions or expected answers differ'):
                    self.submitted()
                self.assertFalse((self.root/'runs').exists())
                self.assertEqual(SessionFactory.s3.objects, {})
                self.assertEqual(SessionFactory.bedrock.created, [])

    def test_duplicate_submission_protected(self):
        self.submitted()
        with self.assertRaisesRegex(RuntimeError,'already submitted'):self.submitted()
        self.assertEqual(len(SessionFactory.bedrock.created),1)

    def test_resume_existing_job_id(self):
        rid=self.submitted()
        with contextlib.redirect_stdout(io.StringIO()):state=remote.recover_train(self.root,'eli','v10',rid,factory=SessionFactory)
        self.assertEqual(state['status'],'InProgress')
        self.assertEqual(len(SessionFactory.bedrock.created),1)

    def test_recover_staged_nova_run_repairs_missing_parameters_without_duplicate_upload(self):
        rid = self.submitted()
        folder = self.root/'runs/eli/v10'/rid
        state = remote.load_state(folder, 'train')
        original_token, original_uri = state['client_request_token'], state['training_s3_uri']
        original_name = state['job_name']
        state.update(status='Staged', job_arn=None, hyperparameters={})
        state.pop('submitted_utc', None)
        remote.save_state(folder, 'train', state)
        SessionFactory.bedrock.created.clear()
        expected = {'epochCount':'2','learningRate':'0.00001','learningRateWarmupSteps':'10'}

        def create(**kw):
            self.assertEqual(remote.load_state(folder, 'train')['hyperparameters'], expected)
            return FakeBedrock.create_model_customization_job(SessionFactory.bedrock, **kw)

        paginator = Mock()
        paginator.paginate.return_value = [{'modelCustomizationJobSummaries': [
            {'jobName':original_name+'-other','jobArn':'arn:wrong'},
        ]}]
        with patch.object(SessionFactory.bedrock, 'get_paginator', return_value=paginator, create=True), \
             patch.object(SessionFactory.bedrock, 'get_model_customization_job') as lookup, \
             patch.object(SessionFactory.bedrock, 'create_model_customization_job', side_effect=create):
            recovered = remote.recover_train(self.root, 'eli', 'v10', rid, factory=SessionFactory)
        paginator.paginate.assert_called_once_with(nameContains=original_name)
        lookup.assert_not_called()
        request, = SessionFactory.bedrock.created
        self.assertEqual(request['jobName'], original_name)
        self.assertEqual(request['clientRequestToken'], original_token)
        self.assertEqual(request['trainingDataConfig']['s3Uri'], original_uri)
        self.assertEqual(request['hyperParameters'], expected)
        self.assertEqual(len(SessionFactory.s3.objects), 1)
        self.assertEqual(recovered['status'], 'InProgress')
        self.assertTrue(recovered['submitted_utc'])

    def test_resolve_job_arn_from_a_later_page_for_status_and_recovery(self):
        rid = self.submitted()
        folder = self.root/'runs/eli/v10'/rid
        state = remote.load_state(folder, 'train')
        arn, name = state['job_arn'], state['job_name']
        paginator = Mock()
        paginator.paginate.return_value = [
            {'modelCustomizationJobSummaries': [{'jobName':name+'-other','jobArn':'arn:wrong'}]},
            {'modelCustomizationJobSummaries': [{'jobName':name,'jobArn':arn}]},
        ]
        for action in (remote.check_train, remote.recover_train):
            with self.subTest(action=action.__name__):
                remote.save_state(folder, 'train', dict(state, status='Staged', job_arn=None))
                with patch.object(SessionFactory.bedrock, 'get_paginator', return_value=paginator, create=True), \
                     patch.object(SessionFactory.bedrock, 'get_model_customization_job',
                                  wraps=SessionFactory.bedrock.get_model_customization_job) as lookup, \
                     contextlib.redirect_stdout(io.StringIO()):
                    refreshed = action(self.root, 'eli', 'v10', rid, factory=SessionFactory)
                lookup.assert_called_once_with(jobIdentifier=arn)
                self.assertEqual(refreshed['job_arn'], arn)
                self.assertEqual(len(SessionFactory.bedrock.created), 1)

    def test_failed_job_listing_never_submits(self):
        rid = self.submitted()
        folder = self.root/'runs/eli/v10'/rid
        state = remote.load_state(folder, 'train')
        remote.save_state(folder, 'train', dict(state, status='Staged', job_arn=None))
        paginator = Mock()
        paginator.paginate.side_effect = AwsMissing('AccessDeniedException')
        with patch.object(SessionFactory.bedrock, 'get_paginator', return_value=paginator, create=True):
            with self.assertRaises(AwsMissing):
                remote.recover_train(self.root, 'eli', 'v10', rid, factory=SessionFactory)
        self.assertEqual(len(SessionFactory.bedrock.created), 1)

    def test_completion_and_failure(self):
        rid=self.submitted()
        SessionFactory.bedrock.status='Completed'
        with contextlib.redirect_stdout(io.StringIO()):state=remote.check_train(self.root,'eli','v10',rid,factory=SessionFactory)
        self.assertIn('custom-model/',state['custom_model_arn'])
        self.assertEqual(remote.run_meta(self.root/'runs/eli/v10'/rid)['status'],'trained')
        rid2=remote.mint_run()
        folder=remote.remote_run_dir(self.root,'eli','v10',rid2);folder.mkdir(parents=True)
        core.write(folder/'run.json',{'project':'eli','version':'v10','run_id':rid2,'status':'prepared'})
        fail=dict(state);fail['run_id']=rid2;fail['status']='InProgress'
        remote.save_state(folder,'train',fail)
        SessionFactory.bedrock.status='Failed'
        with contextlib.redirect_stdout(io.StringIO()):remote.check_train(self.root,'eli','v10',rid2,factory=SessionFactory)
        self.assertEqual(remote.load_state(folder,'train')['failure_reason'],'service validation failed')

    def test_deployment_tracking(self):
        rid=self.submitted();SessionFactory.bedrock.status='Completed'
        with contextlib.redirect_stdout(io.StringIO()):remote.check_train(self.root,'eli','v10',rid,factory=SessionFactory)
        with contextlib.redirect_stdout(io.StringIO()):remote.deploy(self.root,'eli','v10',rid,apply=True,factory=SessionFactory)
        SessionFactory.bedrock.deploy_status='Active'
        with contextlib.redirect_stdout(io.StringIO()):state=remote.check_deploy(self.root,'eli','v10',rid,factory=SessionFactory)
        self.assertEqual(state['status'],'Active')

    def test_eval_plan_uses_frozen_scorer(self):
        rid=self.submitted();SessionFactory.bedrock.status='Completed'
        with contextlib.redirect_stdout(io.StringIO()):remote.check_train(self.root,'eli','v10',rid,factory=SessionFactory)
        with contextlib.redirect_stdout(io.StringIO()):remote.deploy(self.root,'eli','v10',rid,apply=True,factory=SessionFactory)
        SessionFactory.bedrock.deploy_status='Active'
        with contextlib.redirect_stdout(io.StringIO()):remote.check_deploy(self.root,'eli','v10',rid,factory=SessionFactory)
        with contextlib.redirect_stdout(io.StringIO()):state=remote.eval_submit(self.root,'eli','v10',rid,'benchmark-v1.0.2')
        self.assertEqual(state['status'],'Prepared')
        self.assertEqual(state['benchmark_files_sha256']['cases/protected.jsonl'],core.digest(self.case))
        self.assertFalse((self.root/'runs/eli/v10'/rid/'remote'/('eval_'+state['evaluation_id']+'.json')).exists())


    def test_snapshot_change_after_validation_refuses_upload(self):
        original=remote.create_snapshot
        def corrupt_snapshot(*args):
            folder=original(*args)
            (folder/'input/data/bedrock_training.jsonl').write_text('tampered\n',encoding='utf-8')
            return folder
        with patch.object(remote,'create_snapshot',corrupt_snapshot):
            with self.assertRaisesRegex(RuntimeError,'changed after preflight'):
                self.submitted()
        self.assertEqual(SessionFactory.s3.objects,{})
        self.assertEqual(SessionFactory.bedrock.created,[])

    def test_eval_worker_submission_records_pid_and_id(self):
        rid=self.submitted();SessionFactory.bedrock.status='Completed'
        with contextlib.redirect_stdout(io.StringIO()):remote.check_train(self.root,'eli','v10',rid,factory=SessionFactory)
        with contextlib.redirect_stdout(io.StringIO()):remote.deploy(self.root,'eli','v10',rid,apply=True,factory=SessionFactory)
        SessionFactory.bedrock.deploy_status='Active'
        with contextlib.redirect_stdout(io.StringIO()):remote.check_deploy(self.root,'eli','v10',rid,factory=SessionFactory)
        with contextlib.redirect_stdout(io.StringIO()):
            ready=remote.prepare_eval(self.root,'eli','v10',rid,'benchmark-v1.0.2')
        fake_process=type('Child',(),{'pid':123456})()
        with patch.object(remote,'prepare_eval',return_value=ready), patch.object(remote.subprocess,'Popen',return_value=fake_process) as popen:
            with contextlib.redirect_stdout(io.StringIO()):
                state=remote.eval_submit(self.root,'eli','v10',rid,'benchmark-v1.0.2',apply=True)
        self.assertEqual(state['pid'],123456)
        self.assertTrue(state['log_file'].endswith('.log'))
        self.assertEqual(state['status'],'Queued')
        self.assertEqual(popen.call_count,1)
        persisted=remote.load_state(self.root/'runs/eli/v10'/rid,'eval_'+state['evaluation_id'])
        self.assertEqual(persisted['pid'],123456)

    def test_inference_bridge_uses_exact_frozen_system_prompt(self):
        runtime=FakeRuntime()
        server,url=remote._http_bridge(runtime,'arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/1234',
            'eli-nova-local',self.prompt,remote.sha(self.prompt.encode()),{'max_tokens':256,'temperature':0})
        def post(payload):
            req=urllib.request.Request(url+'/api/generate',json.dumps(payload).encode(),
                headers={'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req) as resp:return json.loads(resp.read())
        try:
            response=post({'model':'eli-nova-local','prompt':'Find datasets about cancer','stream':False})
            self.assertIn('sparql',response['response'])
            sent=runtime.requests[-1]
            self.assertEqual(sent['system'],[{'text':self.prompt}])
            self.assertEqual(sent['messages'][0]['content'][0]['text'],'Find datasets about cancer')
            self.assertEqual(sent['inferenceConfig']['temperature'],0)
            with self.assertRaises(Exception):
                post({'model':'eli-nova-local','prompt':'Find datasets about cancer',
                    'system':'ALTERED','stream':False})
            self.assertEqual(len(runtime.requests),1)
        finally:
            server.shutdown();server.server_close()


    def test_eval_revalidates_benchmark_before_inference(self):
        rid=self.submitted();SessionFactory.bedrock.status='Completed'
        with contextlib.redirect_stdout(io.StringIO()):remote.check_train(self.root,'eli','v10',rid,factory=SessionFactory)
        with contextlib.redirect_stdout(io.StringIO()):remote.deploy(self.root,'eli','v10',rid,apply=True,factory=SessionFactory)
        SessionFactory.bedrock.deploy_status='Active'
        with contextlib.redirect_stdout(io.StringIO()):remote.check_deploy(self.root,'eli','v10',rid,factory=SessionFactory)
        with contextlib.redirect_stdout(io.StringIO()):
            state=remote.prepare_eval(self.root,'eli','v10',rid,'benchmark-v1.0.2')
        self.assertTrue(remote.verify_evaluation_assets(self.root,state))
        self.case.write_text(json.dumps({'question':'MODIFIED AFTER EVAL SUBMITTED'})+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Protected benchmark file changed'):
            remote.verify_evaluation_assets(self.root,state)


if __name__=='__main__':unittest.main()
