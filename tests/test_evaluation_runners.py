"""Offline regression tests for frozen custom evaluation runners."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from toolkit.core import digest, write
from toolkit.evaluation_plugins import (load_registry, freeze_evaluation_package,
                                        resolve_model, run, read_cases)
from toolkit.data_package import DEFAULT, validate_package


SCORER = '''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--cases');p.add_argument('--predictions');p.add_argument('--output');p.add_argument('--context');args=p.parse_args()
rows=[json.loads(x) for x in Path(args.predictions).read_text().splitlines()]
ctx=json.loads(Path(args.context).read_text())
if (Path(args.output)/'fail_me').exists():raise SystemExit(4)
summary={'cases':len(rows),'suite':ctx['suite'],'mode':ctx['mode'], 'accurate':sum(x.get('exact_match') is True for x in rows)}
Path(args.output,'summary.json').write_text(json.dumps(summary))
Path(args.output,'failures.jsonl').write_text(''.join(json.dumps(x)+'\\n' for x in rows if x.get('exact_match') is False))
Path(args.output,'report.md').write_text('# Synthetic scorer\\n')
'''


class EvalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.snapshot = self.path/'run/input'
        self.data = self.snapshot/'data'
        self.data.mkdir(parents=True)
        cfg = dict(DEFAULT)
        cfg['required'] = dict(DEFAULT['required'], eval=True)
        write(self.data/'package.json',cfg)
        write(self.snapshot/'version.json',{'base_model':{'id':'fake/model','revision':'fakeSHA'}})
        (self.data/'system_prompt.txt').write_text('Return JSON',encoding='utf-8')
        self._jsonl(self.data/'train.jsonl',[{'id':'t1','question':'Training question','expected':{'answer':1}}])
        self._jsonl(self.data/'eval.jsonl',[{'id':'e1','question':'Holdout question','expected':{'answer':2}}])
        self.external=self.path/'package'
        (self.external/'evaluation').mkdir(parents=True)
        (self.external/'evaluation/scorer.py').write_text(SCORER,encoding='utf-8')
        self._jsonl(self.external/'evaluation/challenge.jsonl',[{'id':'c1','question':'Challenge question'}])
        self.registry={
          'format':'mtt-eval-runners-v1', 'generation':{'max_new_tokens':40,'precision':'fp32'},
          'runners':[{'name':'test_scorer','description':'Check synthetic outputs',
                      'script':'evaluation/scorer.py','stages':['baseline','regression','diagnostic'],
                      'datasets':['train.jsonl','eval.jsonl','evaluation/challenge.jsonl'],
                      'model_inputs':['adapter','merged','checkpoint'],'output_dir':'score'}],
          'stages':[
            {'name':'baseline','suites':[{'name':'training_all','dataset':'train.jsonl','mode':'gold','runner':'test_scorer'}]},
            {'name':'regression','suites':[{'name':'frozen','dataset':'eval.jsonl','mode':'gold','runner':'test_scorer'}]},
            {'name':'diagnostic','suites':[{'name':'challenge','dataset':'evaluation/challenge.jsonl','mode':'diagnostic','runner':'test_scorer'}]},
          ]}
        write(self.external/'evaluation/runners.json',self.registry)
        hashes={str(p.relative_to(self.snapshot)).replace('\\','/'):digest(p)
                for p in self.snapshot.rglob('*') if p.is_file()}
        write(self.snapshot/'input_hashes.json',hashes)
        self.model={'type':'adapter','path':'fake-model','base':{'id':'fake/model'},'sha256':{'adapter_model.safetensors':'xxx'}}
        self.calls=[]
    def _jsonl(self,path,items):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(item)+'\n' for item in items),encoding='utf-8')
    def predictor(self,question,prompt,limit):
        self.calls.append(question)
        if question=='Training question':return '{"answer":1}'
        if question=='Holdout question':return '{"answer":2}'
        return '{}'
    def exec_run(self, **kwargs):
        return run(self.snapshot,self.path/'results',model_spec=self.model,
                   evaluation_package=self.external,predictor=self.predictor,**kwargs)

    def test_registry_discovery_and_validation(self):
        reg=load_registry(self.snapshot,self.external)
        self.assertEqual([s['name'] for s in reg['stages']],['baseline','regression','diagnostic'])
        self.assertEqual(len(reg['stages'][0]['suites'][0]['cases']),1)
        self.assertEqual(validate_package(self.snapshot)['train'],1)
    def test_stage_order_and_standard_outputs(self):
        self.exec_run(suite='regression')
        self.assertEqual(self.calls,['Training question','Holdout question'])
        target=list((self.path/'results').glob('*/score/regression/frozen'))[0]
        for name in ('summary.json','failures.jsonl','predictions.jsonl','report.md'):
            self.assertTrue((target/name).is_file(),name)
        self.assertFalse(list((self.path/'results').glob('*/score/diagnostic/*')))
    def test_resume_skips_completed_predictions_and_adds_stages(self):
        self.exec_run(suite='baseline')
        self.assertEqual(len(self.calls),1)
        self.exec_run(suite='all',resume=True)
        self.assertEqual(self.calls,['Training question','Holdout question','Challenge question'])
        self.exec_run(suite='all',resume=True)
        self.assertEqual(len(self.calls),3)
        summary=list((self.path/'results').glob('*/summary.json'))[0]
        self.assertEqual(len(json.loads(summary.read_text())['stages']),3)
    def test_cannot_overwrite_without_resume(self):
        self.exec_run(suite='baseline')
        with self.assertRaises(FileExistsError):self.exec_run(suite='all')
    def test_interrupted_prediction_recovers_without_repeating(self):
        def once(q,p,m):
            if q=='Holdout question':raise RuntimeError('synthetic interruption')
            return self.predictor(q,p,m)
        with self.assertRaisesRegex(RuntimeError,'interruption'):
            run(self.snapshot,self.path/'results',model_spec=self.model,evaluation_package=self.external,
                predictor=once,suite='all')
        self.exec_run(suite='all',resume=True)
        self.assertEqual(self.calls,['Training question','Holdout question','Challenge question'])
    def test_corruption_fails_closed(self):
        self.exec_run(suite='baseline')
        prediction=list((self.path/'results').glob('*/score/baseline/training_all/predictions.jsonl'))[0]
        prediction.write_text(prediction.read_text().replace('Training question','UNSAFE QUESTION'))
        with self.assertRaisesRegex(RuntimeError,'mismatch'):
            self.exec_run(suite='all',resume=True)
    def test_failing_scorer_does_not_delete_predictions(self):
        original=self.external/'evaluation/scorer.py'
        original.write_text(original.read_text().replace("if (Path(args.output)/'fail_me').exists():", "if True:"))
        with self.assertRaisesRegex(RuntimeError,'failed'):
            self.exec_run(suite='baseline')
        prediction=list((self.path/'results').glob('*/score/baseline/training_all/predictions.jsonl'))[0]
        self.assertTrue(prediction.exists())
        self.assertEqual(len(prediction.read_text().splitlines()),1)
    def test_source_mutation_changes_identity(self):
        self.exec_run(suite='baseline')
        self.external.joinpath('evaluation/scorer.py').write_text(SCORER+'\n# Updated evaluator\n')
        # A new evaluator uses its own output; cannot mix its results into old run.
        self.exec_run(suite='baseline')
        self.assertEqual(len(list((self.path/'results').glob('*/evaluation_manifest.json'))),2)
    def test_freeze_package_non_destructive(self):
        frozen=freeze_evaluation_package(self.path/'run',self.external)
        self.assertTrue((frozen/'evaluation/runners.json').exists())
        self.assertEqual(frozen,freeze_evaluation_package(self.path/'run',self.external))
        self.assertFalse((self.snapshot/'data/evaluation').exists())
    def test_merged_and_adapter_model_validation(self):
        folder=self.path/'merged';folder.mkdir()
        (folder/'config.json').write_text('{}')
        (folder/'model.safetensors').write_bytes(b'weights')
        self.assertEqual(resolve_model(self.path/'run',self.snapshot,'merged',model_path=str(folder))['type'],'merged')
        adapter=self.path/'adapter';adapter.mkdir()
        (adapter/'adapter_config.json').write_text('{}')
        (adapter/'adapter_model.safetensors').write_bytes(b'weights')
        self.assertEqual(resolve_model(self.path/'run',self.snapshot,str(adapter))['type'],'adapter')
    def test_resume_after_partial_scorer_output(self):
        self.exec_run(suite='baseline')
        target=list((self.path/'results').glob('*/score/baseline/training_all'))[0]
        (target/'score_receipt.json').unlink()
        self.exec_run(suite='baseline',resume=True)
        self.assertTrue((target/'summary.json.incomplete').is_file())
        self.assertTrue((target/'score_receipt.json').is_file())
        self.assertEqual(len(self.calls),1)

    def test_recover_last_incomplete_jsonl_line(self):
        self.exec_run(suite='baseline')
        target=list((self.path/'results').glob('*/score/baseline/training_all'))[0]
        pred=target/'predictions.jsonl'
        with pred.open('ab') as f:f.write(b'{broken partial')
        self.exec_run(suite='baseline',resume=True)
        self.assertTrue((target/'predictions.jsonl.incomplete_tail').is_file())
        self.assertEqual(len(pred.read_text().splitlines()),1)

    def test_complete_checkpoint_selection_and_integrity(self):
        run=self.path/'run'
        ck=run/'checkpoints/step_0002000'
        ck.mkdir(parents=True)
        (ck/'adapter_config.json').write_text('{}')
        (ck/'adapter_model.safetensors').write_bytes(b'lora')
        (ck/'training_state.pt').write_bytes(b'opaque optimizer state; never unpickled')
        files={f.name:digest(f) for f in ck.iterdir() if f.is_file()}
        write(ck/'checkpoint_info.json',{'format':'mtt-resume-v1','complete':True,'step':2000,
                                         'sha256':files})
        spec=resolve_model(run,self.snapshot,'checkpoint',checkpoint='step_0002000')
        self.assertEqual(spec['type'],'checkpoint')
        (ck/'adapter_model.safetensors').write_bytes(b'corrupted')
        with self.assertRaisesRegex(RuntimeError,'corrupt'):
            resolve_model(run,self.snapshot,'checkpoint',checkpoint='step_0002000')

    def test_unlabeled_suite_never_in_accuracy(self):
        self.exec_run(suite='all')
        target=list((self.path/'results').glob('*/score/diagnostic/challenge/summary.json'))[0]
        self.assertEqual(json.loads(target.read_text())['mode'],'diagnostic')
        pred=list((self.path/'results').glob('*/score/diagnostic/challenge/predictions.jsonl'))[0]
        self.assertIsNone(json.loads(pred.read_text())['exact_match'])

if __name__=='__main__': unittest.main()
