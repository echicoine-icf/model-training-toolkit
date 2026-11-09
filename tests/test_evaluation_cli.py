"""End-to-end CLI argument tests without a GPU, using injected model predictions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from toolkit import cli
from toolkit.core import digest, write
from toolkit.data_package import DEFAULT


SCORER = '''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('cases','predictions','output','context'): p.add_argument('--'+name,required=True)
a=p.parse_args();o=Path(a.output)
rows=[json.loads(x) for x in Path(a.predictions).read_text().splitlines()]
(o/'summary.json').write_text(json.dumps({'count':len(rows),'exact':sum(r['exact_match'] is True for r in rows)}))
(o/'failures.jsonl').write_text('')
(o/'report.md').write_text('# synthetic scorer')
'''


class CLIIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.run_id='run_20261108T000000000000Z_test'
        self.run=self.root/'runs/test/v0'/self.run_id
        snapshot=self.run/'input'
        data=snapshot/'data';data.mkdir(parents=True)
        p=dict(DEFAULT);p['required']=dict(DEFAULT['required'],eval=True)
        write(data/'package.json',p)
        write(snapshot/'version.json',{'project':'test','version':'v0','base_model':{'id':'fake'}})
        (data/'system_prompt.txt').write_text('JSON only')
        (data/'train.jsonl').write_text(json.dumps({'id':'t','question':'Training','expected':{'x':1}})+'\n')
        (data/'eval.jsonl').write_text(json.dumps({'id':'e','question':'Holdout','expected':{'x':2}})+'\n')
        write(snapshot/'input_hashes.json',{str(q.relative_to(snapshot)):digest(q) for q in snapshot.rglob('*') if q.is_file()})
        adapter=self.run/'adapter';adapter.mkdir()
        (adapter/'adapter_config.json').write_text('{}')
        (adapter/'adapter_model.safetensors').write_bytes(b'test')
        write(self.run/'run.json',{'project':'test','version':'v0','run_id':self.run_id,'status':'trained','run_kind':'training'})
        self.pack=self.root/'eval_package';(self.pack/'evaluation').mkdir(parents=True)
        (self.pack/'evaluation/scorer.py').write_text(SCORER)
        write(self.pack/'evaluation/runners.json',{
          'format':'mtt-eval-runners-v1','generation':{'max_new_tokens':10,'precision':'fp32'},
          'runners':[{'name':'demo','description':'test','script':'evaluation/scorer.py',
                      'stages':['baseline','generalization'],'datasets':['train.jsonl','eval.jsonl'],
                      'model_inputs':['adapter'],'output_dir':'custom'}],
          'stages':[{'name':'baseline','suites':[{'name':'train_all','runner':'demo','dataset':'train.jsonl','mode':'gold'}]},
                    {'name':'generalization','suites':[{'name':'frozen','runner':'demo','dataset':'eval.jsonl','mode':'gold'}]}]})

    def test_cli_dry_run_does_not_copy_or_modify_training(self):
        with patch.object(cli,'ROOT',self.root):
            x=cli.evaluate('test','v0',self.run_id,runner='custom',suite='baseline',
                          eval_package=str(self.pack),dry_run=True)
        self.assertTrue(x['dry_run'])
        self.assertFalse((self.run/'evaluations_custom').exists())
        self.assertFalse((self.run/'evaluation_packages').exists())
        self.assertEqual(json.loads((self.run/'run.json').read_text())['status'],'trained')

    def test_cli_baseline_and_all_resume(self):
        with patch.object(cli,'ROOT',self.root),patch('toolkit.evaluation_plugins.load_predictor',
                 return_value=lambda question,system,limit: '{"x":1}' if question=='Training' else '{"x":2}'):
            cli.evaluate('test','v0',self.run_id,runner='demo',suite='baseline',eval_package=str(self.pack))
            cli.evaluate('test','v0',self.run_id,runner='demo',suite='all',resume_eval=True)
        outputs=list((self.run/'evaluations_custom').glob('*/summary.json'))
        self.assertEqual(len(outputs),1)
        self.assertEqual(len(json.loads(outputs[0].read_text())['stages']),2)
        self.assertEqual(json.loads((self.run/'run.json').read_text())['status'],'evaluated')
        self.assertTrue((self.run/'adapter/adapter_model.safetensors').is_file())
        self.assertTrue((self.run/'evaluation_packages').exists())

    def test_disallow_evaluation_while_training(self):
        record=json.loads((self.run/'run.json').read_text());record['status']='training';write(self.run/'run.json',record)
        with patch.object(cli,'ROOT',self.root):
            with self.assertRaisesRegex(RuntimeError,'Cannot evaluate'):
                cli.evaluate('test','v0',self.run_id,eval_package=str(self.pack),dry_run=True)

if __name__=='__main__':unittest.main()
