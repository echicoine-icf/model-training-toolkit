"""Noninteractive train flag validation and original wizard compatibility."""
import argparse
import unittest
from unittest.mock import patch
from toolkit import cli

BASE_ARGS=dict(
    epochs=1,max_length=4096,learning_rate=2e-5,
    gradient_accumulation=1,seed=42,weight_decay=.01,warmup_ratio=.03,
    max_grad_norm=1.0,log_every=25,lora_r=16,lora_alpha=32,
    lora_dropout=.05,lora_target_modules='q_proj,v_proj',
    auto_eval=False,no_auto_eval=True,yes=True,
)

class TestExplicitTraining(unittest.TestCase):
    def args(self, **kwargs):
        return argparse.Namespace(**(BASE_ARGS|kwargs))

    def test_valid_all_parameters(self):
        tr,auto=cli.explicit_training_config(self.args())
        self.assertFalse(auto)
        self.assertEqual(tr['epochs'],1)
        self.assertEqual(tr['lora']['target_modules'],['q_proj','v_proj'])
        self.assertEqual(tr['max_length'],4096)

    def test_explicit_auto_eval(self):
        _,auto=cli.explicit_training_config(self.args(auto_eval=True,no_auto_eval=False))
        self.assertTrue(auto)

    def test_missing_required_argument(self):
        with self.assertRaisesRegex(RuntimeError,'--epochs'):
            cli.explicit_training_config(self.args(epochs=None))

    def test_needs_eval_decision(self):
        with self.assertRaisesRegex(RuntimeError,'--auto-eval / --no-auto-eval'):
            cli.explicit_training_config(self.args(no_auto_eval=False))

    def test_refuses_unapproved_training(self):
        with self.assertRaisesRegex(RuntimeError,'--yes'):
            cli.explicit_training_config(self.args(yes=False))

    def test_rejects_invalid_values(self):
        for kw in (dict(epochs=0),dict(max_length=100),dict(warmup_ratio=1),dict(lora_dropout=1),dict(lora_target_modules='q_proj,'),dict(learning_rate=-1)):
            with self.subTest(kw=kw),self.assertRaises(RuntimeError):
                cli.explicit_training_config(self.args(**kw))

    def test_fail_fast_before_any_project_file_access(self):
        with patch.object(cli,'version_path',side_effect=AssertionError('read version prematurely')):
            with self.assertRaisesRegex(RuntimeError,'--epochs'):
                cli.train('eli','v9.0',guided=False,train_args=self.args(epochs=None))

    def test_local_project_launcher_forwards_flags(self):
        self.assertIn('"$v" "$@"',cli.PROJECT_LAUNCHER)

if __name__=='__main__':unittest.main()

class TestNoninteractiveRunFlow(unittest.TestCase):
    def test_explicit_run_uses_no_wizard_and_records_selected_values(self):
        import tempfile
        import json
        from pathlib import Path
        from toolkit.core import write
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); v=root/'projects/eli/versions/v9.0';v.mkdir(parents=True)
            write(v/'version.json',{'base_model':{'id':'Qwen/Qwen2.5-0.5B-Instruct','revision':'test_sha'},'training':{},'project':'eli','version':'v9.0'})
            args=argparse.Namespace(**BASE_ARGS)
            called=[]
            folder=root/'runs/eli/v9.0/run_fake';folder.mkdir(parents=True)
            meta={'run_id':'run_fake','project':'eli','version':'v9.0','status':'prepared'}
            snap=folder/'input';snap.mkdir();write(snap/'version.json',{'base_model':{'id':'Qwen/Qwen2.5-0.5B-Instruct','revision':'test_sha'},'training':{}})
            def fake_new_run(project,version,training_overrides=None,base_override=None):
                called.append(('new_run',training_overrides,base_override));return folder,meta
            def fake_update(folder,record,status,other=None):
                d=dict(record,status=status);d.update(other or {});write(folder/'run.json',d);called.append(('status',status));return d
            def fake_train(path,out):
                called.append(('engine',path,out));return {'steps':1,'loss':0.1}
            with patch.object(cli,'version_path',return_value=v),\
                 patch.object(cli,'inspect',return_value={'missing':[]}),\
                 patch.object(cli,'version_validate',return_value={'train':1,'eval':0}),\
                 patch.object(cli,'new_run',side_effect=fake_new_run),\
                 patch.object(cli,'update_run',side_effect=fake_update),\
                 patch.object(cli,'assert_snapshot'),\
                 patch('toolkit.train_engine.run',side_effect=fake_train),\
                 patch.object(cli.subprocess,'check_output',return_value='torch==mock\n'),\
                 patch.object(cli,'training_wizard',side_effect=AssertionError('wizard must not run')):
                cli.train('eli','v9.0',guided=False,train_args=args)
            self.assertEqual(called[0][0],'new_run')
            self.assertEqual(called[0][1]['epochs'],1)
            self.assertEqual([x[1] for x in called if x[0]=='status'],['training','trained'])
            self.assertTrue((folder/'run.json').exists())
            self.assertFalse(cli.read(folder/'run.json')['auto_evaluate'])
