"""Crash-safe checkpoint tests, optimizer/RNG parity, and resume CLI safety."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from toolkit.checkpoints import (signature, save_checkpoint, select_checkpoint,
                                 restore_checkpoint, available_checkpoints, training_lock)
from toolkit import cli

try:
    import torch
except ImportError:
    torch = None


class CheckpointUnitTests(unittest.TestCase):
    def test_signature_rejects_changes_to_frozen_inputs(self):
        from toolkit.checkpoints import SIGNATURE_FIELDS
        report={k:None for k in SIGNATURE_FIELDS}
        self.assertEqual(signature(report,{'train.jsonl':'abc'}),signature(report,{'train.jsonl':'abc'}))
        self.assertNotEqual(signature(report,{'train.jsonl':'abc'}),signature(report,{'train.jsonl':'changed'}))
        other=dict(report,optimizer_effective='different')
        self.assertNotEqual(signature(report,{}),signature(other,{}))

    def test_corrupted_or_adapter_only_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            ck=root/'checkpoints/step_0002000';ck.mkdir(parents=True)
            (ck/'adapter_model.safetensors').write_bytes(b'fake')
            with self.assertRaisesRegex(RuntimeError,'Incomplete checkpoint'):
                select_checkpoint(root)
            with self.assertRaisesRegex(RuntimeError,'step_'):
                select_checkpoint(root,'step_0000001')
            with self.assertRaisesRegex(RuntimeError,'Use --checkpoint'):
                select_checkpoint(root,'../evil')

    def test_lock_rejects_simultaneous_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            with training_lock(Path(temp)):
                with self.assertRaisesRegex(RuntimeError,'already being trained'):
                    with training_lock(Path(temp)):
                        pass
            with training_lock(Path(temp)):
                pass


@unittest.skipIf(torch is None,'Optional PyTorch not installed')
class CheckpointTorchRoundTrip(unittest.TestCase):
    class ToyModel(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.lora_weight=torch.nn.Parameter(torch.tensor(0.75))
        def save_pretrained(self,where,safe_serialization=True):
            folder=Path(where)
            (folder/'adapter_config.json').write_text('{}')
            (folder/'adapter_model.safetensors').write_bytes(b'toy-adapter-fixture')

    class ToyTokenizer:
        def save_pretrained(self,where):
            (Path(where)/'tokenizer.json').write_text('{}')

    def fresh(self):
        model=self.ToyModel()
        opt=torch.optim.AdamW([model.lora_weight],lr=.05)
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        return model,opt,scaler

    def train_step(self, model, opt, scaler):
        # Random PyTorch input makes a missing RNG restore show up as different results.
        x=torch.rand(1).squeeze()
        opt.zero_grad(set_to_none=True)
        ((model.lora_weight*x-1.25)**2).backward()
        opt.step()

    def test_saved_state_restarts_exactly_at_next_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            random.seed(900);torch.manual_seed(900)
            model,opt,scaler=self.fresh()
            for _ in range(3):self.train_step(model,opt,scaler)
            selected=save_checkpoint(torch=torch,model=model,tokenizer=self.ToyTokenizer(),
                optimizer=opt,scaler=scaler,trainable=list(model.named_parameters()),
                out=root,step=3,micro=3,elapsed_seconds=7.5,run_signature='test',
                last_event={'step':3,'loss':.123},keep_last=2)
            self.assertEqual(selected,select_checkpoint(root))
            for _ in range(3):self.train_step(model,opt,scaler)
            uninterrupted=model.lora_weight.detach().clone()
            # Recreate base and optimizer, then recover tensor/optimizer/RNG state.
            model2,opt2,scaler2=self.fresh()
            restored=restore_checkpoint(torch=torch,checkpoint=selected,
                expected_signature='test',optimizer=opt2,scaler=scaler2,
                trainable=list(model2.named_parameters()),total_micro=6,total_steps=6,accum=1)
            self.assertEqual(restored['micro_step'],3)
            self.assertEqual(restored['step'],3)
            self.assertEqual(restored['last_event']['loss'],.123)
            for _ in range(3):self.train_step(model2,opt2,scaler2)
            self.assertTrue(torch.equal(model2.lora_weight.detach(),uninterrupted))
            self.assertTrue(torch.equal(opt2.state[model2.lora_weight]['exp_avg'],
                                        opt.state[model.lora_weight]['exp_avg']))
            with self.assertRaisesRegex(RuntimeError,'configuration or frozen inputs'):
                restore_checkpoint(torch=torch,checkpoint=selected,expected_signature='wrong',
                    optimizer=opt2,scaler=scaler2,trainable=list(model2.named_parameters()),
                    total_micro=6,total_steps=6,accum=1)
            # File integrity guard catches partial/corrupted writes.
            with (selected/'training_state.pt').open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(RuntimeError,'corrupt'):
                select_checkpoint(root)

    def test_retains_last_three_confirmed_snapshots(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            model,opt,scaler=self.fresh()
            for step in range(1,6):
                self.train_step(model,opt,scaler)
                save_checkpoint(torch=torch,model=model,tokenizer=self.ToyTokenizer(),
                    optimizer=opt,scaler=scaler,trainable=list(model.named_parameters()),
                    out=root,step=step,micro=step,elapsed_seconds=float(step),
                    run_signature='test',keep_last=3)
            self.assertEqual([p.name for p in available_checkpoints(root)],
                             ['step_0000003','step_0000004','step_0000005'])


class ResumeCliTests(unittest.TestCase):
    def test_resume_uses_existing_run_and_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            run=root/'runs/eli/v9.0/run_foo'
            (run/'input').mkdir(parents=True)
            (run/'training_plan.json').write_text('{}')
            (run/'run.json').write_text(json.dumps({
                'project':'eli','version':'v9.0','run_id':'run_foo',
                'status':'failed','run_kind':'training','auto_evaluate':False}))
            with patch.object(cli,'ROOT',root), \
                 patch.object(cli,'assert_snapshot'), \
                 patch('toolkit.checkpoints.select_checkpoint',return_value=run/'checkpoints/step_0000003'), \
                 patch('toolkit.train_engine.run',return_value={'steps':6,'loss':0.1}) as engine, \
                 patch.object(cli,'new_run',side_effect=AssertionError('new run prohibited')), \
                 patch.object(cli,'update_run',side_effect=lambda folder,m,status,other=None:{**m,'status':status,**(other or {})}):
                cli.resume('eli','v9.0','run_foo')
                engine.assert_called_once_with(run/'input',run,resume_checkpoint='step_0000003',lock_already_held=True)

    def test_cannot_resume_completed_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            run=root/'runs/eli/v9.0/run_foo'
            run.mkdir(parents=True)
            (run/'run.json').write_text(json.dumps({
                'project':'eli','version':'v9.0','run_id':'run_foo',
                'status':'trained','run_kind':'training'}))
            with patch.object(cli,'ROOT',root):
                with self.assertRaisesRegex(RuntimeError,'not resumable'):
                    cli.resume('eli','v9.0','run_foo')

if __name__=='__main__':unittest.main()
