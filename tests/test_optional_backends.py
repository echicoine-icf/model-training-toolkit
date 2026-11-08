"""Unsloth opt-in backend tests: stdlib only; no GPU or downloads required."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from toolkit import cli
from toolkit.backends import (prepare_backend, load_unsloth_base,
                              attach_unsloth_lora)
from toolkit.train_engine import optimizer_schedule


class TestBackendSelection(unittest.TestCase):
    def test_default_never_imports_unsloth(self):
        with patch('toolkit.backends.importlib.import_module', side_effect=AssertionError('unsloth imported')):
            self.assertIsNone(prepare_backend('pytorch'))

    def test_missing_unsloth_does_not_fallback(self):
        with patch('toolkit.backends.importlib.import_module',side_effect=ImportError('unavailable')):
            with self.assertRaisesRegex(RuntimeError,'NOT substituted'):
                prepare_backend('unsloth')

    def test_invalid_backend_fails(self):
        with self.assertRaisesRegex(ValueError,'Unsupported training backend'):
            prepare_backend('invalid')

    def test_loader_keeps_pinned_repo_full_precision_and_attention(self):
        fast = Mock()
        model, tokenizer = object(), object()
        fast.from_pretrained.return_value = (model, tokenizer)
        output = load_unsloth_base(
            fast, base={'id':'Org/model','revision':'a'*40},
            max_length=6144, precision='BF16_DTYPE', attention='sdpa')
        self.assertEqual(output,(model,tokenizer))
        kw=fast.from_pretrained.call_args.kwargs
        self.assertEqual(kw['model_name'],'Org/model')
        self.assertEqual(kw['revision'],'a'*40)
        self.assertTrue(kw['use_exact_model_name'])
        self.assertFalse(kw['load_in_4bit'])
        self.assertFalse(kw['load_in_8bit'])
        self.assertEqual(kw['max_seq_length'],6144)
        self.assertEqual(kw['attn_implementation'],'sdpa')

    def test_arbitrary_lora_targets_pass_through(self):
        fast=Mock()
        model=object()
        targets=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
        output=attach_unsloth_lora(
            fast,model,lora={'r':16,'alpha':32,'dropout':0.05,'target_modules':targets},
            seed=8002026,max_length=6144,gradient_checkpointing=True)
        self.assertEqual(output,fast.get_peft_model.return_value)
        kw=fast.get_peft_model.call_args.kwargs
        self.assertEqual(kw['target_modules'],targets)
        self.assertEqual(kw['r'],16)
        self.assertEqual(kw['lora_alpha'],32)
        self.assertEqual(kw['lora_dropout'],.05)
        self.assertTrue(kw['init_lora_weights'])
        self.assertTrue(kw['use_gradient_checkpointing'])
        self.assertEqual(kw['random_state'],8002026)
        self.assertFalse(kw['use_rslora'])
        self.assertIsNone(kw['loftq_config'])
        self.assertEqual(targets,['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])

    def test_checkpointing_is_opt_in(self):
        fast=Mock()
        lora={'r':8,'alpha':16,'dropout':0.0,'target_modules':['arbitrary_module']}
        for enabled, mode, expected in [(True,'standard',True),(True,'unsloth','unsloth'),
                                        (False,'standard',False),(False,'unsloth',False)]:
            with self.subTest(enabled=enabled,mode=mode):
                attach_unsloth_lora(fast,object(),lora=lora,seed=42,max_length=512,
                                    gradient_checkpointing=enabled,checkpointing_mode=mode)
                self.assertEqual(fast.get_peft_model.call_args.kwargs['use_gradient_checkpointing'],expected)

    def test_cli_records_selected_backend_without_altering_training_methods(self):
        args=argparse.Namespace(
            epochs=4,max_length=6144,learning_rate=8e-5,min_learning_rate=5e-6,
            gradient_accumulation=1,seed=8002026,weight_decay=.01,warmup_ratio=.03,
            max_grad_norm=1.,log_every=50,lora_r=16,lora_alpha=32,lora_dropout=.05,
            lora_target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
            precision='bf16',attention_implementation='sdpa',optimizer='fused-adamw',
            gradient_checkpointing=True,no_gradient_checkpointing=False,
            save_every=2000,backend='unsloth',unsloth_checkpointing='standard',
            no_auto_eval=True,auto_eval=False,yes=True)
        tr,auto=cli.explicit_training_config(args)
        self.assertEqual(tr['backend'],'unsloth')
        self.assertEqual(tr['unsloth_checkpointing'],'standard')
        self.assertFalse(auto)
        self.assertEqual(tr['learning_rate'],8e-5)
        self.assertEqual(tr['lora']['target_modules'],['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
        f,warmup,floor=optimizer_schedule(tr,50004)
        self.assertEqual(warmup,1500)
        self.assertEqual(floor,5e-6)
        self.assertAlmostEqual(f(1499),8e-5)

    def test_cli_rejects_backend_specific_settings_on_other_backend(self):
        args=argparse.Namespace(
            epochs=1,max_length=2048,learning_rate=1e-4,
            gradient_accumulation=1,seed=42,weight_decay=.01,warmup_ratio=.03,
            max_grad_norm=1.,log_every=10,lora_r=8,lora_alpha=16,lora_dropout=0,
            lora_target_modules='q_proj',backend='pytorch',unsloth_checkpointing='unsloth',
            no_auto_eval=True,auto_eval=False,yes=True)
        with self.assertRaisesRegex(RuntimeError,'requires --backend unsloth'):
            cli.explicit_training_config(args)

    def test_backend_import_is_before_torch_in_real_run(self):
        # A sentinel at backend import proves no transformer/PEFT imports or
        # writes occur before optional backend activation.
        from toolkit import train_engine
        with tempfile.TemporaryDirectory() as tmp:
            snap=Path(tmp)
            (snap/'version.json').write_text(json.dumps({
                'training':{'backend':'unsloth'},'base_model':{'id':'Org/model'}}))
            with patch.object(train_engine,'prepare_backend',side_effect=RuntimeError('BACKEND IMPORT FIRST')):
                with self.assertRaisesRegex(RuntimeError,'BACKEND IMPORT FIRST'):
                    train_engine.run(snap,snap/'run')
            self.assertFalse((snap/'run').exists())

if __name__=='__main__':
    unittest.main()
