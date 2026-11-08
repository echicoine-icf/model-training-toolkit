"""LoRA target auditing and reproducibility checks (runs without GPU downloads)."""
import argparse
import unittest
from toolkit import cli
from toolkit.train_engine import (verify_lora_modules, optimizer_schedule,
                                  supervised_messages, module_matches)

SEVEN=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

class FakeLoRA:
    pass

class FakeModel:
    def __init__(self,names):self.names=names
    def named_modules(self):return [(n,FakeLoRA()) for n in self.names]

class TestAdapterCoverage(unittest.TestCase):
    def test_qwen_seven_actual_layers(self):
        names=[f'base_model.model.layers.{layer}.self_attn.{target}' if target in SEVEN[:4]
               else f'base_model.model.layers.{layer}.mlp.{target}'
               for layer in range(2) for target in SEVEN]
        result=verify_lora_modules(FakeModel(names),SEVEN,FakeLoRA)
        self.assertEqual(result['requested_targets'],SEVEN)
        self.assertEqual(result['matched_targets'],{s:2 for s in SEVEN})
        self.assertEqual(result['adapter_layer_count'],14)

    def test_missing_adapter_fails(self):
        with self.assertRaisesRegex(RuntimeError,'missing targets'):
            verify_lora_modules(FakeModel(['model.layers.0.q_proj']),SEVEN,FakeLoRA)

    def test_extra_adapted_layer_fails(self):
        with self.assertRaisesRegex(RuntimeError,'unexpected'):
            verify_lora_modules(FakeModel([f'x.{x}' for x in SEVEN]+['x.lm_head']),SEVEN,FakeLoRA)

    def test_no_implicit_partial_suffix(self):
        self.assertFalse(module_matches('x.nq_proj','q_proj'))
        self.assertTrue(module_matches('x.q_proj','q_proj'))

    def test_cosine_absolute_minimum(self):
        tr={'learning_rate':8e-5,'min_learning_rate':5e-6,'warmup_ratio':0.03}
        func,warmup,floor=optimizer_schedule(tr,1000)
        self.assertEqual(warmup,30)
        self.assertEqual(floor,5e-6)
        self.assertAlmostEqual(func(29),8e-5)
        self.assertAlmostEqual(func(1000),5e-6)

    def test_original_assistant_text_preserved(self):
        expected={'a':1,'b':2}
        text='{"a": 1, "b": 2}'
        row={'question':'Example','expected':expected,'messages':[
            {'role':'system','content':'Prompt'}, {'role':'user','content':'Example'},
            {'role':'assistant','content':text}]}
        prefix,full=supervised_messages(row,'Prompt',1)
        self.assertEqual(full[2]['content'],text)
        with self.assertRaisesRegex(RuntimeError,'system prompt mismatch'):
            supervised_messages(row,'Different prompt',1)
        bad=dict(row,expected={'a':999})
        with self.assertRaisesRegex(RuntimeError,'differs from gold'):
            supervised_messages(bad,'Prompt',1)

    def test_cli_preserves_exact_seven(self):
        params=dict(epochs=4,max_length=6144,learning_rate=8e-5,
            gradient_accumulation=1,seed=8002026,weight_decay=0.01,
            warmup_ratio=0.03,max_grad_norm=1.0,log_every=25,
            lora_r=16,lora_alpha=32,lora_dropout=0.05,
            lora_target_modules=','.join(SEVEN),auto_eval=False,no_auto_eval=True,
            yes=True,min_learning_rate=5e-6,precision='bf16',
            attention_implementation='sdpa',optimizer='fused-adamw',
            gradient_checkpointing=True,no_gradient_checkpointing=False,
            save_every=2000)
        train,auto=cli.explicit_training_config(argparse.Namespace(**params))
        self.assertEqual(train['lora']['target_modules'],SEVEN)
        self.assertEqual(train['min_learning_rate'],5e-6)
        self.assertEqual(train['attention_implementation'],'sdpa')
        self.assertTrue(train['gradient_checkpointing'])
        self.assertFalse(auto)

    def test_cli_rejects_malformed_targets(self):
        params=dict(epochs=4,max_length=6144,learning_rate=8e-5,
            gradient_accumulation=1,seed=8002026,weight_decay=0.01,
            warmup_ratio=0.03,max_grad_norm=1.0,log_every=25,
            lora_r=16,lora_alpha=32,lora_dropout=0.05,
            lora_target_modules='q_proj,q_proj',auto_eval=False,no_auto_eval=True,yes=True)
        with self.assertRaisesRegex(RuntimeError,'unique'):
            cli.explicit_training_config(argparse.Namespace(**params))

if __name__=='__main__':unittest.main()
