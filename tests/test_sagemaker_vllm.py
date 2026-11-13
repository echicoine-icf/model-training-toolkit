"""Offline tests: never download model weights, use AWS, or run GPU inference."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.sagemaker_vllm import (autoscale, autoscaling_plan, build_parser, deploy, digest, endpoint_plan, model_path,
                                   package_model, publish_package, smoke_invoke, validate_checkpoint)
from toolkit.quantization import quantize
from scripts.benchmark_sagemaker import measure as benchmark_measure
from scripts.vllm_eval_bridge import model_digest, unquantized_model_digest


class SageMakerVllmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'runs' / 'eli' / 'v9.2' / 'run_fake' / 'exports'
        self.base.mkdir(parents=True)
        (self.base.parent / 'run.json').write_text(json.dumps({'status': 'exported'}))
        self.hf = self.base / 'merged_hf'
        self.hf.mkdir()
        (self.hf / 'config.json').write_text('{}')
        (self.hf / 'model.safetensors').write_bytes(b'fake weights')
        (self.hf / 'tokenizer.json').write_text('{}')
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts' / 'quantize_vllm.py').write_text('print(1)')
        self.external = self.root / 'separate-python.exe'
        self.external.write_bytes(b'fake external')

    def test_package_f16_roundtrip_and_integrity(self):
        dst = self.root / 'eli-f16.tar.gz'
        output = package_model(self.root, 'eli', 'v9.2', 'run_fake', 'hf-f16', dst)
        self.assertEqual(output['artifact_sha256'], digest(dst))
        self.assertTrue((self.root / 'eli-f16.tar.gz.receipt.json').is_file())
        import tarfile
        with tarfile.open(dst) as f:
            self.assertEqual(sorted(f.getnames()), ['config.json', 'model.safetensors', 'tokenizer.json'])
        with self.assertRaises(FileExistsError):
            package_model(self.root, 'eli', 'v9.2', 'run_fake', 'hf-f16', dst)

    def test_vllm_w4a16_plan_and_receipt(self):
        plan = quantize('eli', 'v9.2', 'run_fake', root=self.root, fmt='vllm-w4a16',
                        quant_python=str(self.external), dry_run=True,
                        runner=lambda *_a, **_k: self.fail('dry-run executed'))
        self.assertEqual(plan['command'][-1], 'W4A16')
        self.assertEqual(plan['output'], str(self.base / 'vllm-w4a16'))
        def fake(cmd, check):
            out = Path(cmd[cmd.index('--output') + 1])
            out.mkdir()
            (out / 'config.json').write_text(json.dumps({'quantization_config': {'method': 'compressed-tensors'}}))
            (out / 'model.safetensors').write_bytes(b'fake four bit')
        quantize('eli', 'v9.2', 'run_fake', root=self.root, fmt='vllm-w4a16',
                 quant_python=str(self.external), runner=fake)
        self.assertEqual(model_path(self.root, 'eli', 'v9.2', 'run_fake', 'vllm-w4a16'),
                         self.base / 'vllm-w4a16')
        self.assertEqual(len(model_digest(self.base / 'vllm-w4a16' / 'MTT_QUANTIZATION.json')), 64)
        artifact = package_model(self.root, 'eli', 'v9.2', 'run_fake', 'vllm-w4a16',
                                 self.root / 'eli-int4.tar.gz')
        self.assertEqual(artifact['format'], 'vllm-w4a16')

    def test_rejects_tampered_quantized_export(self):
        self.test_vllm_w4a16_plan_and_receipt()
        (self.base / 'vllm-w4a16' / 'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs from'):
            validate_checkpoint(self.base / 'vllm-w4a16', require_receipt=True)

    def test_f16_baseline_vllm_bridge_identity_changes_with_weights(self):
        before = unquantized_model_digest(self.hf)
        self.assertEqual(len(before), 64)
        (self.hf / 'model.safetensors').write_bytes(b'changed weights')
        after = unquantized_model_digest(self.hf)
        self.assertNotEqual(before, after)

    def test_rejects_symlink(self):
        outside = self.root / 'sensitive'
        outside.write_text('secret')
        (self.hf / 'linked').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            package_model(self.root, 'eli', 'v9.2', 'run_fake', 'hf-f16', self.root / 'bad.tar.gz')

    def test_s3_upload_is_dry_by_default_and_validates_sha(self):
        dst = self.root / 'eli-f16.tar.gz'
        package_model(self.root, 'eli', 'v9.2', 'run_fake', 'hf-f16', dst)
        with patch('scripts.sagemaker_vllm.get_boto3', side_effect=AssertionError('network')):
            plan = publish_package(dst, 's3://bucket/path/eli-f16.tar.gz', None, 'us-east-1')
        self.assertFalse(plan['apply'])
        dst.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            publish_package(dst, 's3://bucket/path/eli-f16.tar.gz', None, 'us-east-1')

    def deploy_args(self):
        return build_parser().parse_args([
            'deploy', '--model-s3-uri', 's3://bucket/eli/v9.2/fp8/model.tar.gz',
            '--image-uri', '763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:server-sagemaker-cuda-v2.8',
            '--role', 'arn:aws:iam::123456789012:role/SageMakerRole',
            '--endpoint', 'eli-v92-fp8', '--model-tag', 'datmm-eli:v9.2-fp8'])

    def test_deploy_safe_plan(self):
        args = self.deploy_args()
        with patch('scripts.sagemaker_vllm.get_boto3', side_effect=AssertionError('network')):
            plan = deploy(args)
        self.assertEqual(plan['model']['PrimaryContainer']['ModelDataUrl'],
                         's3://bucket/eli/v9.2/fp8/model.tar.gz')
        self.assertEqual(plan['endpoint_config']['ProductionVariants'][0]['InitialInstanceCount'], 1)
        self.assertEqual(plan['endpoint_config']['ProductionVariants'][0]['InferenceAmiVersion'],
                         'al2-ami-sagemaker-inference-gpu-3-1')
        self.assertEqual(plan['model']['PrimaryContainer']['Environment']['SM_VLLM_MAX_NUM_SEQS'], '64')
        self.assertEqual(plan['endpoint']['EndpointName'], 'eli-v92-fp8')

    def test_deploy_rejects_missing_fields_or_unknown_kv_type(self):
        args = self.deploy_args()
        args.kv_cache_dtype = 'arbitrary'
        with self.assertRaises(ValueError):
            endpoint_plan(args)
        args.kv_cache_dtype = 'fp8'
        args.image_uri = 'example:latest'
        with self.assertRaises(ValueError):
            endpoint_plan(args)
        args.image_uri = 'example:stable'
        args.gpu_memory_utilization = 1.0
        with self.assertRaises(ValueError):
            endpoint_plan(args)

    def test_autoscale_has_always_on_minimum_and_concurrency_metric(self):
        plan = autoscaling_plan('eli-v92-fp8', 1, 3, 8.0)
        self.assertEqual(plan['target']['MinCapacity'], 1)
        self.assertEqual(plan['target']['MaxCapacity'], 3)
        self.assertEqual(plan['policy']['TargetTrackingScalingPolicyConfiguration']
                         ['PredefinedMetricSpecification']['PredefinedMetricType'],
                         'SageMakerVariantConcurrentRequestsPerModelHighResolution')
        with self.assertRaises(ValueError):
            autoscaling_plan('eli-v92-fp8', 0, 3, 8)
        with self.assertRaises(ValueError):
            autoscaling_plan('eli-v92-fp8', 4, 3, 8)
        args = build_parser().parse_args(['autoscale', '--endpoint', 'eli-v92-fp8'])
        with patch('scripts.sagemaker_vllm.get_boto3', side_effect=AssertionError('network')):
            self.assertEqual(autoscale(args)['target']['MinCapacity'], 1)

    def test_invoke_plan_is_safe_and_has_deterministic_payload(self):
        args = build_parser().parse_args(['invoke', '--endpoint', 'eli-v92-fp8',
                  '--model-tag', 'datmm-eli:v9.2-fp8', '--system-prompt', 'Return a plan',
                  '--message', 'Find cancer studies'])
        with patch('scripts.sagemaker_vllm.get_boto3', side_effect=AssertionError('network')):
            plan = smoke_invoke(args)
        self.assertFalse(plan['apply'])
        self.assertEqual(plan['request']['temperature'], 0)
        self.assertEqual(plan['request']['model'], 'datmm-eli:v9.2-fp8')

    def test_concurrency_benchmark_reports_failures(self):
        class Dummy:
            def __init__(self):
                self.calls = 0
            def invoke_endpoint(self, **_kw):
                import io
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError('over capacity')
                return {'Body': io.BytesIO(json.dumps({
                    'choices': [{'finish_reason': 'stop'}],
                    'usage': {'completion_tokens': 10}}).encode())}
        result = benchmark_measure(Dummy(), 'eli-v92', 'eli-v92-fp8', 'system',
                                   ['q1', 'q2', 'q3', 'q4'], 1, 20, 42)
        self.assertEqual(result['attempted'], 4)
        self.assertEqual(result['succeeded'], 3)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['output_tokens'], 30)
        self.assertIsNotNone(result['p95_latency_s'])

    def test_name_path_validation(self):
        with self.assertRaises(ValueError):
            model_path(self.root, 'eli', 'v9.2', '..', 'hf-f16')


if __name__ == '__main__':
    unittest.main()
