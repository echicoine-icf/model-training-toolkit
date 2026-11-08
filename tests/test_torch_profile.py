"""Offline tests for CUDA/CPU installer selection and safety guards."""
import importlib.util
import pathlib
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('mtt_bootstrap',pathlib.Path(__file__).resolve().parents[1]/'scripts/bootstrap.py')
module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

class TestTorchProfile(unittest.TestCase):
    def test_auto_detects_nvidia(self):
        with patch.object(module,'nvidia_gpu_detected',return_value=True):
            self.assertEqual(module.select_torch_profile('auto'),'cu128')
        with patch.object(module,'nvidia_gpu_detected',return_value=False):
            self.assertEqual(module.select_torch_profile('auto'),'cpu')
        self.assertEqual(module.select_torch_profile('cpu'),'cpu')
        self.assertEqual(module.select_torch_profile('cu128'),'cu128')

    def test_existing_cuda_wheel_is_preserved(self):
        with patch.object(module,'check_import',return_value=True), patch.object(module,'query_torch',return_value={
            'version':'2.10.0+cu128','cuda_build':'12.8','cuda_available':True
        }),patch.object(module,'call') as call,patch.object(module,'verify_torch') as verify:
            module.install_ml_stack('virtualenv-python','cu128')
            self.assertEqual(call.call_count,1)
            self.assertIn('requirements-ml.txt',str(call.call_args.args))
            verify.assert_called_once_with('virtualenv-python','cu128')

    def test_cpu_wheel_is_replaced_for_nvidia(self):
        with patch.object(module,'check_import',return_value=True), patch.object(module,'query_torch',return_value={
            'version':'2.15.0+cpu','cuda_build':None,'cuda_available':False
        }),patch.object(module,'call') as call,patch.object(module,'verify_torch'):
            module.install_ml_stack('virtualenv-python','cu128')
            self.assertEqual(call.call_count,2)
            self.assertIn('https://download.pytorch.org/whl/cu128',call.call_args_list[0].args[0])

    def test_cuda_verification_rejects_cpu_fallback(self):
        from json import dumps
        with patch.object(module.subprocess,'check_output',return_value=dumps({
            'version':'2.15.0+cpu','cuda_build':None,'cuda_available':False
        })):
            with self.assertRaisesRegex(RuntimeError,'refusing CPU-only training'):
                module.verify_torch('python','cu128')

    def test_cuda_verification_accepts_5080(self):
        from json import dumps
        with patch.object(module.subprocess,'check_output',return_value=dumps({
            'version':'2.10.0+cu128','cuda_build':'12.8','cuda_available':True,
            'device':'NVIDIA GeForce RTX 5080','gpu_sum':1024.0
        })):
            module.verify_torch('python','cu128')

if __name__=='__main__':unittest.main()
