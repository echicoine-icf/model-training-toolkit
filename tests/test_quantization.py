"""Quantizer tests use fake tools and temp files: no GPU, Hugging Face, or Ollama."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from toolkit.quantization import quantize
from scripts.vllm_eval_bridge import model_digest, handler_factory

class QuantizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.run='run_fake'
        self.exports=self.root/'runs'/'eli'/'v9.2'/self.run/'exports'
        self.exports.mkdir(parents=True)
        self.exports.parent.joinpath('run.json').write_text(json.dumps({'status':'exported'}))
        (self.exports/'model-f16.gguf').write_bytes(b'GGUF abcdefghi')
        (self.exports/'merged_hf').mkdir()
        (self.exports/'merged_hf'/'config.json').write_text('{}')
        (self.exports/'merged_hf'/'model.safetensors').write_bytes(b'fp16 source')
        self.quantizer=self.root/'llama-quantize.exe'
        self.quantizer.write_bytes(b'fake')
        self.python=self.root/'external-python.exe'
        self.python.write_bytes(b'fake python')
        (self.root/'scripts').mkdir()
        (self.root/'scripts'/'quantize_vllm.py').write_text('pass')

    def runq(self,**kw):
        return quantize('eli','v9.2',self.run,root=self.root,**kw)

    def test_gguf_q2_k_end_to_end_and_no_overwrite(self):
        seen = []
        def fake(cmd, check):
            seen.append(cmd)
            Path(cmd[2]).write_bytes(b'GGUF quantized Q2_K')
        plan = self.runq(fmt='gguf-q2_k', quantizer=str(self.quantizer),
                         ollama_model='datmm-eli:v9.2-q2', dry_run=True,
                         runner=lambda *_a, **_kw: self.fail('dry-run executed'))
        self.assertEqual(plan['command'][-1], 'Q2_K')
        self.assertEqual(plan['output'], str(self.exports/'model-q2_k.gguf'))
        result = self.runq(fmt='gguf-q2_k', quantizer=str(self.quantizer),
                           runner=fake)
        self.assertEqual(seen[0][-1], 'Q2_K')
        self.assertTrue((self.exports/'model-q2_k.gguf').is_file())
        self.assertTrue((self.exports/'model-q2_k.mtt-quantization.json').is_file())
        self.assertEqual((self.exports/'model-f16.gguf').read_bytes(), b'GGUF abcdefghi')
        self.assertEqual(result['format'], 'gguf-q2_k')
        with self.assertRaises(FileExistsError):
            self.runq(fmt='gguf-q2_k', quantizer=str(self.quantizer), runner=fake)

    def test_gguf_q2_k_ollama_registration(self):
        seen = []
        def fake(cmd, check):
            seen.append(cmd)
            if cmd[0] != 'ollama':
                Path(cmd[2]).write_bytes(b'GGUF quantized Q2_K')
        result = self.runq(fmt='gguf-q2_k', quantizer=str(self.quantizer),
                           ollama_model='datmm-eli:v9.2-q2', runner=fake)
        self.assertEqual(seen[0][-1], 'Q2_K')
        self.assertEqual(seen[1][:3], ['ollama', 'create', 'datmm-eli:v9.2-q2'])
        self.assertIn('model-q2_k.gguf',
                      (self.exports/'Modelfile.q2_k').read_text())
        self.assertEqual(result['ollama_created'], 'datmm-eli:v9.2-q2')

    def test_gguf_q8_provenance_and_no_overwrite(self):
        captured=[]
        def fake(cmd,check):
            captured.append(cmd)
            Path(cmd[2]).write_bytes(b'GGUF new Q8_0')
        result=self.runq(fmt='gguf-q8_0',quantizer=str(self.quantizer),runner=fake)
        self.assertEqual(captured[0][-1],'Q8_0')
        self.assertTrue((self.exports/'model-q8_0.gguf').is_file())
        self.assertTrue((self.exports/'model-q8_0.mtt-quantization.json').exists())
        self.assertEqual((self.exports/'model-f16.gguf').read_bytes(),b'GGUF abcdefghi')
        self.assertEqual(result['output_integrity']['bytes'],len(b'GGUF new Q8_0'))
        with self.assertRaises(FileExistsError):
            self.runq(fmt='gguf-q8_0',quantizer=str(self.quantizer),runner=fake)

    def test_gguf_q4_k_m_uses_f16_source_and_own_receipt(self):
        commands=[]
        def fake(cmd,check):
            commands.append(cmd)
            Path(cmd[2]).write_bytes(b'GGUF Q4_K_M output')
        # A previously completed Q8 export does not block a new Q4 export.
        (self.exports/'model-q8_0.gguf').write_bytes(b'previous Q8')
        result=self.runq(fmt='gguf-q4_k_m',quantizer=str(self.quantizer),runner=fake)
        self.assertEqual(commands[0][-1],'Q4_K_M')
        self.assertEqual(Path(commands[0][1]),self.exports/'model-f16.gguf')
        self.assertEqual(result['format'],'gguf-q4_k_m')
        self.assertTrue((self.exports/'model-q4_k_m.gguf').is_file())
        self.assertTrue((self.exports/'model-q4_k_m.mtt-quantization.json').is_file())
        self.assertEqual((self.exports/'model-q8_0.gguf').read_bytes(),b'previous Q8')
        self.assertEqual((self.exports/'model-f16.gguf').read_bytes(),b'GGUF abcdefghi')
        with self.assertRaises(FileExistsError):
            self.runq(fmt='gguf-q4_k_m',quantizer=str(self.quantizer),runner=fake)

    def test_gguf_q4_k_m_ollama_registration_and_dry_run(self):
        plan=self.runq(fmt='gguf-q4_k_m',quantizer=str(self.quantizer),
                       ollama_model='datmm-eli:v9.2-q4',dry_run=True,
                       runner=lambda *_args,**_kwargs:self.fail('dry run executes nothing'))
        self.assertEqual(plan['command'][-1],'Q4_K_M')
        self.assertFalse((self.exports/'Modelfile.q4_k_m').exists())
        calls=[]
        def fake(cmd,check):
            calls.append(cmd)
            if cmd[-1]=='Q4_K_M':
                Path(cmd[2]).write_bytes(b'GGUF Q4')
        result=self.runq(fmt='gguf-q4_k_m',quantizer=str(self.quantizer),
                         ollama_model='datmm-eli:v9.2-q4',runner=fake)
        self.assertIn('FROM ./model-q4_k_m.gguf',(self.exports/'Modelfile.q4_k_m').read_text())
        self.assertEqual(result['ollama_created'],'datmm-eli:v9.2-q4')
        self.assertEqual(calls[1][:3],['ollama','create','datmm-eli:v9.2-q4'])

    def test_gguf_ollama_registration_writes_modelfile(self):
        calls=[]
        def fake(cmd,check):
            calls.append(cmd)
            if cmd[-1]=='Q8_0':
                Path(cmd[2]).write_bytes(b'GGUF Q8')
        result=self.runq(fmt='gguf-q8_0',quantizer=str(self.quantizer),
                         ollama_model='datmm-eli:v9.2-q8',num_ctx=4096,runner=fake)
        self.assertEqual(calls[1][:3],['ollama','create','datmm-eli:v9.2-q8'])
        self.assertIn('FROM ./model-q8_0.gguf',(self.exports/'Modelfile.q8_0').read_text())
        self.assertEqual(result['ollama_created'],'datmm-eli:v9.2-q8')

    def test_invalid_ollama_model_rejected_before_export(self):
        with self.assertRaisesRegex(ValueError,'Invalid Ollama model tag'):
            self.runq(fmt='gguf-q8_0',quantizer=str(self.quantizer),ollama_model='bad model')
        self.assertFalse((self.exports/'model-q8_0.gguf').exists())

    def test_fp8_produces_compressed_artifact(self):
        def fake(cmd,check):
            out=Path(cmd[cmd.index('--output')+1]);out.mkdir()
            (out/'config.json').write_text(json.dumps({'quantization_config':{'quant_method':'compressed-tensors'}}))
            (out/'model.safetensors').write_bytes(b'FP8 compressed')
        result=self.runq(fmt='vllm-fp8',quant_python=str(self.python),runner=fake)
        self.assertEqual(result['format'],'vllm-fp8')
        self.assertTrue((self.exports/'vllm-fp8'/'MTT_QUANTIZATION.json').is_file())
        self.assertEqual(len(model_digest(self.exports/'vllm-fp8'/'MTT_QUANTIZATION.json')),64)
        self.assertEqual((self.exports/'merged_hf'/'model.safetensors').read_bytes(),b'fp16 source')

    def test_int8_scheme_and_no_fp8_fallback(self):
        def fake(cmd,check):
            self.assertEqual(cmd[-1],'W8A16')
            out=Path(cmd[cmd.index('--output')+1]);out.mkdir()
            (out/'config.json').write_text('{}');(out/'model.safetensors').write_bytes(b'x')
        with self.assertRaisesRegex(RuntimeError,'lacks quantization_config'):
            self.runq(fmt='vllm-int8',quant_python=str(self.python),runner=fake)
        self.assertFalse((self.exports/'vllm-int8').exists())
        self.assertTrue((self.exports/'vllm-int8.partial').exists())

    def test_dry_run_does_not_modify_anything(self):
        result=self.runq(fmt='gguf-q8_0',quantizer=str(self.quantizer),dry_run=True)
        self.assertEqual(result['format'],'gguf-q8_0')
        self.assertFalse((self.exports/'model-q8_0.gguf').exists())

    def test_no_active_python_for_vllm(self):
        import sys
        with self.assertRaisesRegex(ValueError,'separate environment'):
            self.runq(fmt='vllm-fp8',quant_python=sys.executable,dry_run=True)

    def test_missing_base_cannot_quantize(self):
        (self.exports/'model-f16.gguf').unlink()
        with self.assertRaises(FileNotFoundError):
            self.runq(fmt='gguf-q8_0',quantizer=str(self.quantizer),dry_run=True)

    def test_bridge_avoids_unverified_weights(self):
        def fake(cmd,check):
            out=Path(cmd[cmd.index('--output')+1]);out.mkdir()
            (out/'config.json').write_text(json.dumps({'quantization_config':{'x':1}}))
            (out/'model.safetensors').write_bytes(b'content')
        self.runq(fmt='vllm-fp8',quant_python=str(self.python),runner=fake)
        receipt=self.exports/'vllm-fp8'/'MTT_QUANTIZATION.json'
        model_digest(receipt)
        (self.exports/'vllm-fp8'/'model.safetensors').write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            model_digest(receipt)

    def test_bridge_chat_reuses_exact_prompt_and_generation(self):
        import threading,urllib.request
        from http.server import ThreadingHTTPServer
        from io import BytesIO
        received=[]
        class Response:
            def __init__(self,payload):self.file=BytesIO(json.dumps(payload).encode())
            def __enter__(self):return self.file
            def __exit__(self,*args):pass
        def fake_upstream(req,timeout):
            data=json.loads(req.data)
            received.append(data)
            return Response({'choices':[{'message':{'content':'{"target":"Dataset"}'},'finish_reason':'stop'}],
                             'usage':{'prompt_tokens':25,'completion_tokens':12}})
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_factory('eli-fp8','sha256-test','http://127.0.0.1:8000',4096))
        t=threading.Thread(target=server.serve_forever,daemon=True)
        with patch('scripts.vllm_eval_bridge.urlopen',side_effect=fake_upstream):
            t.start()
            try:
                req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/chat',
                  data=json.dumps({'model':'eli-fp8','stream':False,
                       'messages':[{'role':'system','content':'S'}, {'role':'user','content':'Q'}],
                       'options':{'num_ctx':4096,'num_predict':768,'temperature':0,'seed':99}}).encode(),
                  headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req) as resp:
                    output=json.load(resp)
                self.assertEqual(received[0]['messages'][0]['content'],'S')
                self.assertEqual(received[0]['max_tokens'],768)
                self.assertEqual(received[0]['seed'],99)
                self.assertEqual(output['message']['content'],'{"target":"Dataset"}')
                self.assertEqual(output['eval_count'],12)
            finally:
                server.shutdown();server.server_close();t.join()

    def test_bridge_output_schema(self):
        import threading,urllib.request
        from http.server import ThreadingHTTPServer
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_factory('eli-fp8','sha256-test','http://127.0.0.1:8000',4096))
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/api/tags') as resp:
                self.assertEqual(json.load(resp)['models'][0]['digest'],'sha256-test')
        finally:
            server.shutdown();server.server_close();t.join()

    def test_missing_quantizer_dry_run_plans_build_without_running(self):
        from unittest.mock import patch
        source=self.root/'tools'/'llama.cpp'
        source.mkdir(parents=True)
        (source/'CMakeLists.txt').write_text('project(llama)')
        with patch('toolkit.quantization.shutil.which',return_value=None):
            result=self.runq(fmt='gguf-q8_0',dry_run=True,
                             runner=lambda *_args,**_kwargs: self.fail('dry run must not build'))
        self.assertEqual(len(result['quantizer_auto_build_commands']),2)
        self.assertEqual(result['quantizer_auto_build_commands'][0][0],'cmake')
        self.assertEqual(result['quantizer_auto_build_commands'][1][-2:],['--parallel','8'])
        self.assertIn('resolved after automatic CMake build',result['command'][0])
        self.assertFalse((source/'build').exists())

    def test_missing_quantizer_is_built_then_used(self):
        from unittest.mock import patch
        source=self.root/'tools'/'llama.cpp'
        source.mkdir(parents=True)
        (source/'CMakeLists.txt').write_text('project(llama)')
        binary=source/'build'/'mtt-quantize'/'bin'/'Release'/'llama-quantize.exe'
        calls=[]
        def fake(cmd,check):
            calls.append(cmd)
            if cmd[0]=='cmake' and '--target' in cmd:
                binary.parent.mkdir(parents=True,exist_ok=True)
                binary.write_bytes(b'compiled utility')
            elif cmd[0]==str(binary):
                Path(cmd[2]).write_bytes(b'GGUF Q8')
        with patch('toolkit.quantization.shutil.which',side_effect=lambda n: 'C:/cmake.exe' if n=='cmake' else None):
            result=self.runq(fmt='gguf-q8_0',runner=fake)
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[0][0],'cmake')
        self.assertEqual(calls[1][calls[1].index('--target')+1],'llama-quantize')
        self.assertEqual(calls[2][0],str(binary))
        self.assertEqual(result['quantizer_binary_sha256'],__import__('hashlib').sha256(b'compiled utility').hexdigest())
        self.assertEqual((self.exports/'model-q8_0.gguf').read_bytes(),b'GGUF Q8')

    def test_disable_auto_build_and_explicit_missing_quantizer(self):
        source=self.root/'tools'/'llama.cpp'
        source.mkdir(parents=True)
        (source/'CMakeLists.txt').write_text('project(llama)')
        with patch('toolkit.quantization.shutil.which',return_value=None):
            with self.assertRaisesRegex(FileNotFoundError,'Auto-build is disabled'):
                self.runq(fmt='gguf-q8_0',auto_build_quantizer=False)
            with self.assertRaisesRegex(FileNotFoundError,'Explicit --quantizer not found'):
                self.runq(fmt='gguf-q8_0',quantizer='does-not-exist.exe')

    def test_missing_cmake_is_actionable_and_preserves_model(self):
        source=self.root/'tools'/'llama.cpp'
        source.mkdir(parents=True)
        (source/'CMakeLists.txt').write_text('project(llama)')
        with patch('toolkit.quantization.shutil.which',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'CMake is not on PATH'):
                self.runq(fmt='gguf-q8_0')
        self.assertTrue((self.exports/'model-f16.gguf').exists())
        self.assertFalse((self.exports/'model-q8_0.gguf').exists())

    def test_failed_cmake_build_reports_compiler_error(self):
        import subprocess
        source=self.root/'tools'/'llama.cpp'
        source.mkdir(parents=True)
        (source/'CMakeLists.txt').write_text('project(llama)')
        def fake(cmd,check):
            raise subprocess.CalledProcessError(1,cmd)
        with patch('toolkit.quantization.shutil.which',side_effect=lambda n: 'C:/cmake.exe' if n=='cmake' else None):
            with self.assertRaisesRegex(RuntimeError,'Automatic llama-quantize build failed'):
                self.runq(fmt='gguf-q8_0',runner=fake)
        self.assertFalse((self.exports/'model-q8_0.gguf').exists())
        self.assertFalse((self.exports/'model-q8_0.gguf.partial').exists())

    def test_auto_build_skips_when_quantizer_exists(self):
        calls=[]
        def fake(cmd,check):
            calls.append(cmd)
            Path(cmd[2]).write_bytes(b'GGUF new Q8_0')
        self.runq(fmt='gguf-q8_0',quantizer=str(self.quantizer),runner=fake)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],str(self.quantizer))

if __name__=='__main__': unittest.main()
