"""Project-level shared evaluation integration, stdlib-only; no GPU/Ollama."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from toolkit import shared_eval, cli
from toolkit.core import digest, write


class SharedEvaluationBridge(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.run_id = 'run_20261109T002925408171Z_a4a5de'
        self.project = self.root/'projects/eli'
        self.bench = self.project/'evaluations/benchmark-v1.0.1'
        self.bench.mkdir(parents=True)
        write(self.project/'project.json',{'name':'eli'})
        data=self.bench/'cases/frozen_715.jsonl'
        data.parent.mkdir()
        data.write_text('{"id":"h1","question":"held out"}\n',encoding='utf-8')
        write(self.bench/'benchmark_manifest.json',{'format':'eli-shared-benchmark-v1',
                                                    'benchmark_id':'eli-contract-v1.0.1',
                                                    'package_version':'1.0.1',
                                                    'suites':[{'name':'frozen_715','dataset':'cases/frozen_715.jsonl','count':1}],
                                                    'frozen_files_sha256':{'cases/frozen_715.jsonl':digest(data)}})
        for name in ('audit_shared.py', 'evaluate_shared_ollama.py', 'gate_report.py'):
            (self.bench/name).write_text('pass\n',encoding='utf-8')
        self.run = self.root/'runs/eli/v9.2'/self.run_id
        snap=self.run/'input'; data=snap/'data';data.mkdir(parents=True)
        (data/'train.jsonl').write_text('{"id":"t1","question":"A","expected":{}}\n')
        (data/'system_prompt.txt').write_text('Original frozen ELI prompt')
        write(snap/'input_hashes.json', {str(p.relative_to(snap)):digest(p)
              for p in data.iterdir() if p.is_file()})
        write(self.run/'run.json',{'project':'eli','version':'v9.2','run_id':self.run_id,
                                   'run_kind':'training','status':'exported'})

    def test_offline_audit_only_uses_frozen_train_and_never_writes_run(self):
        with patch('toolkit.shared_eval.subprocess.run') as call:
            ret=shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',
                                   audit_only=True,workspace=self.root)
        self.assertIsNone(ret)
        call.assert_called_once()
        cmd=call.call_args.args[0]
        self.assertEqual(cmd[1],str(self.bench/'audit_shared.py'))
        self.assertEqual(cmd[-1],str(self.run/'input/data/train.jsonl'))
        self.assertFalse((self.run/'shared_evaluations').exists())
        self.assertEqual(json.loads((self.run/'run.json').read_text())['status'],'exported')

    def test_delegate_dry_run_frozen_prompt_and_separate_results(self):
        with patch('toolkit.shared_eval.subprocess.run') as call:
            out=shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',
                                    model='datmm-eli:v9.2-f16',stage='generalization',
                                    dry_run=True,workspace=self.root)
        self.assertEqual(call.call_count,2)
        cmd=call.call_args.args[0]
        self.assertEqual(cmd[1],str(self.bench/'evaluate_shared_ollama.py'))
        self.assertEqual(cmd[cmd.index('--prompt')+1],str(self.run/'input/data/system_prompt.txt'))
        self.assertEqual(cmd[cmd.index('--model')+1],'datmm-eli:v9.2-f16')
        self.assertEqual(cmd[cmd.index('--out')+1],str(out))
        self.assertEqual(out,self.run/'shared_evaluations/benchmark-v1.0.1/datmm-eli_v9.2-f16')
        self.assertIn('--dry-run',cmd)
        self.assertFalse(out.exists())

    def test_delegate_resume_options(self):
        with patch('toolkit.shared_eval.subprocess.run') as call:
            shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',
                model='datmm-eli:v9.2-f16',stage='all',resume=True,
                url='http://127.0.0.1:11434',num_predict=768,seed=8002026,
                workspace=self.root)
        cmd=call.call_args.args[0]
        self.assertIn('--resume',cmd)
        self.assertEqual(cmd[cmd.index('--num-predict')+1],'768')
        self.assertEqual(cmd[cmd.index('--seed')+1],'8002026')

    def test_gate_only_is_read_only_and_checks_selected_model_folder(self):
        out=self.run/'shared_evaluations/benchmark-v1.0.1/datmm-eli_v9.2-f16'
        write(out/'run_config.json',{'benchmark_id':'eli-contract-v1.0.1'})
        with patch('toolkit.shared_eval.subprocess.run') as call:
            shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',
                model='datmm-eli:v9.2-f16',gate_only=True,workspace=self.root)
        self.assertEqual(call.call_count,2)
        cmd=call.call_args.args[0]
        self.assertEqual(cmd[1],str(self.bench/'gate_report.py'))
        self.assertEqual(cmd[-1],str(out))

    def test_delegates_one_or_several_targeted_suites_without_stage(self):
        with patch('toolkit.shared_eval.subprocess.run') as call:
            shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',
                model='m:v1',suites=['v72_graph_26','v81_corrective_16,v9_compositional_160'],
                dry_run=True,workspace=self.root)
        cmd=call.call_args.args[0]
        self.assertNotIn('--stage',cmd)
        self.assertEqual([cmd[i+1] for i,item in enumerate(cmd[:-1]) if item=='--suite'],
                         ['v72_graph_26','v81_corrective_16,v9_compositional_160'])
        self.assertIn('--dry-run',cmd)

    def test_list_suites_readonly_without_ollama_or_model(self):
        # Add a targeted suite so --list-suites displays both suite classes.
        from contextlib import redirect_stdout
        import io
        base=json.loads((self.bench/'benchmark_manifest.json').read_text(encoding='utf-8'))
        base['targeted_suites']=[{'name':'mini','dataset':'cases/frozen_715.jsonl',
                                  'count':1,'stage':'generalization','mode':'gold',
                                  'source_filter':'one','subset_of':'frozen_715'}]
        write(self.bench/'benchmark_manifest.json',base)
        capture=io.StringIO()
        with patch('toolkit.shared_eval.subprocess.run') as call:
            with redirect_stdout(capture):
                self.assertIsNone(shared_eval.execute('eli','v9.2',self.run_id,
                                  'benchmark-v1.0.1',list_suites=True,workspace=self.root))
        self.assertEqual(call.call_count,1) # integrity audit only; no inference process
        self.assertIn('CANONICAL frozen_715',capture.getvalue())
        self.assertIn('TARGETED mini',capture.getvalue())
        self.assertFalse((self.run/'shared_evaluations').exists())

    def test_reject_ambiguous_or_incompatible_selection(self):
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',
                                model='m:v1',stage='all',suites=['frozen_715'],workspace=self.root)
        with self.assertRaisesRegex(ValueError,'read-only'):
            shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',
                                model='m:v1',list_suites=True,workspace=self.root)
        with self.assertRaisesRegex(ValueError,'--model is required'):
            shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',
                                suites=['frozen_715'],workspace=self.root)

    def test_targeted_dataset_digest_checked(self):
        original=self.bench/'cases/frozen_715.jsonl'
        target=self.bench/'cases/target.jsonl'
        target.write_bytes(original.read_bytes())
        manifest=json.loads((self.bench/'benchmark_manifest.json').read_text())
        manifest['targeted_suites']=[{'name':'mini','dataset':'cases/target.jsonl',
                                     'count':1,'stage':'generalization','mode':'gold'}]
        manifest['frozen_files_sha256']['cases/target.jsonl']=digest(target)
        write(self.bench/'benchmark_manifest.json',manifest)
        self.assertEqual(shared_eval.plan('eli','v9.2',self.run_id,'benchmark-v1.0.1',workspace=self.root)['package_version'],'1.0.1')
        target.write_text('tampered',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            shared_eval.plan('eli','v9.2',self.run_id,'benchmark-v1.0.1',workspace=self.root)

    def test_reject_active_run(self):
        meta=json.loads((self.run/'run.json').read_text());meta['status']='training'
        write(self.run/'run.json',meta)
        with self.assertRaisesRegex(RuntimeError,'requires a completed training run'):
            shared_eval.plan('eli','v9.2',self.run_id,'benchmark-v1.0.1',workspace=self.root)

    def test_reject_modified_frozen_snapshot(self):
        (self.run/'input/data/system_prompt.txt').write_text('Tampered')
        with self.assertRaisesRegex(RuntimeError,'Frozen run input changed'):
            shared_eval.plan('eli','v9.2',self.run_id,'benchmark-v1.0.1',workspace=self.root)

    def test_reject_dataset_change_and_missing_package_version(self):
        data=self.bench/'cases/frozen_715.jsonl'
        data.write_text('{"id":"changed"}\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            shared_eval.plan('eli','v9.2',self.run_id,'benchmark-v1.0.1',workspace=self.root)
        manifest=self.bench/'benchmark_manifest.json'
        obj=json.loads(manifest.read_text())
        obj.pop('package_version')
        write(manifest,obj)
        with self.assertRaisesRegex(ValueError,'package_version'):
            shared_eval.plan('eli','v9.2',self.run_id,'benchmark-v1.0.1',workspace=self.root)

    def test_reject_path_traversal_and_missing_benchmark(self):
        with self.assertRaises(ValueError):
            shared_eval.plan('eli','v9.2',self.run_id,'../benchmark',workspace=self.root)
        with self.assertRaises(FileNotFoundError):
            shared_eval.plan('eli','v9.2',self.run_id,'benchmark-v2.0.0',workspace=self.root)

    def test_reject_missing_model_and_invalid_option_combinations(self):
        with self.assertRaisesRegex(ValueError,'--model is required'):
            shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',workspace=self.root)
        with self.assertRaisesRegex(ValueError,'--smoke and --resume'):
            shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',model='m:v1',
                                smoke=5,resume=True,workspace=self.root)
        with self.assertRaisesRegex(ValueError,'--reference-results requires --gate-only'):
            shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',model='m:v1',
                                reference_results='/somewhere',workspace=self.root)

    def test_cli_dispatch_preserves_project_run_binding(self):
        with patch('toolkit.shared_eval.subprocess.run') as call:
            with patch.object(cli,'ROOT',self.root):
                # CLI uses run/project validation; no real Ollama process.
                out=shared_eval.execute('eli','v9.2',self.run_id,'benchmark-v1.0.1',
                                        audit_only=True,workspace=cli.ROOT)
        self.assertIsNone(out)
        self.assertEqual(call.call_count,1)

if __name__=='__main__':unittest.main()
