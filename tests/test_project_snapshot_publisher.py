"""Offline, standard-library regression tests for source-only Git publication."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'publish_project_snapshot.py'
spec = importlib.util.spec_from_file_location('snapshot_publisher', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def git(*args, cwd):
    return subprocess.run(['git', *map(str, args)], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


class ProjectSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'source'
        self.root.mkdir()
        git('init', '-b', 'main', cwd=self.root)
        git('config', 'user.email', 'test@example.com', cwd=self.root)
        git('config', 'user.name', 'Tester', cwd=self.root)
        put(self.root / '.gitignore', 'runs/\nprojects/*/versions/*/data/*.jsonl\n/WORKSPACE_STATE.json\n')
        put(self.root / 'README.md', 'MTT toolkit')
        put(self.root / 'toolkit/cli.py', 'print("cli")\n')
        put(self.root / 'projects/eli/project.sh', '#!/usr/bin/env bash\n')
        put(self.root / 'projects/eli/project.json', '{"name":"eli"}\n')
        put(self.root / 'projects/eli/versions/v9.2/version.json', '{"training":{"epochs":1}}\n')
        put(self.root / 'projects/eli/versions/v9.2/data/train.jsonl', '{"id":"1","question":"test"}\n')
        put(self.root / 'projects/eli/versions/v9.2/data/eval.jsonl', '{"id":"eval1"}\n')
        put(self.root / 'projects/eli/evaluations/benchmark-v1.0.2/benchmark_manifest.json', '{"version":"1.0.2"}\n')
        put(self.root / 'projects/eli/evaluations/benchmark-v1.0.2/cases/gold.jsonl', '{"id":"g1"}\n')
        put(self.root / 'projects/eli/versions/v9.1/version.json', '{"old":true}\n')
        put(self.root / 'projects/other/project.json', '{"other":true}\n')
        put(self.root / 'WORKSPACE_STATE.json', '{"status":"ready"}\n')
        git('add', '-A', cwd=self.root)
        git('commit', '-m', 'initial', cwd=self.root)
        self.run_id = 'run_test123'
        run = self.root / 'runs/eli/v9.2' / self.run_id
        frozen = run / 'input'
        put(frozen / 'version.json', '{"training":{"epochs":4},"base_model":{"revision":"pinned"}}\n')
        put(frozen / 'data/train.jsonl', '{"id":"1","question":"test"}\n')
        put(frozen / 'data/eval.jsonl', '{"id":"eval1"}\n')
        h = {name: hashlib.sha256((frozen/name).read_bytes()).hexdigest()
             for name in ['version.json', 'data/train.jsonl', 'data/eval.jsonl']}
        put(frozen / 'input_hashes.json', json.dumps(h))
        put(run / 'run.json', '{"run_id":"run_test123","status":"trained"}\n')
        put(run / 'training_plan.json', '{"epochs":4}\n')
        put(run / 'adapter/adapter_model.safetensors', 'weights')
        put(run / 'checkpoints/step_000001/model.safetensors', 'weights')
        self.remote = self.base / 'target.git'
        git('init', '--bare', '-b', 'main', self.remote, cwd=self.base)

    def args(self, **kw):
        d = dict(repo_url=str(self.remote), project='eli', version='v9.2',
                 run=self.run_id, benchmark=['benchmark-v1.0.2'],
                 external_eval=[], push=True, visibility='private',
                 allow_public=False, allow_local_remote=True)
        d.update(kw)
        return argparse.Namespace(**d)

    def published_checkout(self):
        dest = self.base / 'clone'
        git('clone', self.remote, dest, cwd=self.base)
        return dest

    def test_publishes_ignored_data_and_run_recipe_preserving_source(self):
        origin = git('remote', '-v', cwd=self.root)
        branch = git('branch', '--show-current', cwd=self.root)
        mod.run(self.args(), root=self.root)
        self.assertEqual(git('remote', '-v', cwd=self.root), origin)
        self.assertEqual(git('branch', '--show-current', cwd=self.root), branch)
        self.assertEqual(git('status', '--porcelain', cwd=self.root), '')
        dest = self.published_checkout()
        self.assertTrue((dest/'projects/eli/versions/v9.2/data/train.jsonl').exists())
        self.assertTrue((dest/'projects/eli/evaluations/benchmark-v1.0.2/cases/gold.jsonl').exists())
        self.assertTrue((dest/f'rebuild_recipes/eli/v9.2/{self.run_id}/input/version.json').exists())
        self.assertFalse((dest/'projects/eli/versions/v9.1').exists())
        self.assertFalse((dest/'projects/other').exists())
        self.assertFalse((dest/'runs').exists())
        self.assertFalse((dest/'WORKSPACE_STATE.json').exists())
        manifest=json.loads((dest/f'snapshots/eli/v9.2/{self.run_id}/manifest.json').read_text())
        self.assertEqual(manifest['project'],'eli')
        self.assertIn('projects/eli/versions/v9.2/data/train.jsonl',manifest['files'])
        self.assertFalse(manifest['model_weights_included'])

    def test_second_publish_is_idempotent(self):
        mod.run(self.args(), root=self.root)
        head1=git('--git-dir', self.remote, 'rev-parse', 'HEAD', cwd=self.base)
        mod.run(self.args(), root=self.root)
        head2=git('--git-dir', self.remote, 'rev-parse', 'HEAD', cwd=self.base)
        self.assertEqual(head1,head2)

    def test_existing_snapshot_cannot_be_overwritten(self):
        mod.run(self.args(), root=self.root)
        put(self.root/'projects/eli/versions/v9.2/data/train.jsonl', '{"changed":true}\n')
        with self.assertRaisesRegex(RuntimeError,'immutable'):
            mod.run(self.args(), root=self.root)

    def test_refuses_corrupt_frozen_input(self):
        put(self.root/'runs/eli/v9.2'/self.run_id/'input/data/train.jsonl', 'corrupt\n')
        with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
            mod.run(self.args(push=False), root=self.root)

    def test_refuses_source_code_dirty(self):
        put(self.root/'README.md', 'modified')
        with self.assertRaisesRegex(RuntimeError,'local modifications'):
            mod.run(self.args(push=False), root=self.root)

    def test_requires_explicit_public_opt_in(self):
        with self.assertRaisesRegex(ValueError,'--allow-public'):
            mod.run(self.args(visibility='public'), root=self.root)

    def test_refuses_secret_in_version_input(self):
        put(self.root/'projects/eli/versions/v9.2/data/keys.txt', 'AK' + 'IA' + 'ABCDEFGHIJKLMNOP\n')
        with self.assertRaisesRegex(RuntimeError,'Potential credential'):
            mod.run(self.args(push=False), root=self.root)

    def test_external_evaluator_excludes_result_files(self):
        ext=self.base/'ELI_VALIDATOR'
        put(ext/'evaluate.py','print(1)\n')
        put(ext/'ollama_results/out/predictions.jsonl','{"bad":"result"}\n')
        put(ext/'ELI_V92_BASELINE_RESULTS.zip','excluded result archive')
        mod.run(self.args(external_eval=[ext]), root=self.root)
        dest=self.published_checkout()
        self.assertTrue((dest/'projects/eli/evaluations/external/ELI_VALIDATOR/evaluate.py').exists())
        self.assertFalse((dest/'projects/eli/evaluations/external/ELI_VALIDATOR/ollama_results').exists())
        self.assertFalse((dest/'projects/eli/evaluations/external/ELI_VALIDATOR/ELI_V92_BASELINE_RESULTS.zip').exists())


if __name__ == '__main__':
    unittest.main()
