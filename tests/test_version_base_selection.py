"""Version creation selects the immutable base identity, not training wizard."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from toolkit import cli
from toolkit.core import write

class TestVersionCreation(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.project=self.root/'projects'/'eli'
        (self.project/'versions').mkdir(parents=True)
        write(self.project/'project.json',{'name':'eli'})
        self.mock_paths=[patch.object(cli,'ROOT',self.root),patch.object(cli,'project_path',lambda n:self.project),patch.object(cli,'version_path',lambda p,v:self.project/'versions'/v)]
        for p in self.mock_paths:p.start()

    def tearDown(self):
        for p in self.mock_paths:p.stop()
        self.tmp.cleanup()

    def test_noninteractive_cannot_silently_choose_default(self):
        with patch.object(cli.sys,'stdin') as fake:
            fake.isatty.return_value=False
            with self.assertRaisesRegex(RuntimeError,'Interactive model selection required'):
                cli.version_new('eli','v9.0')
        self.assertFalse((self.project/'versions/v9.0').exists())

    def test_explicit_base_version(self):
        cli.version_new('eli','v9.0',model_id='Qwen/Qwen2.5-1.5B-Instruct',revision='main')
        dest=self.project/'versions/v9.0'
        cfg=json.loads((dest/'version.json').read_text())
        self.assertEqual(cfg['base_model'],{'id':'Qwen/Qwen2.5-1.5B-Instruct','revision':'main'})
        self.assertFalse((dest/'data/train.jsonl').exists())
        self.assertTrue((dest/'data/package.json').exists())
        # Existing version must not overwrite any model or pre-supplied data.
        (dest/'data/train.jsonl').write_text('SAMPLE')
        cli.version_new('eli','v9.0',model_id='Qwen/Qwen2.5-7B-Instruct')
        self.assertEqual(json.loads((dest/'version.json').read_text())['base_model']['id'],'Qwen/Qwen2.5-1.5B-Instruct')
        self.assertEqual((dest/'data/train.jsonl').read_text(),'SAMPLE')

    def test_interactive_selection_and_no_path_prompts(self):
        with patch.object(cli.sys,'stdin') as fake, \
             patch('toolkit.wizard.display_models',return_value={'primary_vram_gb':16}), \
             patch('toolkit.wizard.ask',side_effect=['1','main']) as prompts:
            fake.isatty.return_value=True
            cli.version_new('eli','v9.0')
        self.assertEqual(prompts.call_count,2)
        cfg=json.loads((self.project/'versions/v9.0/version.json').read_text())
        self.assertEqual(cfg['base_model']['id'],'Qwen/Qwen2.5-0.5B-Instruct')

if __name__=='__main__':unittest.main()
