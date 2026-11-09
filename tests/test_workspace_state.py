"""Workspace state diagnosis and conservative metadata-only repair; stdlib only."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from toolkit import cli, workspace_state
from toolkit.core import write


class WorkspaceStateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        write(self.root/'config/workspace.json',{'toolkit_version':'0.4.0'})
        py=self.root/'.venv/Scripts/python.exe'
        py.parent.mkdir(parents=True)
        py.write_bytes(b'fake interpreter')
        (self.root/'.venv/pyvenv.cfg').write_text('home = fake\n')
        for name in ('projects','runs','registry'):
            (self.root/name).mkdir()
        self.state=self.root/'WORKSPACE_STATE.json'

    def test_doctor_inspection_never_changes_state(self):
        original=b'{"status":"not_initialized","toolkit_version":"0.3.9","message":"run installer"}\n'
        self.state.write_bytes(original)
        check=workspace_state.inspect(self.root)
        self.assertTrue(check.environment_ready)
        self.assertTrue(check.repairable)
        self.assertFalse(check.healthy)
        self.assertEqual(self.state.read_bytes(),original)
        self.assertFalse((self.root/'.workspace_state_backups').exists())

    def test_missing_state_recreated_only_on_explicit_repair(self):
        self.assertFalse(self.state.exists())
        self.assertTrue(workspace_state.inspect(self.root).repairable)
        self.assertFalse(self.state.exists())
        check,backup,changed=workspace_state.repair(self.root)
        self.assertTrue(changed)
        self.assertTrue(check.healthy)
        self.assertIsNone(backup)
        self.assertEqual(json.loads(self.state.read_text())['status'],'ready')

    def test_repair_preserves_other_metadata_and_backs_up(self):
        old={'status':'not_initialized','toolkit_version':'0.3.9',
             'message':'Run bash run_me_first.sh', 'installed_utc':'original',
             'llama_cpp':{'commit':'known-commit'}}
        write(self.state,old)
        check,backup,changed=workspace_state.repair(self.root)
        self.assertTrue(changed)
        self.assertTrue(check.healthy)
        self.assertIsNotNone(backup)
        self.assertEqual(json.loads(backup.read_text()),old)
        new=json.loads(self.state.read_text())
        self.assertEqual(new['status'],'ready')
        self.assertEqual(new['toolkit_version'],'0.4.0')
        self.assertEqual(new['installed_utc'],'original')
        self.assertEqual(new['llama_cpp'],old['llama_cpp'])
        self.assertNotIn('message',new)
        self.assertIn('last_repair_utc',new)

    def test_corrupt_state_is_backed_up_before_repair(self):
        original=b'{ incomplete json!'
        self.state.write_bytes(original)
        self.assertTrue(workspace_state.inspect(self.root).repairable)
        check,backup,changed=workspace_state.repair(self.root)
        self.assertTrue(check.healthy)
        self.assertEqual(backup.read_bytes(),original)

    def test_already_healthy_is_read_only(self):
        original=b'{"status":"ready","toolkit_version":"0.4.0"}\n'
        self.state.write_bytes(original)
        check,backup,changed=workspace_state.repair(self.root)
        self.assertTrue(check.healthy)
        self.assertFalse(changed)
        self.assertIsNone(backup)
        self.assertEqual(self.state.read_bytes(),original)

    def test_broken_venv_blocks_repair_and_preserves_old_state(self):
        self.state.write_bytes(b'{"status":"not_initialized"}\n')
        (self.root/'.venv/pyvenv.cfg').unlink()
        with self.assertRaisesRegex(RuntimeError,'Missing .venv'):
            workspace_state.repair(self.root)
        self.assertNotIn('"ready"',self.state.read_text())
        self.assertFalse((self.root/'.workspace_state_backups').exists())

    def test_missing_projects_blocks_repair(self):
        self.state.write_text('{"status":"not_initialized"}')
        (self.root/'projects').rmdir()
        with self.assertRaisesRegex(RuntimeError,'Missing projects'):
            workspace_state.repair(self.root)
        self.assertEqual(json.loads(self.state.read_text())['status'],'not_initialized')

    def test_symlink_state_refuses_repair(self):
        other=self.root/'other.json';write(other,{'status':'not_initialized'})
        try:
            self.state.symlink_to(other)
        except (OSError, NotImplementedError):
            self.skipTest('Symlink privileges unavailable')
        with self.assertRaisesRegex(RuntimeError,'symlink'):
            workspace_state.repair(self.root)
        self.assertEqual(json.loads(other.read_text())['status'],'not_initialized')

    def test_upgrade_repairs_stale_state_after_success_without_touching_runs(self):
        write(self.state,{'status':'not_initialized','toolkit_version':'0.3.9', 'message':'old'})
        run=self.root/'runs/eli/v9.2/longrun/adapter/model.safetensors'
        run.parent.mkdir(parents=True);run.write_bytes(b'preserve-model')
        with patch.object(cli,'ROOT',self.root),contextlib.redirect_stdout(io.StringIO()):
            cli.upgrade()
        state=json.loads(self.state.read_text())
        self.assertEqual(state['status'],'ready')
        self.assertEqual(state['toolkit_version'],'0.4.0')
        self.assertIn('last_upgrade_utc',state)
        self.assertEqual(run.read_bytes(),b'preserve-model')
        self.assertEqual(len(list((self.root/'.workspace_state_backups').glob('*'))),1)

    def test_upgrade_missing_state_recreates_after_prerequisite_check(self):
        with patch.object(cli,'ROOT',self.root),contextlib.redirect_stdout(io.StringIO()):
            cli.upgrade()
        self.assertTrue(workspace_state.inspect(self.root).healthy)
        self.assertIn('last_upgrade_utc',json.loads(self.state.read_text()))

    def test_failed_upgrade_never_repairs_metadata(self):
        write(self.state,{'status':'not_initialized'})
        (self.root/'runs').rmdir()
        with patch.object(cli,'ROOT',self.root),self.assertRaisesRegex(RuntimeError,'unhealthy'):
            cli.upgrade()
        self.assertNotIn('toolkit_version',json.loads(self.state.read_text()))

    def test_cli_doctor_default_is_read_only(self):
        original=b'{"status":"not_initialized","toolkit_version":"0.3.9"}\n'
        self.state.write_bytes(original)
        buf=io.StringIO()
        with patch.object(cli,'ROOT',self.root),contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as e:
                cli.doctor()
        self.assertEqual(e.exception.code,1)
        self.assertIn('WORKSPACE_STATE_REPAIR_AVAILABLE',buf.getvalue())
        self.assertEqual(self.state.read_bytes(),original)

    def test_cli_parser_accepts_doctor_repair(self):
        with patch('sys.argv',['workspace.sh','doctor','--repair']),patch.object(cli,'doctor') as doctor:
            cli.main()
        doctor.assert_called_once_with(repair=True)

    def test_doctor_repair_corrects_metadata(self):
        write(self.state,{'status':'not_initialized','toolkit_version':'0.3.9'})
        buf=io.StringIO()
        # Source tests run outside .venv; patch the VENV-specific process identity.
        with patch.object(cli,'ROOT',self.root),patch.object(cli.sys,'prefix','fake-venv'),contextlib.redirect_stdout(buf):
            cli.doctor(repair=True)
        self.assertIn('WORKSPACE_STATE_REPAIRED',buf.getvalue())
        self.assertIn('READY',buf.getvalue())
        self.assertTrue(workspace_state.inspect(self.root).healthy)


if __name__ == '__main__':
    unittest.main()
