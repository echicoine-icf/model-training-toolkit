"""Standard library checks for the v0.3 version data contract."""
import json, tempfile, unittest
from pathlib import Path
from toolkit.data_package import initialize, inspect, validate_package, migrate_legacy
from toolkit.core import write

class DataPackageTests(unittest.TestCase):
    def test_new_empty_version_reports_missing(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);initialize(r)
            self.assertEqual(inspect(r)['missing'],['train.jsonl','system_prompt.txt'])
    def test_package_validation_and_missing_eval(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);initialize(r)
            (r/'data/system_prompt.txt').write_text('Output JSON only.')
            (r/'data/train.jsonl').write_text('{"id":"1","question":"asthma","expected":{"ok":true}}\n')
            self.assertEqual(validate_package(r)['train'],1)
            with self.assertRaisesRegex(ValueError,'eval.jsonl'):validate_package(r,need_eval=True)
            (r/'data/eval.jsonl').write_text('{"id":"2","question":"stroke","expected":{"ok":true}}\n')
            self.assertEqual(validate_package(r,need_eval=True)['eval'],1)
    def test_invalid_manifest_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);initialize(r)
            p=r/'data/package.json';x=json.loads(p.read_text());x['prompt_assembly']='append_schema';write(p,x)
            with self.assertRaisesRegex(ValueError,'requires schema'):inspect(r)
    def test_migration_copies_preserving_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);write(r/'version.json',{'include_schema_in_prompt':True})
            (r/'prompt.txt').write_text('legacy prompt')
            (r/'schema.json').write_text('{}')
            self.assertTrue(migrate_legacy(r))
            self.assertEqual((r/'data/system_prompt.txt').read_text(),'legacy prompt')
            self.assertEqual((r/'prompt.txt').read_text(),'legacy prompt')
            self.assertFalse(migrate_legacy(r))
if __name__=='__main__':unittest.main()
