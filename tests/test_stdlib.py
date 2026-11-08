"""No GPU or external packages required; run with: python -m unittest discover -s tests"""
import json, tempfile, unittest
from pathlib import Path
from toolkit.dataset import validate, rows, expected_text
from toolkit.core import validated_name

class ToolkitTests(unittest.TestCase):
    def test_version_identifiers(self):
        self.assertEqual(validated_name('v9.0'),'v9.0')
        for val in ['../evil','..','bad/name','']:
            with self.assertRaises(ValueError):validated_name(val)
    def test_split_validation(self):
        with tempfile.TemporaryDirectory() as d:
            t=Path(d)/'train.jsonl';v=Path(d)/'eval.jsonl'
            t.write_text(json.dumps({'id':'train1','question':'hello','expected':{'x':1}})+'\n')
            v.write_text(json.dumps({'id':'eval1','question':'goodbye','expected':{'x':2}})+'\n')
            self.assertEqual(validate(t,v),{'train':1,'eval':1,'exact_question_overlap':0})
            v.write_text(json.dumps({'id':'eval1','question':'HELLO','expected':{'x':1}})+'\n')
            with self.assertRaisesRegex(ValueError,'overlap'):validate(t,v)
    def test_expected_text(self):
        self.assertEqual(json.loads(expected_text({'answer':'ok'})),{'answer':'ok'})

if __name__=='__main__':unittest.main()
