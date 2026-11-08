"""MTT v0.3 canonical, version-local training-data package contract.

A data package is independent of training implementation; it may be provided by
another developer or another ChatGPT conversation. No silent placeholder data.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
from .core import read, write, digest
from .dataset import rows

FORMAT='mtt-data-v1'
FILENAMES={
    'train':'train.jsonl',
    'eval':'eval.jsonl',
    'system_prompt':'system_prompt.txt',
    'schema':'schema.json',
}
DEFAULT={
    'format': FORMAT,
    'task': 'supervised_chat_json',
    'files': FILENAMES,
    'required': {'train':True,'system_prompt':True,'eval':False,'schema':False},
    'prompt_assembly': 'prompt_only',
    'evaluation':{'scorer':'exact_json','max_new_tokens':384},
    'description':'Version-specific supervised JSON input package. Frozen per run.',
}


def manifest_path(folder):return Path(folder)/'data/package.json'


def initialize(folder):
    folder=Path(folder);(folder/'data').mkdir(parents=True,exist_ok=True)
    p=manifest_path(folder)
    if not p.exists():write(p,DEFAULT)
    return p


def contract(folder):
    p=manifest_path(folder)
    if not p.is_file():raise ValueError(f'Missing version data manifest: {p}. Run data wizard or supply a data ZIP.')
    cfg=read(p)
    if cfg.get('format')!=FORMAT:raise ValueError(f'Unsupported data package format: {cfg.get("format")!r}; expected {FORMAT}')
    if cfg.get('task')!='supervised_chat_json':raise ValueError('Only supervised_chat_json is supported by this trainer')
    if cfg.get('files')!=FILENAMES:raise ValueError('v0.3 requires standard fixed filenames in data/package.json; see docs/DATA_PACKAGE_CONTRACT.md')
    req=cfg.get('required',{})
    if any(k not in FILENAMES for k in req):raise ValueError('Unknown required input key in package.json')
    if not req.get('train') or not req.get('system_prompt'):
        raise ValueError('train and system_prompt must be required')
    if any(not isinstance(v,bool) for v in req.values()):raise ValueError('required values must be booleans')
    if cfg.get('prompt_assembly') not in ('prompt_only','append_schema'):
        raise ValueError('prompt_assembly must be prompt_only or append_schema')
    if cfg.get('prompt_assembly')=='append_schema' and not req.get('schema'):
        raise ValueError('append_schema requires schema to be marked required')
    ev=cfg.get('evaluation',{})
    if ev.get('scorer')!='exact_json':raise ValueError('v0.3 supports exact_json only; ELI semantic graph scorer is not built in')
    if not isinstance(ev.get('max_new_tokens'),int) or ev['max_new_tokens']<1:raise ValueError('Invalid max_new_tokens')
    return cfg


def inspect(folder,need_eval=False):
    folder=Path(folder);p=manifest_path(folder)
    if not p.exists():return {'missing':['package.json'],'present':[],'ready':False}
    cfg=contract(folder)
    missing=[];present=[]
    for key,name in FILENAMES.items():
        f=folder/'data'/name
        necessary=cfg['required'].get(key,False) or (key=='eval' and need_eval)
        if not f.exists() or f.stat().st_size==0:
            if necessary:missing.append(name)
            continue
        if key=='system_prompt' and not f.read_text(encoding='utf-8').strip():
            if necessary:missing.append(name)
            continue
        present.append(name)
    return {'missing':missing,'present':present,'ready':not missing,'format':cfg['format']}


def validate_package(folder,need_eval=False):
    folder=Path(folder);cfg=contract(folder)
    report=inspect(folder,need_eval=need_eval)
    if report['missing']:
        raise ValueError('Missing version data: '+', '.join(report['missing'])+'; run bash workspace.sh data wizard PROJECT VERSION or supply a package')
    root=folder/'data'
    if cfg['prompt_assembly']=='append_schema' or (root/'schema.json').is_file():
        if (root/'schema.json').is_file():json.loads((root/'schema.json').read_text(encoding='utf-8'))
    train=rows(root/'train.jsonl')
    ev=rows(root/'eval.jsonl') if (root/'eval.jsonl').is_file() and (root/'eval.jsonl').stat().st_size>0 else []
    if need_eval and not ev:raise ValueError('Evaluation requested but eval.jsonl has no cases')
    trainq={' '.join(x['question'].casefold().split()) for x in train}
    evalq={' '.join(x['question'].casefold().split()) for x in ev}
    intersection=trainq&evalq
    if intersection:raise ValueError(f'Train/eval overlap: {len(intersection)} normalized questions')
    return {'train':len(train),'eval':len(ev),'exact_question_overlap':0,'package_format':FORMAT,
            'files_sha256':{p.name:digest(p) for p in sorted(root.iterdir()) if p.is_file()}}


def assistant_wizard(folder,need_eval=False):
    """Interactive missing-file importer. Never invent a label or prompt."""
    folder=Path(folder);data=folder/'data';data.mkdir(parents=True,exist_ok=True)
    if not manifest_path(folder).exists():
        choice=input('No package.json found. Create standard mtt-data-v1 manifest? [y/N]: ').strip().lower()
        if choice not in ('y','yes'):return False
        initialize(folder)
    report=inspect(folder,need_eval=need_eval)
    if report['ready']:
        print('All necessary version inputs detected.');return True
    print('Missing version package inputs:',', '.join(report['missing']))
    print('Import each file from an existing path, or exit and copy a prepared version data ZIP.')
    for name in report['missing']:
        if name=='package.json':continue
        while True:
            value=input(f'Path to {name} (Enter to stop): ').strip().strip('"').strip("'")
            if not value:return False
            # Windows native paths on Windows, including spaces, are accepted.
            src=Path(value).expanduser()
            if not src.is_file():print('File not found:',src);continue
            dest=data/name
            if src.resolve()==dest.resolve():break
            if dest.exists():
                print('Existing input will not be overwritten:',dest)
                return False
            shutil.copy2(src,dest)
            print('Imported',dest)
            break
    return inspect(folder,need_eval=need_eval)['ready']


def migrate_legacy(folder):
    """Non-destructive migration of v0.2.x prompt/schema/data layout."""
    folder=Path(folder)
    if manifest_path(folder).exists():return False
    data=folder/'data';data.mkdir(parents=True,exist_ok=True)
    old_prompt=folder/'prompt.txt';new_prompt=data/'system_prompt.txt'
    if old_prompt.is_file() and not new_prompt.exists():shutil.copy2(old_prompt,new_prompt)
    old_schema=folder/'schema.json';new_schema=data/'schema.json'
    if old_schema.is_file() and not new_schema.exists():shutil.copy2(old_schema,new_schema)
    legacy=read(folder/'version.json')
    cfg=json.loads(json.dumps(DEFAULT))
    cfg['prompt_assembly']='append_schema' if legacy.get('include_schema_in_prompt') else 'prompt_only'
    cfg['required']['schema']=cfg['prompt_assembly']=='append_schema'
    if new_prompt.exists() and (new_prompt.read_text(encoding='utf-8').strip() or '') and (data/'train.jsonl').is_file():
        cfg['description']='Migrated from v0.2.x. Confirm prompt/schema are correct and replace old training data if needed.'
    write(manifest_path(folder),cfg)
    return True
