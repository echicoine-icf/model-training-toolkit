"""Interactive, stdlib-only onboarding and conservative hardware model recommendations."""
from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
from .core import ROOT, read, write, utc, validated_name, project_path, version_path

MODEL_CATALOG=[
 {'id':'Qwen/Qwen2.5-0.5B-Instruct','parameters_b':0.5,'label':'Qwen2.5 0.5B (starter)','vram_guideline_gb':4},
 {'id':'Qwen/Qwen2.5-1.5B-Instruct','parameters_b':1.5,'label':'Qwen2.5 1.5B','vram_guideline_gb':8},
 {'id':'Qwen/Qwen2.5-3B-Instruct','parameters_b':3,'label':'Qwen2.5 3B','vram_guideline_gb':12},
 {'id':'Qwen/Qwen2.5-7B-Instruct','parameters_b':7,'label':'Qwen2.5 7B (advanced)','vram_guideline_gb':24},
]

def hardware():
    out={'gpus':[],'disk_free_gb':round(shutil.disk_usage(ROOT).free/(1024**3),2)}
    exe=shutil.which('nvidia-smi')
    if exe:
        p=subprocess.run([exe,'--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if p.returncode==0:
            for line in p.stdout.splitlines():
                try:
                    name,mb=line.rsplit(',',1)
                    out['gpus'].append({'name':name.strip(),'vram_gb':round(float(mb.strip())/1024,2)})
                except ValueError:pass
    out['primary_vram_gb']=out['gpus'][0]['vram_gb'] if out['gpus'] else None
    return out

def model_suggestions(machine=None):
    machine=machine or hardware()
    vram=machine['primary_vram_gb']
    out=[]
    for m in MODEL_CATALOG:
        status='unknown GPU' if vram is None else ('plausible' if vram>=m['vram_guideline_gb'] else 'caution')
        out.append({**m,'status':status})
    return out

def display_models():
    h=hardware()
    for gpu in h['gpus']:print(f'GPU: {gpu["name"]} ({gpu["vram_gb"]} GiB)')
    if not h['gpus']:print('GPU: not detected by nvidia-smi. Compatibility unknown; training requires CUDA.')
    print(f'Disk free: {h["disk_free_gb"]} GiB')
    print('Conservative LoRA (not QLoRA) planning guidelines, not guarantees:')
    for i,m in enumerate(model_suggestions(h),1):
        print(f' {i}. {m["label"]} — {m["status"]} (guide >= {m["vram_guideline_gb"]} GiB)')
    print(' 0. Custom Hugging Face model ID (hardware compatibility unverified)')
    print('Model size alone does not determine trainability: sequence length, precision, LoRA/QLoRA, batch and architecture matter.')
    return h

def ask(prompt,default='',*,choices=None):
    suffix=f' [{default}]' if default!='' else ''
    while True:
        value=input(f'{prompt}{suffix}: ').strip()
        value=value if value else str(default)
        if choices is None or value.lower() in choices:return value
        print('Select one of:',', '.join(choices))

def confirm(prompt,default=False):
    return ask(prompt,'Y' if default else 'N',choices={'y','n','yes','no'}).lower() in ('y','yes')

def positive_int(prompt,default):
    while True:
        x=ask(prompt,default)
        try:
            n=int(x)
            if n>=1:return n
        except ValueError:pass
        print('Enter a positive whole number.')

def positive_float(prompt,default):
    while True:
        x=ask(prompt,default)
        try:
            n=float(x)
            if 0<n<1:return n
        except ValueError:pass
        print('Enter a positive learning rate smaller than 1.')

def import_file(dest,description):
    src=ask(f'Path to {description} (Enter to keep existing/empty)', '')
    if not src:return
    source=Path(src.strip('"').strip("'" )).expanduser().resolve()
    if not source.is_file():raise ValueError(f'Input file not found: {source}')
    if source==dest.resolve():return
    if source.suffix.lower()!=dest.suffix.lower():
        raise ValueError(f'Expected {dest.suffix} file for {description}; got {source.name}')
    shutil.copyfile(source,dest)
    print('Imported:',source,'->',dest)

def create_version_wizard(project,version,version_create):
    """Version identity + inputs only. Run-specific hyperparameters live in train wizard."""
    root=project_path(project)
    if not (root/'project.json').exists():raise RuntimeError(f'Unknown project: {project}')
    dst=version_path(project,version)
    resuming=dst.exists()
    if resuming:
        if not (dst/'version.json').is_file():
            raise RuntimeError(f'Incomplete version directory missing version.json: {dst}')
        state=read(dst/'version_state.json') if (dst/'version_state.json').is_file() else {'stage':'draft'}
        if state.get('stage') not in ('draft','validated'):
            raise RuntimeError(f'Cannot edit version {project}/{version}: state={state.get("stage")}')
        runs=ROOT/'runs'/project/version
        if runs.exists() and any(r.is_file() for r in runs.glob('*/run.json')):
            raise RuntimeError(f'Cannot reconfigure {project}/{version}: training/evaluation runs already exist')
        if not confirm(f'Resume editing existing draft {project}/{version}?',True):
            print('CANCELED: existing draft preserved');return
        print('RESUMING DRAFT:',dst)
    else:
        version_create(project,version)
        state={'stage':'draft','created_utc':utc()}
    cfg=read(dst/'version.json')
    try:
        existing=sorted(x.name for x in (root/'versions').iterdir() if x.name!=version and (x/'version.json').exists())
        if existing and not resuming and confirm('Copy model/prompt/schema SETTINGS from a previous version? (never weights/datasets)',False):
            print('Available:',', '.join(existing))
            copied=ask('Version to copy settings from',existing[-1])
            if copied not in existing:raise ValueError(f'Unknown source version: {copied}')
            parent=version_path(project,copied)
            orig=read(parent/'version.json')
            for key in ('base_model','include_schema_in_prompt','evaluation'):
                if key in orig:cfg[key]=orig[key]
            for f in ('prompt.txt','schema.json'):
                shutil.copyfile(parent/f,dst/f)
            cfg['derived_settings_from']=copied
        print('\nMODEL SELECTION')
        h=display_models()
        existing_id=cfg.get('base_model',{}).get('id')
        default_pick=next((str(i) for i,x in enumerate(MODEL_CATALOG,1) if x['id']==existing_id),'1')
        pick=ask('Model number or 0 for custom',default_pick,choices={str(i) for i in range(len(MODEL_CATALOG)+1)})
        if pick=='0':
            mid=ask('Hugging Face model ID',existing_id or '')
            if not mid or '/' not in mid:raise ValueError('Provide an organization/model Hugging Face ID')
        else:
            selected=MODEL_CATALOG[int(pick)-1]
            mid=selected['id']
            if h['primary_vram_gb'] is not None and h['primary_vram_gb'] < selected['vram_guideline_gb']:
                if not confirm('Model may exceed GPU memory; continue?',False):
                    print('Model selection canceled. Draft retained.');return
        print('\nBase model revision: main branch fetches the latest available model.')
        print('For reproducible results, enter a specific commit SHA instead.')
        revision=ask('Hugging Face branch or commit SHA',cfg.get('base_model',{}).get('revision','main'))
        cfg['base_model']={'id':mid,'revision':revision}
        write(dst/'version.json',cfg)
        print('\nINPUT FILES (copies saved inside this version)')
        for name,desc in [('prompt.txt','system prompt .txt'),('schema.json','schema .json'),('data/train.jsonl','training JSONL'),('data/eval.jsonl','evaluation JSONL')]:
            import_file(dst/name,desc)
        json.loads((dst/'schema.json').read_text(encoding='utf-8'))
        write(dst/'version_state.json',{'stage':'draft','created_utc':state.get('created_utc',utc()),'configured_utc':utc()})
        print('\nCONFIGURED VERSION',dst)
        print('NEXT: bash workspace.sh dataset validate',project,version)
        print('TRAIN: bash',str(root/'project.sh'),'train',version)
        print('All training hyperparameters will be reviewed at the START of each training run.')
    except (EOFError,KeyboardInterrupt):
        print('\nWizard interrupted. Draft version retained:',dst)
        raise
