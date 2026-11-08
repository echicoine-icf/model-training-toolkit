#!/usr/bin/env python3
"""Stable CLI for Model Training Toolkit. All commands use .venv automatically."""
import argparse, json, os, platform, shutil, subprocess, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from toolkit.core import ROOT, read, write, digest, utc, validated_name, project_path, version_path, files_for_version, snapshot_inputs, assert_snapshot, record
from toolkit.dataset import validate
from toolkit.wizard import hardware, display_models, confirm
from toolkit.run_wizard import wizard as training_wizard
from toolkit.data_package import initialize, inspect, validate_package, assistant_wizard, migrate_legacy


PROJECT_LAUNCHER = '#!/usr/bin/env bash\nset -Eeuo pipefail\nP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\nROOT="$(cd "$P/../.." && pwd)"\nNAME="$(basename "$P")"\nACTION="${1:-help}"\nshift || true\ncase "$ACTION" in\n new-version) [[ $# -ge 1 ]] || { echo \'Usage: bash project.sh new-version VERSION\' >&2; exit 2; }; exec bash "$ROOT/workspace.sh" version create "$NAME" "$@" ;;\n train) [[ $# -ge 1 ]] || { echo \'Usage: bash project.sh train VERSION\' >&2; exit 2; }; v="$1"; shift; exec bash "$ROOT/workspace.sh" train "$NAME" "$v" "$@" ;;\n evaluate) [[ $# -ge 1 ]] || { echo \'Usage: bash project.sh evaluate VERSION\' >&2; exit 2; }; v="$1"; shift; exec bash "$ROOT/workspace.sh" evaluate "$NAME" "$v" "$@" ;;\n status) exec bash "$ROOT/workspace.sh" status "$NAME" "$@" ;;\n handoff) exec bash "$ROOT/workspace.sh" handoff "$NAME" ;;\n *) echo \'Usage: bash project.sh {new-version VERSION|train VERSION|evaluate VERSION [--run ID]|status|handoff}\' ;;\nesac\n'

def scaffold_project_launcher(dest):
    dest=Path(dest)
    if not (dest/"project.sh").exists():
        (dest/"project.sh").write_text(PROJECT_LAUNCHER,encoding="utf-8")


def require(cond,message):
    if not cond:raise RuntimeError(message)


def project_new(name):
    dest=project_path(name)
    require(not dest.exists(),f'Project exists: {name}')
    (dest/'versions').mkdir(parents=True)
    write(dest/'project.json',{'name':name,'created_utc':utc(),'description':'Set project description here','toolkit_version':'0.3.4'})
    (dest/'PROJECT_NOTES.md').write_text('# Project Notes\n\nDescribe goals, schema assumptions, open decisions, and evaluation contract.\n',encoding='utf-8')
    # Project-local entrypoint, independent of current working directory and active venv.
    scaffold_project_launcher(dest)
    print('CREATED PROJECT',dest)
    print('PROJECT COMMANDS: bash',dest/'project.sh','new-version v1.0')


def version_new(project,version,*,model_id=None,revision=None):
    """Register a new version with an explicitly chosen base; inputs live in data/."""
    root=project_path(project)
    require((root/'project.json').is_file(),f'Unknown project: {project}')
    dest=version_path(project,version)
    if (dest/'version.json').is_file():
        print('EXISTING VERSION',dest)
        cfg=read(dest/'version.json')
        print('BASE MODEL',cfg.get('base_model',{}).get('id'),'REVISION',cfg.get('base_model',{}).get('revision'))
        print('Existing version preserved. To change its model, create a new version.')
    else:
        if model_id is None:
            require(sys.stdin.isatty(), 'Interactive model selection required when creating a version; or pass --model ORG/MODEL --revision REV')
            from toolkit.wizard import display_models, MODEL_CATALOG, ask, confirm, hardware
            print('\nBASE MODEL SELECTION (for this version)')
            machine=display_models()
            options={str(i) for i in range(len(MODEL_CATALOG)+1)}
            while True:
                pick=ask('Choose a model number (0 for custom Hugging Face ID)',choices=options)
                if pick=='0':
                    model_id=ask('Hugging Face organization/model ID')
                    if not model_id or '/' not in model_id:
                        print('Enter an ID such as Qwen/Qwen2.5-0.5B-Instruct.')
                        continue
                else:
                    selected=MODEL_CATALOG[int(pick)-1]
                    model_id=selected['id']
                    vram=machine.get('primary_vram_gb')
                    if vram is not None and vram<selected['vram_guideline_gb']:
                        if not confirm('Model may exceed available VRAM; choose it anyway?',False):
                            model_id=None
                            continue
                break
            print('\nBase model revision:')
            print('  main = latest version from the Hugging Face main branch')
            print('  Commit SHA = fixed revision for reproducible training')
            revision=ask('Branch or commit SHA','main')
        require(isinstance(model_id,str) and '/' in model_id and model_id.strip() == model_id,
                'Base model must be a nonempty Hugging Face organization/model ID')
        revision=revision or 'main'
        require(isinstance(revision,str) and bool(revision.strip()),'A branch or commit SHA is required')
        # No directory/config written until model selection completes successfully.
        (dest/'data').mkdir(parents=True,exist_ok=True)
        settings={
          'version':version,'project':project,'created_utc':utc(),'initialization':'fresh_base',
          'base_model':{'id':model_id,'revision':revision},
          'training':{'seed':42,'epochs':1,'max_length':4096,'gradient_accumulation':1,'learning_rate':2e-5,
             'weight_decay':0.01,'warmup_ratio':0.03,'max_grad_norm':1.0,'log_every':25,
             'lora':{'r':16,'alpha':32,'dropout':0.05,'target_modules':['q_proj','v_proj']}},
          'evaluation':{'max_new_tokens':384,'scorer':'exact_json'},
          'workflow':{'auto_evaluate_default':False},
          'notes':'Model explicitly selected during version creation. Version inputs live in data/. Per-run training parameters reviewed each time. Fresh base only.'}
        write(dest/'version.json',settings)
        write(dest/'version_state.json',{'stage':'draft','created_utc':utc()})
        print('CREATED VERSION',dest)
        print('BASE MODEL',model_id,'REVISION',revision)
    if not (dest/'data/package.json').exists():initialize(dest)
    check=inspect(dest)
    print('DATA STATUS:',json.dumps(check))
    if check['missing']:
        print('NEXT: supply version data ZIP under',dest/'data')
        print('OR: bash workspace.sh data wizard',project,version)
    else:
        try:
            result=validate_package(dest)
            print('DATA PACKAGE READY:',json.dumps({'train':result['train'],'eval':result['eval']}))
            if sys.stdin.isatty() and confirm('Data package detected. Start training configuration now?',False):
                train(project,version,guided=True)
        except (ValueError,FileNotFoundError) as ex:
            print('DATA PACKAGE NEEDS ATTENTION:',ex)
            print('Try bash workspace.sh data wizard',project,version)


def version_validate(project,version,*,need_eval=False):
    v=version_path(project,version)
    config=read(v/'version.json')
    require(config.get('project')==project and config.get('version')==version,'Version identity mismatch')
    require(config.get('initialization')=='fresh_base','Only fresh_base is supported')
    require(bool(config['base_model']['id']),'Base model identifier missing')
    result=validate_package(v,need_eval=need_eval)
    print('DATASET VALIDATION',json.dumps(result))
    write(v/'version_state.json',{'stage':'validated','checked_utc':utc(),'dataset_counts':{'train':result['train'],'eval':result['eval']}})
    return result


def run_path(project,version,run_id):
    validated_name(run_id)
    return ROOT/'runs'/validated_name(project)/validated_name(version)/run_id


def resolved_model(base):
    # Resolve mutable branch names before downloading weights, so subsequent eval/export
    # reconstruct the exact same original model even if the upstream branch changes.
    base=dict(base)
    revision=base.get('revision') or 'main'
    if len(revision)==40 and all(c in '0123456789abcdefABCDEF' for c in revision):
        base['revision']=revision;return base
    try:
        from huggingface_hub import HfApi
        info=HfApi().model_info(repo_id=base['id'],revision=revision)
        require(bool(info.sha),f'Cannot resolve Hugging Face revision for {base["id"]}')
    except ImportError as ex:
        raise RuntimeError('Hugging Face revision resolution requires ML dependencies. Run bash run_me_first.sh --install-ml.') from ex
    except Exception as ex:
        raise RuntimeError(f'Cannot pin {base["id"]} revision {revision!r} to a commit: {ex}. Check internet or enter a 40-character commit SHA.') from ex
    base['requested_revision']=revision
    base['revision']=info.sha
    print('PINNED BASE REVISION',base['id'],revision,'=>',info.sha)
    return base


def new_run(project,version,*,training_overrides=None,base_override=None):
    validated_name(project);validated_name(version)
    version_validate(project,version)
    pinned_base=resolved_model(base_override) if base_override else None
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    # Avoid collision even if multiple runs begin in the same millisecond.
    import secrets
    run_id='run_'+stamp+'_'+secrets.token_hex(3)
    folder=run_path(project,version,run_id)
    snapshot_inputs(project,version,folder/'input')
    if pinned_base:
        snap=folder/'input';cfg=read(snap/'version.json')
        cfg['base_model']=pinned_base
        write(snap/'version.json',cfg)
        hashes=read(snap/'input_hashes.json');hashes['version.json']=digest(snap/'version.json');write(snap/'input_hashes.json',hashes)
    if training_overrides:
        snap=folder/'input'
        cfg=read(snap/'version.json')
        cfg['training'].update(training_overrides)
        write(snap/'version.json',cfg)
        hashes=read(snap/'input_hashes.json')
        hashes['version.json']=digest(snap/'version.json')
        write(snap/'input_hashes.json',hashes)
    meta={'run_id':run_id,'project':project,'version':version,'created_utc':utc(),'status':'prepared','toolkit_version':'0.3.4','initialization':'fresh_base','run_kind':'training','training_overrides':training_overrides or {}}
    write(folder/'run.json',meta)
    return folder,meta


def update_run(folder,meta,status,other=None):
    meta=dict(meta);meta['status']=status;meta['updated_utc']=utc()
    if other:meta.update(other)
    write(folder/'run.json',meta)
    write(version_path(meta['project'],meta['version'])/'version_state.json',{'stage':status,'last_run_id':meta['run_id'],'updated_utc':meta['updated_utc']})
    record({'run_id':meta['run_id'],'project':meta['project'],'version':meta['version'],'status':status,'updated_utc':meta['updated_utc'],'run_json':str((folder/'run.json').relative_to(ROOT))})
    return meta


# Deliberately require all values in automated runs. A forgotten flag must never
# silently fall back to a prior version's hyperparameter or evaluation policy.
TRAIN_FLAG_ATTRS = ('epochs','max_length','learning_rate','gradient_accumulation',
                    'seed','weight_decay','warmup_ratio','max_grad_norm','log_every',
                    'lora_r','lora_alpha','lora_dropout','lora_target_modules')


def explicit_training_config(args):
    require(args is not None,'Missing CLI training arguments')
    missing=[f'--{k.replace("_","-")}' for k in TRAIN_FLAG_ATTRS if getattr(args,k,None) is None]
    if getattr(args,'auto_eval',False) == getattr(args,'no_auto_eval',False):
        missing.append('exactly one of --auto-eval / --no-auto-eval')
    if not getattr(args,'yes',False):missing.append('--yes (explicitly authorize training)')
    require(not missing,'Noninteractive training requires: '+', '.join(missing))
    lmods=[x.strip() for x in args.lora_target_modules.split(',')]
    controls={'epochs':args.epochs,'max_length':args.max_length,'learning_rate':args.learning_rate,
        'gradient_accumulation':args.gradient_accumulation,'seed':args.seed,
        'weight_decay':args.weight_decay,'warmup_ratio':args.warmup_ratio,
        'max_grad_norm':args.max_grad_norm,'log_every':args.log_every,
        'lora':{'r':args.lora_r,'alpha':args.lora_alpha,
                'dropout':args.lora_dropout,'target_modules':lmods}}
    require(args.epochs>=1,'--epochs must be >= 1')
    require(args.max_length>=128,'--max-length must be >= 128')
    require(0<args.learning_rate<1,'--learning-rate must be between 0 and 1 (exclusive)')
    require(args.gradient_accumulation>=1,'--gradient-accumulation must be >= 1')
    require(args.seed>=0,'--seed must be >= 0')
    require(0<=args.weight_decay<1,'--weight-decay must be in [0,1)')
    require(0<=args.warmup_ratio<1,'--warmup-ratio must be in [0,1)')
    require(args.max_grad_norm>0,'--max-grad-norm must be > 0')
    require(args.log_every>=1,'--log-every must be >= 1')
    require(args.lora_r>=1 and args.lora_alpha>=1,'--lora-r and --lora-alpha must be >= 1')
    require(0<=args.lora_dropout<1,'--lora-dropout must be in [0,1)')
    require(bool(lmods) and all(lmods) and len(set(lmods))==len(lmods),
            '--lora-target-modules must be nonempty unique comma-separated names')
    # Optional advanced settings are validated if supplied, otherwise their
    # defaults are explicitly captured into the run snapshot and training plan.
    min_lr = getattr(args,'min_learning_rate',None)
    if min_lr is not None:
        require(0 < min_lr <= args.learning_rate,
                '--min-learning-rate must be positive and <= --learning-rate')
        controls['min_learning_rate'] = min_lr
    else:
        controls['min_learning_rate'] = args.learning_rate * 0.1
    opts = {
        'precision': ('auto', 'bf16', 'fp16'),
        'attention_implementation': ('sdpa', 'eager', 'flash_attention_2'),
        'optimizer': ('adamw', 'fused-adamw'),
    }
    defaults = {'precision':'auto','attention_implementation':'sdpa','optimizer':'fused-adamw'}
    for key, allowed in opts.items():
        value = getattr(args,key,None)
        value = defaults[key] if value is None else value
        require(value in allowed, f'--{key.replace("_","-")} must be one of {allowed}')
        controls[key] = value
    ck = getattr(args,'gradient_checkpointing',None)
    no_ck = getattr(args,'no_gradient_checkpointing',None)
    require(not (ck and no_ck),'Choose only one of --gradient-checkpointing and --no-gradient-checkpointing')
    controls['gradient_checkpointing'] = False if no_ck else True
    save_every = getattr(args,'save_every',None)
    save_every = 0 if save_every is None else save_every
    require(save_every >= 0,'--save-every must be >=0')
    controls['save_every'] = save_every
    return controls,bool(args.auto_eval)


def train(project,version,*,guided=True,train_args=None):
    """Train with either guided review or fully explicit, recorded CLI arguments."""
    if not guided:
        explicit_training_config(train_args)  # fail fast before reading data or creating a run
    v=version_path(project,version)
    config=read(v/'version.json')
    print('\nCHECKING VERSION DATA PACKAGE',v/'data')
    package_status=inspect(v)
    if package_status['missing']:
        print('Version inputs missing:',', '.join(package_status['missing']))
        if not guided or not sys.stdin.isatty():raise RuntimeError('Training inputs missing; supply the version data package before noninteractive training. Run data wizard manually if needed.')
        if not assistant_wizard(v):
            print('CANCELED: supply missing files, then rerun training');return
    version_validate(project,version)
    prior=None
    runs=ROOT/'runs'/validated_name(project)/validated_name(version)
    if runs.is_dir():
        options=sorted(runs.glob('*/run.json'),reverse=True)
        for path in options:
            rec=read(path)
            if rec.get('reviewed_training') and rec.get('training_overrides'):
                prior=rec['training_overrides']
                break
    if guided:
        print(f'\nTRAINING WIZARD {project}/{version}')
        selected=training_wizard(config,previous=prior)
        if selected is None:return
        controls,auto_eval,base_override=selected
    else:
        controls,auto_eval=explicit_training_config(train_args)
        base_override=config.get('base_model')
        require(bool(base_override and base_override.get('id') and base_override.get('revision')),
                'Base model must be selected at version creation')
        print('NONINTERACTIVE TRAINING PLAN',json.dumps({
          'project':project,'version':version,'base_model':base_override,
          'initialization':'fresh_base','training':controls,'auto_evaluate':auto_eval},indent=2))
    if auto_eval:
        try:version_validate(project,version,need_eval=True)
        except ValueError as exc:
            print('AUTO-EVALUATION INPUTS REQUIRED:',exc)
            if guided and sys.stdin.isatty() and assistant_wizard(v,need_eval=True):
                version_validate(project,version,need_eval=True)
            else:
                print('CANCELED: no training started because automatic evaluation lacks its dataset')
                return
    # Perform validation and snapshot after review, before any ML initialization.
    folder,meta=new_run(project,version,training_overrides=controls,base_override=base_override)
    meta.update({'reviewed_training':True,'auto_evaluate':auto_eval,'run_configuration':{'training':controls,'base_model':read(folder/'input/version.json')['base_model']}})
    write(folder/'run.json',meta)
    meta=update_run(folder,meta,'training')
    try:
        from toolkit.train_engine import run as engine
        frozen=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        (folder/'pip_freeze.txt').write_text(frozen,encoding='utf-8')
        assert_snapshot(folder/'input')
        result=engine(folder/'input',folder)
        meta=update_run(folder,meta,'trained',{'result':result})
        print('TRAINED',meta['run_id'],'at',folder)
        if auto_eval:
            print('AUTO-EVALUATION REQUESTED:',meta['run_id'])
            try:evaluate(project,version,meta['run_id'])
            except Exception as ex:
                print('AUTO-EVALUATION FAILED:',str(ex),file=sys.stderr)
                raise
        print('RUN FINISHED:',meta['run_id'])
    except Exception as exc:
        current=read(folder/'run.json')
        if current['status']=='training':update_run(folder,current,'failed',{'error':str(exc)})
        raise


def chosen_run(project,version,run_id=None):
    parent=ROOT/'runs'/validated_name(project)/validated_name(version)
    if run_id:
        r=run_path(project,version,run_id)
        require((r/'run.json').is_file(),f'Unknown run {run_id}')
        require(read(r/'run.json').get('run_kind','training')=='training','Requested run is a base-only baseline without a trained adapter')
        return r
    candidates=[r for r in parent.iterdir() if (r/'run.json').is_file() and read(r/'run.json').get('status') in ('trained','evaluated','exported','evaluation_failed') and read(r/'run.json').get('run_kind','training')=='training'] if parent.is_dir() else []
    require(bool(candidates),f'No trained runs for {project}/{version}. Use --run to choose an existing run.')
    return sorted(candidates)[-1]


def evaluate(project,version,run_id=None,*,base_only=False):
    if base_only:
        require(run_id is None,'--base-only cannot be combined with --run')
        folder,meta=new_run(project,version)
        meta=update_run(folder,meta,'evaluating_base',{'run_kind':'baseline'})
        adapter=None
    else:
        folder=chosen_run(project,version,run_id)
        meta=read(folder/'run.json');require(meta['status'] in ('trained','evaluated','exported','evaluation_failed'),'Run not trained')
        adapter=folder/'adapter';require(adapter.is_dir(),'Missing adapter output')
    assert_snapshot(folder/'input')
    validate_package(folder/'input',need_eval=True)
    out=folder/'evaluation'
    require(not out.exists(),f'Evaluation already exists for this run: {out}; create a new run instead of overwriting')
    from toolkit.eval_engine import run as engine
    meta=update_run(folder,meta,'evaluating')
    try:
        score=engine(folder/'input',out,adapter)
        update_run(folder,meta,'evaluated',{'evaluation_summary':score})
        print('RESULTS',out)
    except Exception as exc:
        update_run(folder,meta,'failed' if base_only else 'evaluation_failed',{'evaluation_error':str(exc)})
        raise


def export_model(project,version,run_id=None,fmt='hf'):
    folder=chosen_run(project,version,run_id);meta=read(folder/'run.json')
    adapter=folder/'adapter'
    require(meta['status'] in ('trained','evaluated','exported','evaluation_failed'),'Run has not completed training')
    require(adapter.is_dir(),'No adapter, run training first')
    snap=folder/'input';assert_snapshot(snap)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM,AutoTokenizer
    except ImportError as exc:raise RuntimeError('Optional ML dependencies missing') from exc
    cfg=read(snap/'version.json');base=cfg['base_model'];destination=folder/'exports/merged_hf'
    require(not destination.exists(),'Merged model already exported; refusing overwrite')
    model=AutoModelForCausalLM.from_pretrained(base['id'],revision=base.get('revision') or 'main',torch_dtype=torch.float32,trust_remote_code=False)
    model=PeftModel.from_pretrained(model,str(adapter)).merge_and_unload()
    destination.mkdir(parents=True)
    model.save_pretrained(destination,safe_serialization=True)
    tokenizer=AutoTokenizer.from_pretrained(str(adapter),trust_remote_code=False)
    tokenizer.save_pretrained(destination)
    result={'format':'safetensors','merged_hf':str(destination),'exported_utc':utc()}
    if fmt=='gguf':
        cpp=ROOT/'tools/llama.cpp';converter=cpp/'convert_hf_to_gguf.py'
        require(converter.is_file(),'llama.cpp converter not available: rerun setup without --skip-llama')
        outfile=folder/'exports/model-f16.gguf'
        require(not outfile.exists(),'GGUF exists; refusing overwrite')
        subprocess.run([sys.executable,str(converter),str(destination),'--outfile',str(outfile),'--outtype','f16'],check=True)
        result.update({'format':'gguf-f16','gguf_path':str(outfile),'gguf_sha256':digest(outfile)})
    write(folder/'export.json',result);update_run(folder,meta,'exported')
    print('EXPORTED',json.dumps(result,indent=2))



def upgrade():
    config=read(ROOT/'config/workspace.json')
    statefile=ROOT/'WORKSPACE_STATE.json'
    require(statefile.exists(),'Workspace not installed: run run_me_first.sh first')
    state=read(statefile)
    require((ROOT/'.venv/pyvenv.cfg').exists(),'Cannot upgrade without local .venv')
    old=state.get('toolkit_version')
    state['toolkit_version']=config['toolkit_version']
    state['last_upgrade_utc']=utc()
    for project in (ROOT/'projects').iterdir():
        if (project/'project.json').is_file():
            (project/'project.sh').write_text(PROJECT_LAUNCHER,encoding='utf-8')
            for folder in (project/'versions').iterdir():
                if (folder/'version.json').exists() and migrate_legacy(folder):
                    print('MIGRATED DATA CONTRACT (preserved original files):',folder)
    write(statefile,state)
    print('UPGRADED WORKSPACE',old,'->',config['toolkit_version'],'(existing .venv, projects, and runs preserved)')


def list_projects():
    for project in sorted((ROOT/'projects').iterdir()):
        if (project/'project.json').is_file():print(project.name)


def list_versions(project):
    parent=project_path(project)/'versions'
    require(parent.is_dir(),f'Unknown project: {project}')
    for v in sorted(parent.iterdir()):
        if (v/'version.json').exists():print(v.name,read(v/'version.json').get('base_model',{}).get('id','NOT SELECTED'))


def status(project,version=None):
    require((project_path(project)/'project.json').is_file(),f'Unknown project: {project}')
    vs=[version] if version else [p.name for p in sorted((project_path(project)/'versions').iterdir()) if (p/'version.json').exists()]
    print('PROJECT',project)
    for v in vs:
        parent=version_path(project,v)
        cfg=read(parent/'version.json')
        print('VERSION',v,'MODEL',cfg.get('base_model',{}).get('id','NOT SELECTED'))
        runs=ROOT/'runs'/project/v
        found=[]
        for f in sorted(runs.glob('*/run.json')) if runs.is_dir() else []:
            m=read(f);found.append(m)
            print('  RUN',m['run_id'],'STATUS',m['status'])
        if not found:print('  STATUS draft (no training runs)')

def doctor():
    state=read(ROOT/'WORKSPACE_STATE.json')
    print('WORKSPACE',ROOT)
    print('TOOLKIT_CODE_VERSION',read(ROOT/'config/workspace.json')['toolkit_version'])
    print('STATE',json.dumps(state,indent=2))
    print('VIRTUALENV',sys.prefix, 'isolated:',sys.prefix!=sys.base_prefix)
    print('PYTHON',sys.version.split()[0]);print('PLATFORM',platform.platform())
    print('DISK_FREE_GB',round(shutil.disk_usage(ROOT).free/1024**3,2))
    cpp=ROOT/'tools/llama.cpp'
    print('LLAMA_CPP_SOURCE',cpp.is_dir())
    if cpp.is_dir() and (cpp/'.git').exists():
        print('LLAMA_CPP_SHA',subprocess.check_output(['git','-C',str(cpp),'rev-parse','HEAD'],text=True).strip())
    try:
        import torch
        print('TORCH',torch.__version__,'CUDA_AVAILABLE',torch.cuda.is_available())
        if torch.cuda.is_available():print('GPU',torch.cuda.get_device_name(0),'CUDA_RUNTIME',torch.version.cuda)
    except ImportError:print('TORCH: not installed (optional profile)')
    print('READY' if sys.prefix!=sys.base_prefix else 'WARNING: not running from isolated venv')


def list_runs(project=None):
    root=ROOT/'runs'
    for path in sorted(root.glob('**/run.json')) if root.exists() else []:
        m=read(path)
        if project and m['project']!=project:continue
        print(m['run_id'],m['project'],m['version'],m['status'],m.get('result',{}).get('loss',''))


def compare(project,versions):
    found=[]
    for version in versions:
        parent=ROOT/'runs'/validated_name(project)/validated_name(version)
        for p in sorted(parent.glob('*/evaluation/summary.json')) if parent.exists() else []:
            x=read(p);r=read(p.parents[1]/'run.json')
            found.append((version,r['run_id'],x['cases'],x['exact_accuracy']))
    if not found:print('No evaluation summaries found')
    print('VERSION\tRUN\tN\tEXACT_ACCURACY')
    for x in found:print('\t'.join(map(str,x)))
    print('IMPORTANT: exact structural match is not semantic equivalence; only compare runs with the same frozen evaluation data.')


def handoff(project):
    path=project_path(project);require((path/'project.json').is_file(),'Unknown project')
    lines=['# Model Training Toolkit — Project Handoff','',f'Generated: {utc()}','',f'Project: **{project}**','', '## Workspace',f'- Toolkit: {read(ROOT/"config/workspace.json")["toolkit_version"]}',f'- Project description: {read(path/"project.json").get("description","")}',f'- Root: {ROOT}','- Command interface: `bash workspace.sh`','- Activate isolated environment: `source ./activate.sh`','']
    lines+=['## Registered versions','', '- Data package contract: `docs/DATA_PACKAGE_CONTRACT.md` (mtt-data-v1)', '']
    for v in sorted((path/'versions').iterdir()):
        if not (v/'version.json').is_file():continue
        cfg=read(v/'version.json')
        lines+=[f'### {v.name}',f'- Version base: `{cfg["base_model"]["id"]}` at `{cfg["base_model"].get("revision")}` (actual commit pinned per run)',f'- Training initialization: `{cfg.get("initialization")}`',f'- Data package: `projects/{project}/versions/{v.name}/data/`']
        package=v/'data/package.json'
        if package.is_file():
            pc=read(package)
            lines.append(f'- Package: `{pc.get("format")}`, task `{pc.get("task")}`, prompt assembly `{pc.get("prompt_assembly")}`')
            lines.append(f'- Evaluator: `{pc.get("evaluation",{}).get("scorer")}` (not semantic graph validation)')
        for f in files_for_version(project,v.name):
            if f.exists():lines.append(f'- `{f.relative_to(path)}` SHA256: `{digest(f)}`')
        versionruns=ROOT/'runs'/project/v.name
        if versionruns.exists():
            for mfile in sorted(versionruns.glob('*/run.json')):
                m=read(mfile)
                lines.append(f'- Run `{m["run_id"]}`: {m["status"]}')
                frozen=mfile.parent/'input/version.json'
                if frozen.is_file():
                    bm=read(frozen).get('base_model',{})
                    lines.append(f'  - Pinned run base: `{bm.get("id")}` at `{bm.get("revision")}`')
                summ=mfile.parent/'evaluation/summary.json'
                if summ.exists():
                    stats=read(summ);lines.append(f'  - Eval: {stats["exact"]}/{stats["cases"]}, exact_accuracy={stats["exact_accuracy"]}')
                training=mfile.parent/'training_result.json'
                if training.exists():lines.append(f'  - Training: {read(training)["step"]} steps')
        lines.append('')
    lines+=['## Rules for future work','', '- Never edit toolkit engine files for a new model version; create a new version configuration and input dataset.', '- Training always starts from the declared base model (no implicit adapter continuation).', '- Preserve frozen run inputs, checkpoints, evaluations and manifests.', '- Do not call exact JSON comparison a semantic evaluation; add a project-specific evaluator when required.', '- Project-specific details and unresolved issues belong in `PROJECT_NOTES.md`.','']
    notes=path/'PROJECT_NOTES.md'
    if notes.exists():lines+=['## Project Notes','',notes.read_text(encoding='utf-8')]
    destination=path/'PROJECT_HANDOFF.md';destination.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('HANDOFF GENERATED',destination)


def main():
    p=argparse.ArgumentParser(prog='workspace.sh',description='Model Training Toolkit 0.3.4')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    sub.add_parser('upgrade')
    sub.add_parser('hardware')
    sub.add_parser('models')
    a=sub.add_parser('status');a.add_argument('project');a.add_argument('version',nargs='?')
    a=sub.add_parser('project');a.add_argument('action',choices=['create','list']);a.add_argument('name',nargs='?')
    a=sub.add_parser('version');a.add_argument('action',choices=['create','wizard','list']);a.add_argument('project');a.add_argument('version',nargs='?');a.add_argument('--model');a.add_argument('--revision')
    a=sub.add_parser('dataset');a.add_argument('action',choices=['validate']);a.add_argument('project');a.add_argument('version')
    a=sub.add_parser('data');a.add_argument('action',choices=['inspect','wizard','validate']);a.add_argument('project');a.add_argument('version');a.add_argument('--require-eval',action='store_true')
    a=sub.add_parser('train',help='Interactive wizard or explicit flag-driven training')
    a.add_argument('project');a.add_argument('version')
    a.add_argument('--guided',action='store_true',help='Interactive configuration wizard')
    a.add_argument('--epochs',type=int)
    a.add_argument('--max-length',type=int)
    a.add_argument('--learning-rate',type=float)
    a.add_argument('--gradient-accumulation',type=int)
    a.add_argument('--seed',type=int)
    a.add_argument('--weight-decay',type=float)
    a.add_argument('--warmup-ratio',type=float)
    a.add_argument('--max-grad-norm',type=float)
    a.add_argument('--log-every',type=int)
    a.add_argument('--lora-r',type=int)
    a.add_argument('--lora-alpha',type=int)
    a.add_argument('--lora-dropout',type=float)
    a.add_argument('--lora-target-modules')
    a.add_argument('--min-learning-rate',type=float,help='Absolute cosine learning-rate floor')
    a.add_argument('--precision',choices=['auto','bf16','fp16'])
    a.add_argument('--attention-implementation',choices=['sdpa','eager','flash_attention_2'])
    a.add_argument('--optimizer',choices=['fused-adamw','adamw'])
    ckpt=a.add_mutually_exclusive_group()
    ckpt.add_argument('--gradient-checkpointing',action='store_true')
    ckpt.add_argument('--no-gradient-checkpointing',action='store_true')
    a.add_argument('--save-every',type=int,help='Adapter-only intermediate checkpoint interval (0 disables)')
    decision=a.add_mutually_exclusive_group()
    decision.add_argument('--auto-eval',action='store_true')
    decision.add_argument('--no-auto-eval',action='store_true')
    a.add_argument('--yes',action='store_true',help='Explicitly authorize the full run plan; no prompts')
    a=sub.add_parser('evaluate');a.add_argument('project');a.add_argument('version');a.add_argument('--run');a.add_argument('--base-only',action='store_true')
    a=sub.add_parser('export');a.add_argument('project');a.add_argument('version');a.add_argument('--run');a.add_argument('--format',choices=['hf','gguf'],default='hf')
    a=sub.add_parser('compare');a.add_argument('project');a.add_argument('versions',nargs='+')
    a=sub.add_parser('runs');a.add_argument('--project')
    a=sub.add_parser('handoff');a.add_argument('project')
    a=p.parse_args()
    try:
        if a.command=='doctor':doctor()
        elif a.command=='upgrade':upgrade()
        elif a.command=='hardware':print(json.dumps(hardware(),indent=2))
        elif a.command=='models':display_models()
        elif a.command=='status':status(a.project,a.version)
        elif a.command=='project':list_projects() if a.action=='list' else project_new(a.name)
        elif a.command=='version':list_versions(a.project) if a.action=='list' else version_new(a.project,a.version,model_id=a.model,revision=a.revision)
        elif a.command=='dataset':version_validate(a.project,a.version)
        elif a.command=='data':
            folder=version_path(a.project,a.version)
            require((folder/'version.json').is_file(),'Version not registered. Run version create first.')
            if a.action=='inspect':print('DATA STATUS',json.dumps(inspect(folder,need_eval=a.require_eval),indent=2))
            elif a.action=='wizard':
                if not assistant_wizard(folder,need_eval=a.require_eval):print('Data wizard ended without all inputs')
                else:print('DATA READY',json.dumps(validate_package(folder,need_eval=a.require_eval)))
            else:version_validate(a.project,a.version,need_eval=a.require_eval)
        elif a.command=='train':
            has_flags=(any(getattr(a,k) is not None for k in TRAIN_FLAG_ATTRS) or a.auto_eval or a.no_auto_eval or a.yes or
                any(getattr(a,k,None) is not None for k in ('min_learning_rate','precision','attention_implementation','optimizer','save_every')) or
                a.gradient_checkpointing or a.no_gradient_checkpointing)
            require(not (a.guided and has_flags),'--guided cannot be combined with noninteractive training flags')
            train(a.project,a.version,guided=(a.guided or not has_flags),train_args=(None if a.guided or not has_flags else a))
        elif a.command=='evaluate':evaluate(a.project,a.version,a.run,base_only=a.base_only)
        elif a.command=='export':export_model(a.project,a.version,a.run,a.format)
        elif a.command=='compare':compare(a.project,a.versions)
        elif a.command=='runs':list_runs(a.project)
        elif a.command=='handoff':handoff(a.project)
    except (ValueError,FileNotFoundError,RuntimeError,KeyError,EOFError) as ex:
        print('ERROR:',ex,file=sys.stderr);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
