"""MTT Bedrock remote SFT lifecycle (optional boto3; never stores credentials).

This module is deliberately independent of MTT's local HF/LoRA trainer.
The existing toolkit.core snapshot and shared benchmark contracts remain authoritative.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import core
from .dataset import rows as dataset_rows

FORMAT = 'mtt-bedrock-nova-sft-v1'
NOVA_2_LITE_RUNTIME_MODEL = 'amazon.nova-2-lite-v1:0'
DEFAULT_MODEL = NOVA_2_LITE_RUNTIME_MODEL + ':256k'
NOVA_2_LITE_SFT_DEFAULTS = {'epochCount':'2','learningRate':'0.00001','learningRateWarmupSteps':'10'}
RESERVED = ('User:', 'Bot:', 'Assistant:', 'System:', '<image>', '<video>', '[EOS]')
TERMINAL_TRAIN = frozenset({'Completed', 'Failed', 'Stopped'})
TERMINAL_DEPLOY = frozenset({'Active', 'Failed'})
SECRET_KEYS = re.compile(r'(secret|password|access[_-]?key|session[_-]?token|authorization|credential)', re.I)
NAME = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized(text):
    return ' '.join(text.casefold().split())


def safe_public(obj):
    """Remove credential-shaped keys recursively, even from future AWS responses."""
    if isinstance(obj, dict):
        return {k: safe_public(v) for k, v in obj.items() if not SECRET_KEYS.search(str(k))}
    if isinstance(obj, list):
        return [safe_public(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def json_sha(obj) -> str:
    return sha(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))


def load_config(project: str, version: str, overrides=None, env=None, root=None):
    """CLI > version > project > MTT env > standard boto credential/region resolution."""
    root = Path(root or core.ROOT)
    v = root/'projects'/core.validated_name(project)/'versions'/core.validated_name(version)
    cfg_path = v/'version.json'
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Missing version: {cfg_path}; create it with remote init')
    version_obj = core.read(cfg_path)
    project_obj = core.read(v.parent.parent/'project.json')
    def section(obj):
        return obj.get('remote', {}).get('bedrock', {})
    vv, pv = section(version_obj), section(project_obj)
    overrides = overrides or {}
    env = os.environ if env is None else env
    env_names = {
        'profile':'MTT_AWS_PROFILE', 'region':'MTT_AWS_REGION',
        'role_arn':'MTT_BEDROCK_ROLE_ARN', 'bucket':'MTT_BEDROCK_S3_BUCKET',
        'prefix':'MTT_BEDROCK_S3_PREFIX', 'training_prefix':'MTT_BEDROCK_TRAINING_PREFIX',
        'output_prefix':'MTT_BEDROCK_OUTPUT_PREFIX', 'base_model_id':'MTT_BEDROCK_BASE_MODEL_ID',
        'model_prefix':'MTT_BEDROCK_MODEL_PREFIX', 'count_model_id':'MTT_BEDROCK_COUNT_MODEL_ID',
    }
    out = {}
    for key, env_name in env_names.items():
        value = next((x[key] for x in (overrides, vv, pv) if x.get(key) not in (None, '')), None)
        out[key] = value if value is not None else env.get(env_name)
    out['base_model_id'] = out['base_model_id'] or DEFAULT_MODEL
    out['prefix'] = out['prefix'] or project
    out['model_prefix'] = out['model_prefix'] or f'{project}-{version}'
    # Do not persist AWS authentication or per-process environment metadata.
    for key in (set(vv) | set(pv) | set(overrides)):
        if SECRET_KEYS.search(key):
            raise ValueError(f'Credential-like field is not allowed in Bedrock config: {key}')
    for field in ('bucket', 'role_arn'):
        if out[field] and not isinstance(out[field], str):
            raise ValueError(f'Expected string for {field}')
    return out


def aws_session(cfg, factory=None):
    if factory is None:
        try:
            import boto3
        except ImportError as ex:
            raise RuntimeError('Bedrock workflow requires optional boto3: pip install -r requirements/bedrock.txt') from ex
        factory = boto3.Session
    kw = {}
    if cfg.get('profile'): kw['profile_name'] = cfg['profile']
    if cfg.get('region'): kw['region_name'] = cfg['region']
    session = factory(**kw)
    if not session.region_name:
        raise RuntimeError('AWS region unresolved; set --region, project/version remote config, MTT_AWS_REGION or AWS_REGION')
    return session


def clients(cfg, factory=None):
    s = aws_session(cfg, factory=factory)
    return s.client('bedrock'), s.client('s3'), s.client('sts'), s.client('bedrock-runtime'), s.region_name


def verified_clients(state, factory=None):
    found=clients({'profile':state.get('profile'),'region':state['region']},factory)
    expected=state.get('account_id')
    if expected and found[2].get_caller_identity().get('Account') != expected:
        raise RuntimeError('AWS account does not match persisted remote job identity')
    return found


def training_questions(data: Path, prompt: str, *, count_tokens=None, model_id=None):
    """Strict text-only Nova 2 SFT validation; returns normalized questions and audit rows.

    Token validation is exact only when count_tokens is supplied. Text-only ELI
    avoids silently accepting multimedia/tool contracts that we haven't audited.
    """
    questions = set()
    audit = []
    line_count = 0
    with data.open('r', encoding='utf-8-sig') as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'Training line {lineno}: blank JSONL line')
            if len(line.encode('utf-8')) > 10 * 1024 * 1024:
                raise ValueError(f'Training line {lineno}: exceeds conservative 10 MiB per-record guard')
            try: row = json.loads(line)
            except (ValueError, TypeError) as ex: raise ValueError(f'Training line {lineno}: invalid JSON: {ex}') from ex
            if not isinstance(row, dict): raise ValueError(f'Training line {lineno}: expected object')
            if not isinstance(row.get('schemaVersion'), str) or not row['schemaVersion']:
                raise ValueError(f'Training line {lineno}: schemaVersion must be present')
            system = row.get('system')
            if system != [{'text': prompt}]:
                raise ValueError(f'Training line {lineno}: system content differs from exact production system prompt')
            turns = row.get('messages')
            if not isinstance(turns, list) or len(turns) < 2 or len(turns) % 2:
                raise ValueError(f'Training line {lineno}: expected alternating user/assistant turns')
            for i, message in enumerate(turns):
                if not isinstance(message, dict) or message.get('role') != ('user' if i % 2 == 0 else 'assistant'):
                    raise ValueError(f'Training line {lineno}: invalid role ordering at turn {i}')
                content = message.get('content')
                if not isinstance(content, list) or not content or any(not isinstance(b, dict) or set(b) != {'text'} or not isinstance(b['text'],str) or not b['text'].strip() for b in content):
                    raise ValueError(f'Training line {lineno}: only nonempty text content blocks supported')
            for text in [prompt, *(b['text'] for m in turns for b in m['content'])]:
                bad = next((x for x in RESERVED if x in text), None)
                if bad: raise ValueError(f'Training line {lineno}: Nova reserved string {bad!r} present')
            q = '\n'.join(b['text'] for b in turns[0]['content'])
            key = normalized(q)
            if key in questions:
                raise ValueError(f'Training line {lineno}: duplicate normalized training question')
            questions.add(key)
            ans = '\n'.join(b['text'] for b in turns[-1]['content'])
            try: expected = json.loads(ans)
            except ValueError: expected = ans
            audit.append({'id': f'nova_train_{lineno:06}', 'question': q, 'expected': expected})
            if count_tokens is not None:
                result = count_tokens(modelId=model_id, input={'converse':{'messages':turns,'system':system}})
                if int(result['inputTokens']) > 32000:
                    raise ValueError(f'Training line {lineno}: {result["inputTokens"]} tokens exceeds 32K context')
            line_count += 1
    if not 8 <= line_count <= 20000:
        raise ValueError(f'Nova 2 Lite SFT requires 8–20,000 training examples; found {line_count}')
    return {'count':line_count, 'training_file_sha256':core.digest(data), 'system_prompt_sha256':sha(prompt.encode('utf-8')), 'questions':questions, 'audit_rows':audit,
            'token_validation':'bedrock-count-tokens' if count_tokens else 'not-verified'}


def extract_questions(path, *, allow_empty=False):
    """Fail closed on unknown protected formats, never mark them validated silently."""
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    try:
        if path.suffix.lower() == '.jsonl':
            data = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        else: data = json.loads(text)
    except Exception as ex:
        raise ValueError(f'Cannot parse protected evaluation dataset {path}: {ex}') from ex
    found = set()
    def walk(value):
        if isinstance(value, list):
            for item in value: walk(item)
            return
        if not isinstance(value, dict):
            raise ValueError(f'Unrecognized protected dataset record in {path}')
        # Common ELI test and MTT package shapes.
        for key in ('question', 'query', 'prompt', 'user_prompt', 'user_query', 'natural_language', 'input'):
            if isinstance(value.get(key), str) and value[key].strip():
                found.add(normalized(value[key])); return
        messages = value.get('messages')
        if isinstance(messages, list):
            user = next((m for m in messages if isinstance(m,dict) and m.get('role')=='user'), None)
            if user:
                content = user.get('content')
                if isinstance(content, str):found.add(normalized(content));return
                if isinstance(content, list):
                    s='\n'.join(b.get('text','') for b in content if isinstance(b,dict))
                    if s.strip():found.add(normalized(s));return
        for key in ('cases', 'examples', 'records', 'questions', 'items'):
            if isinstance(value.get(key), list): walk(value[key]);return
        raise ValueError(f'Cannot locate question field in protected dataset {path}; add a supported extractor before submission')
    walk(data)
    if not found and not (allow_empty and data == []):
        raise ValueError(f'Empty protected question set: {path}')
    return found


def protected_files(root, project, benchmark_names=(), extra_paths=()):
    project_root = Path(root)/'projects'/project
    records=[]
    for name in benchmark_names:
        name=core.validated_name(name)
        folder=project_root/'evaluations'/name
        manifest=folder/'benchmark_manifest.json'
        if not manifest.is_file(): raise FileNotFoundError(f'Benchmark manifest absent: {manifest}')
        obj=core.read(manifest)
        hashes=obj.get('frozen_files_sha256')
        if not isinstance(hashes, dict) or not hashes: raise ValueError(f'Missing protected hashes: {manifest}')
        suites = [s for s in obj.get('suites', []) + obj.get('targeted_suites', []) if isinstance(s, dict)]
        datasets = {s.get('dataset') for s in suites}
        declared_empty = {s.get('dataset') for s in suites if type(s.get('count')) is int and s['count'] == 0}
        declared_empty -= {s.get('dataset') for s in suites if type(s.get('count')) is not int or s['count'] != 0}
        for relative, expected in hashes.items():
            f=(folder/relative).resolve()
            if not f.is_relative_to(folder.resolve()) or not f.is_file() or f.is_symlink():raise ValueError(f'Unsafe benchmark file: {relative}')
            if core.digest(f)!=expected:raise ValueError(f'Protected benchmark hash mismatch: {f}')
            records.append({'path':str(f),'sha256':expected,'benchmark':name,'is_dataset':relative in datasets,
                            'allow_empty':relative in declared_empty})
        records.append({'path':str(manifest),'sha256':core.digest(manifest),'benchmark':name,'manifest':True})
    for item in extra_paths:
        f=Path(item).expanduser().resolve()
        if not f.is_file():raise FileNotFoundError(f'Protected evaluation file not found: {f}')
        records.append({'path':str(f),'sha256':core.digest(f),'is_dataset':True})
    unique={x['path']:x for x in records}
    if not unique:raise ValueError('At least one immutable protected benchmark or dataset is required for training preflight')
    return list(unique.values())


def preflight(data, prompt_file, protected, *, count_tokens=None, count_model_id=None):
    data=Path(data).expanduser().resolve();prompt_file=Path(prompt_file).expanduser().resolve()
    if not data.is_file() or not prompt_file.is_file():raise FileNotFoundError('Training dataset and system_prompt.txt must exist')
    prompt=prompt_file.read_text(encoding='utf-8')
    if not prompt.strip():raise ValueError('System prompt is empty')
    result=training_questions(data,prompt,count_tokens=count_tokens,model_id=count_model_id)
    train=result.pop('questions'); audit=result.pop('audit_rows')
    overlaps={}
    for entry in protected:
        if entry.get('manifest') or not entry.get('is_dataset'):continue
        path=entry['path']
        if core.digest(path)!=entry['sha256']:
            raise ValueError(f'Protected input changed: {path}')
        questions=extract_questions(path,allow_empty=entry.get('allow_empty',False))
        intersection=questions & train
        if intersection:overlaps[path]=len(intersection)
    if overlaps:raise ValueError('Training/protected evaluation question overlap: '+json.dumps(overlaps))
    result['protected_sha256']={x['path']:x['sha256'] for x in protected}
    return result,audit


def remote_run_dir(root, project, version, run_id):
    return Path(root)/'runs'/core.validated_name(project)/core.validated_name(version)/core.validated_name(run_id)


def version_location(root, project, version):
    return Path(root)/'projects'/core.validated_name(project)/'versions'/core.validated_name(version)


def mint_run():
    return 'run_'+stamp()+'_'+secrets.token_hex(3)


def state_path(folder, kind):
    return Path(folder)/'remote'/f'{kind}.json'


def save_state(folder, kind, state):
    core.write(state_path(folder,kind),safe_public(state))
    history = Path(folder)/'remote'/'events.jsonl'
    history.parent.mkdir(parents=True,exist_ok=True)
    with history.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'at':core.utc(),'kind':kind,'status':state.get('status'),'job':state.get('job_arn') or state.get('deployment_arn')},ensure_ascii=False)+'\n')


def load_state(folder, kind):
    target=state_path(folder,kind)
    if not target.is_file():raise FileNotFoundError(f'No {kind} state at {target}')
    return core.read(target)


@contextlib.contextmanager
def exclusive(folder, kind):
    path=Path(folder)/'remote'/f'.{kind}.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    try:fd=os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as ex:raise RuntimeError(f'{kind} already changing; lock present: {path}') from ex
    try:
        with os.fdopen(fd,'w') as f:f.write(f'{os.getpid()}\n')
        yield
    finally:path.unlink(missing_ok=True)


def run_meta(folder, status=None):
    file=Path(folder)/'run.json'
    m=core.read(file)
    if status:
        m['status']=status
        m['updated_utc']=core.utc()
        core.write(file,m)
        core.record({'project':m['project'],'version':m['version'],'run_id':m['run_id'],'status':status,'updated_utc':m['updated_utc'], 'run_json':core.rel(file)})
    return m


def ensure_version(root, project, version, *, base_model_id=DEFAULT_MODEL):
    """One-time remote version creation; never edits an existing ELI version."""
    if not base_model_id.strip():
        raise ValueError('Bedrock base model ID must not be blank')
    base_model_id = base_model_id.strip()
    v=version_location(root,project,version)
    if not (v.parent.parent/'project.json').is_file():raise FileNotFoundError('Create MTT project before remote version initialization')
    if v.exists():raise FileExistsError(f'Existing MTT version must not be changed: {v}')
    (v/'data').mkdir(parents=True,exist_ok=False)
    core.write(v/'version.json',{'project':project,'version':version,'created_utc':core.utc(),'initialization':'remote_customization',
        'base_model':{'provider':'bedrock','id':base_model_id},'remote':{'bedrock':{'base_model_id':base_model_id}},
        'notes':'Remote Bedrock SFT. Configure remote.bedrock and place exact production system prompt in data/system_prompt.txt.'})
    core.write(v/'version_state.json',{'stage':'draft','created_utc':core.utc()})
    return v


def validate_existing_audit(path, audit):
    """Match SFT questions and labels while allowing source IDs and metadata."""
    supplied = {row['question']: json_sha(row['expected']) for row in dataset_rows(path)}
    projected = {row['question']: json_sha(row['expected']) for row in audit}
    if supplied != projected:
        raise ValueError(f'{path}: questions or expected answers differ from the Bedrock training dataset')


def create_snapshot(root,project,version,run_id,data,prompt,audit):
    folder=remote_run_dir(root,project,version,run_id)
    if folder.exists():raise FileExistsError(f'Run already exists: {folder}')
    core.snapshot_inputs(project,version,folder/'input')
    snap=folder/'input'
    # Raw Bedrock bytes are staged exactly as supplied; no reserialization.
    (snap/'data').mkdir(exist_ok=True)
    sft=snap/'data'/'bedrock_training.jsonl'
    shutil.copyfile(data,sft)
    frozen_prompt=snap/'data'/'system_prompt.txt'
    if frozen_prompt.exists():
        if frozen_prompt.read_bytes()!=Path(prompt).read_bytes():
            raise ValueError('Version system prompt and --system-prompt bytes differ; refusing alternate contract')
    else:shutil.copyfile(prompt,frozen_prompt)
    audit_path=snap/'data'/'train.jsonl'
    if audit_path.exists():
        validate_existing_audit(audit_path, audit)
    else:
        with audit_path.open('x',encoding='utf-8',newline='\n') as f:
            for row in audit:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    hashes=core.read(snap/'input_hashes.json')
    for file in (sft,frozen_prompt,audit_path):
        hashes[file.relative_to(snap).as_posix()]=core.digest(file)
    core.write(snap/'input_hashes.json',hashes)
    meta={'project':project,'version':version,'run_id':run_id,'created_utc':core.utc(),'status':'prepared',
          'run_kind':'training','backend':'bedrock-nova-sft','toolkit_version':'0.4.1'}
    core.write(folder/'run.json',meta)
    return folder


def s3_paths(cfg,version,run_id,sha256):
    if not cfg.get('bucket'):raise ValueError('Bedrock S3 bucket not configured')
    bucket=cfg['bucket']
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]',bucket):raise ValueError('Invalid S3 bucket name')
    pref=(cfg.get('prefix') or 'eli').strip('/')
    train=(cfg.get('training_prefix') or f'{pref}/{version}/{run_id}/training').strip('/')
    out=(cfg.get('output_prefix') or f'{pref}/{version}/{run_id}/output').strip('/')
    for key in (train,out):
        if '..' in key.split('/') or '\\' in key:raise ValueError('Unsafe S3 prefix')
    return f's3://{bucket}/{train}/{sha256}.jsonl',f's3://{bucket}/{out}'


def split_s3(uri):
    if not uri.startswith('s3://') or '/' not in uri[5:]:raise ValueError(f'Bad S3 URI: {uri}')
    b,k=uri[5:].split('/',1)
    return b,k


def immutable_put(s3,file,uri,sha256):
    if core.digest(file) != sha256:
        raise RuntimeError('S3 staging source changed since preflight; refusing mutation')
    bucket,key=split_s3(uri)
    try:
        head=s3.head_object(Bucket=bucket,Key=key)
    except Exception as ex:
        if ex.__class__.__name__ not in ('ClientError', 'NotFound'):
            raise
        if getattr(ex,'response',{}).get('Error',{}).get('Code') not in ('404','NoSuchKey','NotFound'):
            raise
    else:
        if head.get('Metadata',{}).get('sha256')!=sha256:
            raise RuntimeError(f'S3 destination occupied by different artifact: {uri}')
        return 'existing'
    try:
        with Path(file).open('rb') as f:
            s3.put_object(Bucket=bucket,Key=key,Body=f,ContentType='application/x-ndjson',Metadata={'sha256':sha256},IfNoneMatch='*')
    except Exception as ex:
        if getattr(ex,'response',{}).get('Error',{}).get('Code') not in ('PreconditionFailed','412'):
            raise
    head=s3.head_object(Bucket=bucket,Key=key)
    if head.get('Metadata',{}).get('sha256')!=sha256 or head.get('ContentLength')!=Path(file).stat().st_size:
        raise RuntimeError('S3 object metadata/length verification failed')
    return 'uploaded'


def git_commit(root):
    result=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def job_name(prefix, run_id, suffix=''):
    safe=re.sub('[^A-Za-z0-9-]+','-',prefix).strip('-')[:25]
    return (safe+'-'+sha(run_id.encode())[:20]+suffix)[:63]


def is_nova_2_lite(model_id):
    return model_id.rsplit('/', 1)[-1] in (DEFAULT_MODEL, NOVA_2_LITE_RUNTIME_MODEL)


def training_hyperparameters(model_id):
    # Bedrock requires an explicit map for Nova 2 Lite FINE_TUNING jobs.
    return NOVA_2_LITE_SFT_DEFAULTS.copy() if is_nova_2_lite(model_id) else {}


def create_training_job(bedrock, state):
    request = {
        'jobName': state['job_name'],
        'customModelName': state['custom_model_name'],
        'roleArn': state['role_arn'],
        'baseModelIdentifier': state['base_model_id'],
        'customizationType': 'FINE_TUNING',
        'trainingDataConfig': {'s3Uri': state['training_s3_uri']},
        'outputDataConfig': {'s3Uri': state['output_s3_uri']},
        'clientRequestToken': state['client_request_token'],
    }
    if state.get('hyperparameters'):
        request['hyperParameters'] = state['hyperparameters']
    return bedrock.create_model_customization_job(**request)


def submit(root,project,version,dataset,prompt_file,*,benchmark_names=(),protected_paths=(),run_id=None,overrides=None,apply=False,
           skip_token_check=False, factory=None):
    root=Path(root)
    cfg=load_config(project,version,overrides,root=root)
    if apply and skip_token_check:
        raise ValueError('--skip-token-check is only valid for offline preflight')
    protected=protected_files(root,project,benchmark_names,protected_paths)
    dataset=Path(dataset).expanduser().resolve()
    prompt_file=Path(prompt_file).expanduser().resolve()
    bedrock=s3=sts=runtime=region=None
    if apply or not skip_token_check:
        bedrock,s3,sts,runtime,region=clients(cfg,factory)
    # Nova 2 Lite does not support CountTokens on Bedrock Runtime. Treat the
    # customization ID and foundation-model ARN identically here.
    nova_2_lite = is_nova_2_lite(cfg['base_model_id'])
    counter = None if skip_token_check or nova_2_lite else runtime.count_tokens
    count_model_id = (cfg.get('count_model_id') or cfg['base_model_id'].replace(':256k', '')) if counter else None
    checked,audit=preflight(dataset,prompt_file,protected,count_tokens=counter,count_model_id=count_model_id)
    existing_audit = version_location(root, project, version) / 'data/train.jsonl'
    if existing_audit.exists():
        validate_existing_audit(existing_audit, audit)
    checked['hyperparameters'] = training_hyperparameters(cfg['base_model_id'])
    if nova_2_lite:
        checked['token_validation_reason'] = (
            'Nova 2 Lite does not support Bedrock Runtime CountTokens. No verified compatible '
            'offline tokenizer is configured. Bedrock validates the 32K context limit during customization.'
        )
    if not apply:
        if sts is not None:
            account=sts.get_caller_identity()['Account']
            print('AWS TARGET',json.dumps({'account_id':account,'profile':cfg.get('profile'),'region':region,'bucket':cfg.get('bucket')}))
        print('REMOTE TRAIN PREFLIGHT',json.dumps({k:v for k,v in checked.items() if k!='protected_sha256'},indent=2))
        print('No AWS changes (add --apply to submit)')
        return None
    if nova_2_lite:
        print(f"WARNING: Token counts are unverified. {checked['token_validation_reason']}", file=sys.stderr)
    if not cfg.get('role_arn'):raise ValueError('Bedrock service role ARN is required')
    if not cfg.get('bucket'):raise ValueError('Bedrock S3 bucket is required')
    identity=sts.get_caller_identity()
    account=identity['Account']
    print('AWS TARGET',json.dumps({'account_id':account,'profile':cfg.get('profile'), 'region':region, 'bucket':cfg.get('bucket')}))
    # Reject duplicate submissions even from concurrent MTT terminals.
    run_parent=root/'runs'/project/version
    run_parent.mkdir(parents=True, exist_ok=True)
    with exclusive(run_parent,'corpus-'+checked['training_file_sha256'][:32]):
        for st in run_parent.glob('*/remote/train.json') if run_parent.is_dir() else ():
            existing=core.read(st)
            if existing.get('training_file_sha256')==checked['training_file_sha256'] and existing.get('status') not in ('Failed','Stopped'):
                raise RuntimeError(f'Same training corpus already submitted in run {st.parent.parent.name}; inspect/resume that run')
        rid=run_id or mint_run()
        folder=create_snapshot(root,project,version,rid,dataset,prompt_file,audit)
        # Fail closed if source files changed between validation and snapshot creation.
        snapshot_train = folder/'input/data/bedrock_training.jsonl'
        snapshot_prompt = folder/'input/data/system_prompt.txt'
        if core.digest(snapshot_train) != checked['training_file_sha256']:
            raise RuntimeError('Training dataset changed after preflight; refusing to upload altered bytes')
        if sha(snapshot_prompt.read_text(encoding='utf-8').encode('utf-8')) != checked['system_prompt_sha256']:
            raise RuntimeError('System prompt changed after preflight; refusing mismatched semantic contract')
        train_uri,out_uri=s3_paths(cfg,version,rid,checked['training_file_sha256'])
        state={'format':FORMAT,'project':project,'version':version,'run_id':rid,'created_utc':core.utc(),'status':'Prepared',
            'account_id':account,'profile':cfg.get('profile'),'region':region,'base_model_id':cfg['base_model_id'],
            'job_name':job_name(cfg['model_prefix'],rid), 'custom_model_name':job_name(cfg['model_prefix'],rid,'-model'),
            'training_s3_uri':train_uri,'output_s3_uri':out_uri,'role_arn':cfg['role_arn'],
            'hyperparameters':checked['hyperparameters'],'training_file_sha256':checked['training_file_sha256'],
            'system_prompt_sha256':checked['system_prompt_sha256'],'token_validation':checked['token_validation'],
            'training_count':checked['count'],'protected_sha256':checked['protected_sha256'],
            'mtt_commit':git_commit(root),'version_config_sha256':core.digest(folder/'input/version.json'),
            'client_request_token':sha((rid+checked['training_file_sha256']).encode())[:48]}
        if 'token_validation_reason' in checked:
            state['token_validation_reason'] = checked['token_validation_reason']
        with exclusive(folder,'train'):
            save_state(folder,'train',state)
            try:
                immutable_put(s3,folder/'input/data/bedrock_training.jsonl',train_uri,checked['training_file_sha256'])
                state['status']='Staged';state['staged_utc']=core.utc();save_state(folder,'train',state)
                resp=create_training_job(bedrock,state)
                state['job_arn']=resp.get('jobArn');state['status']='InProgress';state['submitted_utc']=core.utc()
                save_state(folder,'train',state)
                run_meta(folder,'remote_training')
            except Exception:
                # Keep Prepared/Staged + idempotency token to recover after interrupted network responses.
                raise
    print('BEDROCK TRAIN SUBMITTED RUN_ID',rid,'JOB',state['job_name'],'ARN',state.get('job_arn'))
    return rid


def find_training_job_arn(bedrock, job_name):
    paginator = bedrock.get_paginator('list_model_customization_jobs')
    for page in paginator.paginate(nameContains=job_name):
        for job in page.get('modelCustomizationJobSummaries', []):
            if job.get('jobName') == job_name:
                if not job.get('jobArn'):
                    raise RuntimeError(f'Bedrock listed job {job_name} without an ARN')
                return job['jobArn']
    return None


def refresh_training_state(bedrock, folder, state, job_arn):
    response = bedrock.get_model_customization_job(jobIdentifier=job_arn)
    discovering_job = not state.get('job_arn')
    state.update(job_arn=job_arn, status=response['status'], last_status_check_utc=core.utc())
    if response.get('outputModelArn'):
        state['custom_model_arn'] = response['outputModelArn']
    if response.get('failureMessage'):
        state['failure_reason'] = response['failureMessage']
    save_state(folder, 'train', state)
    if state['status'].casefold() == 'completed':
        if not state.get('custom_model_arn'):
            raise RuntimeError('Bedrock reports Completed without an outputModelArn')
        run_meta(folder, 'trained')
    elif state['status'] in ('Failed', 'Stopped'):
        run_meta(folder, 'remote_training_failed')
    elif discovering_job:
        run_meta(folder, 'remote_training')
    print('BEDROCK TRAIN',state['run_id'],state['status'],state.get('custom_model_arn') or state.get('failure_reason') or '')
    return state


def check_train(root,project,version,rid,*,factory=None):
    folder = remote_run_dir(root, project, version, rid)
    if not state_path(folder, 'train').is_file():
        raise FileNotFoundError(f"No train state at {state_path(folder, 'train')}")
    with exclusive(folder, 'train'):
        state = load_state(folder, 'train')
        bedrock = verified_clients(state, factory)[0]
        job_arn = state.get('job_arn') or find_training_job_arn(bedrock, state['job_name'])
        if not job_arn:
            raise RuntimeError(f"No Bedrock job found for {state['job_name']}; use recover to submit the saved run")
        return refresh_training_state(bedrock, folder, state, job_arn)


def watch(status_fn,done,interval=90):
    if interval<30:raise ValueError('Polling interval must be at least 30 seconds')
    while True:
        state=status_fn()
        if state['status'] in done or state['status'].casefold() in {s.casefold() for s in done} or state['status']=='Interrupted':return state
        time.sleep(interval)


def deploy(root,project,version,rid,*,apply=False,factory=None):
    folder=remote_run_dir(root,project,version,rid)
    train=load_state(folder,'train')
    if train['status'].casefold()!='completed' or not train.get('custom_model_arn'):
        raise RuntimeError('Completed training and custom model ARN are required before deployment')
    path=state_path(folder,'deployment')
    if path.exists():
        current=core.read(path)
        print('EXISTING DEPLOYMENT',current.get('status'),current.get('deployment_arn'))
        return current
    if not apply:
        print('DEPLOY PLAN',train['custom_model_arn'],'--apply required to create on-demand deployment')
        return None
    bedrock=verified_clients(train,factory)[0]
    with exclusive(folder,'deployment'):
        name=job_name('mtt-deploy',rid)
        state={'status':'Prepared','deployment_name':name,'custom_model_arn':train['custom_model_arn'],
               'account_id':train['account_id'],'profile':train.get('profile'),'region':train['region'],
               'client_request_token':sha(('deploy'+rid).encode())[:48], 'created_utc':core.utc()}
        save_state(folder,'deployment',state)
        resp=bedrock.create_custom_model_deployment(modelDeploymentName=name,modelArn=train['custom_model_arn'],
                                                       clientRequestToken=state['client_request_token'])
        state['deployment_arn']=resp['customModelDeploymentArn'];state['status']='Creating'
        save_state(folder,'deployment',state)
    print('DEPLOYMENT CREATED',state['deployment_arn'])
    return state


def check_deploy(root,project,version,rid,*,factory=None):
    folder=remote_run_dir(root,project,version,rid)
    state=load_state(folder,'deployment')
    bedrock=verified_clients(state,factory)[0]
    resp=bedrock.get_custom_model_deployment(customModelDeploymentIdentifier=state.get('deployment_arn') or state['deployment_name'])
    state['status']=resp['status']
    state['last_status_check_utc']=core.utc()
    if resp.get('failureMessage'):state['failure_reason']=resp['failureMessage']
    if resp.get('customModelDeploymentArn'):state['deployment_arn']=resp['customModelDeploymentArn']
    with exclusive(folder,'deployment'):save_state(folder,'deployment',state)
    print('BEDROCK DEPLOY',rid,state['status'],state.get('deployment_arn'))
    return state


def recover_train(root,project,version,rid,*,factory=None):
    """Find the saved job by exact name, or replay the request with the same token."""
    folder = remote_run_dir(root, project, version, rid)
    if not state_path(folder, 'train').is_file():
        raise FileNotFoundError(f"No train state at {state_path(folder, 'train')}")
    with exclusive(folder, 'train'):
        state = load_state(folder, 'train')
        bedrock, s3, _, _, _ = verified_clients(state, factory)
        job_arn = state.get('job_arn') or find_training_job_arn(bedrock, state['job_name'])
        if job_arn:
            return refresh_training_state(bedrock, folder, state, job_arn)
        if state['status'] in TERMINAL_TRAIN:
            raise RuntimeError(f"No Bedrock job found for terminal run {rid}")

        file = folder / 'input/data/bedrock_training.jsonl'
        core.assert_snapshot(folder / 'input')
        immutable_put(s3, file, state['training_s3_uri'], state['training_file_sha256'])
        # Older Nova 2 Lite receipts omitted the required map.
        if not state.get('hyperparameters') and is_nova_2_lite(state['base_model_id']):
            state['hyperparameters'] = training_hyperparameters(state['base_model_id'])
            save_state(folder, 'train', state)
        response = create_training_job(bedrock, state)
        state.update(job_arn=response['jobArn'], status='InProgress', submitted_utc=core.utc())
        save_state(folder, 'train', state)
        run_meta(folder, 'remote_training')
    print('BEDROCK TRAIN SUBMITTED RUN_ID',rid,'JOB',state['job_name'],'ARN',state['job_arn'])
    return state


def recover_deploy(root,project,version,rid,*,factory=None):
    folder=remote_run_dir(root,project,version,rid)
    state=load_state(folder,'deployment')
    if state.get('deployment_arn'):return check_deploy(root,project,version,rid,factory=factory)
    bedrock=verified_clients(state,factory)[0]
    with exclusive(folder,'deployment'):
        try:
            resp=bedrock.get_custom_model_deployment(customModelDeploymentIdentifier=state['deployment_name'])
            state['deployment_arn']=resp.get('customModelDeploymentArn')
            state['status']=resp['status']
        except Exception as ex:
            if getattr(ex,'response',{}).get('Error',{}).get('Code') not in ('ResourceNotFoundException','404'):
                raise
            resp=bedrock.create_custom_model_deployment(modelDeploymentName=state['deployment_name'],
                modelArn=state['custom_model_arn'],clientRequestToken=state['client_request_token'])
            state['deployment_arn']=resp['customModelDeploymentArn']
            state['status']='Creating'
        save_state(folder,'deployment',state)
    return state


def _http_bridge(client, model_id, model_tag, prompt, prompt_hash, generation):
    """Loopback-only Ollama API shim. No alternate scorer or SPARQL compiler."""
    import http.server
    import threading
    from urllib.parse import urlparse

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def reply(self, data, status=200):
            raw=json.dumps(data,ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if urlparse(self.path).path in ('/','/api/version'):
                self.reply({'version':'mtt-bedrock-adapter-1'});return
            if urlparse(self.path).path=='/api/tags':
                self.reply({'models':[{'name':model_tag,'model':model_tag,'digest':prompt_hash,'modified_at':core.utc(),'size':0,
                    'details':{'format':'bedrock','family':'nova-2-lite'}}]});return
            self.reply({'error':'endpoint unsupported'},404)

        def do_POST(self):
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<1 or n>1024*1024*4:raise ValueError('Invalid request length')
                request=json.loads(self.rfile.read(n))
                if request.get('model') not in (None,model_tag):
                    raise ValueError('Model identity mismatch')
                url=urlparse(self.path).path
                if url=='/api/show':
                    self.reply({'model':model_tag,'digest':prompt_hash,'modified_at':core.utc(),
                        'details':{'format':'bedrock','family':'nova-2-lite','parameter_size':'managed'},'parameters':''})
                    return
                if url=='/api/generate':
                    user=str(request.get('prompt',''))
                    provided=request.get('system')
                    if provided is not None and provided!=prompt:
                        raise ValueError('Evaluator system prompt differs from frozen training prompt')
                    if not user.strip():raise ValueError('Missing prompt')
                    if user.startswith(prompt):
                        raise ValueError('Request appears to embed system prompt into user message; refusing duplication')
                    messages=[{'role':'user','content':[{'text':user}]}]
                elif url=='/api/chat':
                    messages=[]
                    system=[]
                    for m in request.get('messages',[]):
                        role=m.get('role')
                        if role=='system':system.append(m.get('content'))
                        elif role in ('user','assistant'):
                            messages.append({'role':role,'content':[{'text':str(m.get('content',''))}]})
                        else:raise ValueError('Unsupported chat role')
                    if system and system!=[prompt]:raise ValueError('Evaluator system prompt mismatch')
                    if not messages or messages[-1]['role']!='user':raise ValueError('Invalid chat conversation')
                else:
                    self.reply({'error':'API endpoint unsupported'},404);return
                if request.get('stream',False):raise ValueError('Only nonstreaming requests are supported')
                options=request.get('options') or {}
                inf={'maxTokens':int(options.get('num_predict') or generation.get('max_tokens',1024)),
                     'temperature':float(options.get('temperature',generation.get('temperature',0)))}
                if 'top_p' in options:inf['topP']=float(options['top_p'])
                started=time.monotonic_ns()
                resp=client.converse(modelId=model_id,system=[{'text':prompt}],messages=messages,inferenceConfig=inf)
                answer=''.join(b.get('text','') for b in resp.get('output',{}).get('message',{}).get('content',[]) if isinstance(b,dict))
                usage=resp.get('usage',{})
                elapsed=time.monotonic_ns()-started
                result={'model':model_tag,'created_at':core.utc(),'done':True,'done_reason':resp.get('stopReason','stop'),
                    'eval_count':usage.get('outputTokens',0),'prompt_eval_count':usage.get('inputTokens',0),
                    'total_duration':elapsed,'eval_duration':elapsed,'load_duration':0}
                if url=='/api/generate':result['response']=answer
                else:result['message']={'role':'assistant','content':answer}
                self.reply(result)
            except Exception as ex:
                self.reply({'error':str(ex)},500)

    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    return server, f'http://127.0.0.1:{server.server_port}'


def evaluation_id(benchmark, deployment_arn, generation, stage, suites):
    return hashlib.sha256(json.dumps([benchmark,deployment_arn,generation,stage,suites],sort_keys=True).encode()).hexdigest()[:16]


def prepare_eval(root,project,version,rid,benchmark,*,stage='all',suites=None,generation=None):
    """No inference. Read-only audit + deterministic benchmark and deployment receipts."""
    from . import shared_eval
    folder=remote_run_dir(root,project,version,rid)
    deployment=load_state(folder,'deployment')
    if str(deployment.get('status')).casefold()!='active' or not deployment.get('deployment_arn'):
        raise RuntimeError('An Active custom model deployment is required for evaluation')
    core.assert_snapshot(folder/'input')
    spec=shared_eval.plan(project,version,rid,benchmark,workspace=root)
    # Use the benchmark's existing domain-specific audit without changing its sources.
    subprocess.run([sys.executable,str(spec['scripts']['auditor']),'--train',str(spec['train'])],cwd=spec['benchmark'],check=True)
    manifest=core.read(spec['benchmark']/'benchmark_manifest.json')
    train=load_state(folder,'train')
    if core.digest(spec['prompt'])!=train['system_prompt_sha256']:
        raise ValueError('Evaluation prompt does not match training prompt hash')
    for file,expected in train['protected_sha256'].items():
        if not Path(file).is_file() or core.digest(file)!=expected:
            raise ValueError(f'Protected evaluation artifact changed after training: {file}')
    generation=generation or {'temperature':0,'max_tokens':1024}
    selection=suites or []
    eid=evaluation_id(benchmark,deployment['deployment_arn'],generation,stage,selection)
    model_tag='bedrock-nova-'+sha(deployment['deployment_arn'].encode())[:16]
    state={'format':'mtt-bedrock-eval-v1','project':project,'version':version,'run_id':rid,
        'evaluation_id':eid,'status':'Prepared','created_utc':core.utc(),'benchmark':benchmark,'stage':stage,
        'suites':selection,'generation':generation,'model_tag':model_tag,
        'deployment_arn':deployment['deployment_arn'],'custom_model_arn':train['custom_model_arn'],
        'system_prompt_sha256':train['system_prompt_sha256'],
        'benchmark_manifest_sha256':core.digest(spec['benchmark']/'benchmark_manifest.json'),
        'benchmark_id':manifest['benchmark_id'],'benchmark_package_version':manifest['package_version'],
        'benchmark_files_sha256':manifest['frozen_files_sha256'],
        'scorer_files_sha256':{str(p.relative_to(spec['benchmark'])):core.digest(p)
            for p in (spec['benchmark']/'scorers').rglob('*') if p.is_file()},
        'account_id':train['account_id'],'profile':train.get('profile'),'region':train['region'],
        'result_dir':str(folder/'shared_evaluations'/benchmark/model_tag),
        'mtt_commit':git_commit(root)}
    return state



def verify_evaluation_assets(root, state):
    """Revalidate all pinned benchmark, scorer and snapshot inputs before inference."""
    folder=remote_run_dir(root,state['project'],state['version'],state['run_id'])
    core.assert_snapshot(folder/'input')
    prompt=folder/'input/data/system_prompt.txt'
    if sha(prompt.read_text(encoding='utf-8').encode('utf-8')) != state['system_prompt_sha256']:
        raise ValueError('Production system prompt no longer matches the training snapshot')
    benchmark=Path(root)/'projects'/state['project']/'evaluations'/state['benchmark']
    if core.digest(benchmark/'benchmark_manifest.json')!=state['benchmark_manifest_sha256']:
        raise ValueError('Published benchmark manifest changed since evaluation planning')
    for key,expected in state['benchmark_files_sha256'].items():
        path=(benchmark/key).resolve()
        if not path.is_relative_to(benchmark.resolve()) or not path.is_file() or core.digest(path)!=expected:
            raise ValueError(f'Protected benchmark file changed: {key}')
    for key,expected in state['scorer_files_sha256'].items():
        path=(benchmark/key).resolve()
        if not path.is_relative_to(benchmark.resolve()) or not path.is_file() or core.digest(path)!=expected:
            raise ValueError(f'Protected scorer source changed: {key}')
    training=load_state(folder,'train')
    for file,expected in training['protected_sha256'].items():
        if not Path(file).is_file() or core.digest(file)!=expected:
            raise ValueError(f'Training-time protected reference changed: {file}')
    return True


def worker_eval(root,project,version,rid,eval_id,*,factory=None):
    root=Path(root)
    folder=remote_run_dir(root,project,version,rid)
    file=state_path(folder,'eval_'+eval_id)
    state=core.read(file)
    with exclusive(folder,'eval_'+eval_id):
        if state['status'].casefold()=='completed':return state
        state['status']='Running';state['started_utc']=core.utc();state['pid']=os.getpid();core.write(file,state)
    try:
        from . import shared_eval
        verify_evaluation_assets(root,state)
        runtime=verified_clients(state,factory)[3]
        prompt=(folder/'input/data/system_prompt.txt').read_text(encoding='utf-8')
        if sha(prompt.encode('utf-8'))!=state['system_prompt_sha256']:
            raise ValueError('Frozen prompt hash mismatch')
        server,url=_http_bridge(runtime,state['deployment_arn'],state['model_tag'],prompt,state['system_prompt_sha256'],state['generation'])
        try:
            # The ELI semantic benchmark and its scorer remain completely unchanged.
            shared_eval.execute(project,version,rid,state['benchmark'],model=state['model_tag'],
                stage=state['stage'] if not state['suites'] else None,
                suites=state['suites'] or None, resume=True,url=url,
                num_predict=state['generation']['max_tokens'],workspace=root)
        finally:server.shutdown();server.server_close()
        state['status']='Completed';state['completed_utc']=core.utc()
    except Exception as ex:
        state['status']='Failed';state['failure_reason']=str(ex);state['failed_utc']=core.utc()
        core.write(file,state)
        raise
    core.write(file,state)
    print('BEDROCK EVALUATION',eval_id,'Completed',state['result_dir'])
    return state


def eval_submit(root,project,version,rid,benchmark,*,stage='all',suites=None,generation=None,apply=False,resume=False):
    state=prepare_eval(root,project,version,rid,benchmark,stage=stage,suites=suites,generation=generation)
    folder=remote_run_dir(root,project,version,rid)
    key='eval_'+state['evaluation_id']
    file=state_path(folder,key)
    if file.exists():
        old=core.read(file)
        for k in ('benchmark_manifest_sha256','deployment_arn','system_prompt_sha256','benchmark_files_sha256','generation','stage','suites'):
            if old[k]!=state[k]:raise ValueError(f'Existing evaluation differs in {k}; cannot mix results')
        if old['status']=='Completed':
            print('EVALUATION ALREADY COMPLETED',old['evaluation_id']);return old
        if old['status'] in ('Running','Queued') and pid_alive(old.get('pid')):
            print('EVALUATION ALREADY RUNNING',old['evaluation_id']);return old
        if not resume:raise RuntimeError('Existing evaluation found; specify --resume to continue its predictions')
        state=old
    if not apply:
        print('EVAL PLAN',json.dumps({k:state[k] for k in ('evaluation_id','benchmark','stage','suites','model_tag','result_dir')},indent=2))
        print('No model invocations (add --apply to start)')
        return state
    with exclusive(folder,key):
        state['status']='Queued';state['queued_utc']=core.utc();core.write(file,state)
        logfile=folder/'remote'/f'eval_{state["evaluation_id"]}.log'
        cmd=[sys.executable,str(root/'toolkit/cli.py'),'remote','eval','worker',project,version,
             '--run',rid,'--eval-id',state['evaluation_id']]
        flags=0
        kw={'cwd':str(root),'stdin':subprocess.DEVNULL}
        if os.name=='nt':
            flags = getattr(subprocess,'DETACHED_PROCESS',0) | getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0)
            kw['creationflags']=flags
        else:kw['start_new_session']=True
        with logfile.open('ab') as output:
            child=subprocess.Popen(cmd,stdout=output,stderr=subprocess.STDOUT,**kw)
        current=core.read(file)
        if current['status']=='Queued':
            current['pid']=child.pid
            current['log_file']=str(logfile)
            core.write(file,current)
    state = core.read(file)
    print('BEDROCK EVAL QUEUED',state['evaluation_id'],'EVAL_ID',state['evaluation_id'],
          'PID',state.get('pid'),'LOG',state.get('log_file'))
    return state


def pid_alive(pid):
    if not isinstance(pid,int) or pid<1:return False
    try:os.kill(pid,0);return True
    except (ProcessLookupError,OverflowError):return False
    except PermissionError:return True
    except OSError:return False


def eval_status(root,project,version,rid,eval_id):
    folder=remote_run_dir(root,project,version,rid)
    state=load_state(folder,'eval_'+eval_id)
    if state['status'] in ('Queued','Running') and not pid_alive(state.get('pid')):
        state['status']='Interrupted'
        state['last_status_check_utc']=core.utc()
        core.write(state_path(folder,'eval_'+eval_id),state)
    print('BEDROCK EVAL',eval_id,state['status'],state.get('failure_reason') or state.get('result_dir'))
    return state


def args_from_namespace(a):
    return {key:getattr(a,key) for key in ('profile','region','role_arn','bucket','prefix','training_prefix',
        'output_prefix','base_model_id','model_prefix','count_model_id') if getattr(a,key,None) is not None}


def dispatch(a,root=None):
    root=Path(root or core.ROOT)
    project,version=a.project,a.version
    if a.kind=='version' and a.action=='init':
        print('REMOTE VERSION',ensure_version(root,project,version,base_model_id=a.base_model_id));return
    if a.kind=='train':
        if a.action in ('preflight','submit'):
            v=version_location(root,project,version)
            cfg=load_config(project,version,args_from_namespace(a),root=root)
            envconfig=core.read(v/'version.json').get('remote',{}).get('bedrock',{})
            dataset=a.dataset or envconfig.get('dataset')
            if not dataset:raise ValueError('Bedrock training dataset not configured; specify --dataset')
            data=Path(dataset).expanduser()
            if not data.is_absolute():data=v/data
            prompt=Path(a.system_prompt).expanduser() if a.system_prompt else v/'data/system_prompt.txt'
            benches=a.benchmark or envconfig.get('protected_benchmarks',[])
            protected=a.protected or envconfig.get('protected_files',[])
            return submit(root,project,version,data,prompt,benchmark_names=benches,protected_paths=protected,
                overrides=args_from_namespace(a),apply=a.apply if a.action=='submit' else False,
                skip_token_check=a.skip_token_check)
        if a.action=='recover':return recover_train(root,project,version,a.run)
        if a.action=='status':return check_train(root,project,version,a.run)
        if a.action=='watch':return watch(lambda:check_train(root,project,version,a.run),TERMINAL_TRAIN,a.interval)
    if a.kind=='deploy':
        if a.action=='submit':return deploy(root,project,version,a.run,apply=a.apply)
        if a.action=='recover':return recover_deploy(root,project,version,a.run)
        if a.action=='status':return check_deploy(root,project,version,a.run)
        if a.action=='watch':return watch(lambda:check_deploy(root,project,version,a.run),TERMINAL_DEPLOY,a.interval)
    if a.kind=='eval':
        if a.action=='worker':return worker_eval(root,project,version,a.run,a.eval_id)
        if a.action=='status':return eval_status(root,project,version,a.run,a.eval_id)
        if a.action=='watch':return watch(lambda:eval_status(root,project,version,a.run,a.eval_id),{'Completed','Failed'},a.interval)
        if a.action=='submit':
            return eval_submit(root,project,version,a.run,a.benchmark,stage=a.stage,suites=a.suite,
                generation={'temperature':a.temperature,'max_tokens':a.max_tokens},apply=a.apply,resume=a.resume)
    raise ValueError(f'Unsupported remote action: {a.kind} {a.action}')


def add_arguments(sub):
    """Attach a native hierarchical `remote` command to MTT's root argparse."""
    p=sub.add_parser('remote',help='Durable Bedrock customization, deployment and semantic evaluation')
    sub2=p.add_subparsers(dest='kind',required=True)
    for kind,actions in {'version':['init'],'train':['preflight','submit','recover','status','watch'],
                         'deploy':['submit','recover','status','watch'],
                         'eval':['submit','worker','status','watch']}.items():
        level=sub2.add_parser(kind)
        s=level.add_subparsers(dest='action',required=True)
        for action in actions:
            a=s.add_parser(action)
            a.add_argument('project');a.add_argument('version')
            if kind=='version' and action=='init':
                a.add_argument('--base-model-id', default=DEFAULT_MODEL,
                               help='Bedrock customization base model ID for this new version')
            if kind!='version' and action not in ('submit','preflight'):
                a.add_argument('--run',required=True)
            if kind=='deploy' or kind=='eval':
                if action=='submit':a.add_argument('--run',required=True)
            if kind=='train' and action in ('preflight','submit'):
                a.add_argument('--dataset');a.add_argument('--system-prompt')
                a.add_argument('--benchmark',action='append',help='Protected shared benchmark release')
                a.add_argument('--protected',action='append',help='Additional immutable protected JSON/JSONL dataset')
                a.add_argument('--skip-token-check',action='store_true',help='Offline preflight only, not allowed with --apply')
                for name in ('profile','region','role_arn','bucket','prefix','training_prefix',
                             'output_prefix','base_model_id','model_prefix','count_model_id'):
                    a.add_argument('--'+name.replace('_','-'))
            if kind=='eval' and action=='submit':
                a.add_argument('--benchmark',required=True)
                group=a.add_mutually_exclusive_group()
                group.add_argument('--stage',default='all')
                group.add_argument('--suite',action='append')
                a.add_argument('--temperature',type=float,default=0)
                a.add_argument('--max-tokens',type=int,default=1024)
                a.add_argument('--resume',action='store_true')
            if kind=='eval' and action in ('worker','status','watch'):
                a.add_argument('--eval-id',required=True)
            if action=='submit':a.add_argument('--apply',action='store_true',help='Explicitly authorize AWS changes or billable inference')
            if action=='watch':a.add_argument('--interval',type=int,default=90)
    return p
