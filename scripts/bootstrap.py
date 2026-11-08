#!/usr/bin/env python3
"""Create a local virtualenv and clone a pinned llama.cpp source tree."""
import argparse, json, os, platform, shutil, subprocess, sys, venv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'WORKSPACE_STATE.json'
CONFIG = json.loads((ROOT / 'config/workspace.json').read_text(encoding='utf-8'))


def call(argv, *, cwd=None, env=None):
    print('+', ' '.join(str(x) for x in argv), flush=True)
    return subprocess.run([str(x) for x in argv], cwd=cwd, env=env, check=True)


def venv_python():
    for p in [ROOT/'.venv/Scripts/python.exe', ROOT/'.venv/bin/python']:
        if p.is_file(): return p
    raise RuntimeError('Virtualenv Python missing')


def git_revision(where):
    return subprocess.check_output(['git','-C',str(where),'rev-parse','HEAD'], text=True).strip()


def main():
    p = argparse.ArgumentParser(description='Model Training Toolkit first-run installer')
    p.add_argument('--skip-llama',action='store_true',help='Do not clone llama.cpp (e.g., offline setup)')
    p.add_argument('--install-ml',action='store_true',help='Install optional PyTorch/Transformers/PEFT stack into .venv')
    p.add_argument('--no-pip-upgrade',action='store_true')
    p.add_argument('--torch-profile',choices=['auto','cu128','cpu'],default='auto',help='PyTorch wheel profile for --install-ml (default: detect NVIDIA GPU)')
    args=p.parse_args()
    if sys.version_info < (3,11): raise SystemExit('Python 3.11 or newer is required')
    envdir=ROOT/'.venv'
    if not venv_exists(envdir):
        print('Creating isolated Python environment:',envdir)
        venv.EnvBuilder(with_pip=True,clear=False,symlinks=False).create(envdir)
    py=venv_python()
    if not args.no_pip_upgrade:
        call([py,'-m','pip','install','--upgrade','pip'])
    if args.install_ml:
        install_ml_stack(py,args.torch_profile)
    llama=ROOT/'tools/llama.cpp'
    llcfg=CONFIG['llama_cpp']
    revision=None
    if not args.skip_llama:
        if not shutil.which('git'):
            raise SystemExit('Git is required to clone llama.cpp; install Git or rerun with --skip-llama')
        if not llama.exists():
            call(['git','clone','--branch',llcfg['ref'],'--depth','1',llcfg['repository'],str(llama)])
        if not (llama/'.git').exists():raise SystemExit('Existing tools/llama.cpp is not a Git checkout; refusing to overwrite')
        actual=subprocess.check_output(['git','-C',str(llama),'describe','--tags','--exact-match'],text=True,stderr=subprocess.DEVNULL).strip() if is_exact_tag(llama) else None
        if actual!=llcfg['ref']:
            raise SystemExit(f'Unexpected llama.cpp checkout tag {actual!r}; expected {llcfg["ref"]!r}; refusing silent update')
        revision=git_revision(llama)
    elif (llama/'.git').exists():
        revision=git_revision(llama)
    for folder in ['cache/huggingface','projects','runs','registry','logs','exports']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    runlog=ROOT/'registry/runs.jsonl';runlog.touch(exist_ok=True)
    packages=subprocess.check_output([str(py),'-m','pip','freeze'],text=True)
    (ROOT/'registry/installed_requirements.txt').write_text(packages,encoding='utf-8')
    existing=json.loads(STATE.read_text()) if STATE.is_file() else {}
    state={
        'toolkit_version':CONFIG['toolkit_version'],'status':'ready','installed_utc':existing.get('installed_utc') or utc(),
        'last_setup_utc':utc(),'os':platform.platform(),'python':subprocess.check_output([str(py),'--version'],text=True).strip(),
        'venv':str(envdir),'llama_cpp':{'repo':llcfg['repository'],'requested_ref':llcfg['ref'],'commit':revision,'source_present':(llama/'.git').exists()},
        'ml_dependencies_installed':check_import(py,'torch') and check_import(py,'transformers') and check_import(py,'peft'),
        'torch_cuda':'not_checked', 'torch_profile':'not_installed' }
    if state['ml_dependencies_installed']:
        status=query_torch(py)
        state['torch_cuda']=str(status['cuda_available'])
        state['torch_profile']='cu128' if status['cuda_build'] and status['cuda_available'] else 'cpu'
        state['torch_version']=status['version']
    STATE.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print('\nSETUP COMPLETE (local venv only)')
    print('  Activate: source ./activate.sh')
    print('  Check:    bash workspace.sh doctor')
    print('  Help:     bash workspace.sh --help')
    if args.skip_llama:print('  NOTE: llama.cpp clone skipped by request')
    if not args.install_ml:print('  NOTE: ML dependencies not installed. Run bash run_me_first.sh --install-ml before training.')
    else:print('  ML packages installed inside .venv; verify GPU with doctor.')


def nvidia_gpu_detected():
    """Return whether a functioning NVIDIA GPU is visible, not merely a stale driver binary."""
    if not shutil.which('nvidia-smi'):
        return False
    p=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True)
    return p.returncode==0 and bool(p.stdout.strip())


def query_torch(py):
    script=(
        'import json,torch;'
        'print(json.dumps({"version":torch.__version__,"cuda_build":torch.version.cuda,'
        '"cuda_available":bool(torch.cuda.is_available())}))'
    )
    out=subprocess.check_output([str(py),'-c',script],text=True)
    return json.loads(out.strip())


def select_torch_profile(requested):
    if requested!='auto':
        return requested
    return 'cu128' if nvidia_gpu_detected() else 'cpu'


def install_ml_stack(py,requested):
    profile=select_torch_profile(requested)
    print(f'ML PyTorch wheel profile: {profile} (requested: {requested})',flush=True)
    current=None
    if check_import(py,'torch'):
        try:current=query_torch(py)
        except Exception:pass
    if profile=='cu128':
        # Official PyTorch CUDA 12.8 wheels support NVIDIA Blackwell (RTX 50 series).
        # Exact version is intentional; do not silently replace a working CUDA wheel.
        keep=(current is not None and current['version'].startswith('2.10.0+cu128')
              and current['cuda_build']=='12.8' and current['cuda_available'])
        if not keep:
            call([py,'-m','pip','install','torch==2.10.0',
                  '--index-url','https://download.pytorch.org/whl/cu128'])
    elif profile=='cpu':
        # If a CPU-only environment is requested, use CPU-only index to avoid CUDA downloads.
        keep=(current is not None and current['version'].startswith('2.10.0+cpu')
              and not current['cuda_available'])
        if not keep:
            call([py,'-m','pip','install','torch==2.10.0',
                  '--index-url','https://download.pytorch.org/whl/cpu'])
    else:
        raise RuntimeError('Unknown torch profile: '+profile)
    call([py,'-m','pip','install','-r',ROOT/'requirements/requirements-ml.txt'])
    # Training must never quietly fall back to CPU if CUDA was requested.
    verify_torch(py,profile)


def verify_torch(py,profile):
    code='''import json, torch
result = {'version': torch.__version__, 'cuda_build': torch.version.cuda,
          'cuda_available': bool(torch.cuda.is_available())}
if result['cuda_available']:
    t = torch.ones(1024, device='cuda')
    result['device'] = torch.cuda.get_device_name(0)
    result['gpu_sum'] = float(t.sum().item())
print(json.dumps(result))
'''
    result=json.loads(subprocess.check_output([str(py),'-c',code],text=True).strip())
    print('Verified PyTorch:',json.dumps(result),flush=True)
    if profile=='cu128':
        if (not result['cuda_available'] or result['cuda_build']!='12.8'
            or result.get('gpu_sum')!=1024.0 or '+cu128' not in result['version']):
            raise RuntimeError('CUDA wheel/profile verification failed: refusing CPU-only training')
    elif result['cuda_available'] or '+cpu' not in result['version']:
        raise RuntimeError('Requested CPU-only torch but installed wheel/profile differs')


def utc():return datetime.now(timezone.utc).isoformat()

def venv_exists(p):return (p/'pyvenv.cfg').is_file()

def check_import(py,mod):
    x=subprocess.run([str(py),'-c',f'import {mod}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return x.returncode==0

def is_exact_tag(path):
    x=subprocess.run(['git','-C',str(path),'describe','--tags','--exact-match'],capture_output=True,text=True)
    return x.returncode==0

if __name__=='__main__':main()
