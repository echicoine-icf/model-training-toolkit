"""Portable, transactional, *fully resumable* PEFT training checkpoints.

Checkpoint state is intentionally private to the run. Never unpickle a checkpoint
from an untrusted source: torch.load(..., weights_only=False) is used for RNG and
optimizer state that contains Python objects.

Torch is injected, not imported: the Unsloth backend must patch Torch first.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import random
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .core import digest, read, utc, write

STEP_NAME = re.compile(r'^step_(\d{7,})$')
SIGNATURE_FIELDS = (
    'toolkit_version', 'training_engine_sha256', 'checkpoint_engine_sha256',
    'transformers_version', 'peft_version', 'numpy_version', 'safetensors_version',
    'base', 'loaded_model_commit_reported', 'backend', 'backend_version',
    'quantization', 'torch_version', 'precision_effective',
    'attention_requested', 'attention_effective', 'gradient_checkpointing',
    'checkpointing_mode', 'optimizer_effective', 'optimizer_weight_decay',
    'scheduler', 'peak_lr', 'min_lr', 'warmup_ratio', 'warmup_steps',
    'total_optimizer_steps', 'training_examples', 'epochs', 'max_length',
    'gradient_accumulation', 'seed', 'max_grad_norm',
    'system_prompt_sha256', 'schema_sha256', 'full_system_prompt_used_sha256',
    'tokenizer_class', 'tokenizer_chat_template_sha256', 'lora_rank',
    'lora_alpha', 'lora_initialization', 'lora_dropout', 'lora',
    'trainable_parameter_count', 'trainable_parameter_names',
)


def signature(report: dict, frozen_hashes: dict) -> str:
    """Fingerprint training identity, not volatile timestamps or GPU speed."""
    material = {'format':'mtt-resume-v1', 'inputs':frozen_hashes,
                'effective':{k:report[k] for k in SIGNATURE_FIELDS}}
    return hashlib.sha256(json.dumps(material,sort_keys=True,ensure_ascii=False,
                                     separators=(',',':')).encode('utf-8')).hexdigest()


def available_checkpoints(run_dir: Path):
    parent = Path(run_dir)/'checkpoints'
    if not parent.exists():
        return []
    return sorted((p for p in parent.iterdir() if p.is_dir() and STEP_NAME.fullmatch(p.name)),
                  key=lambda p:int(STEP_NAME.fullmatch(p.name).group(1)))


def select_checkpoint(run_dir: Path, name: str = 'latest') -> Path:
    if name != 'latest' and not STEP_NAME.fullmatch(name):
        raise RuntimeError('Use --checkpoint latest or step_0002000 (checkpoint directory name only)')
    candidates = available_checkpoints(run_dir)
    if not candidates:
        raise RuntimeError('No resumable checkpoints in this run. Older adapter-only checkpoints '
                           '(v0.3.5 and earlier) cannot resume optimizer/RNG state.')
    selected = candidates[-1] if name == 'latest' else next((p for p in candidates if p.name==name),None)
    if selected is None:
        raise RuntimeError('Requested checkpoint does not exist: '+name)
    info_file = selected/'checkpoint_info.json'
    if not info_file.is_file():
        raise RuntimeError(f'Incomplete checkpoint (missing commit marker): {selected}')
    info = read(info_file)
    if info.get('format') != 'mtt-resume-v1' or not info.get('complete'):
        raise RuntimeError(f'Checkpoint is not resumable/complete: {selected}')
    if info.get('step') != int(STEP_NAME.fullmatch(selected.name).group(1)):
        raise RuntimeError(f'Checkpoint step mismatch: {selected}')
    files = info.get('sha256')
    if not isinstance(files,dict) or not files or 'training_state.pt' not in files or 'adapter_config.json' not in files or 'adapter_model.safetensors' not in files:
        raise RuntimeError(f'Checkpoint artifact manifest invalid: {selected}')
    for relative,expected in files.items():
        f = selected/relative
        if not f.is_file() or digest(f)!=expected:
            raise RuntimeError(f'Checkpoint corrupt/missing file: {f}')
    return selected


@contextmanager
def training_lock(folder: Path):
    """OS-managed file lock, automatically released even after a process crash."""
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.training.lock').open('a+b') as fp:
        fp.seek(0);fp.write(b'1');fp.flush();fp.seek(0)
        if os.name == 'nt':
            import msvcrt
            try: msvcrt.locking(fp.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:raise RuntimeError('This run is already being trained or resumed') from exc
            try:yield
            finally:
                fp.seek(0)
                msvcrt.locking(fp.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            try:fcntl.flock(fp.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as exc:raise RuntimeError('This run is already being trained or resumed') from exc
            try:yield
            finally:fcntl.flock(fp.fileno(),fcntl.LOCK_UN)


def save_checkpoint(*, torch, model, tokenizer, optimizer, scaler, trainable,
                    out, step, micro, elapsed_seconds, run_signature, last_event=None, keep_last=3):
    """Write to a staging folder, publish by atomic rename only when all files exist."""
    root=Path(out)/'checkpoints'
    root.mkdir(parents=True,exist_ok=True)
    dest=root/f'step_{step:07d}'
    if dest.exists():
        raise RuntimeError(f'Checkpoint already exists (will not overwrite): {dest}')
    tmp=Path(tempfile.mkdtemp(prefix=f'.{dest.name}.incomplete_',dir=root))
    try:
        model.save_pretrained(tmp,safe_serialization=True)
        if not (tmp/'adapter_config.json').is_file() or not (tmp/'adapter_model.safetensors').is_file():
            raise RuntimeError('PEFT did not produce adapter_config.json and adapter_model.safetensors')
        tokenizer.save_pretrained(tmp)
        params={name:param.detach().cpu().clone() for name,param in trainable}
        import numpy as np
        state = {
            'format':'mtt-resume-v1', 'signature':run_signature,
            'step':step,'micro_step':micro,'elapsed_seconds':elapsed_seconds,
            'last_event':last_event,
            'trainable_parameters':params, 'optimizer':optimizer.state_dict(),
            'scaler':scaler.state_dict(),
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),
            'python_rng':random.getstate(),
            'numpy_rng':np.random.get_state(),
        }
        torch.save(state,tmp/'training_state.pt')
        files={str(p.relative_to(tmp)).replace('\\','/'):digest(p)
               for p in sorted(tmp.rglob('*')) if p.is_file()}
        info={'format':'mtt-resume-v1','complete':True,
              'step':step,'micro_step':micro,'elapsed_seconds':elapsed_seconds,
              'signature':run_signature,'saved_utc':utc(),'sha256':files}
        write(tmp/'checkpoint_info.json',info)
        # Flush checkpoint bytes before atomic publication where the OS allows.
        # This is especially important for an unexpected power loss during save.
        for artifact in tmp.rglob('*'):
            if artifact.is_file():
                with artifact.open('rb') as source:
                    os.fsync(source.fileno())
        # Rename on the same filesystem is atomic. A crash before this point
        # leaves a hidden .incomplete_ directory, never a published checkpoint.
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True)
        raise
    if keep_last > 0:
        previous=available_checkpoints(out)
        for older in previous[:-keep_last]:
            shutil.rmtree(older)
    return dest


def restore_checkpoint(*, torch, checkpoint, expected_signature, optimizer,
                       scaler, trainable, total_micro, total_steps, accum):
    """Validate entire checkpoint before restoring any optimizer/model state."""
    checkpoint=Path(checkpoint)
    info=read(checkpoint/'checkpoint_info.json')
    if info['signature'] != expected_signature:
        raise RuntimeError('Checkpoint training configuration or frozen inputs differ from current run')
    # Local trusted checkpoints only; torch.load with weights_only=False may unpickle.
    state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
    if state.get('format')!='mtt-resume-v1' or state.get('signature')!=expected_signature:
        raise RuntimeError('Checkpoint state does not match frozen training plan')
    micro=int(state['micro_step']); step=int(state['step'])
    if step!=info['step'] or micro!=info['micro_step']:
        raise RuntimeError('Checkpoint position metadata mismatch')
    if not (0 < micro <= total_micro and 0 < step <= total_steps and
            (micro % accum == 0 or micro == total_micro) and
            step == (micro + accum - 1)//accum):
        raise RuntimeError('Checkpoint not at a completed optimizer update')
    expected={name:parameter for name,parameter in trainable}
    actual=state['trainable_parameters']
    if set(actual)!=set(expected):
        raise RuntimeError('Checkpoint LoRA parameter names differ from current model')
    for name,param in expected.items():
        if tuple(actual[name].shape)!=tuple(param.shape):
            raise RuntimeError('Checkpoint LoRA tensor shape mismatch: '+name)
    with torch.no_grad():
        for name,param in expected.items():
            param.copy_(actual[name].to(device=param.device,dtype=param.dtype))
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    import numpy as np
    np.random.set_state(state['numpy_rng'])
    return {'step':step,'micro_step':micro,'elapsed_seconds':float(state['elapsed_seconds']),
            'last_event':state.get('last_event')}
