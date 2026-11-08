"""Mandatory per-run review of every trainer-controlled hyperparameter (stdlib only)."""
from __future__ import annotations
from copy import deepcopy
from .wizard import ask, positive_int, positive_float, confirm, hardware, display_models, MODEL_CATALOG


def entry(label, suggested, parser, error, *, check=lambda v: True):
    """Require an explicit entry, not silent acceptance by pressing Enter."""
    while True:
        value=input(f'{label} (suggested: {suggested}; enter value): ').strip()
        if not value:
            print('Required for every run. Type the suggested value to use it.')
            continue
        try:
            parsed=parser(value)
            if check(parsed):return parsed
        except (ValueError,TypeError):pass
        print(error)


def yesno(label,default):
    suggestion='yes' if default else 'no'
    return entry(label,suggestion,lambda v:v.lower(),'Enter yes or no',check=lambda x:x in ('y','n','yes','no')) in ('y','yes')


def positive_float_value(s):return float(s)


def wizard(config, *,previous=None):
    """Review / explicitly enter all training controls; never mutate version config."""
    current=deepcopy(config['training'])
    if previous:current.update({k:v for k,v in previous.items() if k!='lora'})
    if previous and 'lora' in previous:current['lora'].update(previous['lora'])
    print('\nTRAINING CONFIGURATION — REQUIRED EACH RUN')
    print('These values apply only to the next run. Blank answers are not accepted.')
    selected_base=deepcopy(config.get('base_model') or {})
    if not selected_base.get('id') or not selected_base.get('revision'):
        raise RuntimeError('Version has no base model. Select one when registering the version.')
    print('BASE MODEL (selected at version creation):',selected_base['id'])
    print('BASE REVISION:',selected_base['revision'])
    print('To change models, create a different version; no silent model swaps per run.')

    print('Initialization: fresh base weights (no previous adapter)')
    machine=hardware()
    for gpu in machine['gpus']:print('GPU:',gpu['name'],'VRAM:',gpu['vram_gb'],'GiB')
    print('Disk free:',machine['disk_free_gb'],'GiB')
    c={}
    c['backend']=entry('Training backend (pytorch/unsloth)',current.get('backend','pytorch'),str,
                       'Enter pytorch or unsloth',check=lambda v:v in ('pytorch','unsloth'))
    c['epochs']=entry('Epochs',current['epochs'],int,'Enter a positive integer',check=lambda v:v>=1)
    c['max_length']=entry('Maximum sequence tokens (prompt + target)',current['max_length'],int,'Enter integer >= 128',check=lambda v:v>=128)
    c['learning_rate']=entry('Learning rate',current['learning_rate'],float,'Enter learning rate greater than 0 and less than 1',check=lambda v:0<v<1)
    c['gradient_accumulation']=entry('Gradient accumulation',current['gradient_accumulation'],int,'Enter positive integer',check=lambda v:v>=1)
    c['seed']=entry('Random seed',current['seed'],int,'Enter nonnegative integer',check=lambda v:v>=0)
    c['weight_decay']=entry('Weight decay',current['weight_decay'],float,'Enter a number in [0,1)',check=lambda v:0<=v<1)
    c['warmup_ratio']=entry('Warmup ratio',current['warmup_ratio'],float,'Enter a number in [0,1)',check=lambda v:0<=v<1)
    c['max_grad_norm']=entry('Max gradient norm',current['max_grad_norm'],float,'Enter a positive number',check=lambda v:v>0)
    c['log_every']=entry('Log every N optimizer updates',current['log_every'],int,'Enter positive integer',check=lambda v:v>=1)
    old_lora=current['lora']
    l={}
    l['r']=entry('LoRA rank (r)',old_lora['r'],int,'Enter positive integer',check=lambda v:v>=1)
    l['alpha']=entry('LoRA alpha',old_lora['alpha'],int,'Enter positive integer',check=lambda v:v>=1)
    l['dropout']=entry('LoRA dropout',old_lora['dropout'],float,'Enter a number from 0 up to (not including) 1',check=lambda v:0<=v<1)
    mods=entry('LoRA target modules (comma-separated)',','.join(old_lora['target_modules']),lambda v:[x.strip() for x in v.split(',')], 'Provide one or more module names',check=lambda x:bool(x) and all(x))
    l['target_modules']=mods
    c['lora']=l
    c['min_learning_rate']=entry('Minimum learning rate (absolute cosine floor)',current.get('min_learning_rate',current['learning_rate']*0.1),float,'Enter >0 and <= peak LR',check=lambda v:0<v<=c['learning_rate'])
    c['precision']=entry('Precision (auto/bf16/fp16)',current.get('precision','auto'),str,'Enter auto, bf16, or fp16',check=lambda v:v in ('auto','bf16','fp16'))
    c['attention_implementation']=entry('Attention implementation (sdpa/eager/flash_attention_2)',current.get('attention_implementation','sdpa'),str,'Enter supported attention backend',check=lambda v:v in ('sdpa','eager','flash_attention_2'))
    c['optimizer']=entry('Optimizer (fused-adamw/adamw)',current.get('optimizer','fused-adamw'),str,'Enter fused-adamw or adamw',check=lambda v:v in ('fused-adamw','adamw'))
    c['gradient_checkpointing']=yesno('Use gradient checkpointing?',current.get('gradient_checkpointing',True))
    c['save_every']=entry('Save FULL resumable checkpoint every N optimizer steps (0=disabled)',current.get('save_every',500),int,'Enter >=0',check=lambda v:v>=0)
    c['keep_last_checkpoints']=entry('Retain last N complete checkpoints (0=all)',current.get('keep_last_checkpoints',3),int,'Enter >=0',check=lambda v:v>=0)
    if not c['save_every']:
        print('WARNING: checkpoint saving is disabled; an interruption will require a complete new run.')
    if c['backend']=='unsloth':
        c['unsloth_checkpointing']=entry('Unsloth checkpoint implementation (standard/unsloth)',
            current.get('unsloth_checkpointing','standard'),str,
            'Enter standard or unsloth',check=lambda v:v in ('standard','unsloth'))
    else:
        c['unsloth_checkpointing']='standard'
    auto=yesno('Evaluate automatically after successful training?','auto_evaluate_default' in config.get('workflow',{}) and config['workflow']['auto_evaluate_default'])
    print('\nFINAL RUN PLAN (recorded in the run manifest)')
    import json
    print(json.dumps({'base_model':selected_base,'initialization':'fresh_base','training':c,'auto_evaluate':auto},indent=2))
    if not confirm('Confirm these settings and START TRAINING?',False):
        print('CANCELED: no training run created')
        return None
    return c,auto,selected_base
