"""Project-neutral fresh-base PEFT LoRA SFT with audited adapter and training plan.

The engine intentionally does not assume a particular architecture or target-module
list. It refuses missing/extra LoRA targets rather than silently accepting a
partially matched configuration. No implicit adapter continuation.
"""
from __future__ import annotations
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from .core import read, write, utc, digest
from .dataset import rows, expected_text
from .backends import prepare_backend, backend_version, load_unsloth_base, attach_unsloth_lora


def module_matches(name: str, target: str) -> bool:
    """PEFT's list-of-module-names semantics (exact name or dotted suffix)."""
    return name == target or name.endswith('.' + target)


def verify_lora_modules(model, requested, lora_layer_type):
    """Return all actual LoRA-wrapped layers or fail on a partial/extra match."""
    requested = list(requested)
    if not requested or len(set(requested)) != len(requested):
        raise RuntimeError('LoRA modules must be unique nonempty names')
    wrapped = [(name, module) for name, module in model.named_modules()
               if isinstance(module, lora_layer_type)]
    if not wrapped:
        raise RuntimeError('PEFT created zero LoRA layers; aborting before optimization')
    matched = {target: [] for target in requested}
    unexpected = []
    for name, _ in wrapped:
        hits = [target for target in requested if module_matches(name, target)]
        if len(hits) != 1:
            unexpected.append(name)
        else:
            matched[hits[0]].append(name)
    missing = [name for name, matches in matched.items() if not matches]
    if unexpected or missing:
        raise RuntimeError('LoRA PEFT layer mismatch: missing targets=' + repr(missing)
                           + ' unexpected/ambiguous layers=' + repr(unexpected[:20]))
    return {'requested_targets':requested,
            'matched_targets':{name:len(paths) for name,paths in matched.items()},
            'layer_names':sorted(name for name, _ in wrapped),
            'adapter_layer_count':len(wrapped)}


def optimizer_schedule(tr, total_steps):
    """Absolute LR floor, warmup, then cosine; compatible with the V8.0 formula."""
    peak = float(tr['learning_rate'])
    floor = float(tr.get('min_learning_rate', peak * 0.1))
    if not (0 < floor <= peak):
        raise ValueError('min_learning_rate must be >0 and <= learning_rate')
    ratio = float(tr['warmup_ratio'])
    if not 0 <= ratio < 1:
        raise ValueError('warmup_ratio must be in [0,1)')
    warmup = max(1, int(total_steps * ratio)) if ratio else 0
    def at(update_index):
        if update_index < warmup:
            return peak * (update_index + 1) / warmup
        progress = min(1.0, max(0.0, (update_index - warmup) / max(1, total_steps - warmup)))
        return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    return at, warmup, floor



def supervised_messages(row, system, index):
    """Return chat-template messages without altering approved assistant text."""
    preceding = [{'role':'system','content':system}, {'role':'user','content':row['question']}]
    if 'messages' not in row:
        return preceding, preceding + [{'role':'assistant','content':expected_text(row['expected'])}]
    messages = row['messages']
    if (not isinstance(messages,list) or len(messages)!=3 or
        [m.get('role') for m in messages] != ['system','user','assistant'] or
        messages[:2]!=preceding or not isinstance(messages[2].get('content'),str)):
        raise RuntimeError(f'Messages/system prompt mismatch on example {index}')
    target = messages[2]['content']
    expected = row['expected']
    if isinstance(expected,str):
        equivalent = target == expected
    else:
        try:equivalent = json.loads(target) == expected
        except (ValueError,TypeError):equivalent = False
    if not equivalent:
        raise RuntimeError(f'Assistant chat text differs from gold expected on example {index}')
    return preceding, messages

def run(snapshot: Path, out: Path):
    # Import Unsloth *before* torch/transformers/PEFT. Its patching relies on
    # first-import order, and failure must never fall back to another backend.
    cfg = read(snapshot/'version.json')
    tr = cfg['training']
    base = cfg['base_model']
    backend = tr.get('backend', 'pytorch')
    fast_model = prepare_backend(backend)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        from peft.tuners.lora import LoraLayer
    except ImportError as exc:
        raise RuntimeError('ML dependencies missing. Run bash run_me_first.sh --install-ml') from exc
    if cfg.get('initialization', 'fresh_base') != 'fresh_base':
        raise ValueError('Only fresh_base training is supported')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; training requires an NVIDIA CUDA device')
    seed = int(tr['seed'])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    prompt = (snapshot/'data/system_prompt.txt').read_text(encoding='utf-8').rstrip('\r\n')
    schema_file = snapshot/'data/schema.json'
    schema = schema_file.read_text(encoding='utf-8') if schema_file.exists() else ''
    package = read(snapshot/'data/package.json')
    system = prompt + '\n\nSchema:\n' + schema if package['prompt_assembly'] == 'append_schema' else prompt
    train = rows(snapshot/'data/train.jsonl')
    if not train:
        raise RuntimeError('Empty training dataset')
    max_len = int(tr['max_length'])
    if max_len < 128:
        raise ValueError('max_length must be >= 128')
    precision_setting = tr.get('precision', 'auto')
    if precision_setting not in ('auto', 'bf16', 'fp16'):
        raise ValueError('precision must be auto, bf16, or fp16')
    if precision_setting == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 requested but not supported on GPU')
    precision = (torch.bfloat16 if precision_setting == 'bf16' or
                 (precision_setting == 'auto' and torch.cuda.is_bf16_supported())
                 else torch.float16)
    attention = tr.get('attention_implementation', 'sdpa')
    if attention not in ('sdpa', 'eager', 'flash_attention_2'):
        raise ValueError('Unsupported attention_implementation')
    checkpointing = bool(tr.get('gradient_checkpointing', True))
    checkpointing_mode = tr.get('unsloth_checkpointing', 'standard')
    if checkpointing_mode not in ('standard', 'unsloth'):
        raise ValueError('unsloth_checkpointing must be standard or unsloth')
    if backend != 'unsloth' and checkpointing_mode != 'standard':
        raise ValueError('unsloth_checkpointing=unsloth requires backend=unsloth')

    if backend == 'unsloth':
        # Load the model and tokenizer together: Unsloth may patch tokenizers,
        # so tokenization MUST use the tokenizer from the effective backend.
        model, tokenizer = load_unsloth_base(
            fast_model, base=base, max_length=max_len,
            precision=precision, attention=attention)
    else:
        model = None
        tokenizer = AutoTokenizer.from_pretrained(
            base['id'], revision=base.get('revision') or 'main', trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    packed = []
    excluded = []
    lengths = []
    for i, row in enumerate(train):
        preceding, full = supervised_messages(row, system, i)
        ids = tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False)
        prefix = tokenizer.apply_chat_template(preceding, tokenize=True, add_generation_prompt=True)
        if ids[:len(prefix)] != prefix:
            raise RuntimeError(f'Chat-template assistant-prefix mismatch on example {i}; refusing wrong labels')
        if len(ids) > max_len or len(ids) <= len(prefix):
            excluded.append({'index':i, 'id':row.get('id'), 'tokens':len(ids)})
            continue
        packed.append((ids, len(prefix)))
        lengths.append(len(ids))
    if excluded:
        raise RuntimeError(f'{len(excluded)} examples overlength or empty target; no silent skipping. First: {excluded[0]}')
    epochs = int(tr['epochs'])
    accum = int(tr['gradient_accumulation'])
    if epochs < 1 or accum < 1:
        raise ValueError('epochs and gradient_accumulation must be >= 1')
    total_micro = len(packed) * epochs
    total_steps = math.ceil(total_micro / accum)
    at_lr, warmup, min_lr = optimizer_schedule(tr, total_steps)

    if backend == 'pytorch':
        model = AutoModelForCausalLM.from_pretrained(
            base['id'], revision=base.get('revision') or 'main',
            torch_dtype=precision, attn_implementation=attention,
            trust_remote_code=False).to('cuda')
    loaded_revision = getattr(model.config, '_commit_hash', None)
    pinned = base.get('revision') or ''
    if len(pinned) == 40 and all(c in '0123456789abcdefABCDEF' for c in pinned):
        if loaded_revision and loaded_revision.lower() != pinned.lower():
            raise RuntimeError('Loaded model commit differs from frozen base revision: '
                               f'{loaded_revision} != {pinned}; refusing to train')
    model.config.use_cache = False
    lora = tr['lora']
    targets = list(lora['target_modules'])
    # Capture the model's eligible modules before PEFT wraps them; if PEFT
    # silently skips any eligible layers, require an explicit failure.
    eligible = {name:[path for path, module in model.named_modules()
                       if module_matches(path,name) and hasattr(module,'weight')]
                for name in targets}
    if any(not paths for paths in eligible.values()):
        raise RuntimeError('Requested LoRA target(s) absent in base model: '+
                           repr([k for k,v in eligible.items() if not v]))
    if backend == 'unsloth':
        model = attach_unsloth_lora(
            fast_model, model, lora=lora, seed=seed, max_length=max_len,
            gradient_checkpointing=checkpointing,
            checkpointing_mode=checkpointing_mode)
    else:
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
                               r=int(lora['r']), lora_alpha=int(lora['alpha']),
                               lora_dropout=float(lora['dropout']), bias='none',
                               init_lora_weights=True, target_modules=targets))
    adapters = verify_lora_modules(model, targets, LoraLayer)
    if any(adapters['matched_targets'][key]!=len(value) for key,value in eligible.items()):
        raise RuntimeError('PEFT adapted fewer/more base layers than requested: '+
                           repr({key:{'eligible':len(value),'adapted':adapters['matched_targets'][key]}
                                 for key,value in eligible.items()}))
    if backend == 'pytorch' and checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if backend == 'unsloth' and hasattr(fast_model, 'for_training'):
        fast_model.for_training(model)
    model.train()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError('No trainable adapter parameters')
    if backend == 'unsloth' and not all(p.device.type == 'cuda' for _,p in trainable):
        raise RuntimeError('Unsloth returned LoRA parameters outside CUDA; refusing CPU training')
    if any('lora_' not in name for name, _ in trainable):
        raise RuntimeError('Found trainable non-LoRA weights; refusing unexpected base-model updates')
    count_trainable = sum(p.numel() for _, p in trainable)
    count_total = sum(p.numel() for p in model.parameters())
    optimizer_choice = tr.get('optimizer', 'fused-adamw')
    if optimizer_choice not in ('adamw', 'fused-adamw'):
        raise ValueError('optimizer must be adamw or fused-adamw')
    optimizer_params = [p for _,p in trainable]
    opt_kw = {'lr':float(tr['learning_rate']), 'weight_decay':float(tr['weight_decay'])}
    if optimizer_choice == 'fused-adamw':
        try:
            optimizer = torch.optim.AdamW(optimizer_params, fused=True, **opt_kw)
            effective_optimizer = 'fused-adamw'
        except (TypeError, RuntimeError, ValueError):
            optimizer = torch.optim.AdamW(optimizer_params, **opt_kw)
            effective_optimizer = 'adamw (fused unavailable)'
    else:
        optimizer = torch.optim.AdamW(optimizer_params, **opt_kw)
        effective_optimizer = 'adamw'
    save_every = int(tr.get('save_every', 0))
    if save_every < 0:
        raise ValueError('save_every must be >=0')
    report = {
        'started_utc':utc(), 'base':base, 'initialization':'fresh_base',
        'loaded_model_commit_reported':loaded_revision,
        'model_commit_verification':('confirmed' if loaded_revision and loaded_revision.lower()==pinned.lower()
                                     else 'backend_did_not_expose_commit'),
        'backend':backend, 'backend_version':backend_version(backend),
        'quantization':'none (16-bit base)',
        'sequence_packing':False,
        'backend_attention_note':('Unsloth may substitute optimized attention kernels; '
                                  'requested attention is not guaranteed numerically identical'
                                  if backend == 'unsloth' else 'Transformers attention selection'),
        'device':torch.cuda.get_device_name(0), 'cuda_runtime':torch.version.cuda,
        'torch_version':torch.__version__,
        'precision_requested':precision_setting, 'precision_effective':str(precision).replace('torch.',''),
        'attention_requested':attention,
        'attention_effective':getattr(model.config, '_attn_implementation',None),
        'gradient_checkpointing':checkpointing,
        'checkpointing_mode': (checkpointing_mode if backend == 'unsloth' else 'pytorch_non_reentrant'),
        'optimizer_requested':optimizer_choice, 'optimizer_effective':effective_optimizer,
        'optimizer_weight_decay':float(tr['weight_decay']),
        'scheduler':'cosine_absolute_minimum', 'peak_lr':float(tr['learning_rate']),
        'min_lr':min_lr, 'warmup_ratio':float(tr['warmup_ratio']),
        'warmup_steps':warmup, 'total_optimizer_steps':total_steps,
        'training_examples':len(packed), 'epochs':epochs,
        'train_tokens_per_epoch':sum(len(ids) for ids,_ in packed),
        'assistant_target_tokens_per_epoch':sum(len(ids)-prefix_len for ids,prefix_len in packed),
        'loss_masking':'assistant_only_prefix_labels_minus_100',
        'max_length':max_len, 'max_observed_tokens':max(lengths),
        'gradient_accumulation':accum, 'effective_micro_batch':1,
        'effective_batch_size':accum,
        'seed':seed, 'max_grad_norm':float(tr['max_grad_norm']), 'log_every':int(tr['log_every']),
        'save_every':save_every, 'intermediate_checkpoints':'adapter_only_non_resumable',
        'system_prompt_sha256':digest(snapshot/'data/system_prompt.txt'),
        'schema_sha256':digest(schema_file) if schema_file.exists() else None,
        'full_system_prompt_used_sha256': __import__('hashlib').sha256(system.encode('utf-8')).hexdigest(),
        'tokenizer_class':tokenizer.__class__.__name__,
        'tokenizer_chat_template_sha256':__import__('hashlib').sha256(str(tokenizer.chat_template).encode('utf-8')).hexdigest(),
        'lora_rank':int(lora['r']), 'lora_alpha':int(lora['alpha']),
        'lora_initialization':('unsloth_get_peft_model_default' if backend == 'unsloth'
                               else 'peft_default_A_kaiming_B_zero'),
        'lora_dropout':float(lora['dropout']), 'lora':adapters,
        'trainable_parameter_count':count_trainable, 'total_parameter_count':count_total,
        'trainable_fraction':round(count_trainable / count_total, 9),
        'trainable_parameter_names':sorted(name for name, _ in trainable),
    }
    print('EFFECTIVE TRAINING CONFIGURATION — VERIFIED BEFORE FIRST OPTIMIZER STEP',
          json.dumps(report,indent=2),flush=True)
    write(out/'training_plan.json',report)
    output = out/'adapter'
    if output.exists():
        raise RuntimeError('Adapter directory already exists; refusing to overwrite')
    output.mkdir(parents=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(precision == torch.float16))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    step = 0
    micro = 0
    event = None
    optimizer.zero_grad(set_to_none=True)
    with (out/'training_progress.jsonl').open('w',encoding='utf-8') as log:
        for epoch in range(epochs):
            order = list(range(len(packed)))
            random.Random(seed+epoch).shuffle(order)
            for idx in order:
                ids, prefix_len = packed[idx]
                x = torch.tensor([ids],dtype=torch.long,device='cuda')
                y = x.clone()
                y[:,:prefix_len] = -100
                with torch.autocast(device_type='cuda', dtype=precision):
                    loss = model(input_ids=x,labels=y).loss
                # Last partial accumulation group is scaled by its actual size.
                group_size = min(accum, total_micro - (micro // accum)*accum)
                scaler.scale(loss/group_size).backward()
                micro += 1
                if micro % accum != 0 and micro != total_micro:
                    continue
                lr = at_lr(step)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(optimizer_params, float(tr['max_grad_norm']))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                event = {'step':step,'loss':round(float(loss.detach().item()),6),
                         'learning_rate':lr,'epoch':epoch+1,
                         'elapsed_seconds':round(time.monotonic()-t0,2)}
                log.write(json.dumps(event)+'\n')
                log.flush()
                if step % int(tr['log_every']) == 0 or step == total_steps:
                    print('TRAIN',event,flush=True)
                if save_every and step % save_every == 0 and step != total_steps:
                    interim = out/'checkpoints'/f'step_{step:07d}'
                    if interim.exists():
                        raise RuntimeError('Checkpoint already exists: '+str(interim))
                    interim.mkdir(parents=True)
                    model.save_pretrained(interim,safe_serialization=True)
                    tokenizer.save_pretrained(interim)
                    write(interim/'checkpoint_info.json',{
                        'step':step,'kind':'adapter_only_not_optimizer_resumable',
                        'training_plan':str(out/'training_plan.json')})
    if step != total_steps:
        raise RuntimeError(f'Internal optimizer-step count mismatch: {step} != {total_steps}')
    model.save_pretrained(output,safe_serialization=True)
    tokenizer.save_pretrained(output)
    result = {'step':step, 'loss':event['loss'],
              'elapsed_seconds':event['elapsed_seconds'], 'adapter':str(output),
              'completed_utc':utc(),
              'base_resolved_revision':getattr(model.base_model.model.config,'_commit_hash',None),
              'trainable_parameter_count':count_trainable,
              'lora_matched_targets':adapters['matched_targets'], 'backend':backend,
              'train_tokens_processed':report['train_tokens_per_epoch']*epochs,
              'training_tokens_per_second':round(report['train_tokens_per_epoch']*epochs / max(1e-9, time.monotonic()-t0), 2),
              'peak_allocated_gpu_gib':round(torch.cuda.max_memory_allocated()/1024**3,3)}
    write(out/'training_result.json',result)
    return {'steps':step,'loss':event['loss'],'elapsed_seconds':event['elapsed_seconds']}
