"""Simple reproducible assistant-only causal LM LoRA SFT, always from configured base.

This is an intentionally compact first-version backend; no TRL and no automatic
continuation from prior adapters. Requires optional ML environment profile.
"""
from __future__ import annotations
import json, math, random, time
from pathlib import Path
from .core import read, write, utc, digest
from .dataset import rows, expected_text


def run(snapshot:Path,out:Path):
    try:
        import torch
        from torch.optim import AdamW
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:raise RuntimeError('ML dependencies missing. Install with bash run_me_first.sh --install-ml') from exc
    cfg=read(snapshot/'version.json')
    tr=cfg['training'];base=cfg['base_model']
    if cfg.get('initialization','fresh_base')!='fresh_base':
        raise ValueError('Toolkit 0.1 intentionally supports only fresh_base training')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; this first training backend requires an NVIDIA CUDA device')
    seed=int(tr['seed']);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    try:torch.use_deterministic_algorithms(True,warn_only=True)
    except Exception:pass
    prompt=(snapshot/'data/system_prompt.txt').read_text(encoding='utf-8').rstrip('\n')
    schema_file=snapshot/'data/schema.json'
    schema=schema_file.read_text(encoding='utf-8') if schema_file.exists() else ''
    if read(snapshot/'data/package.json')['prompt_assembly']=='append_schema':
        system=prompt+'\n\nSchema:\n'+schema
    else:system=prompt
    train=rows(snapshot/'data/train.jsonl')
    if not train:raise RuntimeError('No training data')
    tokenizer=AutoTokenizer.from_pretrained(base['id'],revision=base.get('revision') or 'main',trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    max_len=int(tr['max_length'])
    if max_len<128:raise ValueError('max_length must be >= 128')
    packed=[];excluded=[]
    for i,row in enumerate(train):
        preceding=[{'role':'system','content':system},{'role':'user','content':row['question']}]
        output=expected_text(row['expected'])
        full=preceding+[{'role':'assistant','content':output}]
        # assistant-only labels: use separate prefix and full template with the same tokenizer.
        ids=tokenizer.apply_chat_template(full,tokenize=True,add_generation_prompt=False)
        prefix=tokenizer.apply_chat_template(preceding,tokenize=True,add_generation_prompt=True)
        if ids[:len(prefix)]!=prefix:
            raise RuntimeError(f'Chat template assistant-prefix mismatch for training example {i}; abort rather than train on wrong labels')
        if len(ids)>max_len or len(ids)<=len(prefix):
            excluded.append({'index':i,'id':row.get('id'),'tokens':len(ids)});continue
        packed.append((ids,[-100]*len(prefix)+ids[len(prefix):]))
    if excluded:raise RuntimeError(f'{len(excluded)} examples exceed max_length or have empty assistant targets; increase max_length. First: {excluded[0]}')
    precision=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=AutoModelForCausalLM.from_pretrained(base['id'],revision=base.get('revision') or 'main',torch_dtype=precision,trust_remote_code=False)
    model.config.use_cache=False
    model.to('cuda')
    lora=tr['lora']
    model=get_peft_model(model,LoraConfig(r=int(lora['r']),lora_alpha=int(lora['alpha']),lora_dropout=float(lora['dropout']),target_modules=lora['target_modules'],task_type='CAUSAL_LM'))
    model.train();scaler=torch.amp.GradScaler('cuda',enabled=(precision==torch.float16));optimizer=AdamW([p for p in model.parameters() if p.requires_grad],lr=float(tr['learning_rate']),weight_decay=float(tr['weight_decay']))
    epochs=int(tr['epochs']);accum=int(tr['gradient_accumulation']);total=math.ceil(len(packed)*epochs/accum)
    if epochs<1 or accum<1:raise ValueError('epochs and gradient_accumulation must be >=1')
    warmup=max(1,round(total*float(tr['warmup_ratio'])))
    output=out/'adapter';output.mkdir(parents=True,exist_ok=False)
    write(out/'training_plan.json',{'samples':len(packed),'epochs':epochs,'optimizer_steps':total,'warmup_steps':warmup,'device':torch.cuda.get_device_name(0),'base':base,'system_prompt_sha256':digest(snapshot/'data/system_prompt.txt'),'schema_sha256':digest(snapshot/'data/schema.json') if schema_file.exists() else None,'started_utc':utc()})
    t0=time.monotonic(); step=0;micro=0;optimizer.zero_grad(set_to_none=True)
    with (out/'training_progress.jsonl').open('w',encoding='utf-8') as log:
        for epoch in range(epochs):
            order=list(range(len(packed)));random.Random(seed+epoch).shuffle(order)
            for j,idx in enumerate(order):
                ids,labels=packed[idx]
                input_ids=torch.tensor([ids],dtype=torch.long,device='cuda')
                label_tensor=torch.tensor([labels],dtype=torch.long,device='cuda')
                with torch.autocast(device_type='cuda',dtype=precision):
                    loss=model(input_ids=input_ids,labels=label_tensor).loss
                batch_start=(j//accum)*accum
                group_size=min(accum,len(order)-batch_start)
                scaler.scale(loss/group_size).backward();micro+=1
                is_update=(j+1)%accum==0 or j==len(order)-1
                if not is_update:continue
                step+=1
                if step<=warmup:lr=float(tr['learning_rate'])*step/warmup
                else:
                    prog=(step-warmup)/max(1,total-warmup)
                    lr=float(tr['learning_rate'])*(0.1+0.9*0.5*(1+math.cos(math.pi*prog)))
                for group in optimizer.param_groups:group['lr']=lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(tr['max_grad_norm']))
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                event={'step':step,'loss':round(float(loss.item()),6),'learning_rate':lr,'epoch':epoch+1,'elapsed_seconds':round(time.monotonic()-t0,2)}
                log.write(json.dumps(event)+'\n');log.flush()
                if step%int(tr['log_every'])==0 or step==total:print('TRAIN',event,flush=True)
    model.save_pretrained(output,safe_serialization=True)
    tokenizer.save_pretrained(output)
    write(out/'training_result.json',{'step':step,'loss':event['loss'],'elapsed_seconds':event['elapsed_seconds'],'adapter':str(output),'completed_utc':utc(),'base_resolved_revision':getattr(model.base_model.model.config,'_commit_hash',None)})
    return {'steps':step,'loss':event['loss'],'elapsed_seconds':event['elapsed_seconds']}
