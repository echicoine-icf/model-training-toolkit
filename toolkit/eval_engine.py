"""Inference evaluator: runs every frozen eval case and stores raw predictions."""
import json, time
from pathlib import Path
from .core import read, write, utc
from .dataset import rows


def normalize(x):
    # Exact structured JSON comparison; list order is intentionally significant.
    if isinstance(x,(dict,list)):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return str(x).strip()


def run(snapshot,out,adapter=None):
    try:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:raise RuntimeError('ML dependencies missing. Install with bash run_me_first.sh --install-ml') from exc
    cfg=read(snapshot/'version.json');base=cfg['base_model']
    cases=rows(snapshot/'data/eval.jsonl')
    if not cases:raise RuntimeError('Frozen evaluation dataset empty')
    prompt=(snapshot/'data/system_prompt.txt').read_text(encoding='utf-8').rstrip('\n')
    schema_file=snapshot/'data/schema.json'
    schema=schema_file.read_text(encoding='utf-8') if schema_file.exists() else ''
    system=prompt+'\n\nSchema:\n'+schema if read(snapshot/'data/package.json')['prompt_assembly']=='append_schema' else prompt
    tokenizer=AutoTokenizer.from_pretrained(adapter or base['id'],revision=None if adapter else (base.get('revision') or 'main'),trust_remote_code=False)
    use_cuda=torch.cuda.is_available()
    dtype=torch.float16 if use_cuda else torch.float32
    model=AutoModelForCausalLM.from_pretrained(base['id'],revision=base.get('revision') or 'main',torch_dtype=dtype,trust_remote_code=False)
    if adapter:model=PeftModel.from_pretrained(model,str(adapter))
    model=model.to('cuda' if use_cuda else 'cpu');model.eval()
    out.mkdir(parents=True,exist_ok=True)
    results=[];stats={'cases':len(cases),'parsed':0,'exact':0,'errors':0}
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in cases:
            messages=[{'role':'system','content':system},{'role':'user','content':row['question']}]
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt').to(model.device)
            start=time.monotonic()
            with torch.no_grad():
                gen=model.generate(ids,max_new_tokens=int(cfg['evaluation']['max_new_tokens']),do_sample=False,pad_token_id=tokenizer.eos_token_id)
            raw=tokenizer.decode(gen[0,ids.shape[1]:],skip_special_tokens=True).strip()
            expected=row['expected'];parsed=None;parse_ok=False
            try:parsed=json.loads(raw);parse_ok=True
            except (json.JSONDecodeError,ValueError):pass
            output=parsed if parse_ok else raw
            same=normalize(output)==normalize(expected)
            if parse_ok:stats['parsed']+=1
            if same:stats['exact']+=1
            if not same:stats['errors']+=1
            item={'id':row.get('id'),'question':row['question'],'expected':expected,'prediction':raw,'parsed':parsed,'json_valid':parse_ok,'exact_match':same,'elapsed_seconds':round(time.monotonic()-start,3)}
            f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
    stats['exact_accuracy']=round(stats['exact']/len(cases),6)
    stats.update({'evaluated_utc':utc(),'scoring':'exact JSON structural match (ordered lists) or exact string; project-specific semantics NOT inferred','model':'adapter' if adapter else 'base'})
    write(out/'summary.json',stats)
    print('EVALUATION',json.dumps(stats,indent=2))
    return stats
