#!/usr/bin/env python3
"""Compare BF16/FP8/INT8 inference latency and throughput via a vLLM OpenAI API.

Uses a fixed question set and system prompt. Does not change an MTT run or score
semantic correctness; use the existing ELI shared benchmark for correctness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen


def percentile(values, percentile_value):
    if not values:
        return None
    arr = sorted(values)
    ix = max(0, math.ceil((percentile_value / 100) * len(arr)) - 1)
    return round(arr[ix], 3)


def read_prompts(path: Path):
    prompts=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        obj=json.loads(line)
        q=obj.get('question') or obj.get('prompt')
        if not isinstance(q,str) or not q:
            raise ValueError('Every JSONL row requires a question or prompt string')
        prompts.append(q)
    if not prompts:
        raise ValueError('No questions in JSONL file')
    return prompts


def inference(url, model, system, question, max_tokens, seed, timeout):
    start=perf_counter()
    payload={'model':model,'messages':[{'role':'system','content':system},
                                       {'role':'user','content':question}],
             'temperature':0,'top_p':1,'max_tokens':max_tokens,'seed':seed,'stream':False}
    req=Request(url.rstrip('/')+'/chat/completions',
                data=json.dumps(payload).encode(),
                headers={'Content-Type':'application/json'},method='POST')
    with urlopen(req,timeout=timeout) as resp:
        result=json.load(resp)
    usage=result.get('usage') or {}
    return {'seconds':perf_counter()-start,
            'output_tokens':int(usage.get('completion_tokens') or 0),
            'input_tokens':int(usage.get('prompt_tokens') or 0),
            'finish_reason':result['choices'][0].get('finish_reason')}


def measure(url, model, system, questions, *, concurrency, max_tokens, seed, timeout,
            inference_fn=inference):
    t0=perf_counter()
    results=[]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        jobs=[pool.submit(inference_fn,url,model,system,q,max_tokens,seed,timeout) for q in questions]
        for job in as_completed(jobs):
            results.append(job.result())
    seconds=perf_counter()-t0
    return {'concurrency':concurrency, 'requests':len(results),
            'wall_seconds':round(seconds,3),
            'requests_per_second':round(len(results)/seconds,3),
            'output_tokens_per_second':round(sum(x['output_tokens'] for x in results)/seconds,3),
            'prompt_tokens_total':sum(x['input_tokens'] for x in results),
            'output_tokens_total':sum(x['output_tokens'] for x in results),
            'latency_seconds_p50':percentile([x['seconds'] for x in results],50),
            'latency_seconds_p95':percentile([x['seconds'] for x in results],95)}


def main():
    p=argparse.ArgumentParser(description='Measure OpenAI-compatible vLLM endpoint throughput with frozen inputs')
    p.add_argument('--url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--model',required=True)
    p.add_argument('--cases',type=Path,required=True,help='JSONL questions from a frozen benchmark')
    p.add_argument('--system-prompt',type=Path,required=True)
    p.add_argument('--requests',type=int,default=32,help='Identical request count in every concurrency trial')
    p.add_argument('--concurrency',default='1,8,32')
    p.add_argument('--max-tokens',type=int,default=768)
    p.add_argument('--seed',type=int,default=8002026)
    p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--output',type=Path,help='Write metrics JSON here; must not exist')
    args=p.parse_args()
    if args.requests<1 or args.max_tokens<1 or args.timeout<1:
        p.error('Requests, max-tokens, and timeout must be positive')
    try:
        cs=[int(x.strip()) for x in args.concurrency.split(',')]
    except ValueError:
        p.error('Expected --concurrency 1,8,32')
    if any(x<1 for x in cs) or len(set(cs))!=len(cs):
        p.error('Concurrency values must be unique, positive integers')
    if args.output and args.output.exists():
        p.error('Output file already exists; refusing overwrite')
    questions=read_prompts(args.cases)
    work=[questions[i%len(questions)] for i in range(args.requests)]
    prompt=args.system_prompt.read_text(encoding='utf-8')
    if not prompt.strip():
        p.error('System prompt is empty')
    results=[]
    for c in cs:
        stats=measure(args.url,args.model,prompt,work,concurrency=c,
                      max_tokens=args.max_tokens,seed=args.seed,timeout=args.timeout)
        print('VLLM_BENCH',json.dumps(stats),flush=True)
        results.append(stats)
    summary={'model':args.model, 'endpoint':args.url,
             'cases_sha256':hashlib.sha256(args.cases.read_bytes()).hexdigest(),
             'system_prompt_sha256':hashlib.sha256(args.system_prompt.read_bytes()).hexdigest(),
             'max_tokens':args.max_tokens,'seed':args.seed,'results':results}
    if args.output:
        args.output.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('BENCHMARK COMPLETE',json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
