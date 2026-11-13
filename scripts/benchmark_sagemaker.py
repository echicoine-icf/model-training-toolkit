#!/usr/bin/env python3
"""Load-test a SageMaker real-time vLLM endpoint using frozen prompts, without scoring.

Use MTT shared semantic evaluation separately to measure accuracy. This script
measures requests handled as concurrency increases, including failures/timeouts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter


def percentile(values, p):
    if not values:
        return None
    return round(sorted(values)[max(0, math.ceil(len(values) * p / 100) - 1)], 3)


def load_cases(path):
    prompts = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        q = obj.get('question') or obj.get('prompt')
        if not isinstance(q, str) or not q.strip():
            raise ValueError('Every case must have a nonempty question or prompt')
        prompts.append(q)
    if not prompts:
        raise ValueError('No cases found')
    return prompts


def invoke(client, endpoint, model_tag, system_prompt, question, max_tokens, seed):
    body = json.dumps({
        'model': model_tag, 'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question},
        ], 'temperature': 0, 'seed': seed, 'max_tokens': max_tokens,
        'stream': False,
    }).encode('utf-8')
    start = perf_counter()
    try:
        response = client.invoke_endpoint(EndpointName=endpoint, ContentType='application/json', Body=body)
        data = json.loads(response['Body'].read())
        choice = data['choices'][0]
        return {'ok': True, 'latency': perf_counter() - start,
                'output_tokens': int((data.get('usage') or {}).get('completion_tokens') or 0),
                'finish_reason': choice.get('finish_reason')}
    except Exception as exc:
        return {'ok': False, 'latency': perf_counter() - start,
                'error': type(exc).__name__ + ': ' + str(exc)[:500]}


def measure(client, endpoint, model_tag, system_prompt, questions, concurrency, max_tokens, seed):
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(invoke, client, endpoint, model_tag, system_prompt, q, max_tokens, seed)
                   for q in questions]
        rows = [f.result() for f in as_completed(futures)]
    duration = perf_counter() - start
    successes = [x for x in rows if x['ok']]
    errors = [x['error'] for x in rows if not x['ok']]
    return {'concurrency': concurrency, 'attempted': len(rows), 'succeeded': len(successes),
            'failed': len(errors), 'errors': errors[:10], 'duration_s': round(duration, 3),
            'p50_latency_s': percentile([x['latency'] for x in successes], 50),
            'p95_latency_s': percentile([x['latency'] for x in successes], 95),
            'output_tokens': sum(x['output_tokens'] for x in successes),
            'requests_per_second': round(len(successes) / duration, 3),
            'note': 'Observed end-to-end service behavior, not a direct GPU memory measurement.'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile'); p.add_argument('--region', default='us-east-1')
    p.add_argument('--endpoint', required=True); p.add_argument('--model-tag', required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--system-prompt', type=Path, required=True)
    p.add_argument('--requests', type=int, default=32)
    p.add_argument('--concurrency', default='1,8,32')
    p.add_argument('--max-tokens', type=int, default=768)
    p.add_argument('--seed', type=int, default=8002026)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.requests < 1 or args.max_tokens < 1 or args.output.exists():
        p.error('Requests and tokens must be positive and output path must not exist')
    try:
        cs = [int(x.strip()) for x in args.concurrency.split(',')]
        if not cs or len(set(cs)) != len(cs) or any(c < 1 for c in cs):
            raise ValueError()
    except ValueError:
        p.error('Concurrency must be positive distinct integers, e.g. 1,8,32')
    questions = load_cases(args.cases)
    prompt = args.system_prompt.read_text(encoding='utf-8')
    if not prompt.strip():
        p.error('Empty system prompt')
    import boto3
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = session.client('sagemaker-runtime', config=__import__('botocore.config', fromlist=['Config']).Config(
        read_timeout=70, connect_timeout=10, max_pool_connections=max(cs) + 2, retries={'max_attempts': 0}))
    sample = [questions[i % len(questions)] for i in range(args.requests)]
    results = [measure(client, args.endpoint, args.model_tag, prompt, sample, c,
                       args.max_tokens, args.seed) for c in cs]
    report = {'endpoint': args.endpoint, 'model': args.model_tag,
              'cases_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              'system_prompt_sha256': hashlib.sha256(args.system_prompt.read_bytes()).hexdigest(),
              'max_tokens': args.max_tokens, 'seed': args.seed, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
