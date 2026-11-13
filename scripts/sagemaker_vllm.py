#!/usr/bin/env python3
"""Package MTT SafeTensors and plan/provision a persistent SageMaker vLLM endpoint.

Standard library for package and plan; boto3 is needed only for --apply operations.
This does not modify the original training run or quantized checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

FORMATS = {'hf-f16': 'merged_hf', 'vllm-fp8': 'vllm-fp8',
           'vllm-int8': 'vllm-int8', 'vllm-w4a16': 'vllm-w4a16'}
AWS_NAME = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,62}$')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def files_in(folder):
    root = folder.resolve()
    found = []
    for f in sorted(folder.rglob('*')):
        if f.is_symlink():
            raise ValueError(f'Cannot package symlink: {f}')
        if f.is_file():
            if not f.resolve().is_relative_to(root):
                raise ValueError(f'Unsafe model file: {f}')
            found.append(f)
    return found


def validate_checkpoint(folder, require_receipt=False):
    if not folder.is_dir() or not (folder / 'config.json').is_file():
        raise FileNotFoundError(f'Missing Hugging Face model/config.json: {folder}')
    if not list(folder.glob('*.safetensors')):
        raise FileNotFoundError(f'Missing model SafeTensors: {folder}')
    manifest = {f.relative_to(folder).as_posix():
                {'sha256': digest(f), 'bytes': f.stat().st_size}
                for f in files_in(folder) if f.name != 'MTT_QUANTIZATION.json'}
    if require_receipt:
        receipt = folder / 'MTT_QUANTIZATION.json'
        if not receipt.is_file():
            raise ValueError(f'Quantized export has no MTT receipt: {receipt}')
        record = json.loads(receipt.read_text(encoding='utf-8'))
        if record.get('output_integrity') != manifest:
            raise ValueError('Quantized checkpoint differs from its original MTT receipt')
        if record.get('format') not in ('vllm-fp8', 'vllm-int8', 'vllm-w4a16'):
            raise ValueError('Not a recognized vLLM quantization receipt')
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        if not config.get('quantization_config'):
            raise ValueError('Checkpoint config lacks quantization_config')
    return manifest


def model_path(root, project, version, run, fmt):
    for label, value in [('project', project), ('version', version), ('run', run)]:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
            raise ValueError(f'Invalid {label}: {value!r}')
    if fmt not in FORMATS:
        raise ValueError(f'Invalid format: {fmt}')
    path = root / 'runs' / project / version / run / 'exports' / FORMATS[fmt]
    validate_checkpoint(path, require_receipt=(fmt != 'hf-f16'))
    return path


def package_model(root, project, version, run, fmt, output):
    source = model_path(Path(root).resolve(), project, version, run, fmt)
    output = Path(output).expanduser().resolve()
    if output.exists() or output.with_name(output.name + '.receipt.json').exists():
        raise FileExistsError(f'Package or receipt already exists: {output}')
    if output.suffixes[-2:] != ['.tar', '.gz']:
        raise ValueError('SageMaker model package must end in .tar.gz')
    if source in output.parents:
        raise ValueError('Cannot create a package inside the source model folder')
    before = validate_checkpoint(source, require_receipt=fmt != 'hf-f16')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + '.partial')
    if tmp.exists():
        raise FileExistsError(f'Partial package exists: {tmp}')
    try:
        with tarfile.open(tmp, 'w:gz') as archive:
            for f in files_in(source):
                # The original integrity record lives in the bundle's external receipt,
                # not in the model payload loaded by the serving container.
                if f.name == 'MTT_QUANTIZATION.json':
                    continue
                archive.add(f, arcname=f.relative_to(source).as_posix(), recursive=False)
        if before != validate_checkpoint(source, require_receipt=fmt != 'hf-f16'):
            raise RuntimeError('Model files changed during packaging')
        artifact = {'project': project, 'version': version, 'run': run,
                    'format': fmt, 'source': str(source),
                    'model_files': before, 'artifact_sha256': digest(tmp),
                    'artifact_bytes': tmp.stat().st_size,
                    'created_utc': datetime.now(timezone.utc).isoformat()}
        tmp.rename(output)
        receipt = output.with_name(output.name + '.receipt.json')
        receipt.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')
    except BaseException:
        # Leave .partial for inspection if packaging failed.
        raise
    return {'package': str(output), 'receipt': str(receipt), **artifact}


def s3_parts(uri):
    p = urlsplit(uri)
    if p.scheme != 's3' or not p.netloc or not p.path.lstrip('/') or p.query or p.fragment:
        raise ValueError('Expected s3://bucket/key/model.tar.gz')
    return p.netloc, p.path.lstrip('/')


def get_boto3(profile, region):
    import boto3
    session = boto3.Session(profile_name=profile, region_name=region)
    if not session.region_name:
        raise ValueError('Specify --region or configure an AWS default region')
    return session


def publish_package(path, destination, profile, region, apply=False):
    path = Path(path).expanduser().resolve()
    receipt = path.with_name(path.name + '.receipt.json')
    if not path.is_file() or not receipt.is_file():
        raise FileNotFoundError('Model tar.gz and its .receipt.json must exist')
    record = json.loads(receipt.read_text(encoding='utf-8'))
    if digest(path) != record.get('artifact_sha256'):
        raise ValueError('Model package differs from its SHA-256 receipt')
    bucket, key = s3_parts(destination)
    plan = {'action': 'upload', 'package': str(path), 's3_uri': destination,
            'sha256': record['artifact_sha256'], 'apply': apply}
    if apply:
        session = get_boto3(profile, region)
        # Existing S3 keys must not be silently replaced. HeadObject requires
        # s3:GetObject; any ambiguous permissions fail closed.
        client = session.client('s3')
        try:
            client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            code = getattr(exc, 'response', {}).get('Error', {}).get('Code', '')
            if code not in ('404', 'NoSuchKey', 'NotFound'):
                raise
        else:
            raise FileExistsError(f'S3 object exists; refusing overwrite: {destination}')
        client.upload_file(str(path), bucket, key, ExtraArgs={'Metadata': {'sha256': record['artifact_sha256']}})
        print('UPLOADED', destination)
    return plan


def endpoint_plan(args):
    if not AWS_NAME.fullmatch(args.endpoint):
        raise ValueError('Invalid SageMaker endpoint name')
    if not args.image_uri or args.image_uri.endswith(':latest'):
        raise ValueError('Specify an explicit pinned SageMaker vLLM DLC --image-uri (no latest tag)')
    if not args.role.startswith('arn:aws:iam::') or ':role/' not in args.role:
        raise ValueError('Provide a SageMaker execution role ARN')
    s3_parts(args.model_s3_uri)
    if not args.instance_type.startswith('ml.g'):
        raise ValueError('Select a SageMaker GPU instance type such as ml.g6.xlarge')
    if not (0.1 <= args.gpu_memory_utilization <= 0.95):
        raise ValueError('GPU memory utilization must be between 0.1 and 0.95')
    if args.max_model_len < 512 or args.max_num_seqs < 1 or args.instance_count < 1:
        raise ValueError('Model length, maximum sequences, and instance count must be positive')
    if args.kv_cache_dtype not in ('auto', 'fp8'):
        raise ValueError('KV cache dtype must be auto or fp8')
    suffix = hashlib.sha256((args.model_s3_uri + args.image_uri + args.model_tag).encode()).hexdigest()[:9]
    stem = args.endpoint[:45].rstrip('-')
    model = f'{stem}-model-{suffix}'
    config = f'{stem}-cfg-{suffix}'
    env = {
        'SM_VLLM_SERVED_MODEL_NAME': args.model_tag,
        'SM_VLLM_MAX_MODEL_LEN': str(args.max_model_len),
        'SM_VLLM_MAX_NUM_SEQS': str(args.max_num_seqs),
        'SM_VLLM_GPU_MEMORY_UTILIZATION': str(args.gpu_memory_utilization),
        'SM_VLLM_KV_CACHE_DTYPE': args.kv_cache_dtype,
        'SM_VLLM_DTYPE': 'auto',
    }
    model_spec = {'ModelName': model,
                  'PrimaryContainer': {'Image': args.image_uri,
                                       'ModelDataUrl': args.model_s3_uri,
                                       'Environment': env},
                  'ExecutionRoleArn': args.role}
    config_spec = {'EndpointConfigName': config,
                   'ProductionVariants': [{
                       'VariantName': 'AllTraffic', 'ModelName': model,
                       'InstanceType': args.instance_type,
                       'InitialInstanceCount': args.instance_count,
                       'InferenceAmiVersion': args.inference_ami_version,
                       'VolumeSizeInGB': args.volume_size,
                   }]}
    create_spec = {'EndpointName': args.endpoint, 'EndpointConfigName': config}
    return {'model': model_spec, 'endpoint_config': config_spec, 'endpoint': create_spec,
            'notes': ['Endpoint stays provisioned and billed while InService.',
                      'This creates one or more always-on instances; autoscaling is not configured.',
                      'Prefer an IAM-restricted invocation path and validate accuracy before production traffic.']}


def deploy(args):
    plan = endpoint_plan(args)
    if not args.apply:
        return plan
    session = get_boto3(args.profile, args.region)
    client = session.client('sagemaker')
    # A remote mutation only occurs behind explicit --apply. If an API call
    # fails, print names so resources can be inspected/cleaned up manually.
    try:
        client.create_model(**plan['model'])
        print('CREATED MODEL', plan['model']['ModelName'], flush=True)
        client.create_endpoint_config(**plan['endpoint_config'])
        print('CREATED CONFIG', plan['endpoint_config']['EndpointConfigName'], flush=True)
        client.create_endpoint(**plan['endpoint'])
        print('CREATED ENDPOINT', args.endpoint, '(provisioning; charges begin once resources run)', flush=True)
    except BaseException:
        print('DEPLOYMENT INTERRUPTED; inspect model/config/endpoint resources:',
              plan['model']['ModelName'], plan['endpoint_config']['EndpointConfigName'],
              args.endpoint, file=sys.stderr)
        raise
    return {'created': True, 'endpoint': args.endpoint, 'model': plan['model']['ModelName'],
            'endpoint_config': plan['endpoint_config']['EndpointConfigName']}


def autoscaling_plan(endpoint, min_instances, max_instances, target_concurrency):
    if not AWS_NAME.fullmatch(endpoint):
        raise ValueError('Invalid SageMaker endpoint name')
    if not (1 <= min_instances <= max_instances) or not (0 < target_concurrency <= 1024):
        raise ValueError('Always-on minimum must be >=1; maximum >= minimum; target >0')
    target = {
        'ServiceNamespace': 'sagemaker',
        'ResourceId': f'endpoint/{endpoint}/variant/AllTraffic',
        'ScalableDimension': 'sagemaker:variant:DesiredInstanceCount',
        'MinCapacity': min_instances, 'MaxCapacity': max_instances,
    }
    policy = {
        'PolicyName': f'{endpoint[:40]}-concurrency',
        'PolicyType': 'TargetTrackingScaling',
        'ServiceNamespace': target['ServiceNamespace'],
        'ResourceId': target['ResourceId'],
        'ScalableDimension': target['ScalableDimension'],
        'TargetTrackingScalingPolicyConfiguration': {
            'TargetValue': target_concurrency,
            'ScaleOutCooldown': 60, 'ScaleInCooldown': 300,
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantConcurrentRequestsPerModelHighResolution'
            },
        },
    }
    return {'target': target, 'policy': policy,
            'notes': ['Apply only after endpoint reaches InService.',
                      'Minimum instance count never reaches zero.',
                      'Tune concurrency target after endpoint load tests.']}


def autoscale(args):
    plan = autoscaling_plan(args.endpoint, args.min_instances,
                            args.max_instances, args.target_concurrency)
    if args.apply:
        client = get_boto3(args.profile, args.region).client('application-autoscaling')
        client.register_scalable_target(**plan['target'])
        client.put_scaling_policy(**plan['policy'])
        print('AUTOSCALING CONFIGURED', args.endpoint, flush=True)
    return plan


def endpoint_status(endpoint, profile, region):
    if not AWS_NAME.fullmatch(endpoint):
        raise ValueError('Invalid endpoint name')
    resp = get_boto3(profile, region).client('sagemaker').describe_endpoint(EndpointName=endpoint)
    return {k: resp.get(k) for k in ('EndpointName', 'EndpointStatus',
            'EndpointConfigName', 'FailureReason', 'ProductionVariants') if k in resp}


def smoke_invoke(args):
    if not args.message.strip() or not args.system_prompt.strip():
        raise ValueError('System prompt and message must not be empty')
    if args.max_tokens < 1:
        raise ValueError('Maximum tokens must be positive')
    payload = {'model': args.model_tag,
               'messages': [{'role': 'system', 'content': args.system_prompt},
                            {'role': 'user', 'content': args.message}],
               'max_tokens': args.max_tokens,
               'temperature': 0, 'seed': args.seed, 'stream': False}
    if not args.apply:
        return {'endpoint': args.endpoint, 'request': payload, 'apply': False}
    session = get_boto3(args.profile, args.region)
    response = session.client('sagemaker-runtime').invoke_endpoint(
        EndpointName=args.endpoint, ContentType='application/json',
        Body=json.dumps(payload).encode('utf-8'))
    return json.loads(response['Body'].read())


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest='action', required=True)
    pkg = cmd.add_parser('package', help='Offline: build a tar.gz from a verified MTT export')
    pkg.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    pkg.add_argument('--project', default='eli'); pkg.add_argument('--version', required=True)
    pkg.add_argument('--run', required=True); pkg.add_argument('--format', choices=FORMATS, required=True)
    pkg.add_argument('--output', required=True, type=Path)
    upload = cmd.add_parser('upload', help='Check a model package then upload to S3 only with --apply')
    upload.add_argument('--package', required=True, type=Path)
    upload.add_argument('--s3-uri', required=True)
    deployment = cmd.add_parser('deploy', help='Plan SageMaker endpoint; provision only with --apply')
    deployment.add_argument('--model-s3-uri', required=True)
    deployment.add_argument('--image-uri', required=True,
                            help='Pinned SageMaker vLLM DLC URI for chosen AWS region')
    deployment.add_argument('--role', required=True, help='SageMaker execution role ARN')
    deployment.add_argument('--endpoint', required=True)
    deployment.add_argument('--model-tag', default='datmm-eli')
    deployment.add_argument('--instance-type', default='ml.g6.xlarge')
    deployment.add_argument('--instance-count', type=int, default=1)
    deployment.add_argument('--volume-size', type=int, default=30)
    deployment.add_argument('--inference-ami-version', default='al2-ami-sagemaker-inference-gpu-3-1')
    deployment.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    deployment.add_argument('--max-model-len', type=int, default=4096)
    deployment.add_argument('--max-num-seqs', type=int, default=64)
    deployment.add_argument('--kv-cache-dtype', choices=['auto', 'fp8'], default='auto')
    scaling = cmd.add_parser('autoscale', help='Keep >=1 replica alive; scale out by concurrent requests')
    scaling.add_argument('--endpoint', required=True)
    scaling.add_argument('--min-instances', type=int, default=1)
    scaling.add_argument('--max-instances', type=int, default=3)
    scaling.add_argument('--target-concurrency', type=float, default=8.0)
    info = cmd.add_parser('status', help='Read-only SageMaker endpoint status')
    info.add_argument('--endpoint', required=True)
    invoke = cmd.add_parser('invoke', help='Deterministic OpenAI chat completion smoke test')
    invoke.add_argument('--endpoint', required=True)
    invoke.add_argument('--model-tag', default='datmm-eli')
    invoke.add_argument('--system-prompt', required=True, help='The same system prompt used in ELI evaluation')
    invoke.add_argument('--message', required=True)
    invoke.add_argument('--max-tokens', type=int, default=768)
    invoke.add_argument('--seed', type=int, default=8002026)
    for c in [upload, deployment, scaling, info, invoke]:
        c.add_argument('--profile', help='AWS CLI profile; omit to use standard credentials chain')
        c.add_argument('--region', default='us-east-1')
        if c is not info:
            c.add_argument('--apply', action='store_true', help='Execute remote AWS operations')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.action == 'package':
        info = package_model(args.root, args.project, args.version, args.run, args.format, args.output)
    elif args.action == 'upload':
        info = publish_package(args.package, args.s3_uri, args.profile, args.region, args.apply)
    elif args.action == 'deploy':
        info = deploy(args)
    elif args.action == 'autoscale':
        info = autoscale(args)
    elif args.action == 'status':
        info = endpoint_status(args.endpoint, args.profile, args.region)
    else:
        info = smoke_invoke(args)
    print(json.dumps(info, indent=2, default=str))
    return info


if __name__ == '__main__':
    main()
