"""Run project-owned shared benchmarks against an existing frozen MTT run.

This is a standard-library subprocess bridge, NOT a scorer or inference engine.
The trusted project Python scripts own the benchmark format, Ollama generation,
resumable predictions, historical scoring and release gate rules.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from . import core

# v1 compatible: the ELI shared benchmark already ships these three scripts.
# Other projects can supply the same script interface without changing MTT.
AUDITOR = 'audit_shared.py'
RUNNER = 'evaluate_shared_ollama.py'
GATES = 'gate_report.py'
APPROVED_RUN_STATES = frozenset({'trained', 'evaluated', 'exported', 'evaluation_failed'})


def _script(benchmark: Path, name: str) -> Path:
    script = benchmark / name
    if script.is_symlink() or not script.is_file() or not script.resolve().is_relative_to(benchmark.resolve()):
        raise FileNotFoundError(f'Shared benchmark is missing safe script {name}: {benchmark}')
    return script


def _model_slug(model: str) -> str:
    if not model or len(model) > 200 or not re.fullmatch(r'[A-Za-z0-9._:/-]+', model):
        raise ValueError('Specify a valid Ollama model tag with --model')
    return re.sub(r'[^A-Za-z0-9._-]', '_', model)


def plan(project: str, version: str, run_id: str, benchmark_name: str,
         *, workspace: Path | None = None) -> dict:
    """Resolve benchmark + immutable snapshot; no writes or Ollama API calls."""
    root = Path(workspace) if workspace is not None else core.ROOT
    project = core.validated_name(project)
    version = core.validated_name(version)
    run_id = core.validated_name(run_id)
    benchmark_name = core.validated_name(benchmark_name)
    project_dir = root / 'projects' / project
    if not (project_dir / 'project.json').is_file():
        raise FileNotFoundError('Unknown MTT project: ' + project)
    run = root / 'runs' / project / version / run_id
    if not (run / 'run.json').is_file():
        raise FileNotFoundError('Unknown MTT training run: ' + str(run))
    meta = core.read(run / 'run.json')
    if meta.get('project') != project or meta.get('version') != version or meta.get('run_id') != run_id:
        raise ValueError('Run identity does not match the requested project/version/run')
    if meta.get('run_kind', 'training') != 'training' or meta.get('status') not in APPROVED_RUN_STATES:
        raise RuntimeError('Shared evaluation requires a completed training run; active training is not eligible')
    snapshot = run / 'input'
    core.assert_snapshot(snapshot)
    prompt = snapshot / 'data' / 'system_prompt.txt'
    training = snapshot / 'data' / 'train.jsonl'
    if not prompt.is_file() or not training.is_file():
        raise FileNotFoundError('Run snapshot is missing system_prompt.txt or train.jsonl')
    eval_root = project_dir / 'evaluations'
    benchmark = eval_root / benchmark_name
    if eval_root.is_symlink() or benchmark.is_symlink() or not benchmark.is_dir() or not benchmark.resolve().is_relative_to(eval_root.resolve()):
        raise FileNotFoundError('Missing or unsafe shared benchmark folder: ' + str(benchmark))
    manifest = benchmark / 'benchmark_manifest.json'
    if not manifest.is_file() or manifest.is_symlink():
        raise FileNotFoundError('Shared benchmark manifest missing: ' + str(manifest))
    data = core.read(manifest)
    if not isinstance(data, dict) or not isinstance(data.get('benchmark_id'), str) or not data['benchmark_id']:
        raise ValueError('Shared benchmark requires a nonempty benchmark_id in benchmark_manifest.json')
    if not isinstance(data.get('package_version'),str) or not data['package_version']:
        raise ValueError('Shared benchmark requires an explicit package_version')
    suites=data.get('suites')
    hashes=data.get('frozen_files_sha256')
    if not isinstance(suites,list) or not isinstance(hashes,dict):
        raise ValueError('Shared benchmark must declare suites and frozen_files_sha256')
    names = set()
    for suite in [*suites, *data.get('targeted_suites', [])]:
        name = suite.get('name') if isinstance(suite,dict) else None
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Duplicate or invalid shared benchmark suite name')
        names.add(name)
        dataset=suite.get('dataset') if isinstance(suite,dict) else None
        if not isinstance(dataset,str) or dataset not in hashes:
            raise ValueError('Shared benchmark suite is missing a dataset SHA-256')
        file=benchmark/dataset
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(benchmark.resolve()):
            raise FileNotFoundError('Unsafe or missing shared benchmark dataset: '+str(dataset))
        if core.digest(file)!=hashes[dataset]:
            raise ValueError('Shared benchmark dataset checksum mismatch: '+str(dataset))
    # Scripts execute trusted local project code. Inspect/review before installation.
    scripts = {'auditor': _script(benchmark, AUDITOR), 'runner': _script(benchmark, RUNNER),
               'gates': _script(benchmark, GATES)}
    return {'benchmark': benchmark, 'benchmark_id': data['benchmark_id'],
            'package_version': data['package_version'], 'run': run,
            'prompt': prompt, 'train': training, 'scripts': scripts}


def execute(project: str, version: str, run_id: str, benchmark_name: str, *,
            model: str | None = None, stage: str | None = None,
            suites: list[str] | None = None, list_suites: bool = False,
            resume: bool = False, dry_run: bool = False, smoke: int = 0,
            audit_only: bool = False, gate_only: bool = False,
            reference_results: str | None = None,
            url: str | None = None, num_ctx: int | None = None,
            num_predict: int | None = None, seed: int | None = None,
            timeout: int | None = None,
            workspace: Path | None = None) -> Path | None:
    """Audit protected holdouts, then delegate to project-owned benchmark scripts.

    All generated predictions live under runs/<project>/<version>/<run>/
    shared_evaluations/<benchmark-folder>/<model-tag>/, not in a Git-tracked
    benchmark folder or the existing V9.2 baseline output.
    """
    if audit_only and (gate_only or model or resume or smoke or reference_results or dry_run or suites or stage or list_suites):
        raise ValueError('--audit-only cannot be combined with execution or gate flags')
    if gate_only and (dry_run or resume or smoke or stage or suites or list_suites):
        raise ValueError('--gate-only cannot be combined with evaluation flags')
    if reference_results and not gate_only:
        raise ValueError('--reference-results requires --gate-only')
    if stage and suites:
        raise ValueError('--stage and --suite are mutually exclusive; choose one selection mode')
    if list_suites and (model or stage or suites or resume or smoke or dry_run or reference_results or url or any(v is not None for v in (num_ctx,num_predict,seed,timeout))):
        raise ValueError('--list-suites is read-only; use without evaluation/model options')
    if not audit_only and not list_suites and not model:
        raise ValueError('--model is required except for --audit-only')
    if smoke < 0 or smoke > 25:
        raise ValueError('--smoke must be between 0 and 25')
    if smoke and resume:
        raise ValueError('--smoke and --resume cannot be combined')
    opts = {'num_ctx': num_ctx, 'num_predict': num_predict, 'seed': seed, 'timeout': timeout}
    if any(v is not None and (type(v) is not int or v <= 0) for v in opts.values()):
        raise ValueError('Inference numeric flags must be positive integers')
    spec = plan(project, version, run_id, benchmark_name, workspace=workspace)
    benchmark = spec['benchmark']
    # Always use the *frozen training snapshot* for overlap checking.
    audit_cmd = [sys.executable, str(spec['scripts']['auditor']), '--train', str(spec['train'])]
    subprocess.run(audit_cmd, cwd=str(benchmark), check=True)
    if audit_only:
        print('SHARED BENCHMARK AUDIT COMPLETE:', spec['benchmark_id'], flush=True)
        return None
    if list_suites:
        import json
        manifest=json.loads((benchmark/'benchmark_manifest.json').read_text(encoding='utf-8'))
        print('SHARED BENCHMARK',spec['benchmark_id'],'PACKAGE',spec['package_version'])
        for suite in manifest['suites']:
            print('CANONICAL',suite['name'], 'stage='+str(suite.get('stage','?')), 'count='+str(suite.get('count','?')), 'mode='+str(suite.get('mode','?')))
        for suite in manifest.get('targeted_suites',[]):
            print('TARGETED',suite['name'], 'source='+suite['source_filter'], 'count='+str(suite['count']), 'mode='+suite['mode'])
        print('Select with --suite NAME or --suite NAME1,NAME2; use --stage all for all canonical suites')
        return None

    output = spec['run'] / 'shared_evaluations' / benchmark_name / _model_slug(model)
    if gate_only:
        if not (output / 'run_config.json').is_file():
            raise FileNotFoundError('Shared evaluation results are not present: ' + str(output))
        cmd = [sys.executable, str(spec['scripts']['gates']), '--results', str(output)]
        if reference_results:
            cmd += ['--reference', str(Path(reference_results).expanduser().resolve())]
        subprocess.run(cmd, cwd=str(benchmark), check=True)
        return output

    cmd = [sys.executable, str(spec['scripts']['runner']), '--model', model,
           '--prompt', str(spec['prompt']), '--out', str(output)]
    if stage:
        cmd += ['--stage', stage]
    if suites:
        for suite in suites:
            cmd += ['--suite', suite]
    if resume:
        cmd.append('--resume')
    if dry_run:
        cmd.append('--dry-run')
    if smoke:
        cmd += ['--smoke', str(smoke)]
    if url:
        cmd += ['--url', url]
    for key, value in opts.items():
        if value is not None:
            cmd += ['--' + key.replace('_', '-'), str(value)]
    print('MTT SHARED BENCHMARK', spec['benchmark_id'],
          'PACKAGE_VERSION',spec['package_version'],'RUN', run_id,
          'OUTPUT', output, flush=True)
    subprocess.run(cmd, cwd=str(benchmark), check=True)
    return output
