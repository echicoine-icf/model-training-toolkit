"""Project-supplied evaluation runners for MTT (mtt-eval-runners-v1).

An evaluation runner is *trusted local Python code*. MTT owns model loading,
inference, checkpoint provenance, per-case durability and resume; the runner
scores immutable predictions. This is NOT a sandbox for untrusted scripts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import shutil
import tempfile
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .core import digest, read, utc, write
from .data_package import contract
from .eval_engine import normalize

FORMAT = 'mtt-eval-runners-v1'
NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,63}$')
MODEL_TYPES = ('adapter', 'merged', 'checkpoint')


def _within(root: Path, relative: str, *, suffix: str | None = None) -> Path:
    """Restrict evaluated files to a frozen package (no symlink traversal)."""
    if not isinstance(relative, str) or not relative or ':' in relative or '\\' in relative:
        raise ValueError(f'Invalid relative evaluation file path: {relative!r}')
    path = Path(relative)
    if path.is_absolute() or any(p in ('.', '..') for p in path.parts):
        raise ValueError(f'Invalid relative evaluation file path: {relative!r}')
    base = root.resolve()
    file = root / path
    if any(p.is_symlink() for p in [file, *list(file.parents)] if p == root or root in p.parents):
        raise ValueError(f'Symlinks are forbidden in evaluation files: {relative}')
    if not file.is_file() or not file.resolve().is_relative_to(base):
        raise ValueError(f'Missing or outside-package evaluation file: {relative}')
    if suffix and file.suffix.lower() != suffix:
        raise ValueError(f'{relative}: expected {suffix} file')
    return file


def registry_path(snapshot: Path) -> Path:
    return Path(snapshot) / 'data/evaluation/runners.json'


def registered(snapshot: Path) -> bool:
    return registry_path(snapshot).is_file()


def freeze_evaluation_package(run_dir: Path, source: Path) -> Path:
    """Immutably capture an external evaluator without altering a training snapshot.

    This enables adding custom evaluation to an already-trained or currently
    training version *after* training completes. Only eval files are copied.
    """
    source = Path(source).expanduser().resolve()
    if not (source / 'evaluation/runners.json').is_file():
        raise ValueError('Evaluation package must contain evaluation/runners.json')
    files = sorted(p for p in source.rglob('*') if p.is_file())
    if not files:
        raise ValueError('Empty external evaluation package')
    for file in files:
        if any(part.is_symlink() for part in [file, *list(file.parents)] if part == source or source in part.parents):
            raise ValueError('Symlinks are not permitted inside evaluation package')
    hashes = _files_hash(source, files)
    hash_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:20]
    target = Path(run_dir) / 'evaluation_packages' / hash_id
    if target.exists():
        for name, expected in hashes.items():
            if not (target / name).is_file() or digest(target / name) != expected:
                raise RuntimeError('Previously frozen evaluation package changed: ' + str(target))
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.eval_package_incomplete_', dir=target.parent))
    try:
        for name in hashes:
            dest = staging / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, dest)
            if digest(dest) != hashes[name]:
                raise RuntimeError('Evaluation package file changed during copy: '+name)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _file_in_sources(data_root: Path, package_root: Path | None, relative: str, suffix: str = '.jsonl') -> Path:
    """Core training inputs always come from the run's frozen data snapshot."""
    if relative in ('train.jsonl', 'eval.jsonl', 'system_prompt.txt', 'schema.json'):
        return _within(data_root, relative, suffix=suffix)
    if package_root is not None and (package_root / relative).is_file():
        return _within(package_root, relative, suffix=suffix)
    return _within(data_root, relative, suffix=suffix)


def load_registry(snapshot: Path, evaluation_package: Path | None = None) -> dict | None:
    """Parse registry without loading a GPU or running project Python."""
    snapshot = Path(snapshot)
    package_root = Path(evaluation_package) if evaluation_package else None
    cfg_path = package_root / 'evaluation/runners.json' if package_root is not None else registry_path(snapshot)
    if not cfg_path.is_file():
        return None
    data = snapshot / 'data'
    cfg = read(cfg_path)
    if cfg.get('format') != FORMAT:
        raise ValueError(f'Expected {FORMAT} in evaluation/runners.json')
    gen = cfg.get('generation', {})
    if not isinstance(gen, dict):
        raise ValueError('generation must be an object')
    max_tokens = gen.get('max_new_tokens', contract(snapshot)['evaluation']['max_new_tokens'])
    if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
        raise ValueError('generation.max_new_tokens must be an integer from 1..32768')
    precision = gen.get('precision', 'bf16')
    if precision not in ('bf16', 'fp16', 'fp32'):
        raise ValueError('Unsupported evaluation precision')
    runners = {}
    expected_hashes = cfg.get('data_sha256', {})
    if not isinstance(expected_hashes, dict):
        raise ValueError('data_sha256 must be a JSON object')
    for relative, expected in expected_hashes.items():
        source = _file_in_sources(data, package_root, relative, suffix=None)
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected) or digest(source) != expected.lower():
            raise ValueError(f'Evaluation input hash mismatch: {relative}')
    for runner in cfg.get('runners', []):
        if not isinstance(runner, dict):
            raise ValueError('Each runner must be a JSON object')
        name = runner.get('name')
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in runners:
            raise ValueError(f'Invalid or duplicate runner name: {name!r}')
        if not isinstance(runner.get('description'), str) or not runner['description'].strip():
            raise ValueError(f'{name}: description required')
        script = _within(package_root or data, runner.get('script'), suffix='.py')
        if not script.resolve().is_relative_to(((package_root or data) / 'evaluation').resolve()):
            raise ValueError(f'{name}: scripts must reside under data/evaluation/')
        supported = runner.get('model_inputs', [])
        if not isinstance(supported, list) or not supported or any(t not in MODEL_TYPES for t in supported):
            raise ValueError(f'{name}: model_inputs must contain adapter, merged or checkpoint')
        stages = runner.get('stages', [])
        if not isinstance(stages, list) or not stages or any(not isinstance(n, str) or not NAME.fullmatch(n) for n in stages):
            raise ValueError(f'{name}: stages must be a nonempty list of names')
        requirements = runner.get('datasets', [])
        if not isinstance(requirements, list):
            raise ValueError(f'{name}: datasets must be a list')
        for item in requirements:
            _file_in_sources(data, package_root, item, suffix='.jsonl')
        output = runner.get('output_dir', name)
        if not isinstance(output, str) or not NAME.fullmatch(output):
            raise ValueError(f'{name}: output_dir must be a simple directory name')
        runners[name] = dict(runner, script_path=script, output_dir=output)
    if not runners:
        raise ValueError('No evaluation runners registered')
    stages = []
    seen = set()
    for stage in cfg.get('stages', []):
        name = stage.get('name') if isinstance(stage, dict) else None
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in seen:
            raise ValueError(f'Invalid or duplicate evaluation stage: {name!r}')
        seen.add(name)
        suites = []
        for suite in stage.get('suites', []):
            name_s = suite.get('name') if isinstance(suite, dict) else None
            if not isinstance(name_s, str) or not NAME.fullmatch(name_s) or any(s['name'] == name_s for s in suites):
                raise ValueError(f'Invalid or repeated suite name in stage {name}: {name_s!r}')
            runner_name = suite.get('runner')
            if runner_name not in runners or name not in runners[runner_name]['stages']:
                raise ValueError(f'{name_s}: runner missing or does not support stage {name}')
            mode = suite.get('mode')
            if mode not in ('gold', 'diagnostic'):
                raise ValueError(f'{name_s}: mode must be gold or diagnostic')
            source = _file_in_sources(data, package_root, suite.get('dataset'), suffix='.jsonl')
            cases = read_cases(source, mode)
            expected_count = suite.get('expected_count')
            if expected_count is not None and (type(expected_count) is not int or expected_count != len(cases)):
                raise ValueError(f'{name_s}: expected_count {expected_count}, found {len(cases)}')
            suites.append(dict(suite, source=source, cases=cases))
        if not suites:
            raise ValueError(f'{name}: each stage requires at least one suite')
        stages.append({'name': name, 'suites': suites})
    if not stages:
        raise ValueError('No evaluation stages registered')
    return {'runners': runners, 'stages': stages, 'precision': precision,
            'max_new_tokens': max_tokens, 'path': cfg_path, 'package_root': package_root}


def read_cases(path: Path, mode: str) -> list[dict]:
    cases = []
    ids = set()
    with Path(path).open(encoding='utf-8-sig') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as ex:
                raise ValueError(f'{path}:{line_num}: invalid JSON') from ex
            if not isinstance(row, dict) or not isinstance(row.get('question'), str) or not row['question'].strip():
                raise ValueError(f'{path}:{line_num}: nonempty question required')
            rid = str(row.get('id', ''))
            if not rid or rid in ids:
                raise ValueError(f'{path}:{line_num}: missing/duplicate ID {rid!r}')
            if mode == 'gold' and 'expected' not in row:
                raise ValueError(f'{path}:{line_num}: expected required for labeled suite')
            ids.add(rid)
            cases.append(row)
    if not cases:
        raise ValueError(f'Empty evaluation dataset: {path}')
    return cases


def _files_hash(root: Path, candidates):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(candidates)}


def resolve_model(run_dir: Path, snapshot: Path, model: str = 'adapter', checkpoint: str = 'latest', model_path: str | None = None):
    """Only constructs a validated read-only model specification. No merging."""
    run_dir = Path(run_dir)
    if model not in MODEL_TYPES:
        if model_path:
            raise ValueError('--model /path and --model-path are mutually exclusive')
        model_path = model
        model = 'auto'
    if model_path:
        path = Path(model_path).expanduser().resolve()
    elif model == 'adapter':
        path = run_dir / 'adapter'
    elif model == 'checkpoint':
        from .checkpoints import select_checkpoint
        path = select_checkpoint(run_dir, checkpoint)
    elif model == 'merged':
        raise ValueError('Merged model requires --model-path /path/to/merged-model')
    else:
        raise ValueError('A model path is required')
    if not path.is_dir():
        raise ValueError(f'Cannot locate model directory: {path}')
    if model == 'auto':
        if (path / 'checkpoint_info.json').is_file():
            model = 'checkpoint'
        elif (path / 'adapter_config.json').is_file():
            model = 'adapter'
        elif (path / 'config.json').is_file():
            model = 'merged'
        else:
            raise ValueError('Unrecognized model directory (expected adapter_config.json or config.json)')
    if model == 'checkpoint':
        from .checkpoints import select_checkpoint
        if path.parent.name != 'checkpoints':
            raise ValueError('Checkpoints must be located in the selected run/checkpoints/')
        verified = select_checkpoint(run_dir, path.name)
        if verified.resolve() != path.resolve():
            raise ValueError('Requested checkpoint is not from the selected run')
        details = {'checkpoint_info_sha256': digest(path / 'checkpoint_info.json')}
    elif model == 'adapter':
        if not (path / 'adapter_config.json').is_file():
            raise ValueError('Missing adapter_config.json')
        weights = list(path.glob('*.safetensors'))
        if not weights:
            raise ValueError('Missing adapter safetensors')
        details = _files_hash(path, [path / 'adapter_config.json', *weights])
    else:
        if not (path / 'config.json').is_file():
            raise ValueError('Missing merged config.json')
        weights = list(path.glob('*.safetensors'))
        if not weights:
            raise ValueError('Missing merged model safetensors')
        details = _files_hash(path, [path / 'config.json', *weights])
    base = read(Path(snapshot) / 'version.json')['base_model']
    return {'type': model, 'path': str(path.resolve()), 'base': base, 'sha256': details}


def _fingerprint(snapshot: Path, spec: dict, model_spec: dict):
    hashes = read(Path(snapshot) / 'input_hashes.json')
    assets = _files_hash(spec['package_root'], [p for p in spec['package_root'].rglob('*') if p.is_file()]) if spec['package_root'] else None
    data = {'format': FORMAT, 'inputs': hashes, 'evaluation_package': assets, 'model': model_spec,
            'generation': {'max_new_tokens': spec['max_new_tokens'], 'precision': spec['precision']}}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(), data


def load_predictor(model_spec: dict, precision: str):
    """Common model loader for adapter, merged HF model, or saved PEFT checkpoint."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[precision]
    if dtype != torch.float32 and not torch.cuda.is_available():
        raise RuntimeError('BF16/FP16 evaluation needs CUDA; do not silently fall back to CPU')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = model_spec['path']
    base = model_spec['base']
    if model_spec['type'] == 'merged':
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, trust_remote_code=False)
    else:
        tokenizer_from = path if (Path(path) / 'tokenizer_config.json').is_file() else base['id']
        tokenizer_kwargs = {} if tokenizer_from == path else {'revision': base.get('revision') or 'main'}
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_from, trust_remote_code=False, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(base['id'], revision=base.get('revision') or 'main',
                                                      torch_dtype=dtype, trust_remote_code=False)
        model = PeftModel.from_pretrained(model, path, is_trainable=False)
    model = model.to(device).eval()
    def predict(question, system, max_new_tokens):
        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}]
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_tensors='pt').to(device)
        with torch.inference_mode():
            answer = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(answer[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return predict


@contextmanager
def evaluation_lock(out: Path):
    """Prevent concurrent evaluation writers; use OS-released locks."""
    from .checkpoints import training_lock
    with training_lock(out):
        yield


def _existing_rows(file: Path, cases: list, fingerprint: str):
    results = []
    if not file.exists():
        return results
    raw = file.read_bytes()
    if raw and not raw.endswith(b'\n'):
        end = raw.rfind(b'\n') + 1
        tail = raw[end:]
        try:
            json.loads(tail.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            # An abrupt shutdown during a single row should not cost the
            # preceding completed predictions. Preserve the damaged tail.
            backup = file.with_name(file.name+'.incomplete_tail')
            if backup.exists():
                raise RuntimeError('Existing incomplete-tail backup; investigate before resume')
            backup.write_bytes(tail)
            with file.open('r+b') as f:
                f.truncate(end)
        else:
            with file.open('ab') as f:
                f.write(b'\n')
    with file.open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except ValueError as e:
                raise RuntimeError(f'Corrupt partial prediction {file}:{n}; refusing unsafe resume') from e
            idx = len(results)
            if idx >= len(cases) or str(row.get('id')) != str(cases[idx]['id']) or \
                    row.get('question') != cases[idx]['question'] or row.get('evaluation_fingerprint') != fingerprint:
                raise RuntimeError(f'Prediction order, case or model mismatch at {file}:{n}')
            results.append(row)
    return results


def _prompts(snapshot: Path):
    pkg = contract(snapshot)
    root = Path(snapshot) / 'data'
    system = (root / 'system_prompt.txt').read_text(encoding='utf-8').rstrip('\r\n')
    if pkg['prompt_assembly'] == 'append_schema':
        system += '\n\nSchema:\n' + (root / 'schema.json').read_text(encoding='utf-8')
    return system


def _execute_script(runner, suite, predictions, output, context):
    command = [sys.executable, str(runner['script_path']), '--cases', str(suite['source']),
               '--predictions', str(predictions), '--output', str(output), '--context', str(context)]
    p = subprocess.run(command, cwd=str(runner['script_path'].parent), check=False)
    if p.returncode:
        raise RuntimeError(f'Custom runner {runner["name"]} failed with exit code {p.returncode}')
    for key in ('summary.json', 'failures.jsonl', 'report.md'):
        if not (output / key).is_file():
            raise RuntimeError(f'Runner {runner["name"]} must write {key}')
    score = read(output / 'summary.json')
    if not isinstance(score, dict):
        raise RuntimeError('Runner summary.json must be a JSON object')
    return score


def run(snapshot: Path, out: Path, adapter: Path | None = None, *, model_spec: dict | None = None,
        runner_name: str | None = None, suite: str = 'all', resume: bool = False,
        dry_run: bool = False, predictor=None, evaluation_package: Path | None = None):
    """Perform all stages through selected stage, preserving previous predictions.

    Inject predictor(question, system, max_new_tokens) for testability. Live
    evaluation uses the shared read-only model loader above.
    """
    snapshot = Path(snapshot)
    out = Path(out)
    spec = load_registry(snapshot, evaluation_package=evaluation_package)
    if spec is None:
        raise ValueError('No project-supplied evaluation/runners.json in frozen data')
    if model_spec is None:
        if adapter is None:
            raise ValueError('Custom evaluation requires model_spec or adapter')
        model_spec = resolve_model(out.parent, snapshot, str(adapter))
    if runner_name and runner_name not in spec['runners']:
        raise ValueError(f'Unknown runner: {runner_name}')
    stage_names = [s['name'] for s in spec['stages']]
    if suite != 'all' and suite not in stage_names:
        raise ValueError(f'Unknown stage: {suite}; choose {stage_names} or all')
    selected = spec['stages'] if suite == 'all' else spec['stages'][:stage_names.index(suite)+1]
    chosen = []
    for stage in selected:
        selected_suites = [s for s in stage['suites'] if runner_name is None or s['runner'] == runner_name]
        if selected_suites:
            chosen.append({'name': stage['name'], 'suites': selected_suites})
    if not chosen:
        raise ValueError('No suites match requested runner and stages')
    for stage in chosen:
        for s in stage['suites']:
            r = spec['runners'][s['runner']]
            if model_spec['type'] not in r['model_inputs']:
                raise ValueError(f'Runner {s["runner"]} does not support {model_spec["type"]} models')
    fingerprint, identity = _fingerprint(snapshot, spec, model_spec)
    root = out / fingerprint[:20]
    if dry_run:
        print('EVALUATION PLAN', json.dumps({'model': model_spec, 'stages': [
            {'name': s['name'], 'suites': [{'name': a['name'], 'count': len(a['cases']), 'mode': a['mode']} for a in s['suites']]}
            for s in chosen], 'output': str(root)}, indent=2), flush=True)
        return {'dry_run': True, 'output': str(root)}
    if root.exists() and not resume:
        raise FileExistsError(f'Evaluation already exists: {root}. Use --resume to continue; never overwrites.')
    with evaluation_lock(root):
        manifest = root / 'evaluation_manifest.json'
        if manifest.exists():
            old = read(manifest)
            if old.get('fingerprint') != fingerprint or old.get('identity') != identity:
                raise RuntimeError('Evaluation fingerprint changed; refuse mixed model/dataset/scorer predictions')
        else:
            write(manifest, {'format': FORMAT, 'fingerprint': fingerprint, 'identity': identity,
                             'created_utc': utc()})
        prompt = _prompts(snapshot)
        predict = predictor  # defer model loading until a suite needs inference
        summary = {}
        for stage in chosen:
            per_suite = {}
            for s in stage['suites']:
                r = spec['runners'][s['runner']]
                target = root / r['output_dir'] / stage['name'] / s['name']
                target.mkdir(parents=True, exist_ok=True)
                outfile = target / 'predictions.jsonl'
                completed = _existing_rows(outfile, s['cases'], fingerprint)
                if len(completed) < len(s['cases']):
                    if predict is None:
                        predict = load_predictor(model_spec, spec['precision'])
                    with outfile.open('a', encoding='utf-8') as handle:
                        for i, case in enumerate(s['cases'][len(completed):], len(completed)+1):
                            begin = time.monotonic()
                            raw = predict(case['question'], prompt, spec['max_new_tokens'])
                            try:
                                parsed = json.loads(raw)
                                valid_json = True
                            except (ValueError, TypeError):
                                parsed, valid_json = None, False
                            match = normalize(parsed if valid_json else raw) == normalize(case['expected']) if s['mode'] == 'gold' else None
                            result = {'id': str(case['id']), 'question': case['question'],
                                      'expected': case.get('expected'), 'prediction': raw, 'parsed': parsed,
                                      'json_valid': valid_json, 'exact_match': match,
                                      'evaluation_fingerprint': fingerprint,
                                      'elapsed_seconds': round(time.monotonic()-begin, 3)}
                            handle.write(json.dumps(result, ensure_ascii=False)+'\n')
                            handle.flush()
                            completed.append(result)
                            if i % 25 == 0 or i == len(s['cases']):
                                print(f'EVALUATION {stage["name"]}/{s["name"]}: {i}/{len(s["cases"])}', flush=True)
                scorefile = target / 'summary.json'
                marker = target / 'score_receipt.json'
                if scorefile.exists() and not marker.exists():
                    # A scorer can crash after writing summary.json. Back up its
                    # partial output and score the already-completed predictions
                    # again. Never repeat inference or silently discard artifacts.
                    for artifact_name in ('summary.json', 'failures.jsonl', 'report.md'):
                        artifact = target / artifact_name
                        if artifact.exists():
                            backup=target / (artifact_name+'.incomplete')
                            if backup.exists():
                                raise RuntimeError('An incomplete scorer backup already exists; inspect it before resuming')
                            artifact.rename(backup)
                if scorefile.exists():
                    if not marker.exists():
                        raise RuntimeError(f'Incomplete scorer receipt for {scorefile}')
                    receipt=read(marker)
                    if receipt.get('predictions_sha256') != digest(outfile) or \
                            receipt.get('scorer_sha256') != digest(r['script_path']):
                        raise RuntimeError('Scored predictions/scorer changed after completion; refusing reuse')
                    for required in ('failures.jsonl','report.md'):
                        if not (target/required).is_file():
                            raise RuntimeError(f'Incomplete scored output: missing {required}')
                    score = read(scorefile)
                else:
                    context = target / 'runner_context.json'
                    write(context, {'format': FORMAT, 'fingerprint': fingerprint,
                                    'stage': stage['name'], 'suite': s['name'], 'mode': s['mode'],
                                    'runner': {k: v for k, v in r.items() if k not in ('script_path',)},
                                    'model': model_spec, 'data_root': str(snapshot / 'data'),
                                    'evaluation_package_root': str(spec['package_root']) if spec['package_root'] else None,
                                    'case_count': len(s['cases'])})
                    before = digest(outfile)
                    score = _execute_script(r, s, outfile, target, context)
                    if digest(outfile) != before:
                        raise RuntimeError('Evaluator script altered immutable MTT predictions')
                    write(marker,{'predictions_sha256':before,'scorer_sha256':digest(r['script_path']),
                                  'completed_utc':utc()})
                per_suite[s['name']] = score
            summary[stage['name']] = per_suite
        aggregate = {'format': FORMAT, 'evaluation_fingerprint': fingerprint,
                     'stages': summary, 'evaluated_utc': utc(),
                     'note': 'Gold accuracy and unlabeled diagnostic statistics must never be combined.'}
        write(root / 'summary.json', aggregate)
        from .core import assert_snapshot
        assert_snapshot(snapshot)
        # Accidental modifications by trusted scorers are detected (not sandboxed).
        if model_spec['path'] != 'fake-model':
            # Do not re-open checkpoint optimizer state via pickle.
            source = Path(model_spec['path'])
            for relative, expected in model_spec['sha256'].items():
                if relative == 'checkpoint_info_sha256':
                    file = source / 'checkpoint_info.json'
                else:
                    file = source / relative
                if not file.is_file() or digest(file) != expected:
                    raise RuntimeError(f'Model artifact changed during evaluation: {file}')
        return aggregate
