"""Non-destructive post-training export: Q8_0 GGUF and vLLM compressed checkpoints.

This module deliberately uses only the Python standard library. ML quantization
runs in an explicitly selected *separate* Python environment.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FORMATS = ('gguf-q8_0', 'vllm-fp8', 'vllm-int8')

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def _file_manifest(folder: Path) -> dict:
    return {
        p.relative_to(folder).as_posix(): {'sha256': _sha256(p), 'bytes': p.stat().st_size}
        for p in sorted(folder.rglob('*')) if p.is_file()
    }

def _available_quantizers(root: Path):
    base = root / 'tools' / 'llama.cpp' / 'build' / 'bin'
    for rel in ('llama-quantize.exe', 'Release/llama-quantize.exe',
                'llama-quantize', 'Release/llama-quantize'):
        yield base / rel
    base = root / 'tools' / 'llama.cpp'
    for rel in ('build/Release/bin/llama-quantize.exe', 'llama-quantize.exe'):
        yield base / rel

def _quantizer_path(root: Path, custom: str | None) -> Path:
    if custom:
        p = Path(custom).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f'Quantizer not found: {p}')
        return p
    for p in _available_quantizers(root):
        if p.is_file():
            return p.resolve()
    found = shutil.which('llama-quantize') or shutil.which('llama-quantize.exe')
    if found:
        return Path(found).resolve()
    raise FileNotFoundError('llama-quantize executable not found. Build llama.cpp quantize tool or pass --quantizer PATH.')

def quantize(project: str, version: str, run: str, *, fmt: str, root: Path,
             quantizer: str | None = None, quant_python: str | None = None,
             dry_run: bool = False, ollama_model: str | None = None,
             num_ctx: int = 4096, runner=subprocess.run) -> dict:
    """Export separate quantized artifacts; never alter original run/model files.

    runner is injectable to unit-test subprocess orchestration without GPU/tools.
    """
    from toolkit.core import validated_name
    project, version, run = (validated_name(x) for x in (project, version, run))
    if fmt not in FORMATS:
        raise ValueError(f'Unsupported format {fmt!r}; choose {FORMATS}')
    root = Path(root).resolve()
    folder = root / 'runs' / project / version / run
    meta_file = folder / 'run.json'
    if not meta_file.is_file():
        raise FileNotFoundError(f'Run not found: {meta_file}')
    meta = json.loads(meta_file.read_text(encoding='utf-8'))
    if meta.get('status') not in ('trained', 'evaluated', 'exported', 'evaluation_failed'):
        raise RuntimeError('Training must complete before quantization.')
    exports = folder / 'exports'
    is_gguf = fmt == 'gguf-q8_0'
    source = exports / ('model-f16.gguf' if is_gguf else 'merged_hf')
    dest = exports / ('model-q8_0.gguf' if is_gguf else ('vllm-fp8' if fmt=='vllm-fp8' else 'vllm-int8'))
    if not source.exists() or (source.is_file() if not is_gguf else not source.is_file()):
        raise FileNotFoundError(f'Source artifact missing: {source}. First export the completed model.')
    if not is_gguf and not (source/'config.json').is_file():
        raise FileNotFoundError(f'Missing Hugging Face config.json in {source}')
    if not is_gguf and not list(source.glob('*.safetensors')):
        raise FileNotFoundError(f'Missing merged Safetensors files in {source}')
    if dest.exists():
        raise FileExistsError(f'Quantized export already exists; refusing overwrite: {dest}')
    if is_gguf:
        engine = _quantizer_path(root, quantizer)
        source_digest = _sha256(source)
        cmd = [str(engine), str(source), str(dest), 'Q8_0']
    else:
        if not quant_python:
            raise ValueError('FP8/INT8 export requires --quant-python pointing to a separate environment with llmcompressor. Do not install it in MTT .venv.')
        python = Path(quant_python).expanduser().resolve()
        if not python.is_file():
            raise FileNotFoundError(f'Quantization Python not found: {python}')
        if python == Path(sys.executable).resolve():
            raise ValueError('--quant-python must be a separate environment, not the running MTT .venv.')
        script = root / 'scripts' / 'quantize_vllm.py'
        if not script.is_file():
            raise FileNotFoundError(f'Missing quantization script: {script}')
        source_digest = _file_manifest(source)
        engine = python
        cmd = [str(python), str(script), '--source', str(source), '--output', str(dest),
               '--scheme', ('FP8_DYNAMIC' if fmt=='vllm-fp8' else 'W8A16')]
    if ollama_model and not is_gguf:
        raise ValueError('--ollama-model is available only for gguf-q8_0')
    if ollama_model and (not ollama_model.strip() or any(c.isspace() for c in ollama_model)):
        raise ValueError('Invalid Ollama model tag')
    if ollama_model and (exports/'Modelfile.q8_0').exists():
        raise FileExistsError('Modelfile.q8_0 already exists; refusing overwrite')
    if num_ctx < 512:
        raise ValueError('--num-ctx must be at least 512')
    result = {
        'project': project, 'version': version, 'run': run, 'format': fmt,
        'input': str(source), 'output': str(dest), 'source_integrity': source_digest,
        'command': cmd, 'ollama_model': ollama_model,
    }
    if dry_run:
        print('QUANTIZATION PLAN', json.dumps(result, indent=2))
        return result
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A failed export must not be mistaken for a completed artifact.
    tmp = dest.with_name(dest.name + '.partial')
    if tmp.exists():
        raise FileExistsError(f'Partial export exists, inspect before retrying: {tmp}')
    work_cmd = [str(tmp) if part == str(dest) else part for part in cmd]
    try:
        runner(work_cmd, check=True)
        if is_gguf:
            if not tmp.is_file() or tmp.stat().st_size == 0:
                raise RuntimeError('Quantizer returned without producing a nonempty GGUF.')
            result['output_integrity'] = {'sha256': _sha256(tmp), 'bytes': tmp.stat().st_size}
            tmp.rename(dest)
            sidecar = dest.with_suffix('.mtt-quantization.json')
        else:
            if not (tmp/'config.json').is_file() or not list(tmp.glob('*.safetensors')):
                raise RuntimeError('Quantizer returned without config.json and compressed .safetensors.')
            config = json.loads((tmp/'config.json').read_text(encoding='utf-8'))
            if not config.get('quantization_config'):
                raise RuntimeError('Quantized checkpoint lacks quantization_config. Refusing to present it as quantized.')
            result['output_integrity'] = _file_manifest(tmp)
            tmp.rename(dest)
            sidecar = dest/'MTT_QUANTIZATION.json'
        result['completed_utc'] = _utc()
        # Sidecar is intentionally not part of output_integrity (non-circular).
        sidecar.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    except BaseException:
        # Preserve partial files for inspection, and never overwrite a good artifact.
        raise
    if ollama_model:
        modelfile = exports / 'Modelfile.q8_0'
        if modelfile.exists():
            raise FileExistsError(f'Modelfile already exists: {modelfile}')
        modelfile.write_text(f'FROM ./{dest.name}\nPARAMETER temperature 0\nPARAMETER num_ctx {num_ctx}\n', encoding='utf-8')
        runner(['ollama', 'create', ollama_model, '-f', str(modelfile)], check=True)
        result['ollama_created'] = ollama_model
        # Update receipt with registration evidence only on success.
        sidecar.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('QUANTIZED', json.dumps({k:v for k,v in result.items() if k != 'source_integrity'}, indent=2))
    return result
