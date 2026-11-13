"""Non-destructive post-training export: GGUF Q8_0/Q4_K_M and vLLM compressed checkpoints.

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

GGUF_QUANT_TYPES = {'gguf-q8_0': 'Q8_0', 'gguf-q4_k_m': 'Q4_K_M'}
FORMATS = (*GGUF_QUANT_TYPES, 'vllm-fp8', 'vllm-int8')

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
    """Common llama.cpp build locations on Linux, Windows/MSVC and Ninja."""
    base = root / 'tools' / 'llama.cpp'
    for build in (base / 'build' / 'mtt-quantize', base / 'build'):
        for rel in ('bin/Release/llama-quantize.exe', 'bin/llama-quantize.exe',
                    'bin/llama-quantize', 'bin/Release/llama-quantize',
                    'Release/bin/llama-quantize.exe', 'Release/llama-quantize.exe',
                    'llama-quantize.exe', 'llama-quantize'):
            yield build / rel
    yield base / 'llama-quantize.exe'


def _find_quantizer(root: Path, custom: str | None) -> Path | None:
    if custom:
        p = Path(custom).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f'Explicit --quantizer not found: {p}')
        return p
    for p in _available_quantizers(root):
        if p.is_file():
            return p.resolve()
    found = shutil.which('llama-quantize') or shutil.which('llama-quantize.exe')
    return Path(found).resolve() if found else None


def _quantizer_build_commands(root: Path) -> list[list[str]]:
    source = root / 'tools' / 'llama.cpp'
    if not (source / 'CMakeLists.txt').is_file():
        raise FileNotFoundError(
            f'llama.cpp CMake source missing: {source}. '
            'Restore the existing llama.cpp checkout or pass --quantizer PATH.')
    build = source / 'build' / 'mtt-quantize'
    return [
        ['cmake', '-S', str(source), '-B', str(build),
         '-DGGML_CUDA=OFF', '-DLLAMA_BUILD_TESTS=OFF',
         '-DCMAKE_BUILD_TYPE=Release'],
        ['cmake', '--build', str(build), '--config', 'Release',
         '--target', 'llama-quantize', '--parallel', '8'],
    ]


def _build_quantizer(root: Path, commands: list[list[str]], runner) -> Path:
    if not shutil.which('cmake'):
        raise RuntimeError(
            'llama-quantize is missing and CMake is not on PATH. '
            'Install CMake and a C++ toolchain (on Windows: Visual Studio '
            'Build Tools with Desktop development with C++), then retry; '
            'or provide --quantizer PATH. No model files were changed.')
    print('QUANTIZER NOT FOUND: building the CPU-only llama-quantize utility automatically.', flush=True)
    for cmd in commands:
        print('QUANTIZER BUILD:', ' '.join(cmd), flush=True)
        try:
            runner(cmd, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                'Automatic llama-quantize build failed. Check the CMake error above '
                'and ensure a C++ compiler is installed and available (Windows: '
                'Visual Studio Build Tools / Developer Command Prompt). '
                'You can also pass --quantizer PATH or '
                '--no-auto-build-quantizer. No model files were changed.') from exc
    result = _find_quantizer(root, None)
    if result is None:
        raise RuntimeError(
            'CMake reported success but llama-quantize executable was not found. '
            f'Inspect {root / "tools" / "llama.cpp" / "build" / "mtt-quantize"} '
            'or pass --quantizer PATH. No model files were changed.')
    print('QUANTIZER READY:', result, flush=True)
    return result

def quantize(project: str, version: str, run: str, *, fmt: str, root: Path,
             quantizer: str | None = None, quant_python: str | None = None,
             dry_run: bool = False, ollama_model: str | None = None,
             num_ctx: int = 4096, auto_build_quantizer: bool = True,
             runner=subprocess.run) -> dict:
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
    is_gguf = fmt in GGUF_QUANT_TYPES
    gguf_suffix = GGUF_QUANT_TYPES[fmt].lower() if is_gguf else None
    source = exports / ('model-f16.gguf' if is_gguf else 'merged_hf')
    dest = exports / (f'model-{gguf_suffix}.gguf' if is_gguf else ('vllm-fp8' if fmt=='vllm-fp8' else 'vllm-int8'))
    if not source.exists() or (source.is_file() if not is_gguf else not source.is_file()):
        raise FileNotFoundError(f'Source artifact missing: {source}. First export the completed model.')
    if not is_gguf and not (source/'config.json').is_file():
        raise FileNotFoundError(f'Missing Hugging Face config.json in {source}')
    if not is_gguf and not list(source.glob('*.safetensors')):
        raise FileNotFoundError(f'Missing merged Safetensors files in {source}')
    if dest.exists():
        raise FileExistsError(f'Quantized export already exists; refusing overwrite: {dest}')
    build_commands = []
    if is_gguf:
        engine = _find_quantizer(root, quantizer)
        if engine is None:
            if not auto_build_quantizer:
                raise FileNotFoundError(
                    'llama-quantize executable not found. Auto-build is disabled; '
                    'build it manually or pass --quantizer PATH.')
            build_commands = _quantizer_build_commands(root)
        source_digest = _sha256(source)
        # A dry run must plan missing builds without executing CMake.
        tool_name = str(engine) if engine else '<llama-quantize: resolved after automatic CMake build>'
        cmd = [tool_name, str(source), str(dest), GGUF_QUANT_TYPES[fmt]]
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
        raise ValueError('--ollama-model is available only for GGUF quantization')
    if ollama_model and (not ollama_model.strip() or any(c.isspace() for c in ollama_model)):
        raise ValueError('Invalid Ollama model tag')
    if ollama_model and (exports/f'Modelfile.{gguf_suffix}').exists():
        raise FileExistsError(f'Modelfile.{gguf_suffix} already exists; refusing overwrite')
    if num_ctx < 512:
        raise ValueError('--num-ctx must be at least 512')
    result = {
        'project': project, 'version': version, 'run': run, 'format': fmt,
        'input': str(source), 'output': str(dest), 'source_integrity': source_digest,
        'command': cmd, 'ollama_model': ollama_model,
    }
    if build_commands:
        result['quantizer_auto_build_commands'] = build_commands
    if dry_run:
        print('QUANTIZATION PLAN', json.dumps(result, indent=2))
        return result
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A failed export must not be mistaken for a completed artifact.
    tmp = dest.with_name(dest.name + '.partial')
    if tmp.exists():
        raise FileExistsError(f'Partial export exists, inspect before retrying: {tmp}')
    if build_commands:
        engine = _build_quantizer(root, build_commands, runner)
        cmd[0] = str(engine)
        result['command'] = cmd
        result['quantizer_binary_sha256'] = _sha256(engine)
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
        modelfile = exports / f'Modelfile.{gguf_suffix}'
        if modelfile.exists():
            raise FileExistsError(f'Modelfile already exists: {modelfile}')
        modelfile.write_text(f'FROM ./{dest.name}\nPARAMETER temperature 0\nPARAMETER num_ctx {num_ctx}\n', encoding='utf-8')
        runner(['ollama', 'create', ollama_model, '-f', str(modelfile)], check=True)
        result['ollama_created'] = ollama_model
        # Update receipt with registration evidence only on success.
        sidecar.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('QUANTIZED', json.dumps({k:v for k,v in result.items() if k != 'source_integrity'}, indent=2))
    return result
