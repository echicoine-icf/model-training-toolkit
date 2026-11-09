#!/usr/bin/env python3
"""Export reproducible MTT project sources + frozen inputs to an independent Git repo.

Never mutates the source checkout's Git config, .gitignore, index, branches,
workspace state, models, datasets, checkpoints, or run results. Git clone/commit
happens only in a temporary destination checkout. Python standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
BLOCKED_COMPONENTS = {
    '.git', '.venv', '__pycache__', '.workspace_state_backups',
    'ollama_results', 'evaluations_custom', 'shared_evaluations',
    'checkpoints', 'adapter', 'merged_hf', 'cache', 'node_modules',
    '.aws', '.ssh', '.idea', '.vscode', 'logs',
}
BLOCKED_SUFFIXES = {
    '.safetensors', '.gguf', '.ckpt', '.pth', '.pt', '.onnx', '.bin',
    '.pem', '.p12', '.pfx', '.key', '.pyc', '.exe', '.dll', '.so', '.dylib',
}
PRIVATE_NAMES = {
    '.env', '.env.local', 'credentials', 'credentials.json',
    'secrets.json', 'secret.json', 'id_rsa', 'id_ed25519',
    'workspace_state.json', '.running.lock',
}
LARGE_FILE_MAX = 90 * 1024 * 1024  # Safely below GitHub's hard 100-MiB-ish object limit.
SCAN_LIMIT = 4 * 1024 * 1024
EXTERNAL_EVAL_EXTENSIONS = {'.py', '.json', '.jsonl', '.md', '.txt', '.yaml', '.yml', '.toml', '.sh', '.csv'}
SECRET_PATTERNS = [
    re.compile(rb'AKIA[0-9A-Z]{16}'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s*(?:[A-Za-z0-9+/=]{40,}\s*){2,}'),
    re.compile(rb'gh[pousr]_[A-Za-z0-9_]{20,}'),
    re.compile(rb'github_pat_[A-Za-z0-9_]{20,}'),
]


def cmd(*args: str, cwd: Path | None = None, capture: bool = False, check: bool = True):
    return subprocess.run([str(x) for x in args], cwd=cwd, check=check,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def output(*args: str, cwd: Path | None = None) -> str:
    return cmd(*args, cwd=cwd, capture=True).stdout.decode('utf-8').strip()


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def safe_name(value: str, label: str, *, run: bool = False):
    if not (RUN_NAME if run else SAFE_NAME).fullmatch(value) or '..' in value:
        raise ValueError(f'Unsafe {label}: {value!r}')
    return value


def acceptable(path: Path, *, relative: Path) -> bool:
    rel = relative
    if any(part in BLOCKED_COMPONENTS for part in rel.parts):
        return False
    if any(part.lower() in PRIVATE_NAMES for part in rel.parts):
        return False
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    if path.name.lower().startswith('.env'):
        return False
    if path.is_symlink():
        raise ValueError(f'Symlinks not permitted in snapshot: {path}')
    if not path.is_file():
        return False
    return True


def validate_payload(src: Path, *, label: str):
    n = src.stat().st_size
    if n > LARGE_FILE_MAX:
        raise RuntimeError(f'File too large for normal Git ({n:,} bytes): {label}; use Git LFS or external storage')
    if n <= SCAN_LIMIT and src.suffix.lower() not in {'.zip', '.png', '.jpg', '.jpeg', '.webp'}:
        data = src.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                raise RuntimeError(f'Potential credential in {label}; review before publishing')


def git_tracked(root: Path):
    # Validate source is a real Git checkout and the commit exists.
    cmd('git', 'rev-parse', '--verify', 'HEAD', cwd=root, capture=True)
    raw = cmd('git', 'ls-files', '-z', cwd=root, capture=True).stdout
    return [Path(os.fsdecode(p)) for p in raw.split(b'\0') if p]


def collect(root: Path, project: str, version: str, run_id: str,
            benchmarks: list[str], external_evals: list[Path]) -> tuple[dict[str, Path], str]:
    src_commit = output('git', 'rev-parse', 'HEAD', cwd=root)
    dirty = cmd('git', 'status', '--porcelain', '--untracked-files=no', cwd=root, capture=True).stdout
    if dirty.strip():
        raise RuntimeError('Tracked MTT files have local modifications. Commit or stash first; refusing ambiguous toolkit snapshot.')
    tracked = git_tracked(root)
    files: dict[str, Path] = {}

    def add(path: Path, dest: Path):
        if path.is_relative_to(root):
            for component in [path, *path.parents]:
                if component == root:
                    break
                if component.is_symlink():
                    raise ValueError(f'Symlinks not permitted in snapshot: {path}')
        if not acceptable(path, relative=dest):
            return
        key = dest.as_posix()
        if key in files and files[key] != path:
            raise RuntimeError(f'Snapshot destination collision: {key}')
        validate_payload(path, label=key)
        files[key] = path

    project_prefix = Path('projects') / project
    ver_prefix = project_prefix / 'versions' / version
    for rel in tracked:
        if rel.parts[0] in {'runs', 'exports', 'registry'}:
            continue
        if rel.parts[0] == 'projects':
            if not rel.is_relative_to(project_prefix):
                continue
            if 'versions' in rel.parts:
                if not rel.is_relative_to(ver_prefix):
                    continue
            if 'evaluations' in rel.parts:
                # Only explicitly selected benchmark versions are published.
                if not any(rel.is_relative_to(project_prefix / 'evaluations' / bench) for bench in benchmarks):
                    continue
        add(root / rel, rel)

    # Include project launcher/configuration even when newly created and not yet
    # tracked upstream. Never capture unrelated projects or other model versions.
    project_dir = root / project_prefix
    if not project_dir.is_dir():
        raise FileNotFoundError(f'Missing project: {project_dir}')
    for path in sorted(project_dir.iterdir()):
        if path.is_file():
            add(path, path.relative_to(root))

    # Include ignored training data and any untracked data-package files explicitly.
    v = root / ver_prefix
    required = [v / 'version.json', v / 'data' / 'train.jsonl']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Missing essential training input: {path}')
    for path in sorted(v.rglob('*')):
        if path.is_file():
            add(path, path.relative_to(root))

    # Immutable benchmark files: include even if not Git-tracked in operational checkout.
    if not benchmarks:
        raise ValueError('Pass at least one --benchmark; evaluation assets must be explicitly selected')
    for bench in benchmarks:
        directory = root / project_prefix / 'evaluations' / bench
        if not (directory / 'benchmark_manifest.json').is_file():
            raise FileNotFoundError(f'Missing benchmark or manifest: {directory}')
        for p in sorted(directory.rglob('*')):
            if p.is_file():
                add(p, p.relative_to(root))

    # Capture the precise frozen training recipe, not only the mutable version defaults.
    r = root / 'runs' / project / version / run_id
    frozen = r / 'input'
    hashes_path = frozen / 'input_hashes.json'
    if not hashes_path.is_file():
        raise FileNotFoundError(f'Missing frozen run input hashes: {hashes_path}')
    hashes = json.loads(hashes_path.read_text(encoding='utf-8'))
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Run input hashes are empty/invalid')
    for rel, expected in hashes.items():
        relpath = Path(rel)
        if relpath.is_absolute() or '..' in relpath.parts:
            raise ValueError('Unsafe run input hashes entry')
        path = frozen / relpath
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f'Frozen input checksum mismatch: {path}')
    recipe = Path('rebuild_recipes') / project / version / run_id
    for p in sorted(frozen.rglob('*')):
        if p.is_file():
            add(p, recipe / 'input' / p.relative_to(frozen))
    for name in ('run.json', 'training_plan.json'):
        path = r / name
        if path.is_file():
            add(path, recipe / name)

    for ext in external_evals:
        ext = ext.expanduser().resolve()
        if not ext.is_dir():
            raise FileNotFoundError(f'Missing external evaluation directory: {ext}')
        safe_name(ext.name, 'external evaluator directory')
        prefix = project_prefix / 'evaluations' / 'external' / ext.name
        for p in sorted(ext.rglob('*')):
            if p.is_file():
                # External directories can include Ollama results and caches: never export them.
                rel = p.relative_to(ext)
                if (any(part in BLOCKED_COMPONENTS for part in rel.parts)
                    or p.suffix.lower() not in EXTERNAL_EVAL_EXTENSIONS):
                    continue
                if p.is_symlink():
                    raise ValueError(f'Symlink in external evaluation directory: {p}')
                if any(part.lower() in PRIVATE_NAMES for part in rel.parts) or any(part.lower().startswith('.env') for part in rel.parts):
                    continue
                add_external = prefix / rel
                validate_payload(p, label=add_external.as_posix())
                files[add_external.as_posix()] = p

    return files, src_commit


def environment_fingerprint():
    packages = ('torch', 'torchvision', 'transformers', 'peft', 'accelerate',
                'safetensors', 'unsloth', 'tokenizers', 'datasets', 'numpy',
                'huggingface-hub', 'triton', 'trl')
    installed = {}
    for pkg in packages:
        try:
            installed[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        'python': platform.python_version(),
        'os': platform.system(),
        'os_release': platform.release(),
        'installed_ml_packages': installed,
    }


def manifest_for(files: dict[str, Path], *, project: str, version: str,
                 run_id: str, source_commit: str, benchmarks: list[str]) -> dict:
    return {
        'format': 'mtt-project-source-snapshot-v1',
        'project': project, 'version': version, 'run_id': run_id,
        'source_toolkit_commit': source_commit,
        'benchmarks': benchmarks,
        'model_weights_included': False,
        'environment': environment_fingerprint(),
        'note': 'Frozen input and training recipe preserved. Retraining may not reproduce identical floating-point weights or eval scores.',
        'files': {name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                  for name, p in sorted(files.items())},
    }


def snapshot_to_checkout(files: dict[str, Path], manifest: dict, dest: Path,
                         *, project: str, version: str, run_id: str):
    # Never mutate an already-published model version or benchmark. Git history is
    # preserved, but a silent overwrite would destroy the meaning of a version tag.
    immutable = [f'projects/{project}/versions/{version}/',
                 f'projects/{project}/evaluations/benchmark-',
                 f'rebuild_recipes/{project}/{version}/{run_id}/']
    changed_immutable = []
    for name, src in files.items():
        dst = dest / name
        if dst.exists() and digest(dst) != digest(src) and any(name.startswith(x) for x in immutable):
            changed_immutable.append(name)
    if changed_immutable:
        raise RuntimeError('Published version/benchmark input differs; do not silently overwrite immutable files: '\
                           + ', '.join(changed_immutable[:8]))

    for name, src in files.items():
        dst = dest / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or digest(dst) != digest(src):
            shutil.copyfile(src, dst)

    manifest_rel = Path('snapshots') / project / version / run_id / 'manifest.json'
    target = dest / manifest_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'
    if target.exists() and target.read_text(encoding='utf-8') != payload:
        raise RuntimeError(f'Snapshot receipt already exists and differs: {manifest_rel}')
    target.write_text(payload, encoding='utf-8')
    paths = sorted([*files, manifest_rel.as_posix()])
    for start in range(0, len(paths), 70):
        cmd('git', 'add', '-f', '--', *paths[start:start+70], cwd=dest)
    return manifest_rel


def validate_url(url: str, *, allow_local: bool = False) -> str:
    if allow_local and (url.startswith('file://') or Path(url).exists()):
        return url
    if re.fullmatch(r'git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git', url):
        return url
    parsed = urlparse(url)
    if (parsed.scheme != 'https' or parsed.hostname != 'github.com' or parsed.username
        or parsed.password or parsed.query or parsed.fragment
        or not re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?', parsed.path)):
        raise ValueError('Use an HTTPS GitHub repository URL without embedded credentials, or git@github.com:OWNER/REPO.git')
    return url


def run(args, *, root: Path = ROOT):
    project = safe_name(args.project, 'project')
    version = safe_name(args.version, 'version')
    run_id = safe_name(args.run, 'run ID', run=True)
    benchmarks = [safe_name(b, 'benchmark') for b in args.benchmark]
    url = validate_url(args.repo_url, allow_local=getattr(args, 'allow_local_remote', False))
    files, source_commit = collect(root, project, version, run_id, benchmarks, args.external_eval)
    manifest = manifest_for(files, project=project, version=version,
                            run_id=run_id, source_commit=source_commit, benchmarks=benchmarks)
    nbytes = sum(v['bytes'] for v in manifest['files'].values())
    print(f'SOURCE: {root}', flush=True)
    print(f'DESTINATION: {url}', flush=True)
    print(f'SNAPSHOT: {project}/{version}/{run_id}', flush=True)
    print(f'SOURCE COMMIT: {source_commit}', flush=True)
    print(f'BENCHMARKS: {", ".join(benchmarks)}', flush=True)
    print(f'FILES: {len(files)}, TOTAL: {nbytes / 1024**2:.1f} MiB', flush=True)
    print('EXCLUDED: model weights, GGUF, checkpoints, caches, .venv, run output, predictions, credentials', flush=True)
    print('WARNING: Training corpora and benchmark cases will be committed. Check GitHub repository visibility.', flush=True)
    if not args.push:
        print('DRY RUN ONLY: no clone, commits, or pushes. Use --push --visibility private|public to publish.')
        return
    if args.visibility is None:
        raise ValueError('--push requires --visibility private|public (confirm the repository settings in GitHub)')
    if args.visibility == 'public' and not args.allow_public:
        raise ValueError('Refusing public push without --allow-public. Review training/evaluation data licensing and sensitivity.')
    with tempfile.TemporaryDirectory(prefix='mtt-publish-') as td:
        staging = Path(td) / 'destination'
        cmd('git', 'clone', '--', url, str(staging))
        branch = output('git', 'branch', '--show-current', cwd=staging)
        if branch and branch != 'main':
            raise RuntimeError(f'Destination default branch is {branch!r}, expected main; refusing to change branches')
        if not branch:
            cmd('git', 'switch', '-c', 'main', cwd=staging)
        # Honor the source checkout's local/global author configuration without
        # modifying the user's global Git settings.
        for field in ('user.name', 'user.email'):
            val = output('git', 'config', '--get', field, cwd=root)
            if not val:
                raise RuntimeError(f'Configure git {field} in the originating checkout first')
            cmd('git', 'config', '--local', field, val, cwd=staging)
        receipt = snapshot_to_checkout(files, manifest, staging,
                                       project=project, version=version, run_id=run_id)
        change = cmd('git', 'diff', '--cached', '--quiet', cwd=staging, check=False)
        if change.returncode == 0:
            print('NO CHANGES: destination already has this exact snapshot.')
            return
        if change.returncode != 1:
            raise RuntimeError('Git index inspection failed')
        cmd('git', 'commit', '-m', f'Snapshot {project} {version} run {run_id}', cwd=staging)
        cmd('git', 'push', 'origin', 'HEAD:refs/heads/main', cwd=staging)
        print(f'PUBLISHED: {url}  receipt={receipt.as_posix()}')
    print('SOURCE UNCHANGED: Git origin, branch, index, and ignore rules were not touched.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo-url', required=True, help='Existing destination GitHub repository HTTPS/SSH URL')
    p.add_argument('--project', required=True, help='MTT project (e.g. eli)')
    p.add_argument('--version', required=True, help='Version to preserve (e.g. v9.2)')
    p.add_argument('--run', required=True, help='Completed run ID used to freeze training inputs')
    p.add_argument('--benchmark', action='append', required=True, help='Project benchmark directory name; repeatable')
    p.add_argument('--external-eval', action='append', type=Path, default=[],
                   help='Optional extra evaluation code/data directory; excludes results and caches')
    p.add_argument('--push', action='store_true', help='Clone destination, commit and push main (no force)')
    p.add_argument('--visibility', choices=['private', 'public'], help='Confirm GitHub repository visibility before publishing')
    p.add_argument('--allow-public', action='store_true', help='Explicit permission to publish datasets to public repository')
    args = p.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
