"""Conservative, stdlib-only MTT workspace-state diagnostics and repair.

State metadata is *not* evidence of a working installation.  Environment
checks must pass before any command may mark a workspace ready.  Nothing here
creates virtualenvs, installs packages, edits projects, or touches run data.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from toolkit.core import write


@dataclass(frozen=True)
class WorkspaceHealth:
    version: str
    state: dict | None
    state_problem: str | None
    environment_problems: tuple[str, ...]
    metadata_problems: tuple[str, ...]

    @property
    def environment_ready(self) -> bool:
        return not self.environment_problems

    @property
    def healthy(self) -> bool:
        return self.environment_ready and not self.metadata_problems

    @property
    def repairable(self) -> bool:
        return self.environment_ready and bool(self.metadata_problems)


def inspect(root: Path) -> WorkspaceHealth:
    """Read-only audit; missing or invalid metadata never crashes doctor."""
    root = Path(root)
    cfg = root / 'config/workspace.json'
    try:
        data = json.loads(cfg.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or not isinstance(data.get('toolkit_version'), str) or not data['toolkit_version']:
            raise ValueError('config/workspace.json has no toolkit_version')
        version = data['toolkit_version']
    except (OSError, ValueError) as exc:
        version = 'UNKNOWN'
        config_problem = f'Invalid toolkit configuration: {exc}'
    else:
        config_problem = None

    issues = []
    if config_problem:
        issues.append(config_problem)
    venv = root / '.venv'
    if not (venv / 'pyvenv.cfg').is_file():
        issues.append('Missing .venv/pyvenv.cfg')
    if not any(p.is_file() for p in (venv / 'Scripts/python.exe', venv / 'bin/python')):
        issues.append('Missing .venv Python interpreter')
    for name in ('projects', 'runs', 'registry'):
        if not (root / name).is_dir():
            issues.append(f'Missing {name}/ directory')

    path = root / 'WORKSPACE_STATE.json'
    state = None
    problem = None
    if path.is_symlink():
        issues.append('WORKSPACE_STATE.json is a symlink; refusing automated repair')
        problem = 'State file is a symlink'
    elif not path.exists():
        problem = 'State file missing'
    elif not path.is_file():
        problem = 'State path is not a regular file'
        issues.append(problem)
    else:
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(state, dict):
                raise ValueError('Expected JSON object')
        except (OSError, ValueError) as exc:
            problem = f'Unreadable/invalid state JSON: {exc}'
            state = None

    metadata = []
    if problem:
        metadata.append(problem)
    elif state is not None:
        if state.get('status') != 'ready':
            metadata.append(f'Status is {state.get("status")!r}, expected "ready"')
        if state.get('toolkit_version') != version:
            metadata.append(f'State version {state.get("toolkit_version")!r} differs from code {version!r}')
    return WorkspaceHealth(version, state, problem, tuple(issues), tuple(metadata))


def _backup_state(root: Path) -> Path | None:
    """Retain existing state before repairing; never overwrite prior backups."""
    source = root / 'WORKSPACE_STATE.json'
    if not source.exists():
        return None
    if not source.is_file() or source.is_symlink():
        raise RuntimeError('Refusing to back up an irregular workspace state path')
    backups = root / '.workspace_state_backups'
    backups.mkdir(exist_ok=True)
    when = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = backups / f'WORKSPACE_STATE.{when}.json'
    # Avoid silently replacing backup on clock collision.
    with target.open('xb') as out, source.open('rb') as inp:
        shutil.copyfileobj(inp, out)
    return target


def repair(root: Path, *, upgrade: bool = False) -> tuple[WorkspaceHealth, Path | None, bool]:
    """Repair only metadata after independent filesystem prerequisites pass.

    upgrade=True stamps the current code version and last upgrade time even
    when the current status was already correct.  This must be invoked only
    after the upgrade's other operations succeed.
    """
    root = Path(root)
    health = inspect(root)
    if not health.environment_ready:
        raise RuntimeError('Cannot repair workspace metadata: ' + '; '.join(health.environment_problems))
    if health.healthy and not upgrade:
        return health, None, False

    state = dict(health.state or {})
    state['status'] = 'ready'
    state['toolkit_version'] = health.version
    state.pop('message', None)  # stale "run first" messages are not retained
    if upgrade:
        state['last_upgrade_utc'] = datetime.now(timezone.utc).isoformat()
    else:
        state['last_repair_utc'] = datetime.now(timezone.utc).isoformat()

    backup = _backup_state(root)
    write(root / 'WORKSPACE_STATE.json', state)  # atomic replace
    result = inspect(root)
    if not result.healthy:
        raise RuntimeError('Workspace state still inconsistent after metadata repair')
    return result, backup, True
