"""Standard-library toolkit primitives: paths, JSON, hashing and snapshots."""
from __future__ import annotations
import hashlib, json, os, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAME=re.compile(r'^[a-zA-Z][a-zA-Z0-9_.-]{0,63}$')

def validated_name(s):
    if not NAME.fullmatch(s) or '..' in s:raise ValueError(f'Invalid name: {s!r}; use letters, digits, hyphens, underscores and dots')
    return s

def utc():return datetime.now(timezone.utc).isoformat()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    os.replace(temp,p)

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def rel(p):return str(Path(p).relative_to(ROOT)).replace('\\','/')

def project_path(name):return ROOT/'projects'/validated_name(name)

def version_path(project,version):return project_path(project)/'versions'/validated_name(version)

def files_for_version(project,version):
    v=version_path(project,version)
    return [v/'version.json'] + sorted(p for p in (v/'data').iterdir() if p.is_file()) if (v/'data').exists() else [v/'version.json']

def record(line):
    p=ROOT/'registry/runs.jsonl';p.parent.mkdir(exist_ok=True)
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(line,ensure_ascii=False)+'\n')

def snapshot_inputs(project,version,out):
    files=files_for_version(project,version)
    for f in files:
        if not f.is_file():raise FileNotFoundError('Missing version input: '+str(f))
    if out.exists():raise FileExistsError('Run snapshot exists: '+str(out))
    out.mkdir(parents=True)
    source=version_path(project,version)
    h={}
    for f in files:
        dest=out/f.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,dest)
        h[str(f.relative_to(source)).replace('\\','/')]=digest(dest)
    write(out/'input_hashes.json',h)
    return h

def assert_snapshot(snapshot):
    hashes=read(snapshot/'input_hashes.json')
    for name,expect in hashes.items():
        p=snapshot/name
        if not p.is_file() or digest(p)!=expect:raise RuntimeError(f'Frozen run input changed: {p}')
    return hashes
