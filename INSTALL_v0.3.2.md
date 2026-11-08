# Model Training Toolkit v0.3.2 — full ZIP and safe in-place update

This full distribution is for both fresh installations and in-place upgrades.
It fixes the v0.3.1 ML installer mistakenly selecting CPU-only PyTorch on Windows with NVIDIA GPUs.

**Existing workspace, Windows Git Bash** (ZIP downloaded into `~/Downloads`):

```bash
cd ~/model-training-toolkit
source ./activate.sh
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

root = Path.home() / 'model-training-toolkit'
archive = Path.home() / 'Downloads/Model_Training_Toolkit_v0.3.2_FULL.zip'
assert (root / '.venv/pyvenv.cfg').is_file(), 'STOP: existing isolated venv not found'
assert (root / 'WORKSPACE_STATE.json').is_file(), 'STOP: existing toolkit state missing'
assert archive.is_file(), f'STOP: ZIP not found: {archive}'
protected = {'.venv', 'projects', 'runs', 'registry', 'cache', 'exports', 'logs'}
with ZipFile(archive) as z:
    for m in z.infolist():
        p = Path(m.filename)
        if m.is_dir() or p.is_absolute() or '..' in p.parts:
            continue
        if not p.parts or p.parts[0] in protected:
            continue
        if p.parts[0] == 'tools' and len(p.parts)>1 and p.parts[1]=='llama.cpp':
            continue
        if str(p).replace('\\','/') in {'WORKSPACE_STATE.json','config/workspace-local.json'}:
            continue
        dst = root / p
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(z.read(m))
print('Toolkit v0.3.2 code installed. Existing .venv and project data preserved.')
PY
bash run_me_first.sh --install-ml
bash workspace.sh doctor
```

Since the RTX 5080 already successfully passed a CUDA operation using `torch 2.10.0+cu128`, setup will **reuse** that exact working wheel and install or check only the remaining ML packages. CUDA will be verified before setup reports success.

For first-time setup after unzipping the complete distribution into a clean directory, run `bash run_me_first.sh --install-ml` directly, then `source ./activate.sh` and `bash workspace.sh doctor`.

The default wheel choice is automatic: NVIDIA GPU visible via `nvidia-smi` → PyTorch 2.10.0 with CUDA 12.8; otherwise CPU wheel. Override with `bash run_me_first.sh --install-ml --torch-profile cu128` or `--torch-profile cpu` when appropriate. Selecting a CUDA profile requires successful GPU computation; a CPU-only build is rejected.
