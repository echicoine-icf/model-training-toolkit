# Model Training Toolkit v0.3.3 — explicit training CLI

This release is a **complete distributable source ZIP**, with no virtualenv, downloaded models, or local ELI data. The original `v0.3.2` training wizard remains available.

## Update Git source repository and deploy to operational workspace

This applies only the changed source files from the full ZIP. Do not replace `.venv`, `WORKSPACE_STATE.json`, `projects/`, `runs/`, `registry/`, or `tools/llama.cpp/`.

```bash
cd ~/model-training-toolkit-repo
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
home=Path.home()
root=home/'model-training-toolkit-repo'
archive=home/'Downloads/Model_Training_Toolkit_v0.3.3_FULL.zip'
files=[
    'toolkit/cli.py', 'config/workspace.json', 'README.md', 'CHANGELOG.md',
    'tests/test_train_cli_flags.py', 'INSTALL_v0.3.3.md', 'RELEASE_MANIFEST.json',
]
assert archive.is_file(), archive
assert (root/'.git').is_dir(), 'Source repository missing'
with ZipFile(archive) as z:
    for name in files:
        dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(z.read(name))
print('Applied v0.3.3 source files to Git repository; no local workspaces changed.')
PY
python -m unittest discover -s tests -v
git add toolkit/cli.py config/workspace.json README.md CHANGELOG.md \
  tests/test_train_cli_flags.py INSTALL_v0.3.3.md RELEASE_MANIFEST.json
git diff --cached --stat
git commit -m "feat: add fully parameterized noninteractive training CLI (v0.3.3)"
git push origin main
git tag -a v0.3.3 -m "Model Training Toolkit v0.3.3"
git push origin v0.3.3

cd ~/model-training-toolkit
git pull --ff-only
bash workspace.sh upgrade
bash workspace.sh doctor
```

The last `upgrade` refreshes the existing `projects/eli/project.sh` wrapper without replacing projects, models, datasets, or runs.

## Example (NOT a tested ELI v9.0 training recipe)

```bash
bash projects/eli/project.sh train v9.0 \
 --epochs 1 --max-length 4096 --learning-rate 2e-5 \
 --gradient-accumulation 1 --seed 42 \
 --weight-decay 0.01 --warmup-ratio 0.03 --max-grad-norm 1.0 \
 --log-every 25 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
 --lora-target-modules q_proj,v_proj --auto-eval --yes
```

Use `--no-auto-eval` rather than `--auto-eval` to skip evaluation. Exactly one of them must be provided for noninteractive runs. All flags and `--yes` are mandatory; a missing/invalid parameter stops the process before it creates a run. The selected model belongs to `version.json` and is not changed by the training command. All settings are saved into the run's immutable input snapshot and run metadata.

This release has been tested with unit and simulated engine tests, not an actual RTX 5080 training job. Tests cannot guarantee the selected model or memory configuration will fit the user's hardware.
