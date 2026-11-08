# v0.3.5 — Optional Unsloth backend

This release changes **source code only** and does not require replacing the operational workspace, `.venv`, CUDA wheel, models, datasets, checkpoints, or existing runs.

## Upgrade with Git

The official source release should be reviewed, committed to a feature branch, and merged through a pull request into `main`. On the operational machine:

```bash
cd ~/model-training-toolkit
git pull --ff-only
bash workspace.sh upgrade
bash workspace.sh doctor
python -m unittest discover -s tests -v
```

**Do not install Unsloth or rerun ML setup while a GPU training job is in progress.** The existing native PyTorch backend continues working without Unsloth.

## Optional Unsloth installation (after the current job finishes)

```bash
cd ~/model-training-toolkit
source ./activate.sh
python -m pip freeze > registry/pip_freeze_before_unsloth.txt
python -m pip install unsloth
python -m pip check
bash workspace.sh doctor
```

Review `docs/UNSLOTH_BACKEND.md` before starting a real model run. For full Windows-specific installation instructions and fallback routes, refer to https://unsloth.ai/docs/get-started/install/windows-installation.

To opt in: add `--backend unsloth` to the existing complete `train` command. Unsloth is never loaded for `--backend pytorch` (the default). A requested but missing Unsloth backend is an error rather than a silent fallback.

Tests validate CLI/config handoff, model identity and nonquantized loading settings, arbitrary LoRA target passthrough, checkpointing selection, and import ordering. **Actual Unsloth GPU training and subsequent export/evaluation remain to be validated on your machine.**
