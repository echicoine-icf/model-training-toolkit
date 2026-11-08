# v0.3.6 — Full resumable checkpoints for both training backends

Review the change summary in `CHANGELOG.md` and detailed instructions in `docs/RESUMABLE_CHECKPOINTS.md`.

**Stop any currently running MTT training before changing source code or dependencies.** An already-running v0.3.4 or v0.3.5 job will **not** acquire resumable checkpoint support by upgrading source files.

## Update from v0.3.5

Apply the source patch to the **source repo**, not your operational workspace. Run `python -m unittest discover -s tests -v`, commit to a feature branch, and merge a PR into `main`. In your operational workspace:

```bash
cd ~/model-training-toolkit
git pull --ff-only
bash workspace.sh upgrade
bash workspace.sh doctor
python -m unittest discover -s tests -v
```

This source-only update preserves `.venv`, CUDA packages, llama.cpp, models, training/evaluation datasets, project versions, previous adapters, logs, and existing checkpoints. No Unsloth install is required if using `pytorch`.

## New training runs

The previously supported one-shot training flags still work. Add `--save-every 500 --keep-last-checkpoints 3` to specify checkpoint intervals, or omit them for those defaults. This writes full LoRA/optimizer/RNG state rather than only an adapter.

After a crash, from the original run's last complete checkpoint:

```bash
bash workspace.sh runs --project eli
bash projects/eli/project.sh resume v9.1 --run RUN_ID
```

**Do not use `train` to resume.** `train` creates a new run from fresh base weights. Resumption uses original run ID, original pinned input snapshot and hyperparameters. Adapter-only snapshots made by v0.3.5 or earlier cannot resume; see `docs/RESUMABLE_CHECKPOINTS.md`.

Tests cover checkpoint integrity and optimizer/RNG parity using CPU PyTorch; GPU and Unsloth recovery must still be smoke-tested on the target RTX 5080 before relying on them for an unattended multi-hour job.
