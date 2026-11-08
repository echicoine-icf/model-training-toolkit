# Changelog

## v0.3.2 — GPU-aware PyTorch installation

- Detect NVIDIA GPUs using `nvidia-smi` and install pinned official `torch==2.10.0` CUDA 12.8 wheels into the toolkit `.venv`.
- No more unqualified `torch` in the shared ML requirements file; CPU-only wheels cannot silently satisfy the GPU training profile.
- Preserve the already-working `2.10.0+cu128` PyTorch environment rather than reinstalling it.
- Explicit installer override: `--torch-profile {auto,cu128,cpu}`.
- Verify CUDA availability and an actual GPU tensor operation before reporting successful GPU installation.
- Save the installed PyTorch wheel version and selected CUDA/CPU profile in the workspace state.
- Full distribution includes safe in-place patching instructions that leave `.venv`, llama.cpp, projects, and runs untouched.


## v0.3.1 — Explicit base model at version creation

- Version creation **requires explicit base model selection**, with GPU-guided catalog or custom Hugging Face ID.
- Prompts, schema, training/evaluation files and hyperparameters remain outside version creation.
- Training shows the version's selected model without asking the user to choose a different one.
- Fresh-base runs still resolve `main` to a commit SHA before loading the model.
- Existing versions remain unchanged if `version create` is invoked again.
- Full release packaging includes safe update instructions to preserve an existing local `.venv`, workspace state and project data.


## v0.3.0 — Data-package-driven project versions

- Full reusable distribution with standardized `mtt-data-v1` data package contract.
- Version registration no longer collects prompt/schema/train/eval paths or training hyperparameters.
- Version registration accepts externally supplied `versions/<version>/data` folders without overwriting data.
- Training inspects required data files first, offers an import wizard for missing inputs, and refuses to invent labels.
- Run-by-run model selection, hyperparameters and auto-evaluation confirmation.
- Input snapshots include all version-local files in `data/` and SHA-256 hashes; mutable Hugging Face branches resolved to an exact commit before training.
- Frozen training and evaluation stay aligned on the same prompt/data package.
- Evaluation requires a trained adapter and a frozen evaluation dataset (unless explicitly base-only).
- Project handoff continues across chats; contract documentation includes a ready-to-copy ELI v9.0 package authoring brief.
- Non-destructive upgrade of legacy v0.2.x project inputs; older frozen snapshots remain intact.

## v0.2.1 and earlier

See archived release ZIPs for historical development notes. v0.3.0 is the first documented data-package contract for handoffs.
