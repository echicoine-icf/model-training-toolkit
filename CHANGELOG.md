## v0.3.6 — Safe, complete optimizer/RNG checkpoint resume

- Windows checkpoint durability hotfix: reopen saved artifacts with writable file descriptors for `os.fsync`; add cross-platform regression test.

- Save fully resumable optimizer, GradScaler, exact LoRA weights, Torch/CUDA/NumPy/Python RNG, microbatch and update-step state for either backend.
- Atomically publish verified complete checkpoints; detect corruption and incompatible frozen inputs/configuration before recovery.
- Add `workspace.sh resume PROJECT VERSION --run RUN_ID [--checkpoint latest|step_XXXXXXX]` and generated project launcher equivalent; resume the existing run, not a new run.
- Default to a checkpoint every 500 optimizer steps and retaining the latest three. Support `--keep-last-checkpoints 0` for unlimited history and `--save-every 0` to disable.
- Add OS-managed concurrent training lock, loss-log resume marker, runtime accounting across attempts, and interruption status.
- Preserve data packages and existing runs. Pre-v0.3.6 adapter-only checkpoints remain non-resumable.
- Include CPU optimizer/RNG interruption parity tests; GPU/Unsloth end-to-end recovery remains to be verified.

## v0.3.5 — Optional Unsloth training backend

- Add `--backend pytorch|unsloth` to CLI and training wizard; default remains the existing PyTorch implementation.
- Import Unsloth before torch; initialize unquantized base and PEFT LoRA through Unsloth when explicitly requested.
- Preserve the common supervised masking, dataset validation, optimizer/schedule, accumulation, snapshots, evaluation interface, and output contract.
- Preserve arbitrary target modules and audit all actual PEFT LoRA layers before optimization; abort on incomplete or unexpected matches.
- Add `--unsloth-checkpointing standard|unsloth` (Unsloth-only) and record backend version and computation caveats in the training plan.
- Document safe optional installation, Windows/CUDA considerations, and smoke-test requirements. No automatic dependency upgrades or existing-run changes.
- Add backend-selection and Unsloth API passthrough tests. Native GPU verification is pending.

## v0.3.4 — LoRA architecture verification and training reproducibility

- Honor arbitrary CLI LoRA target-module lists; verify exact PEFT matched-layer coverage before optimization.
- Record LoRA target names, matched layer counts, trainable parameter count, effective precision/attention/optimizer/LR schedule, checkpoint behavior, and prompt/tokenizer hashes.
- Add explicit minimum cosine learning rate, BF16/FP16 precision, attention backend, gradient checkpointing, and intermediate adapter checkpoint controls.
- Preserve original validated assistant message text rather than reserializing training JSON targets.
- Correct gradient accumulation optimizer-step counting across epoch boundaries.
- Add unit tests that fail on partial/extra target modules, output mismatches, and training-option regressions.
- No changes to project datasets, evaluation datasets, or existing training runs.

## 0.3.3

- Training supports noninteractive, explicit CLI flags for every hyperparameter and auto-eval policy; `--yes` required.
- Invalid or missing flags fail before any training run is created.
- Guided wizard remains available.
- Project-local launcher forwards train arguments (refreshed by `workspace.sh upgrade`).

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
