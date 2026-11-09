# Model Training Toolkit v0.3.8

An isolated, repeatable **version-package-driven** workspace for supervised LoRA fine-tuning and evaluation. Share this full ZIP with another developer; they can initialize in any directory without using `datmm-model`.

## Install (Windows Git Bash; Python 3.11+)

Extract the complete release archive so that `run_me_first.sh` is in the `model-training-toolkit` directory:

```bash
cd ~/model-training-toolkit
bash run_me_first.sh
source ./activate.sh
bash workspace.sh doctor
```

This creates `.venv` and clones a pinned llama.cpp source tree. GPU training packages are **not** installed by default:

```bash
bash run_me_first.sh --install-ml
bash workspace.sh doctor
```

To upgrade an existing workspace, copy distribution files from the full archive **without overwriting** `WORKSPACE_STATE.json`, `.venv`, `projects/`, `runs/`, `tools/llama.cpp`, or registry data. See [INSTALL_v0.3.3.md](INSTALL_v0.3.3.md) for the Git-based source-to-operational update workflow. Then:

```bash
bash workspace.sh upgrade
bash workspace.sh doctor
```

Legacy project prompt/schema files are copied non-destructively to the new version-data layout when `upgrade` runs. Existing frozen run snapshots are never rewritten; evaluate older runs with the matching historical toolkit implementation.

## New project and version

```bash
bash workspace.sh project create eli
bash projects/eli/project.sh new-version v9.0
```

Version creation **requires selecting the base model and Hugging Face revision**. It does **not** ask for paths, prompts, epochs, or tokens. The training wizard confirms the selected model but does not allow swapping it.

A contributor can send a ZIP containing `v9.0/data/` (see **[DATA_PACKAGE_CONTRACT.md](docs/DATA_PACKAGE_CONTRACT.md)**). Extract its contents to `projects/eli/versions/` and then:

```bash
bash workspace.sh data inspect eli v9.0
bash workspace.sh dataset validate eli v9.0
bash projects/eli/project.sh train v9.0
```

When required data is missing, `train` offers an import wizard before the run configuration questions. The training wizard requires explicit review and entry of every hyperparameter and asks whether to evaluate automatically after a successful training run. When you select `main` for the Hugging Face revision during version creation, the toolkit attempts to resolve its immutable commit SHA before training.

```bash
bash projects/eli/project.sh evaluate v9.0   # Only after a trained adapter exists
bash workspace.sh status eli
bash workspace.sh runs --project eli
bash workspace.sh handoff eli
```

The handoff document is at `projects/eli/PROJECT_HANDOFF.md`. Upload it with the data contract when starting a new ChatGPT conversation.

## Version layout

```text
model-training-toolkit/
├── .venv/                 # created locally; never distribute
├── run_me_first.sh
├── activate.sh
├── workspace.sh
├── toolkit/               # stable engine
├── docs/DATA_PACKAGE_CONTRACT.md
├── tools/llama.cpp/       # pinned source; created at setup
├── projects/eli/
│   ├── project.sh
│   ├── PROJECT_HANDOFF.md  # generated on request
│   └── versions/v9.0/
│       ├── version.json
│       └── data/
│           ├── package.json
│           ├── train.jsonl
│           ├── system_prompt.txt
│           ├── eval.jsonl
│           └── schema.json
├── runs/eli/v9.0/<run-id>/
│   ├── input/             # frozen data and config, SHA-256 manifest
│   ├── adapter/           # produced by successful LoRA training
│   ├── evaluation/        # optional, exact JSON scorer
│   ├── run.json
│   └── training_progress.jsonl
└── registry/runs.jsonl
```

## Project-supplied evaluation runners (v0.3.7)

In addition to the built-in exact-JSON evaluator, a project can register scripts and
ordered evaluation stages in `data/evaluation/runners.json`. Complete evaluation
runs can be resumed without regenerating predictions. Existing training snapshots
can instead attach an external immutable evaluation package with `--eval-package`.

See [CUSTOM_EVALUATION_RUNNERS.md](docs/CUSTOM_EVALUATION_RUNNERS.md).

## Project-level shared evaluation benchmarks (v0.3.8)

Project-owned benchmarks live at `projects/<project>/evaluations/<benchmark-name>/`,
independently of model version. Each package declares an explicit **package version**
and per-dataset SHA-256 hashes in its benchmark manifest; the historical semantic
scoring contract can remain unchanged as more reviewed cases are added.

`shared-eval` audits the benchmark against the selected run's **frozen** training
snapshot (including protected-holdout leakage), uses the run's frozen system prompt,
and stores model-specific results under the existing run's `shared_evaluations/`.
No training artifacts or earlier reports are overwritten. The project-owned Python
runner (Ollama for ELI) performs inference; MTT only handles safe selection, audit,
and delegation. Compare only results with matching benchmark package, case hashes,
scorer and model/inference provenance.

```bash
bash projects/eli/project.sh shared-eval v9.2 \
  --run RUN_ID --benchmark benchmark-v1.0.1 \
  --audit-only

bash projects/eli/project.sh shared-eval v9.2 \
  --run RUN_ID --benchmark benchmark-v1.0.1 \
  --model datmm-eli:v9.2-f16 --stage generalization --dry-run

bash projects/eli/project.sh shared-eval v9.2 \
  --run RUN_ID --benchmark benchmark-v1.0.1 \
  --model datmm-eli:v9.2-f16 --stage generalization
```

See `docs/SHARED_EVALUATIONS.md`. The original V9.2 training-baseline results
remain version-specific and are **not** repeated by the shared command.

## Sanity checks

```bash
python -m unittest discover -s tests -v
bash workspace.sh --help
```

## Important boundaries

The model suitability list is conservative guidance, **not** a guarantee the GPU can train a selected architecture. CUDA LoRA fine-tuning, not QLoRA or full-model training, is currently supported. The built-in evaluator uses exact JSON matching; project-supplied evaluators can implement DATMM semantic scoring without changing MTT. Actual training, Hugging Face downloads, and inference require the ML optional dependencies and have not been verified on every GPU architecture.

## GPU-aware PyTorch setup (v0.3.2)

`bash run_me_first.sh --install-ml` chooses the CUDA 12.8 PyTorch 2.10.0 build when an NVIDIA GPU is detected, checks a real CUDA tensor operation, and aborts on CUDA failure. The existing Python `.venv` remains isolated. An already verified `2.10.0+cu128` wheel is preserved. See `INSTALL_v0.3.2.md` for the non-destructive upgrade.

## Single-command training (v0.3.3)

The guided wizard still works: `bash projects/eli/project.sh train v9.0`.
For automation or an ELI development chat, provide **every** run parameter explicitly:

```bash
bash workspace.sh train eli v9.0 \
  --epochs 1 --max-length 4096 --learning-rate 2e-5 \
  --gradient-accumulation 1 --seed 42 --weight-decay 0.01 \
  --warmup-ratio 0.03 --max-grad-norm 1.0 --log-every 25 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj \
  --no-auto-eval --yes
```

Switch `--no-auto-eval` to `--auto-eval` to require an evaluation dataset and automatically run evaluation after successful training. The command fails before creating a run if any flag is missing, conflicting, or invalid. **These values are examples, not ELI-specific recommendations.** The fixed model remains defined at version creation. A fully parameterized run is recorded in a frozen run snapshot. `--yes` explicitly authorizes execution without prompts.

The project-local entrypoint also accepts flags after the version: `bash projects/eli/project.sh train v9.0 [FLAGS...]`. Existing project launchers are refreshed by `bash workspace.sh upgrade`.


## Architecture-audited training (v0.3.4)

Single-command training also supports `--min-learning-rate`, `--precision`,
`--attention-implementation`, `--optimizer`, `--gradient-checkpointing`
(or `--no-gradient-checkpointing`), and `--save-every`.
Every run records the **actual** PEFT-adapted layer paths, per-target counts,
trainable and total parameters, attention, precision, scheduler, optimizer,
prompt/tokenizer hashes, and runtime environment in `training_plan.json`
**before the first optimizer update**. Requested target modules must all match;
missing or unexpected PEFT LoRA targets abort the run.

If a dataset provides validated `messages` with system/user/assistant text,
that exact approved assistant text is used for loss masking and tokenization;
it is not silently regenerated from `expected`.

Intermediate checkpoints contain **adapter weights and tokenizer only**;
optimizer/RNG state resume is **not yet supported** by this general-purpose
backend. A new launch always starts fresh, creates a new run ID, and does not
replace older runs. `--no-auto-eval` is recommended for work requiring semantic
rather than exact-JSON evaluation until a suitable evaluator is installed.

See `docs/V80_TRAINING_COMPARISON.md` for a precise comparison with the ELI
V8.0 reference trainer supplied for the audit.

## Optional Unsloth backend (v0.3.6)

Training now offers `--backend pytorch` (the unchanged default) and `--backend unsloth`. The latter uses Unsloth for full-precision base loading and LoRA initialization, **without replacing MTT's masking, optimizer, scheduler, data package, or evaluation interface**. It accepts the same `--lora-target-modules` lists and refuses any missing targets or unexpected adapted layers.

Install Unsloth only after stopping active training and backing up the existing pip environment; on an already GPU-enabled Windows installation, its official installation command is `python -m pip install unsloth` inside the MTT `.venv`. The optional `--unsloth-checkpointing standard|unsloth` chooses Unsloth's checkpointing strategy. See **[docs/UNSLOTH_BACKEND.md](docs/UNSLOTH_BACKEND.md)** and **[INSTALL_v0.3.6.md](INSTALL_v0.3.6.md)** for full prerequisites, a complete example command, and parity caveats.

Unsloth is **not** automatically installed or selected, and the feature has not been benchmarked or GPU-smoke-tested on an RTX 5080 as part of this release.

## Fully resumable training (v0.3.6)

The native PyTorch and optional Unsloth training backends now write complete checkpoints (adapter, LoRA tensor state, AdamW optimizer, AMP scaler, RNG states, and exact microbatch position), defaulting to one every **500 optimizer updates**. Keep the latest three complete checkpoints by default; opt out with `--save-every 0` or retain all with `--keep-last-checkpoints 0`.

```bash
# Add to a full one-shot train command if a custom interval/retention is desired:
--save-every 500 --keep-last-checkpoints 3

# After stopping an interrupted process, resume the *original* run:
bash workspace.sh runs --project eli
bash projects/eli/project.sh resume v9.1 --run RUN_ID
# For a retained older checkpoint:
bash workspace.sh resume eli v9.1 --run RUN_ID --checkpoint step_0000500
```

The `resume` command reuses the frozen input snapshot and settings; it neither starts another run nor changes the dataset. Only checkpoints created by v0.3.6 or later are fully resumable. See [Resumable checkpoints](docs/RESUMABLE_CHECKPOINTS.md) for safety, disk-space, and verification details. Existing adapter-only checkpoints cannot resume optimizer state.
