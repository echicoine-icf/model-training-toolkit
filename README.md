# Model Training Toolkit v0.3.2

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

To upgrade an existing v0.3.0 workspace, copy distribution files from the full archive **without overwriting** `WORKSPACE_STATE.json`, `.venv`, `projects/`, `runs/`, `tools/llama.cpp`, or registry data. See [INSTALL_v0.3.2.md](INSTALL_v0.3.2.md) for exact safe commands. Then:

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

## Sanity checks

```bash
python -m unittest discover -s tests -v
bash workspace.sh --help
```

## Important boundaries

The model suitability list is conservative guidance, **not** a guarantee the GPU can train a selected architecture. CUDA LoRA fine-tuning, not QLoRA or full-model training, is currently supported. Evaluation is exact JSON matching only; semantic scoring of DATMM graph queries is future work. Actual training, Hugging Face downloads, and inference require the ML optional dependencies and have not been verified on every GPU architecture.

## GPU-aware PyTorch setup (v0.3.2)

`bash run_me_first.sh --install-ml` chooses the CUDA 12.8 PyTorch 2.10.0 build when an NVIDIA GPU is detected, checks a real CUDA tensor operation, and aborts on CUDA failure. The existing Python `.venv` remains isolated. An already verified `2.10.0+cu128` wheel is preserved. See `INSTALL_v0.3.2.md` for the non-destructive upgrade.
