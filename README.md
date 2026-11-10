# Model Training Toolkit (MTT)

A reusable, project-based toolkit for **supervised fine-tuning of task-specific language-model assistants**. MTT standardizes training inputs, LoRA configuration, checkpoints, exports, and evaluations so a model can be developed and compared across iterations without rebuilding the workflow each time.

MTT trains compatible causal instruction models to follow an **application-defined response contract**, such as natural language in and structured JSON out. It does not decide the semantics of your application: your project provides the examples, labels, system prompt, and (when needed) semantic evaluator.

## What it does

- **Prepare and validate** versioned datasets (`mtt-data-v1`) with prompt, expected assistant answer, and optional schema.
- **Fine-tune** from a selected Hugging Face base with PEFT LoRA using the PyTorch backend or the optional Unsloth backend. Assistant-only loss masking trains on assistant responses while retaining system/user context.
- **Freeze every run**: source model revision, dataset and prompt hashes, training settings, adapter, and run metadata.
- **Resume training** from verified checkpoints containing optimizer and RNG state; keep a configurable number of checkpoints.
- **Export** a trained adapter into a merged Hugging Face model and use supported export tooling for GGUF workflows.
- **Evaluate** training recall and unseen cases, with built-in exact-JSON evaluation, project-defined semantic scorers, or a project-level, versioned shared benchmark.
- **Publish reproducible project snapshots** to a separate repository, explicitly including selected training inputs and benchmarks while excluding weights and local runtime artifacts.

## Quick start

Prerequisites: Python 3.11+, Git, and sufficient disk space. CUDA training additionally requires a compatible NVIDIA GPU and driver. Examples below use Windows Git Bash.

```bash
# From a new checkout of this repository
bash run_me_first.sh
source ./activate.sh
bash workspace.sh doctor

# Opt in to ML dependencies only when training is required
bash run_me_first.sh --install-ml
bash workspace.sh doctor
```

**Existing workspace?** Do not rerun initialization or replace `.venv`, `projects/`, `runs/`, or downloaded model files. Pull approved source changes and run:

```bash
git pull --ff-only
bash workspace.sh upgrade
bash workspace.sh doctor
```

`doctor` is read-only; `doctor --repair` repairs validated machine-local workspace metadata without reinstalling dependencies. See [Workspace health](docs/WORKSPACE_HEALTH.md).

## Project workflow

```bash
# Create a project; select its model at version creation
bash workspace.sh project create example
bash workspace.sh version create example v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct --revision main

# Supply a validated data package in projects/example/versions/v1/data/
bash workspace.sh data inspect example v1
bash workspace.sh data validate example v1 --require-eval

# Interactive training (a fully specified noninteractive CLI is also supported)
bash projects/example/project.sh train v1

# Inspect and manage completed runs
bash workspace.sh runs --project example
bash workspace.sh status example
```

`main` is a moving Hugging Face revision; the training run resolves an exact base revision when possible. Prefer a pinned commit for repeatable training. Training requires explicit configuration review, including learning rate, epochs, LoRA targets, precision, and evaluation choice. A model may be evaluated only after the corresponding training run has finished.

## Data and application contract

A typical `data/` directory contains:

```text
projects/example/versions/v1/data/
├── package.json         # mtt-data-v1 contract and prompt assembly
├── system_prompt.txt    # assistant instructions
├── train.jsonl          # labeled training prompts
├── eval.jsonl           # independent labeled evaluation prompts (optional)
└── schema.json          # optional output contract/schema
```

Each JSONL line contains an `id`, `question`, and `expected` response:

```json
{"id":"train_001","question":"Put buying groceries on my list.","expected":{"title":"Buy groceries","priority":"medium"}}
```

The example output is illustrative, not MTT's own schema. Your **application** defines required fields, permitted values, defaults, and post-inference validation. Evaluation questions must stay out of training if you want an independent generalization measurement.

See [Data package contract](docs/DATA_PACKAGE_CONTRACT.md) and [Training a task-specific assistant](docs/TRAINING_TASK_SPECIFIC_ASSISTANTS.md).

## Evaluation options

| Option | Appropriate use |
|---|---|
| Built-in `evaluate` | Basic exact-JSON regression checks |
| Custom evaluation runner | Project-specific semantic correctness (e.g., equivalent plans, domain rules) |
| Shared evaluation benchmark (`shared-eval`) | Reusable, versioned suites with targeted selection, historical comparison, and dataset hashes |

```bash
# Evaluate a completed run with the built-in/custom evaluation entry point
bash projects/example/project.sh evaluate v1 --run RUN_ID

# A project with an installed shared benchmark and its own evaluator can:
bash projects/example/project.sh shared-eval v1 \
  --run RUN_ID --benchmark BENCHMARK_NAME --list-suites

bash projects/example/project.sh shared-eval v1 \
  --run RUN_ID --benchmark BENCHMARK_NAME \
  --model PROJECT_MODEL --suite SUITE_NAME --dry-run
```

Shared benchmarks are **optional project assets**; MTT does not bundle an ELI-specific semantic scorer or Ollama model. `--suite` on `shared-eval` selects one or several named suites, whereas `--suite` on the older custom `evaluate` workflow denotes its ordered stages. They are distinct interfaces. Both keep result provenance separate from training artifacts.

Read [Custom runners](docs/CUSTOM_EVALUATION_RUNNERS.md) and [Shared benchmarks](docs/SHARED_EVALUATIONS.md).

## Exports, snapshots, and reproducibility

```bash
# Export the completed LoRA run as merged Hugging Face SafeTensors
bash workspace.sh export example v1 --run RUN_ID --format hf
```

Weights, GGUF files, checkpoints, caches, and the local `.venv` are **not committed to the framework repository**. The [project snapshot publisher](docs/PROJECT_SNAPSHOTS.md) can preserve a selected project's authored data, benchmark release, and frozen run recipe in a separate Git repository; trained weights require separate artifact storage. Model rebuilds may not be numerically bit-identical across differing hardware and library versions.

## Documentation

- [Training a task-specific assistant](docs/TRAINING_TASK_SPECIFIC_ASSISTANTS.md) — concepts and end-to-end worked example
- [Architecture](docs/ARCHITECTURE.md) — core ownership and data flow
- [Data package contract](docs/DATA_PACKAGE_CONTRACT.md) — validated input format
- [Training backends](docs/UNSLOTH_BACKEND.md) — native PyTorch versus optional Unsloth
- [Resumable checkpoints](docs/RESUMABLE_CHECKPOINTS.md) — safe recovery
- [Custom evaluation runners](docs/CUSTOM_EVALUATION_RUNNERS.md) — executable scorer contract
- [Shared evaluation benchmarks](docs/SHARED_EVALUATIONS.md) — suites, hashes, and release gates
- [Project snapshot publishing](docs/PROJECT_SNAPSHOTS.md) — portable source and data snapshots
- [Workspace health](docs/WORKSPACE_HEALTH.md) — diagnostics and repair
- [Changelog](CHANGELOG.md) — release history

Run offline tests with `python -m unittest discover -s tests -q`. Review project-provided Python evaluation runners before executing them: **they are trusted code, not sandboxed plugins**.
