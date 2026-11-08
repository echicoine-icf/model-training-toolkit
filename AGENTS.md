# Model Training Toolkit — Agent Operating Guide

**Applies to:** Model Training Toolkit v0.3.5 (the code in this repository).  
**Purpose:** Give a coding agent enough verified context to install a workspace, register projects and model versions, create version-specific training/evaluation packages, execute runs, and hand work to another agent **without inventing version-specific training scripts**.

> **Source of truth:** Inspect the checked-out code and `docs/DATA_PACKAGE_CONTRACT.md` before changing anything. This file documents v0.3.5, not hypothetical future capabilities. If the installed toolkit version differs, inspect its implementation first.

## 1. Architecture and ownership

- **Toolkit source:** `toolkit/`, `scripts/`, `config/`, `requirements/`, `tests/`, `docs/`, `run_me_first.sh`, `workspace.sh`. This is reusable infrastructure, independent of ELI.
- **Operational workspace:** A checkout of the toolkit plus an isolated `.venv`, downloaded `tools/llama.cpp`, local `projects/`, `runs/`, registry files, and caches. Never commit local artifacts to the source repository.
- **Project:** `projects/<project>/` contains a generated `project.sh` launcher, `project.json`, notes, and versions.
- **Model version:** `projects/<project>/versions/<version>/version.json` identifies a Hugging Face base model/revision and always initializes training from the **fresh base**. Version inputs live in `data/`.
- **Training run:** `runs/<project>/<version>/<run-id>/` has frozen inputs/configuration and SHA-256 fingerprints, adapter weights, logs, optional evaluation, and `run.json` status. Do not modify completed run snapshots.
- **Version data package:** Authored independently of the toolkit; **not** a Python training script. Authoritative format: `docs/DATA_PACKAGE_CONTRACT.md` (`mtt-data-v1`).

## 2. Install a clean workspace (Windows Git Bash)

Prerequisites: Python 3.11+, Git, adequate disk space, and NVIDIA drivers for CUDA training.

```bash
cd ~/model-training-toolkit
bash run_me_first.sh
source ./activate.sh
bash workspace.sh doctor

# Before GPU training (optional for project registration / data authoring):
bash run_me_first.sh --install-ml
bash workspace.sh doctor
```

- Installer creates **only a workspace-local `.venv`**, not system-wide Python dependencies. `workspace.sh` uses its Python automatically.
- `run_me_first.sh` clones a pinned llama.cpp source checkout to `tools/llama.cpp`; record its actual Git commit.
- On NVIDIA hardware the installer selects the PyTorch CUDA 12.8 profile and checks a GPU tensor calculation. Verify `TORCH ...+cu128 CUDA_AVAILABLE True` and the detected GPU before training; do not silently accept a CPU-only PyTorch wheel.
- `bash workspace.sh models` and `bash workspace.sh hardware` inspect curated suggestions/hardware. Model sizing is advisory, **not** guaranteed to fit VRAM.
- Never reinitialize, delete, or overwrite `.venv`, local `projects/`, `runs/`, registry data, model caches, or `tools/llama.cpp` merely to fix a source-code issue.

## 3. Source development and deployment

The established workflow uses two separate checkouts:

| Directory | Use |
|---|---|
| `~/model-training-toolkit-repo` | Edit/test/commit/push reusable source to GitHub |
| `~/model-training-toolkit` | Run models and preserve local data; pull approved changes |

```bash
# In source checkout, after tests/review:
cd ~/model-training-toolkit-repo
python -m unittest discover -s tests -v
git status --short
git switch -c feat/describe-change
git add <explicit-source-paths>
git commit -m "Describe change"
git push -u origin feat/describe-change
# Open a pull request and merge through the protected main branch

# In operational checkout:
cd ~/model-training-toolkit
git pull --ff-only
bash workspace.sh upgrade
bash workspace.sh doctor
```

Never use `git reset --hard`, `git clean -fd`, `git add .` blindly, force-push, or delete training artifacts. Keep operational machine state out of Git; update `.gitignore` when new local artifacts appear. Tag tested releases separately; **do not rewrite existing release tags**.

## 4. Create a project and model version

```bash
bash workspace.sh project create eli

# Interactive: choose the base model and revision at version creation
bash projects/eli/project.sh new-version v9.0

# Or noninteractive (model selection is REQUIRED, not silently guessed):
bash workspace.sh version create eli v9.0 \
  --model Qwen/Qwen2.5-0.5B-Instruct --revision main
```

- A model version's base model and Hugging Face revision are selected **at version creation**, not each training run. `main` means latest at the time of resolution; exact commit SHA is better for reproducibility. The run snapshot attempts to resolve the concrete SHA.
- Version creation **does not** request prompt paths, training data paths, epochs, learning rate, or sequence length.
- Existing version registrations are preserved. For a different base model, create a **new version**. Do not mutate existing `version.json` on a used version to change its identity.
- The toolkit may create a draft `data/package.json`; inspect its settings before replacing it with an authored data package.

## 5. Data package contract (`mtt-data-v1`)

A model/data author supplies one ZIP containing exactly the relative version layout (example):

```text
v9.0/
  data/
    package.json           # required
    train.jsonl            # required
    system_prompt.txt      # required
    eval.jsonl             # required only if evaluating
    schema.json            # optional unless append_schema
```

Extract `v9.0/` under `projects/eli/versions/`, **not** at repository root and never over a frozen run. The user can extract files before or after version registration.

`package.json` example:

```json
{
  "format": "mtt-data-v1",
  "task": "supervised_chat_json",
  "files": {
    "train": "train.jsonl",
    "eval": "eval.jsonl",
    "system_prompt": "system_prompt.txt",
    "schema": "schema.json"
  },
  "required": {
    "train": true,
    "system_prompt": true,
    "eval": false,
    "schema": false
  },
  "prompt_assembly": "prompt_only",
  "evaluation": {
    "scorer": "exact_json",
    "max_new_tokens": 384
  },
  "description": "Explain the project task and data provenance."
}
```

**Every train/eval JSONL line** is an independent UTF-8 JSON object with `id`, nonempty `question`, and `expected` (JSON object, array, or string). Example:

```json
{"id":"train_0001","question":"Find datasets about asthma.","expected":{"target":"Dataset","conditions":[{"field":"dataset.subject.label","operator":"equals","value":"asthma","id":"condition1"}],"related":[],"relationshipGraph":null,"conditionBindings":[]}}
```

- Training and evaluation examples must be **separate**. No duplicate normalized questions across splits. Preserve provenance and exact gold labels; **never invent expected JSON from model output**.
- `prompt_assembly: "prompt_only"`: use `system_prompt.txt` as-is. If the schema is already embedded, **do not append it again**.
- `prompt_assembly: "append_schema"`: append `schema.json` to the prompt and mark schema required.
- `eval.jsonl` can be optional for training, but is mandatory for automatic evaluation or normal evaluation.
- For ELI, ensure the RDF schema, canonical fields, supported relationships, scope/bindings and unsupported requests agree with the DATMM compiler. The toolkit's current evaluator **cannot** establish those semantics; use a separately reviewed ELI-specific evaluation process where necessary.
- When authoring data packages, deliver the ZIP plus a compact data README: counts, source/provenance, schema/prompt version, SHA-256 hashes, holdout definition, known limitations, and a full recommended training command. **Do not add version-specific training Python scripts.**

Validate before training:

```bash
bash workspace.sh data inspect eli v9.0
bash workspace.sh dataset validate eli v9.0
# If evaluation is planned, require its presence:
bash workspace.sh data validate eli v9.0 --require-eval
```

If required files are missing, `bash workspace.sh data wizard eli v9.0` can assist with importing existing files; it must not manufacture gold labels.

## 6. Train — wizard or complete one-command CLI

Interactive:

```bash
bash projects/eli/project.sh train v9.0
```

Noninteractive for coding agents/automation: **every flag is required**; the command must include one explicit evaluation decision (`--auto-eval` or `--no-auto-eval`) and `--yes`. Example values below are **illustrative, not an approved ELI training configuration**:

```bash
bash workspace.sh train eli v9.0 \
  --epochs 1 \
  --max-length 4096 \
  --learning-rate 2e-5 \
  --gradient-accumulation 1 \
  --seed 42 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --log-every 25 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj \
  --auto-eval \
  --yes
```

Project-local equivalent: `bash projects/eli/project.sh train v9.0 [FLAGS...]`.

Before launch: confirm project/version, correct base model and revision, `mtt-data-v1` validation, GPU compatibility, corpus/holdout counts, prompt assembly, and **explicit** training settings. Do not launch just because the user provided a ZIP; obtain authorization to execute the run. With all flags and `--yes`, the user has explicitly authorized the run.

- Training uses the declared **fresh base** plus PEFT LoRA with PyTorch/Transformers. It **does not** continue from the previous adapter.
- Successful training saves an adapter and records a trained run; `--auto-eval` then evaluates it if eval data was supplied. An evaluation failure should not erase trained weights.
- Every run freezes version/config/data inputs with SHA-256. Retain the frozen snapshot, logs and run metadata.
- This toolkit currently supports **LoRA on compatible causal LMs with NVIDIA CUDA**, not QLoRA, full-model training, distributed training, or arbitrary architectures. No optimizer-state resume is guaranteed.

## 7. Evaluate, export, inspect and hand off

```bash
bash workspace.sh evaluate eli v9.0
bash workspace.sh runs --project eli
bash workspace.sh status eli
bash workspace.sh compare eli v8.0 v9.0
bash workspace.sh export eli v9.0 --format hf
bash workspace.sh export eli v9.0 --format gguf
bash workspace.sh handoff eli
```

Important conditions:

- The evaluator refuses to evaluate an untrained model unless `--base-only` is explicitly requested. Use `--run <run-id>` to select a specific trained run when needed.
- **Do not evaluate the same run twice by overwriting its output**. The engine protects the existing evaluation directory. `--base-only` is an explicit baseline mode, not a trained-adapter evaluation.
- `--format hf` produces merged Hugging Face weights; `--format gguf` uses the pinned llama.cpp converter. Export only from a completed trained adapter. Outputs may consume substantial disk space.
- The built-in evaluator currently scores **exact JSON**, including array order. It is **not semantic equivalence** for DATMM relationships, commutative conditions or conditionBindings. Do not report exact scores as semantic accuracy. Only compare results evaluated on identical frozen holdout data.
- Generate `projects/eli/PROJECT_HANDOFF.md` with `bash workspace.sh handoff eli`; include that file **and** this `AGENTS.md` and `docs/DATA_PACKAGE_CONTRACT.md` in a new chat/agent session.

## 8. Agent guardrails and stop conditions

1. **Inspect first; change second.** Do not rewrite a working toolkit, trainer or evaluator because a new version has different data. A change to the generic engine requires tests in the source checkout and a controlled deployment.
2. **Keep data separate from code.** Future ELI v9.1 should normally add only `projects/eli/versions/v9.1/data/` plus a deliberate base-model selection—not another trainer.
3. **Preserve provenance.** Record exact hashes, immutable model revisions, source/heldout partitions, hyperparameters, tokenizer/ML dependency versions, and evaluation results. No speculative success claims.
4. **Never silently change evaluation criteria** across versions. Record scorer version, evaluation set identity and limitations.
5. **Do not delete or overwrite** `.venv`, local data, model weights, runs, registered versions, or llama.cpp checkouts to repair a routine error. No destructive Git commands without explicit confirmation and a verified backup plan.
6. **Stop when a contract is unsupported.** For ELI: universal/disjunctive quantifiers, unsupported graph semantics and invented canonical fields require an explicit representation/compiler decision—not fabricated labels or another blind training cycle.
7. **Report concise next steps.** Give the user exact Git Bash commands, what should happen, and what output to return. For new conversations, read `PROJECT_HANDOFF.md` before proposing new scripts.

## 9. Fast context restoration for a new coding agent

Read these in order:

1. `AGENTS.md` (this document — architecture, commands, safety).
2. `README.md` and `docs/DATA_PACKAGE_CONTRACT.md` (exact data format and current capabilities).
3. `config/workspace.json`, `toolkit/cli.py`, `toolkit/train_engine.py`, `toolkit/eval_engine.py` (actual implementation).
4. `projects/<project>/PROJECT_HANDOFF.md` and `PROJECT_NOTES.md` (project-specific decisions and current state).
5. `projects/<project>/versions/<version>/version.json` and `data/package.json`, then relevant `runs/.../run.json` and evaluation summaries.

**Initial agent response in a new chat:** State the detected toolkit version, project/version, selected base model, current dataset readiness, last completed run and evaluation status, and the *single next executable action*. If any of those facts are unavailable, say so rather than guessing.

## Optional Unsloth training backend (v0.3.5)

Read `docs/UNSLOTH_BACKEND.md` before enabling Unsloth. The default `--backend pytorch` never imports it. With `--backend unsloth`, MTT uses Unsloth for fresh-base full-precision model loading and PEFT initialization but retains the shared loss masking, optimizer, LR schedule, and output contract. Do not claim GPU compatibility, parity, or speedup until tested on the target installation. Import Unsloth **before** Torch/Transformers/PEFT when selected; never silently fall back to PyTorch. Do not change a frozen project's data package just to enable a backend. A new run may use the same immutable inputs.
