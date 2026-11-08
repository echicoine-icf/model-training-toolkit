# Model Training Toolkit — Data Package Contract (mtt-data-v1)

**This is the authoritative interface for future training packages.**

A model author supplies the data package; the toolkit owns the training and evaluation engines. For ELI v9.0, the author should create a ZIP with this relative layout:

```text
v9.0/
└── data/
    ├── package.json          # REQUIRED, package format and behavior
    ├── train.jsonl           # REQUIRED, approved training cases
    ├── system_prompt.txt     # REQUIRED, exact training + inference system message
    ├── eval.jsonl            # Required if you elect automatic/manual evaluation
    └── schema.json           # Optional unless package.json uses append_schema
```

**Destination:** extract `v9.0/data/` into `model-training-toolkit/projects/eli/versions/`. You may extract before *or* after creating the version. Never extract a new ZIP over a frozen run snapshot. New revisions require a new named version or explicitly managed new run.

## package.json

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
  "description": "Describe the task and the training corpus."
}
```

`prompt_assembly` is **critical**:
- `prompt_only`: use exactly `system_prompt.txt` as the system message. If it includes the schema, do **not** append another copy.
- `append_schema`: append `\n\nSchema:\n` followed by the complete `schema.json` file. Set `required.schema=true`. Always check that resulting prompt matches what the intended model learned.

The manifest's `required.eval` may be false so training can proceed without evaluation. Selecting auto-evaluation requires a populated `eval.jsonl`, regardless of this flag. Declaring true requires eval during ordinary validation.

## train.jsonl and eval.jsonl

Both are UTF-8 JSON Lines: **one JSON object per line**, each with `id`, nonempty `question`, and `expected`. `expected` can be a JSON object, JSON array or string. Extra descriptive metadata is allowed; it is not passed to the model. Do not insert an already-rendered chat prompt into `question`.

```jsonl
{"id":"train_0001","question":"Find datasets about asthma.","expected":{"target":"Dataset","conditions":[{"field":"dataset.subject.label","operator":"equals","value":"asthma","id":"condition1"}],"related":[],"relationshipGraph":null,"conditionBindings":[]}}
{"id":"train_0002","question":"Find datasets about stroke.","expected":{"target":"Dataset","conditions":[{"field":"dataset.subject.label","operator":"equals","value":"stroke","id":"condition1"}],"related":[],"relationshipGraph":null,"conditionBindings":[]}}
```

Store the evaluation questions in **eval.jsonl**, not train.jsonl. Repeating an identical normalized question between splits fails validation. Exact-match scoring is order-sensitive for arrays, including DATMM condition lists; ELI will eventually need its own semantic validator as a separately versioned evaluator.

## Required version package behavior

1. `version create` registers the version without asking for data paths or hyperparameters.
2. `data inspect` checks the package's required inputs.
3. `train` checks the package **before** any training parameters. If inputs are missing, it offers to import prepared files (never generates invented gold data).
4. Training requires explicit per-run model selection, epochs, max sequence tokens, learning rate, accumulation, seed, optimizer/LoRA settings and auto-evaluation decision.
5. A run **freezes every file** under `data/` plus `version.json` with SHA-256. `main` is resolved to a concrete Hugging Face commit SHA in the frozen run.
6. The trainer starts from the selected fresh base model, never silently layers adapters.
7. Evaluate only a successfully trained run with an adapter (or an explicitly requested `--base-only` baseline). A missing eval dataset prevents automatic evaluation from starting.
8. The engine reports per-run results, versions and frozen files in the generated `PROJECT_HANDOFF.md`.

## Current supported capabilities and limits

- Model family: Hugging Face causal language models compatible with Transformers + PEFT LoRA, NVIDIA CUDA for training.
- Evaluation scorer: `exact_json` only, not semantic equivalence. ELI's graph/condition-binding contract scorer is **not part of v0.3.0**.
- Not implemented: QLoRA, distributed training, arbitrary model architecture compatibility, robust VRAM prediction, resume optimizer checkpoints, artifact uploads to S3, non-CUDA training.
- Virtualenv isolation applies to Python dependencies installed by the toolkit, not NVIDIA GPU drivers or external Git executables.

## Copy/paste handoff for the ELI data-authoring conversation

> Please create an ELI v9.0 **data package only**, with ZIP-relative paths `v9.0/data/package.json`, `train.jsonl`, `system_prompt.txt`, `eval.jsonl`, and `schema.json` if needed. Follow `docs/DATA_PACKAGE_CONTRACT.md` (mtt-data-v1). Each JSONL row needs `id`, `question`, and `expected`. Preserve approved historical semantics, explicitly identify compiler-unsupported cases, and keep evaluation questions separate. Set `prompt_assembly=prompt_only` when the system prompt already embeds the schema. The toolkit handles model selection, hyperparameters, training and output packaging; do not include new trainer scripts. Provide hashes, corpus counts, label provenance and a concise README inside the data folder.
