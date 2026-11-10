# Data package contract (`mtt-data-v1`)

A project author supplies the examples, expected assistant responses, and system prompt. The toolkit validates inputs and handles training. The **application** owns the meaning of a correct output.

## Files

```text
projects/<project>/versions/<version>/data/
  package.json         # required
  train.jsonl          # required
  system_prompt.txt    # required
  eval.jsonl           # optional unless required by your evaluation workflow
  schema.json          # optional unless append_schema is selected
```

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
  "required": {"train": true, "system_prompt": true, "eval": false, "schema": false},
  "prompt_assembly": "prompt_only",
  "evaluation": {"scorer": "exact_json", "max_new_tokens": 384},
  "description": "Describe the task, labels and data provenance."
}
```

`prompt_assembly` must match training and inference: `prompt_only` uses the system prompt as-is; `append_schema` appends the schema and requires `schema.json`. Do not embed the schema twice.

## JSONL example

Each row is a separate UTF-8 JSON object with a unique `id`, nonblank `question`, and `expected` (JSON object, list or string):

```jsonl
{"id":"train_001","question":"Add buying groceries to my list.","expected":{"title":"Buy groceries","priority":"medium","due_date":null}}
{"id":"train_002","question":"Make the report urgent.","expected":{"title":"Complete report","priority":"high","due_date":null}}
```

These examples are illustrative. Your schema may require additional fields. `question` is the user's request, not a pre-rendered chat template. A dataset can also supply approved chat messages; the trainer masks system and user tokens from the assistant-response loss when that format is used. Keep evaluation examples separate from training and avoid duplicate normalized questions across splits.

## Validation and freezing

```bash
bash workspace.sh data inspect PROJECT VERSION
bash workspace.sh data validate PROJECT VERSION --require-eval
```

`--require-eval` is appropriate when that evaluation input is needed; otherwise a data package can be validated for training without a local eval file. Training freezes version data, settings and model revision under a distinct `runs/` directory with hashes. It does not rewrite a completed run's inputs or invent labels.

## Scoring boundaries

- The built-in `exact_json` scorer is a basic structural/value comparison, not domain-aware semantic equivalence.
- Custom Python scorers can implement project-defined matching rules and diagnose malformed model outputs; see [Custom evaluation runners](CUSTOM_EVALUATION_RUNNERS.md).
- Shared benchmark packages are project-level assets with independent package versions, frozen cases and hashes; see [Shared evaluations](SHARED_EVALUATIONS.md).
- **Training recall** on known examples and **blind generalization** on held-out examples are different metrics. Do not combine them into one accuracy claim.

## Author checklist

Document your application's schema/contract; example counts and provenance; how relative dates and missing optional values are handled; holdout definitions; prompt/schema hashes; expected response format; known ambiguities; and the intended evaluator. The toolkit's data contract describes packaging, not application semantics.
