# Custom evaluation runners

This is an **additive** feature. Without an evaluation registry, MTT's historical exact-JSON evaluator and training/checkpoint behavior are unchanged. Custom evaluation uses trusted project Python scripts; it does **not** sandbox arbitrary code.

## Configuration: `mtt-eval-runners-v1`

Put `evaluation/runners.json` and Python scripts in a version's `data/` folder **before training**, or build an external evaluation package in a separate directory to attach after a run completes. This second option is useful when attaching an evaluator to an existing completed run whose frozen training snapshot must not change.

```text
projects/<project>/versions/<version>/data/
  package.json
  train.jsonl
  eval.jsonl
  system_prompt.txt
  schema.json
  evaluation/
    runners.json
    scorer.py
    challenge.jsonl
```

Example `evaluation/runners.json`:

```json
{
  "format": "mtt-eval-runners-v1",
  "generation": {"precision": "bf16", "max_new_tokens": 768},
  "runners": [{
    "name": "project_scorer",
    "description": "Project-supplied semantic scorer",
    "script": "evaluation/scorer.py",
    "stages": ["baseline", "generalization", "diagnostic"],
    "datasets": ["train.jsonl", "eval.jsonl", "evaluation/challenge.jsonl"],
    "model_inputs": ["adapter", "merged", "checkpoint"],
    "output_dir": "project_scorer"
  }],
  "stages": [
    {"name": "baseline", "suites": [
      {"name": "training_all", "runner": "project_scorer", "dataset": "train.jsonl", "mode": "gold"}
    ]},
    {"name": "generalization", "suites": [
      {"name": "frozen", "runner": "project_scorer", "dataset": "eval.jsonl", "mode": "gold"}
    ]},
    {"name": "diagnostic", "suites": [
      {"name": "challenge", "runner": "project_scorer", "dataset": "evaluation/challenge.jsonl", "mode": "diagnostic"}
    ]}
  ]
}
```

Set optional `expected_count` on any suite and optional top-level `data_sha256` mapping of relative file paths to expected SHA-256. If specified, mismatches stop evaluation before inference. The runner's `datasets` list is checked at registry discovery, as are scripts, requested model types, supported stages, names and file paths.

Stage names and order are project-defined. For this **custom `evaluate` command**, `--suite NAME` executes stages **through** NAME, preserving baseline-first ordering. This is distinct from `shared-eval --suite`, which selects named individual suites. `--suite all` executes the complete sequence. `--runner NAME` restricts execution to one registered runner; `--runner custom` means all registered runners.

## Scorer subprocess contract

MTT owns model loading and inference. A registered scorer is invoked with:

```text
python evaluation/scorer.py --cases FROZEN_CASES.jsonl \
  --predictions PREDICTIONS.jsonl --output OUTPUT_DIRECTORY \
  --context OUTPUT_DIRECTORY/runner_context.json
```

The scorer must write the following files in its output directory:

- `summary.json` — arbitrary machine-readable metrics, including suite and category breakdowns
- `failures.jsonl` — one JSON object per failure, or an empty file if no failures
- `report.md` — explanatory report

MTT generates `predictions.jsonl` with case ID, question, raw model output, parsed JSON (when possible), expected gold if labeled, and timing. Do not rewrite this file. The context JSON contains `stage`, `suite`, `mode` (`gold` or `diagnostic`), model identity, `data_root`, optional `evaluation_package_root`, and runner metadata. Unlabeled diagnostics must not be reported as gold accuracy. Historical V7+ scoring, compiler checks and other domain-specific criteria belong in these scripts, not in the MTT core.

## Model selection (read only)

- `--model adapter` — finished adapter in `runs/PROJECT/VERSION/RUN_ID/adapter`
- `--model checkpoint --checkpoint step_0002000` — verified complete training checkpoint (for selected run)
- `--model merged --model-path PATH` — merged Hugging Face directory containing config and safetensors
- `--model PATH` — detect an adapter, merged model, or selected-run checkpoint by files

A shared model loader opens the model for inference with its appropriate tokenizer; it never performs a merge or writes to weights. Input hashes, model weights/checkpoint marker, registry/scripts, generation configuration and run model identity are recorded in the evaluation manifest.

## Commands (Windows Git Bash)

```bash
# Frozen registry from before training
bash projects/PROJECT/project.sh evaluate VERSION --run RUN_ID --runner custom --suite baseline --dry-run
bash projects/PROJECT/project.sh evaluate VERSION --run RUN_ID --runner custom --suite baseline
bash projects/PROJECT/project.sh evaluate VERSION --run RUN_ID --runner custom --suite all --resume

# For an already-trained run whose original snapshot lacks the registry:
bash projects/PROJECT/project.sh evaluate VERSION --run RUN_ID \
  --runner custom --suite baseline --eval-package /path/to/project-evaluation-package --dry-run
bash projects/PROJECT/project.sh evaluate VERSION --run RUN_ID \
  --runner custom --suite baseline --eval-package /path/to/project-evaluation-package
bash projects/PROJECT/project.sh evaluate VERSION --run RUN_ID \
  --runner custom --suite all --resume
```

The first non-dry external evaluation copies the standalone package into `runs/.../evaluation_packages/<digest>/`. It does **not** modify training inputs, checkpoints, or LoRA weights. Later `--resume` automatically selects the one frozen package, or requires `--eval-package` when multiple exist.

Each evaluation identity has its own `runs/.../evaluations_custom/<fingerprint>/` directory. Existing results are never overwritten by default. `--resume` skips verified completed predictions and scoring; interrupted prediction writes have recoverable tails and incomplete scorer outputs are backed up and rerun. Results are organized as `<fingerprint>/<runner-output-dir>/<stage>/<suite>/` with the four standard files and a top-level `summary.json`.

`--auto-eval` can automatically invoke this pipeline only when `data/evaluation/runners.json` **was already included before the run snapshot**. An evaluation package attached afterward must be invoked separately. No toolkit upgrade or evaluator package should be installed during an active training process.

## Integrity, limitations and safety

- Run evaluations only after training has stopped; CLI rejects actively training runs.
- The frozen package and model are checked by SHA-256. Checkpoint evaluation checks `checkpoint_info.json` and its artifact manifests first.
- Registered scripts are trusted executable Python, not isolated sandboxes. Audit scripts before enabling them.
- Generic MTT inference uses deterministic `do_sample=False`; custom scoring doesn't imply execution against domain-specific external services.
- `--resume` requires the same frozen model, evaluator and data. Deliberate changes create a separate evaluation identity, never a mixed score.
- The checkpoint and merged-model loader has been unit-tested with mock model files; full GPU validation with specific model versions remains necessary.
- Missing output (e.g. scorer crashes) causes an evaluation failure, not model deletion or retraining.

## Related files

Project-specific semantic scorers and their test datasets are not included in MTT core. Review evaluator licensing, trusted Python execution, and data privacy before publishing.
