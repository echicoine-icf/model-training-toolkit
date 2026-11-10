# Project-level shared evaluation benchmarks

A shared benchmark is a **project-owned, independently versioned collection of evaluation cases and scoring code**. The same benchmark can evaluate multiple trained versions, avoiding repeated copies of historical tests in each model's training package.

MTT's `shared-eval` command audits the project's benchmark and completed run, resolves the frozen system prompt, and delegates inference/scoring to the benchmark's trusted Python runner. A benchmark is optional; the toolkit does **not** ship ELI-specific evaluators or trained models.

## Layout and contract

```text
projects/<project>/evaluations/<benchmark-release>/
  benchmark_manifest.json    # benchmark package version, datasets and SHA-256 hashes
  cases/                     # immutable labeled or unlabeled cases
  scorers/                   # application-specific semantic scoring
  audit_shared.py            # package and overlap audit
  evaluate_shared_ollama.py  # example project runner, if supplied
  gate_report.py             # release criteria, if supplied
```

The exact files are defined by the project benchmark. Project scripts are **trusted code**, not sandboxed. Do not use a benchmark against an unfinished training run, and do not train on its protected holdouts.

## Discover, audit and run

```bash
# Example for a project with an installed benchmark and matching Ollama model
bash projects/PROJECT/project.sh shared-eval VERSION \
  --run RUN_ID --benchmark BENCHMARK --audit-only

bash projects/PROJECT/project.sh shared-eval VERSION \
  --run RUN_ID --benchmark BENCHMARK --list-suites

# One named suite
bash projects/PROJECT/project.sh shared-eval VERSION \
  --run RUN_ID --benchmark BENCHMARK --model MODEL_TAG \
  --suite SUITE_NAME --dry-run

# Two specific suites, in one selection
bash projects/PROJECT/project.sh shared-eval VERSION \
  --run RUN_ID --benchmark BENCHMARK --model MODEL_TAG \
  --suite SUITE_A,SUITE_B --resume

# The whole canonical benchmark
bash projects/PROJECT/project.sh shared-eval VERSION \
  --run RUN_ID --benchmark BENCHMARK --model MODEL_TAG \
  --stage all --resume
```

`--stage` and `--suite` are mutually exclusive. Named suites can be supplied as comma-separated values or repeated flags. `--list-suites`, `--audit-only` and `--dry-run` do not perform the full inference. Use `--resume` to reuse only compatible completed predictions; never mix identities or force results into a different model's output directory.

**Note:** For the `shared-eval` command, `--suite` selects actual named subsets. The *separate* custom `evaluate` pipeline uses `--suite` for ordered stages instead.

## Scores and comparability

Each result belongs to a particular model/run, prompt, decoding configuration, scorer and **benchmark package version**. Record exact dataset SHA-256 for each suite and manifest/hash identity in report provenance. Adding cases creates a **new benchmark package**, even when the semantic scoring contract remains unchanged. Rerun both models on the same release before claiming an improvement.

- Training recall is version-specific and **not part of independent generalization**.
- Labeled holdouts provide semantic accuracy; unlabeled diagnostics can be inspected but **must not** inflate accuracy.
- A targeted suite score is diagnostic. Full release gates should use the designated canonical holdout and verified complete results, not sum overlapping slices.
- A project's gate policy may declare per-category thresholds and critical-case nonregression rules; values are not universal toolkit defaults.

Inference provider, canonical suite composition, scoring rules, and integration with an external compiler are **project responsibilities**; the generic toolkit does not guarantee any particular domain semantics.
