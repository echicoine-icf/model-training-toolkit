# MTT — agent operating guide

This file describes stable **safety and repository** rules for coding assistants. Check the installed toolkit version with `bash workspace.sh doctor`, inspect the current CLI before using a flag, and consult the linked docs rather than relying on historical chat handoffs.

## Boundaries

- MTT core (`toolkit/`, `scripts/`, `config/`, `requirements/`, `tests/`, `docs/`) is general-purpose. Project-specific prompts, gold labels, schema rules, scoring semantics and benchmarks belong under `projects/<project>/` or in explicitly attached evaluation packages.
- Do not silently change a model's base revision, LoRA targets, training data, tokenizer/chat template, hyperparameters or benchmark contents. Create a new version/experiment for changed scientific assumptions.
- Never modify or delete completed run snapshots, model weights, checkpoints, frozen datasets, results, `.venv`, caches, or local `WORKSPACE_STATE.json` to fix a source-code issue.
- Distinguish **training recall** from unseen **generalization**. Semantic correctness is project-specific; exact JSON equality is not a substitute. Labeled and unlabeled suites must not be combined into a gold accuracy denominator.
- The historical semantic scorer and versioned benchmark dataset hashes must remain stable for comparable reports. If cases change, publish a new benchmark package; compare models on the same package and scorer identity.
- Project-provided evaluation runners execute trusted Python code, not sandboxed scripts.

## Source changes

Use a feature branch and PR in the **source checkout**, run `python -m unittest discover -s tests -q`, and merge to main before updating an operational checkout with `git pull --ff-only`. Treat `WORKSPACE_STATE.json` as machine-local and Git-ignored. Avoid `git reset --hard`, `git clean`, `git add .` on a workspace containing training artifacts, and force-pushes.

For an installed workspace:

```bash
bash workspace.sh doctor           # read-only
bash workspace.sh doctor --repair  # only metadata repair, with backup
bash workspace.sh upgrade          # refreshes launchers and state; no GPU reinstall
```

Do not run bootstrap or reinstall an existing CUDA environment merely to fix a state inconsistency. Review [Workspace health](docs/WORKSPACE_HEALTH.md).

## Feature references

- [README](README.md): quick start and user-facing overview
- [Architecture](docs/ARCHITECTURE.md): components and state boundaries
- [Data contract](docs/DATA_PACKAGE_CONTRACT.md): `mtt-data-v1` and input validation
- [Training backends](docs/UNSLOTH_BACKEND.md): optional Unsloth and CUDA risks
- [Resumable checkpoints](docs/RESUMABLE_CHECKPOINTS.md): safe resume and integrity
- [Custom evaluation](docs/CUSTOM_EVALUATION_RUNNERS.md): registered scorer interface
- [Shared benchmarks](docs/SHARED_EVALUATIONS.md): suite selection and provenance
- [Project snapshots](docs/PROJECT_SNAPSHOTS.md): publishing selected project assets

## Training and evaluation execution

Training requires explicit authorization and validated data packages. A project version specifies the base model/revision; runs freeze data, prompt, configuration, model metadata and hashes. Model training starts from the specified base unless the explicit resume command is used against a verified checkpoint of the same run.

Use `bash workspace.sh data inspect PROJECT VERSION`, `bash workspace.sh data validate PROJECT VERSION --require-eval` when appropriate, and `bash workspace.sh runs --project PROJECT` to discover run IDs. Do not guess run IDs or trust stale paths.

Use project-specific evaluation instructions to run canonical full benchmarks and targeted suites. `shared-eval --suite` selects specific named subsets (can repeat or comma-separate); the custom `evaluate --suite` interface selects a sequence through the named stage. Use `--dry-run` / `--audit-only` where available before expensive inference. Keep benchmark and model receipts intact during resume.
