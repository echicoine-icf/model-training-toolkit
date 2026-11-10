# Architecture

MTT separates reusable infrastructure from project-specific **training and evaluation contracts**.

## Components

| Path | Responsibility |
|---|---|
| `toolkit/cli.py` and `workspace.sh` | CLI, project/version/run lifecycle, evaluation and export commands |
| `toolkit/data_package.py` | Validate and freeze version data packages |
| `toolkit/train_engine.py` and `toolkit/backends.py` | Common supervised fine-tuning loop, LoRA setup and optional backend loading |
| `toolkit/checkpoints.py` | Atomic resumable checkpoints with optimizer, model and RNG state |
| `toolkit/eval_engine.py`, `toolkit/evaluation_plugins.py` | Built-in and custom evaluation execution |
| `toolkit/shared_eval.py` | Project-level benchmark discovery, integrity audit and delegated inference |
| `toolkit/workspace_state.py` | Read-only diagnostics and explicit repair of local workspace state |
| `scripts/publish_project_snapshot.py` | Publish source, version inputs and frozen run recipe into a separate Git repository |
| `projects/<project>/versions/<version>/data/` | Authored inputs, not toolkit code |
| `runs/<project>/<version>/<run-id>/` | Frozen inputs, checkpoints, model adapters and provenance |

## Lifecycle

1. Select a base model and immutable revision for a project version.
2. Author `mtt-data-v1` training examples and an application output contract.
3. Validate data and holdout separation.
4. Review model/LoRA/optimizer settings; freeze data, prompt and configuration at run creation.
5. Train from the selected base using assistant-only loss and save resumable checkpoints.
6. Export adapter/merged model as needed for the serving environment.
7. Measure **training recall** separately from **independent generalization** and targeted regression suites.
8. Record evaluator version, data hashes, model identity, prompt and decoding settings with each evaluation.

## State and safety boundaries

- `WORKSPACE_STATE.json`, `.venv`, `runs/`, caches, model weights and downloaded llama.cpp source are machine-local; never publish as framework source.
- The optional project snapshot publisher explicitly exports selected training and benchmark data to a separate destination, but excludes weights. Inspect licensing and privacy before publishing.
- Evaluation code is supplied by the project and **trusted**; MTT doesn't sandbox it.
- A project benchmark's package version, dataset hashes and semantic scorer identity must agree before comparing model scores. Adding cases means a new benchmark package release.
- The built-in exact-JSON scorer is not equivalent to a project's semantic evaluator or compiler integration.

See [Data package](DATA_PACKAGE_CONTRACT.md), [Custom evaluation](CUSTOM_EVALUATION_RUNNERS.md), [Shared benchmarks](SHARED_EVALUATIONS.md), and [Checkpoints](RESUMABLE_CHECKPOINTS.md).
