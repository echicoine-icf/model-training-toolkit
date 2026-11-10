# Publishing reproducible project snapshots

`python scripts/publish_project_snapshot.py` copies the **current committed MTT framework source** plus explicitly selected project/version data into a **separate Git repository**. It uses a temporary checkout, not a change to your operational repository's `origin`, `.gitignore`, branch or index.

This is for reconstructing a training run from its inputs and recipe. It is **not** a backup of a trained model's weights.

## Published assets

- Toolkit source and tests from a clean, committed checkout.
- Selected project's version configuration and authored training/evaluation data (including intentionally ignored JSONL datasets).
- Selected project benchmark releases.
- Chosen completed run's immutable frozen input snapshot and training recipe.
- Optional external evaluation source/cases (not prediction output).
- SHA-256 snapshot receipt recording source commit, copied file hashes, project/version/run and environment details.

Excluded: weights, adapters, GGUF, checkpoints, `.venv`, caches, credential files, `WORKSPACE_STATE.json`, prediction logs, and unrelated projects/versions. Back up weights to separate secure artifact storage if you need to **use the already-trained model** rather than retrain it.

Publishing training prompts or evaluations may expose protected data. **Confirm rights and destination repository visibility before pushing**. The publisher's visibility flag is an acknowledgement; it does not verify remote GitHub settings.

## Publish a new project snapshot

Create an **empty destination Git repository** first. From your operational MTT checkout, with a completed run and a clean source Git state:

```bash
python scripts/publish_project_snapshot.py \
  --repo-url 'https://github.com/OWNER/PROJECT-SNAPSHOT.git' \
  --project PROJECT --version VERSION --run RUN_ID \
  --benchmark BENCHMARK_RELEASE

# After checking the preview and confirming the destination is private:
python scripts/publish_project_snapshot.py \
  --repo-url 'https://github.com/OWNER/PROJECT-SNAPSHOT.git' \
  --project PROJECT --version VERSION --run RUN_ID \
  --benchmark BENCHMARK_RELEASE \
  --push --visibility private
```

`--benchmark` can be repeated. If external evaluator definitions are needed, supply `--external-eval PATH` for the same path in both commands. The exporter does not automatically include all local evaluation results or every dataset it finds on disk.

For a public destination, the publisher requires explicit `--visibility public --allow-public`; first review the data and its distribution restrictions. No operation force-pushes Git.

## Update toolkit source in an existing snapshot repository

Once a project snapshot exists, use **`--sync-toolkit`** to refresh *framework source only*, leaving frozen training inputs, benchmarks and the original receipt unchanged:

```bash
python scripts/publish_project_snapshot.py \
  --repo-url 'https://github.com/OWNER/PROJECT-SNAPSHOT.git' \
  --project PROJECT --version VERSION --run RUN_ID \
  --sync-toolkit

python scripts/publish_project_snapshot.py \
  --repo-url 'https://github.com/OWNER/PROJECT-SNAPSHOT.git' \
  --project PROJECT --version VERSION --run RUN_ID \
  --sync-toolkit --push --visibility private
```

The publisher compares source/destination Git contents and records a distinct toolkit refresh receipt. It guards against overwriting independent remote changes and accounts for Git's CRLF/LF normalization on Windows. It cannot change previously published training inputs. The source must first be committed and merged/pulled into the originating workspace.

## Rebuild on another machine

1. Clone the snapshot repository and initialize MTT with the installation instructions in its README.
2. Review the frozen `rebuild_recipes/PROJECT/VERSION/RUN_ID/input/`, `training_plan.json` and model revision before starting training.
3. Validate package data and required holdout/benchmark SHA-256 hashes.
4. Supply compatible GPU dependencies and training flags as recorded in the run recipe. A new training run does not restore identical optimizer state or promise identical weights across environments.

Do not overwrite an existing frozen run snapshot or change benchmark datasets in place. New benchmark lessons belong in a new benchmark package version.
