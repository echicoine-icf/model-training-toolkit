# Publish a reproducible project snapshot to a separate Git repository

The `scripts/publish_project_snapshot.py` utility creates a standalone **source and
training-input** snapshot in another Git repository. It leaves the operational
MTT checkout, its `origin`, its `.gitignore`, its branch/index, its weights, and
its runs unchanged. You can continue pulling framework updates from the
original MTT project afterward.

It does **not** create a GitHub repository: create a blank repository first. Set
it to **private** if training questions, patient-related examples, or evaluation
materials have any distribution restrictions. Review data licensing and the
contents before publishing; the tool checks common credential signatures but
is not a comprehensive security scanner. Confirm repository visibility on GitHub.

## What is exported

- MTT code, configuration, requirements, CLI, utilities, tests, and docs from
  the current clean Git checkout.
- `projects/PROJECT/project.json` and `project.sh` when present, plus
  `projects/PROJECT/versions/VERSION/` including ignored `data/*.jsonl`.
- Explicitly selected versioned benchmarks under
  `projects/PROJECT/evaluations/BENCHMARK/` (including ignored files).
- The **selected run's frozen `input/`**, with `input_hashes.json`, under
  `rebuild_recipes/PROJECT/VERSION/RUN/input/`, plus `run.json` and
  `training_plan.json` if available. Frozen hashes must all validate before
  publication. Use its `version.json` when reproducing the exact run settings.
- Optional extra evaluation definitions/cases from `--external-eval DIRECTORY`,
  stored under `projects/PROJECT/evaluations/external/NAME/`. Prediction logs and
  the evaluator's `ollama_results/` are excluded.
- A per-run SHA-256 snapshot receipt at
  `snapshots/PROJECT/VERSION/RUN/manifest.json`, with selected benchmark names,
  source commit, Python/OS and installed ML package versions, file sizes and digests.

Excluded by design: `.venv`, package caches, secrets/credentials, live
workspace state, `runs/` model weights and optimizer checkpoints, merged models,
GGUF, evaluation predictions, and other versions/projects. No GPU dependencies
are reinstalled on the originating machine.

A rebuild downloads the pinned pretrained base model and retrains; it will not
necessarily reproduce bit-identical weights or exactly the same scores across
hardware/library versions. `requirements/` and the run recipe document the
training environment; include a `pip freeze`/CUDA details separately if exact
environment recreation is critical, after screening for sensitive paths.

## Usage — ELI V9.2 example (Git Bash on Windows)

From the **operational** MTT workspace after installing this script through a
source-code PR:

```bash
cd ~/model-training-toolkit
source .venv/Scripts/activate

REPO=https://github.com/echicoine-icf/datmm-model-training-toolkit.git
RUN=run_20261109T002925408171Z_a4a5de
EVAL="$HOME/Downloads/ELI_V92_MTT_CUSTOM_EVALUATION_CORRECTED"

# Read-only preview; checks frozen run hashes and enumerates included files.
python scripts/publish_project_snapshot.py \
  --repo-url "$REPO" --project eli --version v9.2 --run "$RUN" \
  --benchmark benchmark-v1.0.2 --external-eval "$EVAL"

# After reviewing the preview and checking GitHub repository privacy:
python scripts/publish_project_snapshot.py \
  --repo-url "$REPO" --project eli --version v9.2 --run "$RUN" \
  --benchmark benchmark-v1.0.2 --external-eval "$EVAL" \
  --push --visibility private
```

If the target is **public**, you must explicitly pass `--visibility public
--allow-public`. The visibility flag is an acknowledgement, not a remote
verification; confirm actual GitHub settings yourself.

`--benchmark` can be repeated to include multiple benchmark releases. The
selected benchmark must already exist at
`projects/eli/evaluations/benchmark-v1.0.2/`. If the corrected external evaluator
is not installed or you don't need to preserve it, omit `--external-eval` from
**both** commands. Source weights/checkpoints are never exported.

A remote GitHub repository with no commits is supported. An existing GitHub
repository is cloned, then updated without force-pushing. Already published
version inputs, benchmark files and run recipes cannot be silently changed:
publish an additional version instead. The script will stop on source code
changes not committed in the originating MTT working tree.

## Recovering on a different PC

```bash
git clone https://github.com/echicoine-icf/datmm-model-training-toolkit.git
cd datmm-model-training-toolkit
bash run_me_first.sh --install-ml
bash workspace.sh doctor
```

Review the selected `rebuild_recipes/eli/v9.2/RUN/input/version.json` (including
pinned Qwen model revision, optimizer/training settings) and
`training_plan.json` before retraining. The `projects/eli/versions/v9.2/`
package contains the original training data and evaluation inputs. Compare
against the frozen snapshot if the live version changed; prefer the frozen
snapshot as the authoritative recipe. Reapply run-specific training overrides.
Verify shared benchmark hashes with `shared-eval --audit-only` after restoring
the required benchmark. Train with the appropriate `projects/eli/project.sh
train v9.2` flags or guided workflow. Training a new run never overwrites the
archived inputs.

For a full **trained model** backup in addition to reproducible inputs, separately
archive the adapter, checkpoint and GGUF/SafeTensors outputs to secure artifact
storage (e.g. private S3). They are intentionally excluded from ordinary Git.

## Safety and failure modes

- The exporter neither modifies Git remotes nor runs `git checkout`, `git
  reset`, or `git push --force` in the source checkout. A temporary clone does
  all staging and pushing; failures leave the source intact.
- GitHub's ordinary Git object size limit applies. Files over 90 MiB fail
  preflight rather than relying on GitHub to reject them late.
- Files containing obvious credentials fail preflight; `.env`, `*.pem`, key
  files, caches, weights and locally generated results are excluded.
- New snapshots are additive. Changes to already-published version data,
  benchmarks or frozen run recipes are refused. Pushes are not forced; if the
  remote advanced while exporting, rerun against its updated state.
- The snapshot is a source/data repository, not a copy of the originating
  repository's entire Git history. The source commit SHA is recorded.

## Refreshing MTT code in a repository that already has a model snapshot

The initial snapshot is **immutable**. Running the original publisher with the
same `--run` after the MTT source commit changes would fail because its
`manifest.json` records the original source commit and file hashes. It also
would not remove deleted toolkit files from the destination.

Use **`--sync-toolkit`** for framework-only updates instead. First merge and pull
any source changes you want to publish (for example, removing obsolete
`INSTALL_*.md` handoff documents from the MTT main branch). Then, from the
original operational MTT checkout:

```bash
cd ~/model-training-toolkit
REPO=https://github.com/echicoine-icf/datmm-model-training-toolkit.git
RUN=run_20261109T002925408171Z_a4a5de

# Local dry run: lists source files without contacting/changing the destination.
python scripts/publish_project_snapshot.py \
  --repo-url "$REPO" --project eli --version v9.2 --run "$RUN" \
  --sync-toolkit

# Sync only the latest committed toolkit source (no model/data modifications).
python scripts/publish_project_snapshot.py \
  --repo-url "$REPO" --project eli --version v9.2 --run "$RUN" \
  --sync-toolkit --push --visibility private
```

This mode updates tracked toolkit source, adds new toolkit files, and removes
obsolete toolkit files **only when they match the last published hashes**.
It refuses to overwrite destination-only edits or collide with existing files.
It never writes to `projects/`, `rebuild_recipes/`, or the original snapshot
receipt. A new immutable source-refresh receipt is written under:

`snaps​hots/PROJECT/VERSION/RUN/toolkit_refreshes/SOURCE_COMMIT.json`

and `toolkit_refreshes/latest.json` points at the most recent receipt. The
original model snapshot receipt remains as a historical record. Ordinary
GitHub pulls in the *operational* source checkout continue to use their usual
`origin` remote; publishing uses a separate temporary clone.

The target must already contain a snapshot published using the original
publisher. If it is still empty, use the standard initial publish command
(including `--benchmark`) first, and then use `--sync-toolkit` for subsequent
framework updates. The tool does not change repository visibility; you must
confirm privacy in GitHub before publishing training data.
