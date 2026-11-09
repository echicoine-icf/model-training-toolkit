# Install the MTT project snapshot publisher

This is a source-only feature patch on top of **MTT v0.4.0**. It does not
upgrade your Python packages, overwrite the active ELI workspace, change Git
remotes, or modify `WORKSPACE_STATE.json`.

## 1. Install the feature through the MTT source repository

Download `MTT_PROJECT_SNAPSHOT_PUBLISHER_PATCH.zip` to your Windows Downloads
folder, then open Git Bash:

```bash
cd ~/model-training-toolkit-repo

git switch main
git pull --ff-only
git switch -c feat/project-snapshot-publisher

python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
root = Path.cwd().resolve()
archive = Path.home() / 'Downloads/MTT_PROJECT_SNAPSHOT_PUBLISHER_PATCH.zip'
with ZipFile(archive) as z:
    for item in z.infolist():
        if item.is_dir():
            continue
        target = (root / item.filename).resolve()
        if not target.is_relative_to(root):
            raise ValueError(item.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(item))
print('MTT project snapshot publisher installed in source checkout')
PY

python -m unittest discover -s tests -v

git add scripts/publish_project_snapshot.py docs/PROJECT_SNAPSHOTS.md \
  tests/test_project_snapshot_publisher.py INSTALL_PROJECT_SNAPSHOT_PUBLISHER.md \
  README.md CHANGELOG.md

git commit -m "Add reusable project source and training snapshot publisher"
git push -u origin feat/project-snapshot-publisher
```

Open and merge the pull request into the *original MTT repository*.

## 2. Update the local operational MTT copy

```bash
cd ~/model-training-toolkit
git switch main
git pull --ff-only
# No workspace.sh upgrade or pip install is needed for this source-only script.
python scripts/publish_project_snapshot.py --help
```

The originating MTT repository `origin` must still point to the general MTT
repository. This script NEVER changes that remote.

## 3. Preview, then publish ELI V9.2 to a separate GitHub repository

First verify on GitHub whether
`https://github.com/echicoine-icf/datmm-model-training-toolkit` is private or
public. Private is recommended if the training/evaluation material has any
restriction on redistribution.

```bash
cd ~/model-training-toolkit

REPO=https://github.com/echicoine-icf/datmm-model-training-toolkit.git
RUN=run_20261109T002925408171Z_a4a5de
EVAL="$HOME/Downloads/ELI_V92_MTT_CUSTOM_EVALUATION_CORRECTED"

# Read-only preflight: enumerates files, validates frozen hashes, no Git push.
python scripts/publish_project_snapshot.py \
  --repo-url "$REPO" --project eli --version v9.2 --run "$RUN" \
  --benchmark benchmark-v1.0.2 --external-eval "$EVAL"

# Push to a GitHub repository you have verified is PRIVATE:
python scripts/publish_project_snapshot.py \
  --repo-url "$REPO" --project eli --version v9.2 --run "$RUN" \
  --benchmark benchmark-v1.0.2 --external-eval "$EVAL" \
  --push --visibility private
```

If the destination is **public**, instead use `--push --visibility public
--allow-public`, but only after reviewing your training/evaluation data for
privacy and licensing. The visibility flag acknowledges your manual check; the
script cannot independently verify the remote repository's privacy status.

If you don't have `ELI_V92_MTT_CUSTOM_EVALUATION_CORRECTED` anymore, you can
omit `--external-eval` from both calls; the run's frozen train/eval data and
shared benchmark v1.0.2 will still be included.

**Never** force-push, delete or copy `runs/`, or modify your original Git remote.
Model weights are intentionally not published. See `docs/PROJECT_SNAPSHOTS.md`
for exactly what is included and how to retrain from the frozen recipe.
