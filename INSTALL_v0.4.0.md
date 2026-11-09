# Install MTT v0.4.0 — Read-only doctor and safe state repair

This source-only patch builds on MTT **v0.3.9**. It does not touch installed CUDA,
PyTorch, Unsloth, llama.cpp, training data, model weights, checkpoints, or results.
Use the established **source checkout → feature branch/PR → operational pull** flow.

## 1. Apply the patch to the source checkout (Git Bash)

Download `Model_Training_Toolkit_v0.4.0_SOURCE_PATCH.zip` into `~/Downloads`.

```bash
cd ~/model-training-toolkit-repo
git switch main
git pull --ff-only
git switch -c fix/workspace-health-v040

python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
root=Path.cwd().resolve()
archive=Path.home()/'Downloads/Model_Training_Toolkit_v0.4.0_SOURCE_PATCH.zip'
with ZipFile(archive) as z:
    for member in z.infolist():
        if member.is_dir() or member.filename == '.gitignore':
            continue  # preserve existing ignores; merge the two state patterns below
        dest=(root/member.filename).resolve()
        if not dest.is_relative_to(root):
            raise ValueError('Unsafe ZIP member: '+member.filename)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(z.read(member))
ignore=root/'.gitignore'
text=ignore.read_text(encoding='utf-8') if ignore.exists() else ''
for line in ('/WORKSPACE_STATE.json','/.workspace_state_backups/'):
    if line not in text.splitlines():
        text += ('\n' if text and not text.endswith('\n') else '')+line+'\n'
ignore.write_text(text,encoding='utf-8')
print('Applied MTT v0.4.0 source patch and merged .gitignore safely')
PY

python -m unittest discover -s tests -v
git status --short
```

Expected: **94 stdlib/offline tests pass**.

If the state file is still tracked in the *source* repository, untrack it
without deleting it on disk (this needs a separate review before pulling into
an operational checkout that also tracks it):

```bash
git ls-files --error-unmatch WORKSPACE_STATE.json >/dev/null 2>&1 && \
    git rm --cached WORKSPACE_STATE.json
```

Commit only the intended source changes:

```bash
git add .gitignore AGENTS.md CHANGELOG.md README.md RELEASE_MANIFEST.json \
  INSTALL_v0.4.0.md config/workspace.json docs/WORKSPACE_HEALTH.md \
  toolkit/cli.py toolkit/workspace_state.py tests/test_workspace_state.py

git commit -m "Diagnose and safely repair workspace state metadata"
git push -u origin fix/workspace-health-v040
```

Review and merge the PR into `main`.

## 2. Update the operational MTT checkout

```bash
cd ~/model-training-toolkit
# Extra precaution in case an older source commit still tracked this local file:
if [ -f WORKSPACE_STATE.json ]; then
    cp WORKSPACE_STATE.json "$HOME/Downloads/WORKSPACE_STATE.before_v040_pull.json"
fi
git switch main
git pull --ff-only
bash workspace.sh upgrade
bash workspace.sh doctor
```

Expected: `TOOLKIT_CODE_VERSION 0.4.0`, `STATE` with `"status": "ready"`,
`READY`. `upgrade` can recreate missing metadata on an otherwise working
workspace. It will not reinstall or change any ML dependencies.

## 3. Verify the new repair path without changing anything

```bash
bash workspace.sh doctor          # read-only
# Only if doctor reports inconsistent metadata:
bash workspace.sh doctor --repair
```

`doctor --repair` makes a backup of any existing metadata under
`.workspace_state_backups/` and changes only `WORKSPACE_STATE.json`.

Check state remains outside Git:

```bash
git status --short
git check-ignore -v WORKSPACE_STATE.json
```

Do **not** run `run_me_first.sh` or reinstall ML dependencies for a state-file
problem. A truly broken environment will be reported rather than automatically
marked ready.
