# Workspace Health and Recovery — MTT v0.4.0

## Commands

```bash
cd ~/model-training-toolkit
bash workspace.sh doctor           # read-only; returns nonzero on unhealthy state
bash workspace.sh doctor --repair  # explicit, metadata-only repair
bash workspace.sh upgrade          # upgrade project launchers, then refresh state
```

`doctor` checks the installed virtualenv (`.venv/pyvenv.cfg` and Python
interpreter) and required `projects/`, `runs/` and `registry/` directories. It
independently reads `config/workspace.json`, then checks whether
`WORKSPACE_STATE.json` is missing, corrupt, stale, or not marked ready. It
prints the current runtime details but **never changes files by default**.

If the environment is healthy but the metadata is missing or invalid,
`doctor` prints `WORKSPACE_STATE_REPAIR_AVAILABLE`. Run `doctor --repair` to
correct metadata; the command preserves any other state fields (such as
original installation timestamp, pinned llama.cpp revision, and environment
information) and writes atomically. Existing JSON (including corrupt JSON)
is backed up under `.workspace_state_backups/`. If a required component is
missing, **repair fails without marking the state ready**.

`upgrade` validates the environment first, performs its existing non-destructive
project-launcher/data-contract operations, then refreshes `toolkit_version`,
`status: ready`, and `last_upgrade_utc`. It is also able to recreate a missing
state file without reinitialization when filesystem prerequisites are valid.

These operations do not install packages or modify `.venv`, models, weights,
training datasets, frozen run inputs, checkpoints, or evaluation results. A
`ready` metadata flag is not a substitute for training/serving-specific tests.

## Git hygiene

`.gitignore` must include these root-level patterns:

```gitignore
/WORKSPACE_STATE.json
/.workspace_state_backups/
```

If state was already tracked, `.gitignore` cannot remove it from Git's index.
Use a dedicated reviewed change in the **source checkout**:

```bash
git rm --cached WORKSPACE_STATE.json  # leaves local file on disk
git add .gitignore
```

**Before pulling a commit which removes a previously tracked state file into a
second operational checkout**, back up that checkout's local state file first.
A pull may delete the tracked copy. Starting with v0.4.0, `doctor --repair`
can recreate it if the local environment is healthy.

Avoid `git reset --hard`, `git clean`, reinitialization, and reinstalling
GPU dependencies to fix a state-file inconsistency.
