# v0.3.4 source-to-deployment update

Maintain the source checkout (`~/model-training-toolkit-repo`) separately from
the operational workspace (`~/model-training-toolkit`). Do not extract a full
release directly over the operational directory. Do not modify `projects/`,
`runs/`, `.venv/`, `tools/llama.cpp/`, `WORKSPACE_STATE.json`, or registry state.

1. Download full release ZIP into Downloads.
2. In the **source checkout**, extract only these source files from the ZIP:
   `toolkit/train_engine.py`, `toolkit/cli.py`, `toolkit/run_wizard.py`,
   `config/workspace.json`, `README.md`, `CHANGELOG.md`,
   `docs/V80_TRAINING_COMPARISON.md`, `tests/test_training_architecture.py`,
   `INSTALL_v0.3.4.md`, `RELEASE_MANIFEST.json`.
3. Run `python -m unittest discover -s tests -v`, commit to a feature branch,
   push and merge through the repo's branch-protection pull-request workflow.
4. In the **operational workspace**, run `git pull --ff-only`,
   `bash workspace.sh upgrade`, and `bash workspace.sh doctor`.
5. Launch a new, explicitly configured fresh-base training run. No existing
   V9.0 data package or previous run is modified.

Important: v0.3.4's intermediate save-every checkpoints do not contain optimizer
state; they are not resumable training checkpoints.
