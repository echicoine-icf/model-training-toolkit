# MTT v0.3.7 — Installing custom evaluation runners

This source-only update supports generic project evaluation scripts without changing training engines, checkpoints, or ELI datasets.

1. Download `Model_Training_Toolkit_v0.3.7_SOURCE_PATCH.zip` to Downloads.
2. Apply its contents **only** in the clean `~/model-training-toolkit-repo` source repository on a feature branch. Do not extract into the operational workspace.
3. Run `python -m unittest discover -s tests -v` in the clean repository; review the changed files; commit and push a PR to GitHub.
4. **Only after current training has stopped**, merge PR and update the operational workspace:

```bash
cd ~/model-training-toolkit
git pull --ff-only
bash workspace.sh upgrade
bash workspace.sh doctor
```

Do not run `run_me_first.sh`, reinstall PyTorch, or reinstall Unsloth for this code-only update. v0.3.7 uses only stdlib dependencies for runner discovery; actual evaluation continues to use the existing ML environment.

For existing training runs, use `--eval-package PATH_TO_EXTRACTED_EVAL_DIR` to freeze a separate project evaluator. For future versions, include `data/evaluation/runners.json` and scripts alongside training data **before** creating the training run. See `docs/CUSTOM_EVALUATION_RUNNERS.md`.

This release does **not** include the user-specific ELI evaluation assets in the public source repository. A separate reference ZIP is available. Full GPU checkpoint/adapter/merged-model evaluation was not executed in the release environment.
