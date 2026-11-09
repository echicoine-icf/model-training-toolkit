# MTT v0.3.9 — Targeted Shared Benchmark Suites

This is a **source-only update for MTT v0.3.8**, followed by a separately
versioned **ELI benchmark v1.0.2** install. No CUDA/torch/Unsloth updates,
training reruns, workspace resets, or model merges are necessary.

## 1. Merge MTT code through a Git feature branch

Download `Model_Training_Toolkit_v0.3.9_SOURCE_PATCH.zip` to `~/Downloads`.
From Git Bash:

```bash
cd ~/model-training-toolkit-repo
git switch main
git pull --ff-only
git switch -c feat/targeted-shared-evals-v039
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
root=Path.cwd().resolve()
archive=Path.home()/'Downloads/Model_Training_Toolkit_v0.3.9_SOURCE_PATCH.zip'
with ZipFile(archive) as z:
    for item in z.infolist():
        if item.is_dir(): continue
        path=(root/item.filename).resolve()
        if not path.is_relative_to(root): raise ValueError(item.filename)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(z.read(item))
print('MTT source-only patch applied')
PY
python -m unittest discover -s tests -v
git add AGENTS.md CHANGELOG.md README.md RELEASE_MANIFEST.json \
  INSTALL_v0.3.9.md config/workspace.json docs/SHARED_EVALUATIONS.md \
  toolkit/cli.py toolkit/shared_eval.py tests/test_shared_eval.py
git commit -m "Support targeted and multi-suite shared benchmark evaluation"
git push -u origin feat/targeted-shared-evals-v039
```

Review and merge the pull request to `main`.

## 2. Pull merged source into the operational workspace

```bash
cd ~/model-training-toolkit
git switch main
git pull --ff-only
bash workspace.sh upgrade
bash workspace.sh doctor
```

Do not run `run_me_first.sh`, `pip install`, or touch existing frozen snapshots.

## 3. Install ELI benchmark v1.0.2 separately

Extract `ELI_SHARED_BENCHMARK_V1.0.2.zip` under
`~/model-training-toolkit/projects/eli/evaluations/`, yielding precisely
`projects/eli/evaluations/benchmark-v1.0.2/`. This does not overwrite
`benchmark-v1.0.1` or any existing evaluation results.

Prefer committing the entire new `benchmark-v1.0.2` directory through a
separate reviewable Git pull request and then pulling into the operational
workspace, so the benchmark remains source-controlled and reproducible.

## 4. Safe inspection and targeted testing

```bash
cd ~/model-training-toolkit
RUN=run_20261109T002925408171Z_a4a5de
B=benchmark-v1.0.2
M=datmm-eli:v9.2-f16
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --audit-only
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --list-suites
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --model "$M" --suite v72_graph_26 --dry-run
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --model "$M" --suite v72_graph_26
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --model "$M" --suite v81_corrective_16,v802_labeled_24 --resume
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --model "$M" --stage all --resume
```

V1.0.2's release-gate policy still uses the canonical 715 and focused gold.
Targeted suites are optional non-additive development views. Run `--gate-only`
after full canonical suites complete. The immutable benchmark, suite datasets,
scorer, model digest, prompt SHA-256 and inference settings are recorded per
run. `--resume` is required when extending an existing evaluation run.

The V9.2 24,501-row baseline is separate and preserved; this upgrade does
not start inference or alter the completed baseline.
