# MTT v0.3.8 — Shared Project Benchmarks (Non-destructive Git workflow)

MTT v0.3.8 adds a generic `shared-eval` command. This source change installs
**before** adding a shared ELI benchmark to the project. Neither this patch nor
`workspace.sh upgrade` changes frozen training input snapshots, model weights,
checkpoints, existing baseline evaluation files, `projects/eli/versions/`,
`.venv`, or CUDA packages.

## A. Apply source patch in source checkout, then PR

1. Download `Model_Training_Toolkit_v0.3.8_SOURCE_PATCH.zip` into `~/Downloads`.
2. From Git Bash:

```bash
cd ~/model-training-toolkit-repo
git switch main
git pull --ff-only
git switch -c feat/shared-evaluation-v038
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
root=Path.cwd().resolve()
archive=Path.home()/'Downloads/Model_Training_Toolkit_v0.3.8_SOURCE_PATCH.zip'
with ZipFile(archive) as z:
    for item in z.infolist():
        if item.is_dir(): continue
        target=(root/item.filename).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f'Unsafe ZIP member: {item.filename}')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(z.read(item))
print('Source patch applied:',archive)
PY
python -m unittest discover -s tests -v
git status --short
git add AGENTS.md CHANGELOG.md README.md RELEASE_MANIFEST.json \
  INSTALL_v0.3.8.md config/workspace.json docs/SHARED_EVALUATIONS.md \
  toolkit/cli.py toolkit/shared_eval.py tests/test_shared_eval.py
git commit -m "Add project-level shared evaluation benchmarks with provenance checks"
git push -u origin feat/shared-evaluation-v038
```

Open a GitHub pull request, review, and merge into protected `main`.

## B. Update operational checkout only after merging

```bash
cd ~/model-training-toolkit
git switch main
git pull --ff-only
bash workspace.sh upgrade
bash workspace.sh doctor
```

No `pip install`, reinitialization, dependency upgrade, or model merge is required.
The upgrade refreshes `projects/eli/project.sh` so it recognizes `shared-eval`.
If your operational source checkout is configured differently, first confirm
`git status` and the remote before pulling.

## C. Install benchmark v1.0.1 as a separate second step

Download **`ELI_SHARED_BENCHMARK_V1.0.1.zip`**. It contains one top-level
`benchmark-v1.0.1/` directory. Extract that directory beneath:

`~/model-training-toolkit/projects/eli/evaluations/`

For reproducibility, also add the shared benchmark folder to the *source* repo
in its own reviewable commit/PR, then pull it into the operational checkout.
Do not use the earlier v1.0.0 ZIP for new evaluations: v1.0.1 adds package
version and dataset hash provenance required by this workflow.

Use the exact completed V9.2 run:

```bash
cd ~/model-training-toolkit
RUN=run_20261109T002925408171Z_a4a5de
bash projects/eli/project.sh shared-eval v9.2 \
  --run "$RUN" --benchmark benchmark-v1.0.1 --audit-only
bash projects/eli/project.sh shared-eval v9.2 \
  --run "$RUN" --benchmark benchmark-v1.0.1 \
  --model datmm-eli:v9.2-f16 --stage generalization --dry-run
bash projects/eli/project.sh shared-eval v9.2 \
  --run "$RUN" --benchmark benchmark-v1.0.1 \
  --model datmm-eli:v9.2-f16 --stage generalization
```

The first command runs no model inference. The second requires the selected
Ollama model to exist, but does not generate predictions. The third runs the
715 protected holdouts. This does **not** regenerate the finished 24,501-row
training baseline. `--resume` resumes a stopped shared benchmark run and checks
its model digest, prompt, benchmark package, scorer, and dataset hashes.

Run `--stage all --resume` only after generalization is complete.
See `docs/SHARED_EVALUATIONS.md` for release-gate comparison rules.
