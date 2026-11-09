# Project-level shared evaluation benchmarks (MTT v0.3.8)

`shared-eval` is a generic bridge for trusted, project-owned evaluation code.
MTT does not load ELI dataset JSON or implement the V7 graph-aware scorer.

Layout:

```
projects/eli/evaluations/benchmark-v1.0.1/
  benchmark_manifest.json  # package_version + frozen_files_sha256 + suites
  cases/                    # immutable
  scorers/                  # historical V7 scorer
  audit_shared.py
  evaluate_shared_ollama.py
  gate_report.py
```

The command uses `runs/eli/<VERSION>/<RUN_ID>/input/data/system_prompt.txt` and
`train.jsonl`, after verifying the snapshot. It requires the project to be
registered and the run completed. It audits protected holdouts against the
run snapshot before any inference. Project scripts are trusted local Python:
review them before installation.

```bash
cd ~/model-training-toolkit
RUN=run_20261109T002925408171Z_a4a5de
BENCH=benchmark-v1.0.1

bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" \
  --benchmark "$BENCH" --audit-only
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" \
  --benchmark "$BENCH" --model datmm-eli:v9.2-f16 \
  --stage generalization --dry-run
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" \
  --benchmark "$BENCH" --model datmm-eli:v9.2-f16 \
  --stage generalization
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" \
  --benchmark "$BENCH" --model datmm-eli:v9.2-f16 \
  --stage all --resume
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" \
  --benchmark "$BENCH" --model datmm-eli:v9.2-f16 \
  --gate-only
```

Generated results are isolated under
`runs/eli/v9.2/<run>/shared_evaluations/benchmark-v1.0.1/<model-tag>/`.
The evaluator stores an immutable `run_config.json` containing the benchmark
**package version**, manifest SHA-256, and **SHA-256 of every case suite**.
Every `summary.json`, `report.md`, `suite_complete.json`, and
`evaluation_provenance.json` records the evaluated suite's package version and
SHA-256. Aggregate release gates include the full suite hash map. Any mismatch
stops resume or cross-model comparison; a new lesson requires a NEW benchmark
package (such as `benchmark-v1.1.0`), never changing files in place.

The V9.2 24,501-row training recall is a separate, version-specific baseline.
Its output remains untouched. Unlabeled suite diagnostics do not contribute to
accuracy. The first V1.0.1 gate policy is provisional, and compiler execution
is not part of this fast Ollama runner.
