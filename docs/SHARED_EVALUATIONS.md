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

## Targeted suites (MTT v0.3.9; ELI benchmark v1.0.2)

The benchmark is immutable. Choose `benchmark-v1.0.2` for individually
addressable V7 513, V7.2 graph 26, V8.1 corrective 16, and V9 compositional
160 cases. The original 715 remains the authoritative release benchmark.
The toolkit delegates selection to the trusted project runner; it never
rewrites test cases or the historical scorer.

```bash
RUN=run_20261109T002925408171Z_a4a5de
B=benchmark-v1.0.2
M=datmm-eli:v9.2-f16
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --list-suites
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --model "$M" --suite v72_graph_26 --dry-run
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --model "$M" --suite v72_graph_26
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --model "$M" --suite v81_corrective_16,v802_labeled_24 --resume
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" --benchmark "$B" --model "$M" --stage all --resume
```

`--suite NAME` is repeatable and accepts comma-separated names. `--stage`
remains cumulative. `--stage` and `--suite` are mutually exclusive; `--suite
all` runs the six canonical suites without duplicate targeted splits. The
benchmark runner validates every name against its immutable manifest, and
reports exact suite count, mode, model digest, prompt SHA-256, per-suite
SHA-256 and package version. Targeted views are not extra independent gold
cases and never increase the 715 denominator. Completed canonical predictions
can be reused for targeted views and all four completed targeted views can
reconstruct the full 715, using verified completion receipts. No existing
v1.0.1 outputs or version-specific training baselines are modified.

`--gate-only` continues to require the canonical original 715 and focused
regressions, regardless of which development subsets ran. Gate policy is
provisional. `--list-suites` requires a completed run for snapshot audit but
never connects to Ollama.
