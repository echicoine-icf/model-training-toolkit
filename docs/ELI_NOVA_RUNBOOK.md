# ELI 10 — Nova 2 Lite remote SFT handoff

This runbook supplements `docs/BEDROCK_NOVA.md`; keep the exact DATMM production system prompt and application/compiler authority unchanged.

## Inputs to freeze

- `projects/eli/versions/v10/data/system_prompt.txt` contains the **complete** production prompt including the DATMM RDF schema. The validator compares the text exactly against each Bedrock JSONL `system` block.
- Prepared `ELI10_NOVA2_SFT_TRAIN.jsonl` is supplied by path and copied byte-for-byte into the new `runs/eli/v10/<run>/input/data/bedrock_training.jsonl`.
- Register hashes of all **independent protected** ELI 10 assets: holdout, cross-entity, RDF-reference evaluation, help-panel/production regressions, and every versioned shared ELI benchmark used for release comparisons. Source files must not be mutated to accommodate Nova.
- Set the Bedrock base ID and AWS config on `version.json` or via `MTT_` environment variables. Do not place AWS credentials/tokens in MTT config.
- Never merge protected holdouts into the SFT corpus, or copy original labeled benchmark cases into prompt examples.

## Release process

1. Run offline and then AWS preflight. Validate 8–20,000 rows, message alternation, reserved literals, the exact system-prompt hash, overlap, and protected-file hashes. Nova 2 Lite has no Bedrock Runtime token-counting support. MTT reports the 32K training-example context limit as unverified until Bedrock validates the job.
2. Submit once with `--apply`; record printed run ID and Bedrock job ARN. `remote train status`, `watch`, or `recover` must reference the **same run**. Do not create an independent AWS job after a lost terminal session.
3. Once `Completed`, confirm the resulting custom model ARN and the `runs/eli/v10/<run>/remote/train.json` provenance.
4. Deploy the custom model on-demand, store the separate deployment ARN, and check for `Active`.
5. Run the original ELI shared benchmark through the MTT Bedrock compatibility adapter. Freeze generation settings, model deployment identity, exact prompt hash, benchmark identity and scorer hashes. Use the existing semantic gate report—**not** a managed AWS generic metric—to decide correctness.
6. Add any ELI 10 frozen suites that are not yet registered with the shared evaluator through a **new versioned benchmark package**, not by editing the published benchmark in place.
7. Stop/clean up unnecessary AWS deployments only through an explicit reviewed administrative action; do not delete custom models, historical MTT runs or protected evaluation sources.

## Evaluation acceptance criteria

- No protected evaluation question overlaps training (normalized whitespace and casefold match); all protected file SHA-256 values match their published receipts.
- Both training and evaluation use the exact same system prompt bytes and the current production semantics.
- Model-generated JSON/SPARQL is evaluated by the existing ELI semantic scorer; do not introduce a second SPARQL compiler.
- Comparison to Qwen v9.2 is performed on an **identical benchmark release and scorer**, with labeled and unlabeled suites reported separately.
- Failure, retry and post-restart recovery preserve AWS identity, job ARN/token, MTT run ID, model ARN, deployment ARN and result identities.

## Required preproduction integration tests

The ELI 10 dataset and benchmark sources are operational assets not included with public MTT main. Before production use, run this code against a staging copy of the real ELI 10 corpus, the real frozen benchmarks, the exact unmodified `evaluate_shared_ollama.py`, and the real Bedrock account. Confirm `/api/generate` response parsing and the original benchmark's resume behavior with a one-case smoke test before submitting a complete evaluation.
