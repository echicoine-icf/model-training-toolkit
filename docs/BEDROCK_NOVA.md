# MTT remote backend: Bedrock Nova 2 Lite SFT

This is an **additive optional backend**, not a replacement for MTT's PyTorch/Unsloth training, quantization, or project-owned semantic scorers. The native command group is `workspace.sh remote KIND ACTION PROJECT VERSION`. Once `bash workspace.sh upgrade` refreshes the project launchers, the project form is `bash projects/eli/project.sh remote KIND ACTION VERSION`.

## Source / operations boundary

- Modify **source** under `~/model-training-toolkit-repo` on a feature branch. Run tests and merge through a PR.
- Only then update **operational** `~/model-training-toolkit` with `git pull --ff-only`, `bash workspace.sh upgrade`, and `bash workspace.sh doctor`.
- Never overwrite `.venv`, run directories, registered model files, frozen benchmark packages, or `projects/eli/versions/v9.*`.
- Install `requirements/bedrock.txt` (`boto3`) if not available. The dependency is loaded lazily by the remote backend; ordinary MTT operations do not import boto3.
- No AWS resources are created until the matching command explicitly receives `--apply`.

## New remote version

Create ELI v10 only if not already present (the command refuses to overwrite an existing version):

```bash
bash workspace.sh remote version init eli v10 --base-model-id amazon.nova-2-lite-v1:0:256k
```

The optional `--base-model-id` flag sets the customization base model in the new version. This creates `projects/eli/versions/v10/version.json` and the `data/` directory. Place the **exact production** `system_prompt.txt` in the version's `data` folder. The prepared Nova training JSONL is supplied with `--dataset`; MTT copies its **original bytes** into the new run snapshot without rewriting the SFT artifact. For the benchmark overlap auditor, MTT derives a question-and-answer projection. An existing `data/train.jsonl` is preserved byte-for-byte in the run snapshot when its questions and expected answers match the Bedrock corpus. Row IDs, metadata, and ordering may differ.

Suggested `remote.bedrock` object inside `version.json` (keep unrelated version fields):

```json
{
  "remote": {
    "bedrock": {
      "base_model_id": "amazon.nova-2-lite-v1:0:256k",
      "protected_benchmarks": ["benchmark-v1.0.2"],
      "protected_files": []
    }
  }
}
```

`protected_benchmarks` and `protected_files` are **examples**. For ELI 10, declare the actual frozen holdout, cross-entity, RDF-reference evaluation and production regression assets after verifying their file layouts. Never silently ignore an unrecognized protected dataset: the validator intentionally fails closed.

## AWS settings and precedence

`CLI > version.remote.bedrock > project.remote.bedrock > MTT environment > standard boto3/AWS resolution`.

Fields: `profile`, `region`, `role_arn`, `bucket`, `prefix`, `training_prefix`, `output_prefix`, `base_model_id`, `model_prefix`, `count_model_id`. The first five correspond to `MTT_AWS_PROFILE`, `MTT_AWS_REGION`, `MTT_BEDROCK_ROLE_ARN`, `MTT_BEDROCK_S3_BUCKET`, and `MTT_BEDROCK_S3_PREFIX`. Optional additional environment names are in `toolkit/remote_bedrock.py`.

Example Git Bash session (not a hard-coded application default):

```bash
export MTT_AWS_PROFILE=nlm-account
export MTT_AWS_REGION=us-east-1
export MTT_BEDROCK_ROLE_ARN='arn:aws:iam::<ACCOUNT-ID>:role/<SERVICE-ROLE>'
export MTT_BEDROCK_S3_BUCKET='<YOUR-BUCKET>'
export MTT_BEDROCK_S3_PREFIX=eli
```

The IAM user/profile needs STS identity, model customization permissions (including `bedrock:ListModelCustomizationJobs` to recover a missing job ARN), on-demand model deployment permissions, S3 write/head permissions, and `Converse` permissions for evaluation. Preflight requires `CountTokens` permission for models where MTT calls it. The Bedrock service role must be assumable by Bedrock and have the relevant input/output bucket/KMS permissions. Credentials, session tokens and access keys are not stored by MTT. Resume commands verify the current AWS account against the persisted account identity.

## Train: preflight, submit, status, recover

All commands below are examples; replace the dataset and benchmark paths with ELI 10's actual frozen artifacts.

Offline validation (no AWS calls):

```bash
bash projects/eli/project.sh remote train preflight v10 \
  --dataset '/c/path/to/ELI10_NOVA2_SFT_TRAIN.jsonl' \
  --benchmark benchmark-v1.0.2 --skip-token-check
```

AWS preflight verifies the account using STS. For countable models, it also checks the 32K training limit per example using Bedrock `CountTokens`. For Nova 2 Lite, MTT reports context length as unverified:

```bash
bash projects/eli/project.sh remote train preflight v10 \
  --dataset '/c/path/to/ELI10_NOVA2_SFT_TRAIN.jsonl' \
  --benchmark benchmark-v1.0.2
```

Submit a new run only after both dataset and protected sets are correct:

```bash
bash projects/eli/project.sh remote train submit v10 \
  --dataset '/c/path/to/ELI10_NOVA2_SFT_TRAIN.jsonl' \
  --benchmark benchmark-v1.0.2 --apply
```

Submission prints `RUN_ID`, AWS identity/region, customization job ARN/name, and records `runs/eli/v10/<RUN_ID>/remote/train.json`. Each S3 training URI contains the run ID and SHA-256, and upload uses `IfNoneMatch='*'` plus metadata/length verification. The original dataset and exact frozen prompt are hashed into `input/input_hashes.json`. MTT explicitly sends the Nova 2 Lite SFT defaults required by Bedrock: two epochs, a learning rate of `0.00001`, and ten warmup steps. The exact parameter map is recorded in the run receipt and shown in preflight.

After the terminal is closed or restarted:

```bash
RUN='<RUN_ID_FROM_SUBMISSION>'
bash projects/eli/project.sh remote train status v10 --run "$RUN"
bash projects/eli/project.sh remote train watch v10 --run "$RUN" --interval 90
```

If the first submission lost its AWS response *after* saving a run, do **not** create another training run. Recover the deterministic remote job identity and idempotency token. When the ARN is missing, MTT lists customization jobs and checks the exact recorded job name. If the list contains no match, recovery replays the original request. For a staged Nova 2 Lite run from older MTT code, recovery also fills in and records the required hyperparameter defaults:

```bash
bash projects/eli/project.sh remote train recover v10 --run "$RUN"
```

The train state persists account, region, profile alias, S3 paths, role ARN, source commit, version hash, prompt hash, training hash, job identifier, status transitions and failure messages; `remote/events.jsonl` records status checkpoints. Completed `outputModelArn` is captured as `custom_model_arn` and the MTT run state becomes `trained`.

## Deploy for on-demand inference

Customization completion alone is **not** an inference endpoint. An on-demand custom-model deployment is a separate Bedrock resource. Its deployment ARN is used as `modelId` for Bedrock `Converse` requests.

```bash
bash projects/eli/project.sh remote deploy submit v10 --run "$RUN" --apply
bash projects/eli/project.sh remote deploy status v10 --run "$RUN"
bash projects/eli/project.sh remote deploy watch v10 --run "$RUN" --interval 90
```

If interrupted at deployment creation time, `remote deploy recover v10 --run "$RUN"` queries the deterministic name or replays the same idempotency token. Deployment creation is a billable AWS-changing action; review service-specific costs and cleanup controls before submission. This implementation does not delete deployments automatically.

## Evaluate with the existing ELI scorer

**Option A implemented.** Direct Bedrock Converse inference runs through an ephemeral localhost compatibility bridge; MTT's existing `shared_eval.execute` launches the original benchmark's `audit_shared.py`, `evaluate_shared_ollama.py` and unchanged semantic scorer. The bridge does **not** generate a second SPARQL contract, change gold labels, rewrite the DATMM RDF schema or alter any benchmark source file. The bridge supplies the exact frozen production system prompt for every Converse call and rejects a divergent prompt if explicitly sent by a benchmark runner.

Evaluation submission starts a *local detached worker*, not an AWS-managed evaluation job. This is intentional: the historical ELI semantic scorer remains authoritative, and local state records are resumable after interruptions. Because this is not an AWS evaluation job, there is no Bedrock evaluation-job ARN to record. Use `--resume` to restart an interrupted worker; the underlying shared benchmark runner also receives its existing `--resume` flag.

```bash
bash projects/eli/project.sh remote eval submit v10 \
  --run "$RUN" --benchmark benchmark-v1.0.2 --stage all
```

The above is read-only planning. Start inference (billable) explicitly:

```bash
bash projects/eli/project.sh remote eval submit v10 \
  --run "$RUN" --benchmark benchmark-v1.0.2 --stage all --apply
```

Record the printed `EVAL_ID` and use:

```bash
EVAL='<EVAL_ID_FROM_SUBMISSION>'
bash projects/eli/project.sh remote eval status v10 --run "$RUN" --eval-id "$EVAL"
bash projects/eli/project.sh remote eval watch v10 --run "$RUN" --eval-id "$EVAL" --interval 90
```

To recover an interrupted evaluation:

```bash
bash projects/eli/project.sh remote eval submit v10 \
  --run "$RUN" --benchmark benchmark-v1.0.2 --stage all --resume --apply
```

Outputs remain under `runs/eli/v10/<run>/shared_evaluations/<benchmark>/<model-tag>/`, with a separate `remote/eval_<id>.json` provenance receipt and log. The scorer's existing `run_config.json`, dataset hashes and gate reports remain authoritative. The matching model tag includes a digest of the unique Bedrock deployment ARN, so results never collide with Qwen/Ollama.

### Cautions / current limitations

1. **Source-based compatibility:** The MTT core contract was inspected on public main 0.4.1. ELI's real v10 benchmark runner and new frozen data packages were not available in this environment. The localhost bridge must be integration-tested with those exact assets before a 977-case/full ELI10 evaluation. It implements Ollama `api/show`, `api/tags`, `api/generate` and `api/chat` non-streaming endpoints; unsupported paths deliberately fail rather than fabricate outputs.
2. **Text-only training:** Preflight currently fails on unsupported multimedia, reasoning, tool-call or mixed-modality records instead of silently discarding them. ELI's current text-only schema-to-SPARQL task should fit. Add validated formats explicitly if needed.
3. **Token checks:** [Nova 2 Lite does not support Bedrock Runtime CountTokens](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-lite.html). MTT runs all structural and contamination checks, records token counts as unverified, and leaves the 32K context check to Bedrock customization. This also applies if `--count-model-id` is supplied for Nova 2 Lite. Other models retain the per-record `CountTokens` check with the configured count-model ID. The offline `--skip-token-check` flag is accepted only without `--apply`.
4. **Protected corpora:** The overlap preflight extracts natural-language questions from supported JSON/JSONL structures and verifies frozen release hashes. Unknown formats fail closed. Make sure *every* ELI10 protected evaluation layer is declared or registered; the existing general shared benchmark does not automatically include new cross-entity/reference/help-panel tests.
5. **Managed Bedrock evaluation:** Intentionally not used for ELI's semantic correctness. A future optional managed job can use a separate durable job-state type without replacing benchmark definitions. No managed-eval job is claimed to exist in this release.
6. **Destructive cleanup:** On-demand deployment deletion and S3 artifact retention are explicit infrastructure management operations and are not automated by this code.

## Tests

```bash
python -m unittest discover -s tests -q
```

No tests rely on the `nlm-account` profile. Unit tests inject mocked AWS clients and create synthetic local project/version/run/benchmark fixtures. Before an actual AWS run, verify `aws sts get-caller-identity --profile ...` and service quotas/permissions with the chosen account.
