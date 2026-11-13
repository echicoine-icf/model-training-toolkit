# Deploy MTT exports with vLLM on Amazon SageMaker AI

MTT can prepare **independent** F16, FP8, INT8 (weight-only), and INT4 (weight-only) Hugging Face SafeTensors packages for an always-on SageMaker real-time endpoint. The toolkit never deploys or uploads without an explicit `--apply`; no GPU deployment has been verified by the source test suite.

The supported official deployment path uses the AWS **SageMaker vLLM Deep Learning Container** (DLC). No custom Dockerfile is needed. Its environment variables (`SM_VLLM_*`) control the vLLM server; it loads a SageMaker `model.tar.gz` into `/opt/ml/model`. Select a *pinned, regional* SageMaker GPU image and its required inference AMI. See [AWS's SageMaker vLLM deployment guide](https://aws.github.io/deep-learning-containers/vllm/deployment/sagemaker/).

## Formats and accuracy

| MTT export | Serving artifact | Method | Evaluation note |
|---|---|---|---|
| `hf-f16` | `exports/merged_hf/` | Original merged SafeTensors; no new compression | Baseline. This is the model's saved dtype, not a guaranteed BF16 conversion |
| `vllm-fp8` | `exports/vllm-fp8/` | `FP8_DYNAMIC`, 8-bit weights / dynamic activations | Compare against baseline on the same vLLM hardware |
| `vllm-int8` | `exports/vllm-int8/` | `W8A16`, 8-bit weight-only RTN | Different from FP8 or W8A8 |
| `vllm-w4a16` | `exports/vllm-w4a16/` | `W4A16`, 4-bit weight-only RTN | **Experimental**: accuracy may degrade; calibrated GPTQ is a possible later improvement |

GGUF `gguf-q8_0` and `gguf-q4_k_m` remain useful Ollama accuracy experiments but are **not** the source for vLLM compression. All vLLM quantization starts from the original `merged_hf/`. 4-bit `W4A16` uses 16-bit activations; it does not halve the vLLM KV-cache footprint. `--kv-cache-dtype fp8` is an *independent* optional inference setting with its own accuracy considerations.

### Step 1: produce selected quantizations

The completed ELI V9.2 run already has `exports/merged_hf/`, suitable for F16. FP8/INT8/W4A16 creation needs an **isolated** Python environment containing compatible Torch, Transformers, SafeTensors, and LLM Compressor. Do **not** change the working MTT CUDA/Unsloth venv. On Windows, WSL2 or a separate Linux compression host is advisable.

```bash
cd ~/model-training-toolkit
RUN=run_20261109T002925408171Z_a4a5de

bash projects/eli/project.sh quantize v9.2 --run "$RUN" \
  --format vllm-fp8 --quant-python /path/to/isolated/python --dry-run
bash projects/eli/project.sh quantize v9.2 --run "$RUN" \
  --format vllm-fp8 --quant-python /path/to/isolated/python

# Optionally create the other variants independently:
bash projects/eli/project.sh quantize v9.2 --run "$RUN" \
  --format vllm-int8 --quant-python /path/to/isolated/python
bash projects/eli/project.sh quantize v9.2 --run "$RUN" \
  --format vllm-w4a16 --quant-python /path/to/isolated/python
```

Each has its own immutable destination and SHA-256 receipt. If one already exists, MTT refuses to overwrite it. Confirm that the actual vLLM version and GPU can load that compressed format **before** production use.

### Step 2: package and stage the chosen export

The following is **offline** until `upload --apply`. Package `hf-f16`, `vllm-fp8`, `vllm-int8`, and `vllm-w4a16` separately, choosing one at a time to deploy. The packager validates receipt hashes for quantized exports and creates a new `model.tar.gz` plus a sidecar receipt.

```bash
python scripts/sagemaker_vllm.py package \
  --project eli --version v9.2 --run "$RUN" --format vllm-fp8 \
  --output "$HOME/Downloads/eli-v92-fp8-model.tar.gz"

python scripts/sagemaker_vllm.py upload \
  --package "$HOME/Downloads/eli-v92-fp8-model.tar.gz" \
  --s3-uri s3://YOUR-PRIVATE-BUCKET/eli/v9.2/fp8/model.tar.gz \
  --profile nlm-account --region us-east-1
# Review the plan. Then repeat with --apply to actually upload.
```

Model artifacts remain private; grant SageMaker's execution role **read-only** access to this S3 object and any KMS key needed. Do not make this bucket public.

### Step 3: create a persistent vLLM endpoint

Use a real SageMaker execution-role ARN and a **pinned SageMaker vLLM GPU image** for the target region. The official image's generic family is `vllm:server-sagemaker-cuda`, but verify the exact region/tag in the [AWS image catalog](https://aws.github.io/deep-learning-containers/vllm/deployment/sagemaker/) instead of assuming a particular version exists.

```bash
SM_ROLE='arn:aws:iam::YOUR-ACCOUNT:role/YOUR-SAGEMAKER-ROLE'
SM_IMAGE='YOUR-REGIONAL-PINNED-SAGEMAKER-VLLM-IMAGE'
MODEL_S3='s3://YOUR-PRIVATE-BUCKET/eli/v9.2/fp8/model.tar.gz'

python scripts/sagemaker_vllm.py deploy \
  --model-s3-uri "$MODEL_S3" --image-uri "$SM_IMAGE" --role "$SM_ROLE" \
  --endpoint datmm-eli-v92-fp8 --model-tag datmm-eli:v9.2-fp8 \
  --instance-type ml.g6.xlarge --instance-count 1 \
  --max-model-len 4096 --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 --kv-cache-dtype auto \
  --profile nlm-account --region us-east-1
# Inspect JSON plan; rerun with --apply only when you intend to incur charges.
```

The endpoint **stays on and incurs SageMaker charges continuously**, including during idle hours. The deployment command never silently updates an existing endpoint and never deletes existing models. It prints created resource names for inspection if a deployment fails. Check status after provisioning:

```bash
python scripts/sagemaker_vllm.py status --endpoint datmm-eli-v92-fp8 \
  --profile nlm-account --region us-east-1
```

`--kv-cache-dtype fp8` and a lower `--max-model-len` can increase effective cache capacity but may affect quality or reject longer requests. Measure maximum active sequences, preemptions, queuing, p95 latency and CloudWatch GPU memory—not just on-disk weights. The appropriate tuning differs by GPU.

### Step 4: autoscaling, smoke test and concurrency check

**Only after the endpoint is `InService`** can you apply autoscaling. The target-tracking metric is concurrent requests per model; keep minimum at 1 for an always-on service:

```bash
python scripts/sagemaker_vllm.py autoscale \
  --endpoint datmm-eli-v92-fp8 --min-instances 1 --max-instances 3 \
  --target-concurrency 8 --profile nlm-account --region us-east-1
# Review, then repeat with --apply.

python scripts/sagemaker_vllm.py invoke --endpoint datmm-eli-v92-fp8 \
  --model-tag datmm-eli:v9.2-fp8 \
  --system-prompt 'YOUR EXACT FROZEN ELI SYSTEM PROMPT' \
  --message 'Find datasets about cancer.' \
  --profile nlm-account --region us-east-1
# Repeat with --apply to actually send the request.
```

To measure behavior under traffic, use `scripts/benchmark_sagemaker.py` with the **same frozen cases/system prompt** used in your ELI benchmark. The script measures successful and failed requests and p50/p95 latency at concurrency 1,8,32; it does **not** substitute for semantic accuracy evaluation. Standard SageMaker real-time requests have a response-time limit, so timeouts must be counted as failures rather than silently excluded.

For local semantic evaluation, serve the exact SafeTensors artifact through a compatible local vLLM server and use `scripts/vllm_eval_bridge.py` with the existing `shared-eval --stage all`. Use `--model-dir /path/to/exports/merged_hf` for the F16 baseline, or `--receipt /path/to/vllm-fp8/MTT_QUANTIZATION.json` (likewise INT8/W4A16) for quantized exports. Each run needs a distinct served model tag and result directory. Compare F16/FP8/INT8/W4A16 under matched generation parameters, scoring version, model tag, and benchmark. **Do not promote a quantized candidate solely because it fits VRAM.**

## Production readiness requirements

Before routing DATMM user traffic: verify model load and inference on the selected SageMaker image/GPU; freeze the image tag and benchmark receipts; run the complete ELI semantic holdout and regressions for the deployed runtime; configure IAM least privilege, VPC/network security, CloudWatch, endpoint auto scaling, realistic concurrency load tests, timeouts, and a rollback plan. This module handles packaging and explicit initial provisioning, **not** those external operational approvals or ongoing deployments. The endpoint is not deployed until `--apply` succeeds in your AWS account.
