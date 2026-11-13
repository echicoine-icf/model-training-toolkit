# Post-training quantization for Ollama and vLLM

MTT can quantize a **completed, separately exported** model. It does not change
training, LoRA adapters, checkpoints, original SafeTensors, original GGUF, or
past evaluations. Each format writes its own artifact and hash receipt.

## Formats

| `--format` | Source | Destination | Serving/runtime | Purpose |
|---|---|---|---|---|
| `gguf-q8_0` | `exports/model-f16.gguf` | `exports/model-q8_0.gguf` | Ollama / llama.cpp | 8-bit comparison against the existing shared benchmark |
| `gguf-q4_k_m` | `exports/model-f16.gguf` | `exports/model-q4_k_m.gguf` | Ollama / llama.cpp | Lower-memory 4-bit comparison using Q4_K_M |
| `vllm-fp8` | `exports/merged_hf/` | `exports/vllm-fp8/` | vLLM + compatible CUDA GPU | 8-bit float weights/activations; data-free FP8_DYNAMIC |
| `vllm-int8` | `exports/merged_hf/` | `exports/vllm-int8/` | vLLM compatible deployment | Experimental 8-bit integer **weight-only** W8A16; not an INT8 W8A8 activation path |

**GGUF Q8_0 is not a vLLM FP8 or INT8 checkpoint.** Quantize the HF model
separately for vLLM deployment. FP8 vs BF16 latency/concurrency must be tested
on the **same hardware, serving options, and evaluation cases**. The 0.5B model's
size may be too small for quantization to improve latency. Lower weight memory
does not mean the KV cache is also quantized.

## Example: ELI v9.2 — Q8_0 and historical semantic accuracy

Run from Git Bash inside the operational MTT workspace:

```bash
cd ~/model-training-toolkit
RUN=run_20261109T002925408171Z_a4a5de

bash projects/eli/project.sh quantize v9.2 --run "$RUN" \
  --format gguf-q8_0 --ollama-model datmm-eli:v9.2-q8 --dry-run

bash projects/eli/project.sh quantize v9.2 --run "$RUN" \
  --format gguf-q8_0 --ollama-model datmm-eli:v9.2-q8
```

The toolkit locates llama.cpp's built `llama-quantize.exe` / `llama-quantize`.
**If missing, a normal (non-`--dry-run`) Q8_0 export automatically runs CMake**
to configure and build the CPU-only quantizer in the ignored
`tools/llama.cpp/build/mtt-quantize/` directory, then uses it immediately.
A `--dry-run` shows the build commands without compiling anything.
CMake and a working C++ toolchain must already be installed; on Windows, use
Visual Studio Build Tools with the *Desktop development with C++* workload.
If you prefer a manually built utility, pass `--quantizer PATH`; use
`--no-auto-build-quantizer` to require an existing binary and fail otherwise.
The F16 GGUF must already exist;
if absent, first convert the `exports/merged_hf` model to F16 GGUF (see standard
llama.cpp conversion instructions). **Do not rerun the LoRA merge** when the
merged SafeTensors already exists.

Once Ollama lists `datmm-eli:v9.2-q8`, run the exact same immutable evaluation
package used for the F16 model:

```bash
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" \
  --benchmark benchmark-v1.0.2 --model datmm-eli:v9.2-q8 \
  --suite v72_graph_26 --dry-run

bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" \
  --benchmark benchmark-v1.0.2 --model datmm-eli:v9.2-q8 \
  --suite frozen_715 --resume
```

The shared runner saves Q8 results under a *different model tag* from F16.
Use F16 `datmm-eli:v9.2-f16` on the same `benchmark-v1.0.2`, same frozen prompt,
`num_ctx`, `num_predict`, `seed`, and Ollama version; compare case-by-case failures.
For completed results use `--resume`. A **Q8 model does not inherit F16 accuracy**.

## FP8 and INT8 vLLM-compatible exports

**Do not `pip install llmcompressor` into the operational MTT `.venv`.** It can
change the carefully pinned CUDA Torch/Unsloth environment. Instead, create a
separate quantization Python environment with compatible `torch`,
`transformers`, `safetensors`, and `llmcompressor`. Linux / WSL2 / a SageMaker
build instance is recommended for vLLM deployments; library/platform support
must be verified there before using an export.

When that interpreter is callable by your MTT terminal, use:

```bash
cd ~/model-training-toolkit
RUN=run_20261109T002925408171Z_a4a5de

bash projects/eli/project.sh quantize v9.2 --run "$RUN" \
  --format vllm-fp8 --quant-python /path/to/isolated/python --dry-run

bash projects/eli/project.sh quantize v9.2 --run "$RUN" \
  --format vllm-fp8 --quant-python /path/to/isolated/python
```

On Windows, it may be simpler to execute the *standalone exporter* inside WSL2
instead (where the MTT checkout is mounted under `/mnt/c`):

```bash
# Run inside a separately provisioned Linux quantization venv in WSL2:
python /mnt/c/Users/echic/model-training-toolkit/scripts/quantize_vllm.py \
  --source /mnt/c/Users/echic/model-training-toolkit/runs/eli/v9.2/RUN_ID/exports/merged_hf \
  --output /mnt/c/Users/echic/model-training-toolkit/runs/eli/v9.2/RUN_ID/exports/vllm-fp8 \
  --scheme FP8_DYNAMIC
```

The standalone script produces the compressed checkpoint, but not the MTT
integrity receipt; for full provenance run via MTT with a separately callable
interpreter. The `vllm-int8` format uses `W8A16` **weights only**, and is a
secondary experiment, not a calibrated W8A8 model.

In an environment with vLLM, serve the compressed model:

```bash
vllm serve /path/to/exports/vllm-fp8 \
  --served-model-name datmm-eli:v9.2-fp8 --max-model-len 4096
```

On recent vLLM versions, an alternative requiring **no separate FP8 export**
is online quantization from the original merged checkpoint, for example:

```bash
vllm serve /path/to/exports/merged_hf \
  --served-model-name datmm-eli:v9.2-fp8-online \
  --quantization fp8_per_tensor --max-model-len 4096
```

Check that your pinned vLLM version supports the chosen mode and accelerator.

### Score the vLLM FP8 model with the same ELI scorer

The shared ELI evaluator speaks Ollama's HTTP protocol. The optional
`vllm_eval_bridge.py` translates those calls to a vLLM OpenAI-compatible
endpoint while keeping the **same frozen suites and historical scorer**.
This is for *development correctness comparison*, not latency benchmarking.

Start vLLM, then start a local loopback-only bridge (separate terminal):

```bash
python scripts/vllm_eval_bridge.py \
  --model datmm-eli:v9.2-fp8 \
  --receipt "/path/to/exports/vllm-fp8/MTT_QUANTIZATION.json" \
  --vllm-url http://127.0.0.1:8000 --port 11435 --max-context 4096
```

Use the same MTT shared suite selection and scoring via the bridge:

```bash
bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" \
  --benchmark benchmark-v1.0.2 --model datmm-eli:v9.2-fp8 \
  --suite frozen_715 --url http://127.0.0.1:11435 --resume
```

The bridge checks the quantized checkpoint file hashes against its receipt.
Because Ollama and vLLM may differ in tokenizer templates, numerical kernels,
seed behavior, and stopping criteria, document those differences in comparisons.
Never compare outputs solely from differing benchmark package versions.

### Measure deployment performance instead of assuming memory savings imply speed

Run tests using the same input dataset, prompt, output token cap, hardware,
server options and concurrency for both BF16 and FP8. Example:

```bash
python scripts/benchmark_vllm.py \
  --model datmm-eli:v9.2-fp8 \
  --cases projects/eli/evaluations/benchmark-v1.0.2/cases/frozen_715.jsonl \
  --system-prompt "runs/eli/v9.2/$RUN/input/data/system_prompt.txt" \
  --requests 64 --concurrency 1,8,32 \
  --output "runs/eli/v9.2/$RUN/vllm_fp8_bench.json"
```

The script reports requests/sec, **end-to-end output tokens/sec**, p50/p95
response latency and actual prompt/completion token counts. It deliberately
does not call those figures GPU decode tokens/sec. To validate semantic
correctness, use the shared ELI evaluator separately.

## Safety and provenance

- Requires a completed training run and existing merged artifact.
- Writes only new exports with separate names and SHA-256 sidecars; refuses overwrite.
- Does not modify the model's adapter, F16 GGUF, merged HF files or run.json.
- A failed conversion leaves an explicitly named `.partial` artifact for review.
- Uses existing project-specific scoring; the quantization exporter itself does not measure accuracy.
- Avoid changing MTT's `.venv`, Unsloth installation, or PyTorch packages.

Official documentation: [llama.cpp GGUF quantization](https://github.com/ggml-org/llama.cpp), [vLLM quantization](https://docs.vllm.ai/en/latest/features/quantization/), [LLM Compressor FP8_DYNAMIC](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_fp8/), and [vLLM online quantization](https://docs.vllm.ai/en/latest/features/quantization/online/).

## Q4_K_M: ELI V9.2 accuracy experiment

Quantize **from the original F16 GGUF**, not from the Q8 GGUF; repeated
quantization can degrade quality. MTT reuses the existing quantizer and preserves
F16, Q8, all previous evaluations, and benchmark files.

```bash
cd ~/model-training-toolkit
RUN=run_20261109T002925408171Z_a4a5de

bash projects/eli/project.sh quantize v9.2 --run "$RUN" \
  --format gguf-q4_k_m --ollama-model datmm-eli:v9.2-q4 --dry-run

bash projects/eli/project.sh quantize v9.2 --run "$RUN" \
  --format gguf-q4_k_m --ollama-model datmm-eli:v9.2-q4

bash projects/eli/project.sh shared-eval v9.2 --run "$RUN" \
  --benchmark benchmark-v1.0.2 --model datmm-eli:v9.2-q4 \
  --stage all --resume
```

The Q4 model has its own file, Modelfile, receipt, Ollama tag, and evaluation
output path. Check semantic accuracy and case-level regressions against F16 and
Q8 using the same benchmark version and decoding settings. GGUF Q4_K_M is a
**llama.cpp / Ollama** format, not a vLLM FP8 checkpoint.

## Production vLLM / SageMaker path

MTT also supports `vllm-w4a16`: weight-only 4-bit **W4A16 RTN** compressed SafeTensors created from `exports/merged_hf/` with `--quant-python`. This is separate from GGUF Q4_K_M and must pass semantic validation independently. For the F16/FP8/INT8/INT4 SafeTensors-to-S3 packaging, a persistent SageMaker vLLM endpoint, and concurrency autoscaling, see [SageMaker vLLM deployment](SAGEMAKER_VLLM.md). No extra release or installation handoff files are used.
