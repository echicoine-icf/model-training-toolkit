# Optional Unsloth training backend (MTT v0.3.5)

MTT supports two explicitly selected engines in the same workspace:

- `--backend pytorch` (default): unchanged PyTorch + Transformers + PEFT model loading and LoRA setup.
- `--backend unsloth`: loads the model with `FastLanguageModel.from_pretrained` and initializes LoRA with `FastLanguageModel.get_peft_model`. **The remainder of the MTT trainer remains common**: data snapshots, exact assistant-only token masking, un-packed examples, ordered per-example microbatches, AdamW optimizer, LR scheduler, gradient accumulation, logging, checkpoint output, and optional evaluation.

This is an optional acceleration path, **not** Unsloth Studio and **not** a migration to TRL's `SFTTrainer`. It has not yet been benchmarked on the intended GPU; speed or model quality improvements are not guaranteed.

## Install without replacing a working workspace

**Stop all running MTT training processes before changing dependencies.** Do not install Unsloth while a training process is active. Within the existing operational MTT workspace, on Windows Git Bash:

```bash
cd ~/model-training-toolkit
source ./activate.sh
python -m pip freeze > registry/pip_freeze_before_unsloth.txt
python -m pip install unsloth
python -m pip check
bash workspace.sh doctor
```

The `pip install unsloth` command is the official Unsloth Windows path **when CUDA-capable PyTorch is already installed**. It can upgrade shared dependencies. If it reports conflicts, **stop**; do not wipe `.venv` or force-downgrade CUDA. Review the documented [Unsloth Windows installation options](https://unsloth.ai/docs/get-started/install/windows-installation) first. The previous `pytorch` backend remains the default.

MTT never installs Unsloth automatically, and importing the PyTorch backend never imports Unsloth. The Unsloth backend imports Unsloth before torch/Transformers/PEFT. Missing Unsloth is a hard failure, not an implicit fallback.

## Training

From the operational workspace, supply the same required one-shot parameters as before, plus:

```bash
--backend unsloth
```

The optional extra setting:

```bash
--unsloth-checkpointing standard   # default; Unsloth's regular gradient checkpointing
# OR
--unsloth-checkpointing unsloth    # experimental optimized Unsloth checkpointing
```

It only applies with `--backend unsloth`. `--gradient-checkpointing` enables checkpointing; `--no-gradient-checkpointing` disables it. Note that Unsloth's `standard` checkpointing mode is **not asserted to use PyTorch's `use_reentrant=False` behavior**, unlike our native backend. Choose backend `pytorch` when reproducing V8.0 execution details exactly.

Example **configuration only** (adapt project/version and training parameters):

```bash
bash projects/eli/project.sh train v9.0 \
  --backend unsloth \
  --epochs 4 --max-length 6144 \
  --learning-rate 8e-5 --min-learning-rate 5e-6 \
  --gradient-accumulation 1 --seed 8002026 \
  --weight-decay 0.01 --warmup-ratio 0.03 \
  --max-grad-norm 1 --log-every 50 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --precision bf16 --attention-implementation sdpa \
  --optimizer fused-adamw --gradient-checkpointing \
  --save-every 2000 --no-auto-eval --yes
```

This command starts a **new run**; it does not overwrite earlier runs or re-use their adapters. Launch only after validating the model version's data and confirming you have sufficient free GPU and disk resources. Never run both GPU backends simultaneously on one GPU.

## Methodology guarantees and differences

**Shared with the native trainer:** same versioned/frozen data files, validated messages and assistant-only masks, LoRA target-name list (verified after attachment against actual PEFT layers), optimizer choice and weight decay, absolute-minimum cosine LR schedule, effective batch size, epochs, no packing, output paths, and adapters saved in PEFT format. No implicit QLoRA: `load_in_4bit=False` and `load_in_8bit=False` are explicit. Hugging Face model ID and resolved commit SHA are forwarded without substituting a different Unsloth mirror (`use_exact_model_name=True`).

**Not guaranteed identical:** Unsloth may patch attention, tokenizers, LoRA backward kernels, or gradient-checkpointing implementations; `attn_implementation=sdpa` is forwarded, but optimized execution may differ. Its use of random state during LoRA setup can change RNG consumption. Its optimized LoRA path is fastest with `lora_dropout=0`; preserving a nonzero dropout (e.g., 0.05) can reduce acceleration. The new backend may use more memory than expected with 16-bit bases. Inspect `training_plan.json` **before trusting comparability**. Some model families or future Unsloth API changes may be incompatible and will fail rather than fall back.

**Remaining common limitations:** full optimizer/RNG checkpoints are available starting in v0.3.6, but require real hardware resumption validation; the built-in evaluation is exact JSON match, **not** task-specific semantic correctness; no benchmarked speedup is claimed. Existing evaluation and export use Transformers/PEFT and should work with a standard PEFT adapter, but **require a real GPU end-to-end smoke test** for the Unsloth output before use in production.

## Safe validation sequence

1. Run `python -m unittest discover -s tests -v` (tests mock Unsloth; no downloads).
2. Verify the installed Unsloth version and CUDA on the actual target system with `bash workspace.sh doctor`.
3. Use a **new, small, independent test version/data package**, not a frozen ELI dataset, for a minimal GPU smoke test. Inspect the training plan: the real LoRA layer paths, precision, optimizer, scheduler, backend, and trainable count must be printed before optimizer steps.
4. Load/evaluate/export that test adapter and compare against the PyTorch backend with the same test examples. Measure tokens/sec and peak VRAM, not just steps/sec.
5. Only then use it for a real project model version.

Source: [Unsloth Windows installation](https://unsloth.ai/docs/get-started/install/windows-installation), [Unsloth project](https://github.com/unslothai/unsloth), [Unsloth LoRA examples](https://github.com/unslothai/unsloth-zoo).
