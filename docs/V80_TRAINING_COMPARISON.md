# General-purpose MTT training versus supplied ELI V8.0 trainer

Reference: `train_v80_fresh.py` (provided by the user). This document is an audit, **not** a claim that MTT runs are numerically identical to the V8.0 runs.

| Dimension | V8.0 reference | Toolkit v0.3.4 |
| --- | --- | --- |
| Initialization | Original Qwen 2.5 0.5B Instruct base | Declared frozen base Hugging Face revision, fresh only |
| LoRA modules | q,k,v,o,gate,up,down | Arbitrary explicit `--lora-target-modules`, exact PEFT coverage verified |
| LoRA parameters | r=16 alpha=32 dropout=.05 bias=none | CLI-configurable, bias=none, recorded |
| Assistant loss | Exact `messages` chat template, prefix masked -100 | Same when `messages` available and verified; otherwise standardized JSON `expected` serialization |
| Model precision | BF16 | auto/BF16/FP16; explicitly select BF16 to compare |
| Attention | SDPA | sdpa/eager/flash_attention_2; explicitly select sdpa |
| Gradient checkpointing | Enabled, use_reentrant=False | Explicit control; same API when enabled |
| Optimizer | fused AdamW if available, fallback regular, decay=.01 | fused-adamw/adamw; actual selection logged |
| LR | warmup .03; cosine from 8e-5 to absolute 5e-6 | configurable peak, warmup, absolute min LR; same schedule expression |
| Gradient accumulation | 1 microbatch default; optimizer steps across epochs | 1 microbatch per forward; accumulation across epoch boundaries |
| Shuffle | random.Random(seed+epoch) | same |
| Clipping | norm 1 | CLI configurable |
| Checkpoints | Adapter, tokenizer, optimizer/RNG state resumable | Adapter + tokenizer intermediate; **not** optimizer/RNG resumable |
| Train/Eval data | V8 corpus & frozen holdouts | Version-local mtt-data-v1; exact JSON scorer, not equivalent to ELI's semantic tests |

## Material caveats

- V8.0 used a specific four-epoch dataset and exact training **system message**. The package's `prompt_assembly` and exact `system_prompt.txt` must match the intended architecture to make an architecture-only comparison. Be sure schema is not appended twice.
- The dataset changed between V8.0 and V9.0, so even identical hyperparameters don't make the experiments otherwise equivalent.
- Different installed PyTorch, PEFT, Transformers, CUDA kernels and optimizer implementations may change numerical results. MTT freezes `pip freeze` per run, and records actual architecture details.
- MTT may use autocast and an FP16 GradScaler; V8.0 BF16 training used direct BF16 weights. These are similar but not guaranteed bitwise equal.
- v0.3.4 does **not** guarantee checkpoint resumption; intermediate save-only snapshots cannot resume an interrupted optimizer. The original V8.0 trainer did implement full optimizer/RNG resume.
- Existing V9.0 two-module adapter remains a valid *separate experiment*, but cannot become a seven-module fresh-base run by resuming it. Begin a **new run** from the base model.
