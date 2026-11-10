# Training backends: PyTorch and optional Unsloth

MTT supports two explicitly selected training backends:

- `--backend pytorch` (default): Transformers, PyTorch and PEFT initialize the selected base model and LoRA layers.
- `--backend unsloth` (optional): Unsloth initializes the model and LoRA layers. MTT retains its shared dataset snapshots, assistant-only masking, optimizer/scheduler, checkpointing, and run management.

Unsloth is **not** TRL `SFTTrainer`; this toolkit uses a custom supervised training loop. Acceleration and numerical parity are hardware/version/configuration dependent and must be measured, not presumed.

## Enable safely

1. Stop active training processes before changing dependencies.
2. Save a dependency record (`python -m pip freeze > <safe-local-file>`), check CUDA functionality and inspect the compatible Unsloth installation instructions for your Python, driver and PyTorch versions.
3. Install/upgrade only after planning dependencies. Some pip installs can unintentionally replace a working CUDA PyTorch wheel with a CPU build; **verify CUDA after any installation**. Do not blindly reinstall or upgrade PyTorch in a functioning environment.
4. Run `bash workspace.sh doctor` and verify `torch.cuda.is_available()` before GPU training.
5. Test with a disposable project/run first; do not modify an established version's frozen datasets to test a backend.

## Training and reproducibility

Use `--backend unsloth` in a fully specified MTT training command. The optional `--unsloth-checkpointing standard|unsloth` affects gradient checkpointing (memory-saving activation recomputation). **Gradient checkpointing is not the same thing as resumable training checkpoints**, which are configured using `--save-every` and `--keep-last-checkpoints`.

Record: pinned base model revision, target LoRA modules, r/alpha/dropout, precision, attention implementation, optimizer, learning rate, scheduler, effective batch size, max sequence length, seed and backend. Explicitly inspect `training_plan.json` for the actual adapted layers and trainable parameter count before comparing runs.

MTT's LoRA use does not imply 4-bit QLoRA. Optimized kernels, RNG consumption and checkpointing can differ across backends; equal CLI flags do **not** guarantee bit-identical weights.

Validation: `python -m unittest discover -s tests -q` checks offline logic; only a real GPU training/inference smoke test establishes local runtime compatibility.
