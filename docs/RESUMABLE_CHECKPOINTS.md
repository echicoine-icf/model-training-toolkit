# Resumable training checkpoints

MTT saves complete training checkpoints for its PyTorch and optional Unsloth backends. These differ from **gradient checkpointing**, which saves VRAM during training rather than preserving optimizer progress.

## Configuration

Add these flags to an explicitly configured training command:

```text
--save-every 500 --keep-last-checkpoints 3
```

- `--save-every 0`: disable periodic checkpointing.
- `--keep-last-checkpoints 0`: retain all complete checkpoints.
- More checkpoints consume additional disk space; each can include sizable optimizer state.

## Contents and integrity

Checkpoints contain LoRA tensor state, optimizer state, applicable AMP scaler state, Python/NumPy/Torch/CUDA RNG state, completed step and training position, configuration and integrity metadata. A checkpoint is marked complete only after its files are written; interrupted/partial saves must not be reused as if valid.

## Resume a run

```bash
bash workspace.sh runs --project PROJECT
bash projects/PROJECT/project.sh resume VERSION --run RUN_ID

# To select a retained checkpoint deliberately:
bash workspace.sh resume PROJECT VERSION --run RUN_ID --checkpoint step_0000500
```

Resuming uses the **original run**, its frozen data, optimizer settings and model revision; do not change the model architecture, dataset, LoRA configuration or training backend. Checkpoint integrity is verified first. The oldest checkpoints created before complete optimizer/RNG state support cannot be magically upgraded into exact-resume checkpoints. If no complete checkpoint exists, the run may need to be restarted.

## Operational safeguards

Never resume into the same run from two processes; stop any active training before invoking resume. Keep backups of important runs before testing new toolkit releases. Avoid editing or deleting run metadata, adapter files and checkpoint state to resolve errors. Inspect logs and checkpoint receipts instead.
