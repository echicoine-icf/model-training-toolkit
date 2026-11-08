# Resumable training checkpoints (MTT v0.3.6+)

Starting with v0.3.6, **both** `pytorch` and `unsloth` backends create complete, resumable checkpoints. These are distinct from Unsloth's **gradient checkpointing**, which reduces training memory usage but does **not** save a resumable training run.

## Initial run

Every `train` run defaults to saving a complete checkpoint **every 500 optimizer steps**. For a different interval, add these optional flags to your ordinary full one-shot training command:

```bash
--save-every 500 --keep-last-checkpoints 3
```

`--save-every 0` disables resumable checkpoints, and `--keep-last-checkpoints 0` retains every completed checkpoint. The default retention is the three latest complete checkpoints, which trades disk usage against rollback flexibility. A checkpoint can be considerably larger than a LoRA adapter because AdamW momentum and variance state are included. On low disk space, reduce retention, **not** the integrity of the checkpoint. Check disk space before multi-hour jobs.

Each checkpoint under `runs/<project>/<version>/<run-id>/checkpoints/step_0000500/` stores:

- Standard PEFT adapter and tokenizer files.
- `training_state.pt` with exact trainable LoRA tensors, optimizer state, AMP GradScaler state, Python/NumPy/Torch/CUDA RNG state, completed optimizer update, processed microbatch position, elapsed time, and last training metrics.
- `checkpoint_info.json` as the completion marker, with SHA-256 hashes of all checkpoint files and a signature of the immutable training inputs and effective hyperparameters.

A checkpoint is first written to a hidden `.step_*.incomplete_*` directory and atomically renamed when complete. A crash during writing leaves that staging directory, **not** a seemingly valid checkpoint. A resume checks the manifest, all SHA-256 hashes, and the frozen training plan before updating the model. It also verifies LoRA parameter names and tensor shapes.

## Recover after interruption, crash, or power outage

1. Ensure the original process is no longer running, and ensure you have the same dependencies/backend available. Do **not** change the model, training data, prompt, or original run inputs.
2. Find the original run ID:

```bash
bash workspace.sh runs --project eli
# Or: bash workspace.sh status eli v9.1
```

3. Resume **the existing run**, from its latest complete checkpoint:

```bash
bash projects/eli/project.sh resume v9.1 --run run_YYYYMMDDTHHMMSSffffffZ_abcdef
# equivalent:
bash workspace.sh resume eli v9.1 --run run_YYYYMMDDTHHMMSSffffffZ_abcdef
```

You can choose an older retained checkpoint explicitly:

```bash
bash workspace.sh resume eli v9.1 --run RUN_ID --checkpoint step_0000500
```

The command **does not** request training flags because it restores the original run's frozen settings. It does not create a new run ID or a new data snapshot. It skips completed examples in the deterministic original epoch order, continuing with the next microbatch and optimizer update. The loss log is appended with an explicit `resume` event; if you crashed after a checkpoint, later logged steps may appear twice because those computations were rolled back to the checkpoint.

If the latest checkpoint is corrupt or incomplete, resume fails safely rather than silently using a different checkpoint. Use `--checkpoint step_XXXXXXX` to select an earlier verified one.

**Important:** v0.3.5 and earlier checkpoints are *adapter-only* and **cannot be converted into exact optimizer/RNG resumable checkpoints**. Upgrading while a training process is still running does not retroactively add the feature. Complete the existing training first; subsequent runs created under v0.3.6 receive complete checkpoints. Checkpoint resumption on the user's GPU and Unsloth backend still requires an end-to-end hardware smoke test.

## Reproducibility and security

- Resume requires matching effective model revision, tokenizer/chat template, LoRA layer names, backend and version, PyTorch version, optimizer, precision, attention, LR schedule, frozen input hashes, and other signed configuration. Different dependencies might fail validation intentionally rather than changing the run silently.
- Reusing a run snapshot does not guarantee bit-for-bit reproducibility across hardware, CUDA kernels, nondeterministic operations, or third-party library behavior. The toolkit restores captured RNG state but cannot control all backend internals.
- The checkpoint's `training_state.pt` is loaded with `torch.load(..., weights_only=False)` to restore optimizer and Python RNG state. **Only resume locally generated, trusted checkpoints. Never open checkpoints from untrusted sources.**
- Changes to the toolkit's training implementation during an existing run can affect continuity. Do not upgrade software or change training dependencies while a job is running; pin/test the runtime before resuming.
- A failure before the first successful checkpoint still requires a fresh run. A checkpoint interval of 500 bounds the recoverable work to approximately the last 500 optimizer updates, not zero.
