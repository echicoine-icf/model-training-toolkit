# Architecture — v0.3.5

**Installer** `run_me_first.sh` → stdlib `scripts/bootstrap.py` → isolated `.venv` and pinned llama.cpp source.

**Workspace CLI** `workspace.sh` → `.venv` Python `toolkit/cli.py` → `project create` / `version create` / `data inspect` / `train` / `evaluate` / `export` / `handoff`.

**Version data** `projects/<project>/versions/<version>/data/` is a stable content contract, independent from any one training experiment or ChatGPT conversation.

**Run lifecycle** version inputs validate → model & hyperparameters reviewed → Hugging Face base commit resolved → configuration and all version data frozen with SHA-256 → fresh LoRA training → adapter saved → optional evaluation → registry and handoff.

**Canonical evaluation** exact JSON only until task-specific semantic scorer modules are added. Do not interpret successful JSON parsing or exact equality as semantic correctness for unordered conditions/graph bindings.

**Storage** `.venv`, downloaded models, llama.cpp source tree, actual training runs and caches are local and never included in a distributable source ZIP.

**Training backends (v0.3.5):** `toolkit/backends.py` handles opt-in Unsloth initialization; `toolkit/train_engine.py` retains the shared token masking, PyTorch optimizer/gradient loop, scheduling, and PEFT audit. `--backend pytorch` requires no Unsloth import. `--backend unsloth` imports Unsloth before Torch/Transformers and loads the exact pinned model in full precision, then verifies the wrapped PEFT modules. Experimental GPU compatibility and performance must be measured on the target hardware before deployment.
