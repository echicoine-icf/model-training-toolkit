"""Optional fine-tuning model backends; no model-specific assumptions.

Keep this module stdlib-only at import time. Unsloth must be imported *before*
Torch/Transformers/PEFT in a fresh training process so its patches take effect.
"""
from __future__ import annotations
import importlib
from importlib import metadata

SUPPORTED_BACKENDS = ('pytorch', 'unsloth')


def prepare_backend(backend: str):
    """Return FastLanguageModel for Unsloth or None for stock PyTorch.

    Never silently fall back to PyTorch if Unsloth is absent or fails to load.
    Call before importing torch, transformers or peft from the training process.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f'Unsupported training backend {backend!r}; choose {SUPPORTED_BACKENDS}')
    if backend == 'pytorch':
        return None
    try:
        unsloth = importlib.import_module('unsloth')
    except ImportError as exc:
        raise RuntimeError(
            'Unsloth backend requested but it is not installed or its dependencies failed to import. '
            'See docs/UNSLOTH_BACKEND.md. The PyTorch backend was NOT substituted.'
        ) from exc
    model_class = getattr(unsloth, 'FastLanguageModel', None)
    if model_class is None:
        raise RuntimeError('Installed Unsloth does not expose FastLanguageModel; no fallback performed')
    return model_class


def backend_version(backend: str):
    if backend == 'pytorch':
        return None
    try:
        return metadata.version('unsloth')
    except metadata.PackageNotFoundError:
        return 'unknown'


def load_unsloth_base(fast_model, *, base, max_length, precision, attention):
    """Load original full-precision model; never change repo or use 4bit weights."""
    kwargs = dict(
        model_name=base['id'],
        revision=base.get('revision') or 'main',
        use_exact_model_name=True,
        max_seq_length=max_length,
        dtype=precision,
        load_in_4bit=False,
        load_in_8bit=False,
        trust_remote_code=False,
        attn_implementation=attention,
    )
    model, tokenizer = fast_model.from_pretrained(**kwargs)
    return model, tokenizer


def attach_unsloth_lora(fast_model, model, *, lora, seed, max_length,
                         gradient_checkpointing, checkpointing_mode='standard'):
    """Apply arbitrary PEFT LoRA targets; leave training loop and losses to MTT."""
    if checkpointing_mode not in ('standard', 'unsloth'):
        raise ValueError('unsloth_checkpointing must be standard or unsloth')
    requested = list(lora['target_modules'])
    if not requested or len(set(requested)) != len(requested) or not all(requested):
        raise ValueError('LoRA target modules must be nonempty, unique names')
    ck = (checkpointing_mode if checkpointing_mode == 'unsloth' else True) if gradient_checkpointing else False
    return fast_model.get_peft_model(
        model,
        r=int(lora['r']),
        target_modules=requested,
        lora_alpha=int(lora['alpha']),
        lora_dropout=float(lora['dropout']),
        bias='none',
        init_lora_weights=True,
        use_gradient_checkpointing=ck,
        random_state=int(seed),
        max_seq_length=int(max_length),
        use_rslora=False,
        loftq_config=None,
    )
