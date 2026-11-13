#!/usr/bin/env python3
"""Produce vLLM compressed-tensors checkpoints in an isolated Python environment.

Install transformers, torch, safetensors and llmcompressor ONLY in that environment.
No calibration is used for FP8_DYNAMIC, W8A16, or W4A16, or experimental W2A16 weight-only quantization.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description='Compress an existing merged HF model for vLLM')
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scheme', choices=['FP8_DYNAMIC', 'W8A16', 'W4A16', 'W2A16'], required=True)
    args = p.parse_args(argv)
    source = Path(args.source).resolve()
    out = Path(args.output).resolve()
    if not (source/'config.json').is_file() or not list(source.glob('*.safetensors')):
        p.error('Source must be a merged Hugging Face SafeTensors model')
    if source == out or source in out.parents or out in source.parents:
        p.error('Source and destination must be separate, non-nested paths')
    if out.exists():
        p.error('Destination exists; refusing overwrite')

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError as ex:
        raise RuntimeError('Install torch, transformers, safetensors and llmcompressor in the SEPARATE quantization environment') from ex

    # Load original merged artifact, not a base checkpoint or PEFT adapter.
    model = AutoModelForCausalLM.from_pretrained(
        str(source), dtype=torch.float16, trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=False)
    oneshot(model=model, recipe=QuantizationModifier(
        targets='Linear', scheme=args.scheme, ignore=['lm_head']))
    out.mkdir(parents=True)
    model.save_pretrained(str(out), safe_serialization=True, save_compressed=True)
    tokenizer.save_pretrained(str(out))
    config = json.loads((out/'config.json').read_text(encoding='utf-8'))
    if not config.get('quantization_config') or not list(out.glob('*.safetensors')):
        raise RuntimeError('Quantization checkpoint not serialized correctly')
    print('COMPRESSED_MODEL', out, 'scheme=', args.scheme, flush=True)

if __name__ == '__main__':
    main()
