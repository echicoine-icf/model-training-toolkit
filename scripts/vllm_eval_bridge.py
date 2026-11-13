#!/usr/bin/env python3
"""Local Ollama-protocol bridge for vLLM's OpenAI-compatible inference server.

Allows an existing MTT project-owned Ollama evaluation runner to assess exactly
its existing cases/scorer against a vLLM-hosted quantized checkpoint.
Do not expose this unauthenticated local bridge to the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def model_digest(receipt: Path) -> str:
    data = json.loads(receipt.read_text(encoding='utf-8'))
    if data.get('format') not in ('vllm-fp8', 'vllm-int8', 'vllm-w4a16'):
        raise ValueError('Receipt is not for an MTT vLLM quantized model')
    files = data.get('output_integrity')
    if not isinstance(files, dict) or not any(k.endswith('.safetensors') for k in files):
        raise ValueError('Missing quantized SafeTensors identity')
    # Verify that the local model still matches the receipt before serving it.
    root = receipt.parent
    for name, info in files.items():
        p=(root/name).resolve()
        if not p.is_file() or not p.is_relative_to(root.resolve()):
            raise ValueError('Quantized model missing source file: ' + name)
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                h.update(block)
        if h.hexdigest()!=info['sha256']:
            raise ValueError('Quantized model checksum mismatch: ' + name)
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def unquantized_model_digest(folder: Path) -> str:
    """Stable identity for uncompressed merged SafeTensors (F16/BF16 baseline)."""
    root = folder.resolve()
    if not (root / 'config.json').is_file() or not list(root.glob('*.safetensors')):
        raise ValueError('F16/BF16 baseline requires merged Hugging Face config and SafeTensors')
    h = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Cannot validate symlink in baseline: {path}')
        if path.is_file():
            h.update(path.relative_to(root).as_posix().encode('utf-8'))
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024*1024), b''):
                    h.update(block)
    return h.hexdigest()


def handler_factory(model_name: str, model_hash: str, api_base: str, max_context: int):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, payload: dict):
            blob=json.dumps(payload).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self):
            if self.path=='/api/tags':
                return self.respond(200,{'models':[{'name':model_name,'model':model_name,'digest':model_hash}]})
            self.respond(404,{'error':'Unknown endpoint'})

        def do_POST(self):
            if self.path!='/api/chat':
                return self.respond(404,{'error':'Unknown endpoint'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size<2 or size>1024*1024:
                    raise ValueError('Request size out of range')
                payload=json.loads(self.rfile.read(size).decode('utf-8'))
                if payload.get('model')!=model_name or payload.get('stream') is not False:
                    raise ValueError('Incorrect model name or streaming unsupported')
                options=payload.get('options') or {}
                if float(options.get('temperature',0))!=0:
                    raise ValueError('Only deterministic temperature=0 evaluations are supported')
                if int(options.get('num_ctx',max_context))>max_context:
                    raise ValueError('Evaluation context exceeds vLLM server max-model-len')
                out={'model':model_name,'messages':payload['messages'],
                     'temperature':0,'top_p':1,'n':1,'stream':False,
                     'max_tokens':int(options.get('num_predict',768)),
                     'seed':int(options.get('seed',42))}
                request=Request(api_base.rstrip('/')+'/v1/chat/completions',
                        data=json.dumps(out).encode(),headers={'Content-Type':'application/json'},method='POST')
                with urlopen(request,timeout=600) as resp:
                    answer=json.load(resp)
                raw=answer['choices'][0]['message']['content']
                if not isinstance(raw,str):
                    raise ValueError('vLLM response must contain plain string content')
                usage=answer.get('usage') or {}
                return self.respond(200,{'model':model_name,'message':{'role':'assistant','content':raw},
                                        'done_reason':answer['choices'][0].get('finish_reason'),
                                        'eval_count':usage.get('completion_tokens'),
                                        'prompt_eval_count':usage.get('prompt_tokens')})
            except HTTPError as exc:
                return self.respond(502,{'error':f'vLLM HTTP error {exc.code}'})
            except (URLError,ValueError,KeyError,IndexError,TypeError) as exc:
                return self.respond(502,{'error':str(exc)})

        def log_message(self, fmt, *args):
            pass
    return Handler


def main():
    p=argparse.ArgumentParser(description='Loopback-only vLLM-to-Ollama evaluation API bridge')
    p.add_argument('--model',required=True,help='vLLM --served-model-name')
    model_source=p.add_mutually_exclusive_group(required=True)
    model_source.add_argument('--receipt',help='Quantized model MTT_QUANTIZATION.json')
    model_source.add_argument('--model-dir',help='Unquantized merged HF model folder for F16/BF16 baseline')
    p.add_argument('--vllm-url',default='http://127.0.0.1:8000')
    p.add_argument('--port',type=int,default=11435)
    p.add_argument('--max-context',type=int,default=4096)
    args=p.parse_args()
    if args.port<=0 or args.port>65535 or args.max_context<512:
        p.error('Invalid local port or maximum context length')
    hash=(model_digest(Path(args.receipt).resolve()) if args.receipt else
          unquantized_model_digest(Path(args.model_dir).resolve()))
    server=ThreadingHTTPServer(('127.0.0.1',args.port),
                   handler_factory(args.model,hash,args.vllm_url,args.max_context))
    print(f'VLLM EVAL BRIDGE 127.0.0.1:{args.port} model={args.model} digest={hash}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__=='__main__':
    main()
