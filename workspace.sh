#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
  PY="$ROOT/.venv/Scripts/python.exe"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  echo "Toolkit .venv is missing. Run bash run_me_first.sh first." >&2
  exit 1
fi
export MTT_HOME="$ROOT"
export HF_HOME="$ROOT/cache/huggingface"
exec "$PY" "$ROOT/toolkit/cli.py" "$@"
