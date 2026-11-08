#!/usr/bin/env bash
# `source ./activate.sh` to modify the current shell. Do not `bash activate.sh`.
_MTT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_MTT_HOME/.venv/Scripts/activate" ]]; then
  # shellcheck source=/dev/null
  source "$_MTT_HOME/.venv/Scripts/activate"
elif [[ -f "$_MTT_HOME/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$_MTT_HOME/.venv/bin/activate"
else
  echo "Missing .venv: run bash run_me_first.sh first" >&2
  return 1 2>/dev/null || exit 1
fi
export MTT_HOME="$_MTT_HOME"
export HF_HOME="$_MTT_HOME/cache/huggingface"
echo "Model Training Toolkit venv active: $VIRTUAL_ENV"
