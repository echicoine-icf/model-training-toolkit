#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MTT_BOOTSTRAP_PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN (install Python 3.11+ and ensure it is on PATH)." >&2
  exit 1
fi
exec "$PYTHON_BIN" "$ROOT/scripts/bootstrap.py" "$@"
