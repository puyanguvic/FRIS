#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first: https://github.com/astral-sh/uv"
  exit 1
fi

if [ ! -d ".venv" ]; then
  uv venv .venv
fi

uv pip install -e .

echo "Setup complete. Activate with: source .venv/bin/activate"
