#!/usr/bin/env bash
# Launch the Gradio demo locally with an optional public *.gradio.live tunnel.
#
# Usage:
#   bash scripts/launch_demo.sh                    # local only
#   bash scripts/launch_demo.sh --share            # also create public *.gradio.live URL (72h)
#   bash scripts/launch_demo.sh --single           # single-model variant (faster, lighter)
#   bash scripts/launch_demo.sh --share --single   # both

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="models/ensemble_final/config.json"
SHARE=""

for arg in "$@"; do
    case $arg in
        --share)
            SHARE=1
            shift
            ;;
        --single)
            CONFIG="models/single_final/config.json"
            shift
            ;;
        *)
            echo "Unknown arg: $arg"
            exit 1
            ;;
    esac
done

if [ ! -f .venv/Scripts/python.exe ] && [ ! -f .venv/bin/python ]; then
    echo ".venv not found. Run: python -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

# Pick the right python
if [ -f .venv/Scripts/python.exe ]; then
    PY=".venv/Scripts/python.exe"
else
    PY=".venv/bin/python"
fi

ENSEMBLE_CONFIG="$CONFIG" SHARE="${SHARE:-}" "$PY" app/space_app.py
