#!/usr/bin/env bash
set -euo pipefail
OUT=${1:?experiment output directory}
DEVICE=${2:-npu:0}
# Analyze is intentionally not repeated: a frozen tree must not be retuned.
if [[ ! -f "$OUT/trees.json" ]]; then
  python tools/qwen3_tts_spec/experiment.py analyze --output "$OUT" --device "$DEVICE"
fi
python tools/qwen3_tts_spec/experiment.py replay --output "$OUT" --device "$DEVICE"
python tools/qwen3_tts_spec/experiment.py report --output "$OUT" --device "$DEVICE"
