#!/usr/bin/env bash
# Run from the repository root in the installed vLLM-Omni / Ascend environment.
set -euo pipefail
MODEL=${1:?model checkpoint path or repository}
OUT=${2:?absolute experiment output directory}
MODE=${3:-collect}
PORT=${4:-8000}
if [[ "$MODE" == stock ]]; then
  unset VLLM_OMNI_TTS_SPEC_CONFIG
else
  case "$MODE" in
    collect|chain|tree8|tree16|tree32) ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
  esac
  export VLLM_OMNI_TTS_SPEC_CONFIG="$OUT/${MODE}_config.json"
  test -f "$VLLM_OMNI_TTS_SPEC_CONFIG"
fi
exec vllm-omni serve "$MODEL" --omni --trust-remote-code \
  --deploy-config "$OUT/deploy.yaml" --host 127.0.0.1 --port "$PORT"
