# SPDX-License-Identifier: Apache-2.0
"""Run hardware-independent tests without importing the platform bootstrap."""

import sys
import types
import uuid
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
for name in ("vllm_omni", "vllm_omni.engine", "vllm_omni.model_executor", "vllm_omni.model_executor.models"):
    package = types.ModuleType(name)
    package.__path__ = [str(root / name.replace(".", "/"))]
    sys.modules[name] = package
paths = [
    "tests/model_executor/models/qwen3_tts/test_code_predictor_adaptive.py",
    "tests/model_executor/models/qwen3_tts/test_code_predictor_dtype.py",
    "tests/model_executor/models/qwen3_tts/test_code_predictor_pipeline.py",
    "tests/core/sched/test_playback_deadline.py",
    "tests/core/sched/test_audio_interaction.py",
    "tests/benchmarks/test_gstreamer_player.py",
]
temporary = root / ".pytest_cache" / ("playback-" + uuid.uuid4().hex)
raise SystemExit(
    pytest.main(["--noconftest", "-o", "addopts=", "--basetemp", str(temporary), "-q", *[str(root / p) for p in paths]])
)
