"""Unit tests for the single-NPU LiveServe experiment driver."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "tts"))
import run_liveserve_audio_sweep as sweep

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_arrival_trace_is_deterministic_and_monotonic():
    first = sweep.generate_arrival_offsets(128, 2.5, 7)
    second = sweep.generate_arrival_offsets(128, 2.5, 7)
    assert first == second
    assert first[0] == 0
    assert all(current >= previous for previous, current in zip(first, first[1:]))


def test_benchmark_command_replays_manifest_and_trace(tmp_path):
    args = Namespace(
        bench_bin="vllm-omni",
        model="/models/Qwen3-TTS-12Hz-1.7B-Base",
        host="127.0.0.1",
        port=8000,
        dataset_path=tmp_path / "dataset",
        num_requests=128,
        num_warmups=2,
        useful_audio_ttfp_ms=1000.0,
        benchmark_extra_arg=[],
    )
    command = sweep.build_benchmark_command(
        args,
        result_path=tmp_path / "result.json",
        manifest_path=tmp_path / "manifest.json",
        arrival_trace=tmp_path / "trace.json",
    )
    assert command[command.index("--model") + 1] == args.model
    assert "--workload-manifest-in" in command
    assert "--arrival-trace-in" in command
    assert "--enable-playback-feedback" in command
    assert command[command.index("--num-prompts") + 1] == "128"


def test_local_model_server_command_omits_revision(tmp_path):
    args = Namespace(
        serve_bin="vllm",
        model="/models/Qwen3-TTS-12Hz-1.7B-Base",
        model_revision=None,
        host="127.0.0.1",
        port=8000,
        server_extra_arg=[],
    )

    command = sweep.build_server_command(args, tmp_path / "deploy.yaml")

    assert command[:3] == ["vllm", "serve", args.model]
    assert "--revision" not in command


def test_verify_results_rejects_audio_duration_drift():
    records = []
    for index, strategy in enumerate(sweep.STRATEGIES):
        records.append(
            {
                "completed": 128,
                "failed": 0,
                "total_audio_duration_s": 100.0 if index < 2 else 110.0,
                "experiment": {
                    "strategy": strategy,
                    "mode": "fixed",
                    "concurrency": 4,
                    "load_factor": None,
                    "repeat": 0,
                    "manifest_sha256": "same",
                    "model": "/models/Qwen3-TTS-12Hz-1.7B-Base",
                    "model_revision": "revision",
                    "common_config_sha256": "config",
                },
            }
        )
    errors = sweep.verify_results(records, "same")
    assert any("3%" in error for error in errors)
