# SPDX-License-Identifier: Apache-2.0
"""Compact single-policy LiveServe audio validity benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand


def build_serve_argv(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / f"seed_tts_en_n{args.requests}_manifest.json"
    result_name = f"liveserve_c{args.concurrency}_n{args.requests}.json"
    argv = [
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--backend",
        "openai-audio-speech",
        "--endpoint",
        "/v1/audio/speech",
        "--dataset-name",
        "seed-tts",
        "--dataset-path",
        args.dataset_path,
        "--seed-tts-locale",
        "en",
        "--extra-body",
        '{"task_type":"Base"}',
        "--seed",
        "0",
        "--num-prompts",
        str(args.requests),
        "--num-warmups",
        "2",
        "--max-concurrency",
        str(args.concurrency),
        "--request-rate",
        "inf",
        "--enable-playback-feedback",
        "--save-audio-timeline",
        "--useful-audio-ttfp-ms",
        "1000",
        "--useful-require-audio-rtf",
        "--useful-audio-rtf-max",
        "1.0",
        "--realtime-min-continuity",
        "0.95",
        "--realtime-max-p90-rtf",
        "1.0",
        "--goodput",
        "audio_ttfp:1000",
        "streaming_audio_rtf:1",
        "audio_continuity:1",
        "--percentile-metrics",
        "e2el,audio_ttfp,audio_rtf,audio_duration,audio_underrun",
        "--metric-percentiles",
        "10,50,90,99",
        "--save-result",
        "--result-dir",
        str(output_dir),
        "--result-filename",
        result_name,
    ]
    argv += ["--workload-manifest-in" if manifest.exists() else "--workload-manifest-out", str(manifest)]
    return argv


class LiveServeAudioBenchmarkSubcommand(OmniBenchmarkSubcommandBase):
    name = "liveserve-audio"
    help = "Run a compact GStreamer-backed LiveServe audio validity benchmark."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", required=True)
        parser.add_argument("--dataset-path", required=True)
        parser.add_argument("--concurrency", type=int, default=8)
        parser.add_argument("--requests", type=int, default=32)
        parser.add_argument("--output-dir", default="results/liveserve-validity")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if args.concurrency <= 0 or args.requests <= 0:
            raise SystemExit("--concurrency and --requests must be positive")
        parser = argparse.ArgumentParser(add_help=False)
        OmniBenchmarkServingSubcommand.add_cli_args(parser)
        serve_args = parser.parse_args(build_serve_argv(args))
        serve_args.omni = True
        OmniBenchmarkServingSubcommand.cmd(serve_args)
