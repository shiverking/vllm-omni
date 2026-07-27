# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from vllm_omni.entrypoints.cli.benchmark.liveserve_audio import build_serve_argv


def _args(tmp_path):
    return Namespace(
        model="/models/Qwen3-TTS-12Hz-1.7B-Base",
        dataset_path="/data/seed-tts-eval",
        concurrency=8,
        requests=32,
        output_dir=str(tmp_path),
        host="127.0.0.1",
        port=8000,
    )


def test_compact_command_expands_fixed_liveserve_protocol(tmp_path):
    argv = build_serve_argv(_args(tmp_path))
    assert argv[argv.index("--backend") + 1] == "openai-audio-speech"
    assert argv[argv.index("--num-warmups") + 1] == "2"
    assert argv[argv.index("--max-concurrency") + 1] == "8"
    assert argv[argv.index("--metric-percentiles") + 1] == "10,50,90,99"
    assert "--enable-playback-feedback" in argv
    assert "--save-audio-timeline" in argv
    assert "--workload-manifest-out" in argv


def test_compact_command_replays_existing_manifest(tmp_path):
    manifest = tmp_path / "seed_tts_en_n32_manifest.json"
    manifest.write_text('{"schema_version":1,"requests":[]}', encoding="utf-8")
    argv = build_serve_argv(_args(tmp_path))
    assert "--workload-manifest-in" in argv
    assert "--workload-manifest-out" not in argv
