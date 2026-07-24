import argparse
import asyncio
import os
from typing import Any

from vllm.benchmarks.serve import main_async

# Import patch to register daily-omni dataset and omni backends
# This monkey-patches vllm.benchmarks.datasets.get_samples before it's used
# Must be imported before any vllm.benchmarks module usage
import vllm_omni.benchmarks.patch.patch  # noqa: F401
from vllm_omni.benchmarks.patch.patch import (
    configure_audio_benchmark,
    maybe_enable_stage_metrics,
    set_print_stage,
    should_request_stage_metrics,
)


def main(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "seed_tts_wer_eval", False):
        os.environ["SEED_TTS_WER_EVAL"] = "1"
    if getattr(args, "seed_tts_wer_save_items", False):
        os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
    if getattr(args, "daily_omni_save_eval_items", False):
        os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"
    set_print_stage(getattr(args, "print_stage", False))
    configure_audio_benchmark(
        save_audio_timeline=getattr(args, "save_audio_timeline", False),
        useful_audio_ttfp_ms=getattr(args, "useful_audio_ttfp_ms", 1000.0),
        useful_require_audio_rtf=getattr(args, "useful_require_audio_rtf", False),
        useful_audio_rtf_max=getattr(args, "useful_audio_rtf_max", 1.0),
        realtime_min_continuity=getattr(args, "realtime_min_continuity", 0.95),
        realtime_max_p90_rtf=getattr(args, "realtime_max_p90_rtf", 1.0),
    )
    args.extra_body = maybe_enable_stage_metrics(
        getattr(args, "extra_body", None),
        enabled=should_request_stage_metrics(args),
    )
    return asyncio.run(main_async(args))
