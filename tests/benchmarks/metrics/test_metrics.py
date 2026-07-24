# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for metrics.py
"""

from types import SimpleNamespace

import pytest
from vllm.benchmarks.serve import TaskType

from vllm_omni.benchmarks.metrics.metrics import calculate_metrics
from vllm_omni.benchmarks.patch.patch import MixRequestFuncOutput, build_audio_request_results, check_goodput_args

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def _make_output(prompt_len: int, output_tokens: int = 10) -> MixRequestFuncOutput:
    """Build a minimal successful MixRequestFuncOutput for metrics aggregation."""
    output = MixRequestFuncOutput()
    output.success = True
    output.prompt_len = prompt_len
    output.output_tokens = output_tokens
    output.generated_text = "x" * output_tokens
    output.ttft = 0.1
    output.text_latency = 1.0
    output.latency = 1.0
    output.start_time = 0.0
    output.itl = [0.1] * max(output_tokens - 1, 0)
    output.audio_ttfp = 0.0
    output.audio_rtf = 0.0
    output.audio_duration = 0.0
    output.audio_frames = 0
    output.input_audio_duration = 0.0
    output.error = ""
    return output


# ============================================================================
# total_input Tests
# ============================================================================


def test_total_input_aggregated_from_output_prompt_len():
    """Test that total_input sums outputs[i].prompt_len, not input_requests[i].prompt_len."""
    outputs = [_make_output(4992), _make_output(3000)]

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=10.0,
        tokenizer=None,
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=[],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
    )

    assert metrics.total_input == 7992, (
        "total_input should aggregate from outputs[i].prompt_len to reflect the true multimodal input token count"
    )


def test_audio_continuity_aggregation():
    """Continuity rate and underrun percentile must aggregate from per-output fields."""
    bad = _make_output(100)
    bad.audio_underrun_s = 0.5
    bad.audio_continuity_ok = False
    good_a = _make_output(100)
    good_a.audio_underrun_s = 0.02
    good_a.audio_continuity_ok = True
    good_b = _make_output(100)
    good_b.audio_underrun_s = 0.0
    good_b.audio_continuity_ok = True

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=[bad, good_a, good_b],
        dur_s=10.0,
        tokenizer=None,
        selected_percentiles=[50.0, 99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=["audio_underrun"],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
    )

    assert metrics.audio_continuity_ok_rate == pytest.approx(2 / 3, abs=1e-6)
    # p99 of [0.5, 0.02, 0.0] is dominated by the 0.5 outlier.
    p99 = dict(metrics.percentiles_audio_underrun_s).get(99.0)
    assert p99 is not None and p99 > 0.4


def test_useful_audio_throughput_and_realtime_capacity():
    useful = _make_tts_output(100)
    useful.audio_ttfp = 0.2
    useful.audio_rtf = 0.8
    useful.audio_continuity_ok = True
    late = _make_tts_output(100)
    late.audio_ttfp = 1.2
    late.audio_rtf = 0.9
    late.audio_continuity_ok = True
    underrun = _make_tts_output(100)
    underrun.audio_ttfp = 0.1
    underrun.audio_rtf = 0.7
    underrun.audio_continuity_ok = False

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=[useful, late, underrun],
        dur_s=2.0,
        tokenizer=_EmptyAwareTokenizer(),
        selected_percentiles=[90.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=[],
        max_concurrency=3,
        request_rate=float("inf"),
        benchmark_duration=2.0,
        useful_audio_ttfp_s=1.0,
    )

    assert metrics.useful_request_count == 1
    assert metrics.useful_request_throughput == pytest.approx(0.5)
    assert metrics.audio_throughput == pytest.approx(7.5)
    assert metrics.realtime_capacity_pass is False


def test_audio_goodput_slos():
    good = _make_tts_output(100)
    good.audio_ttfp = 0.2
    good.audio_rtf = 0.8
    good.audio_continuity_ok = True
    bad = _make_tts_output(100)
    bad.audio_ttfp = 0.4
    bad.audio_rtf = 1.2
    bad.audio_continuity_ok = True

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=[good, bad],
        dur_s=2.0,
        tokenizer=_EmptyAwareTokenizer(),
        selected_percentiles=[90.0],
        goodput_config_dict={"audio_ttfp": 300.0, "audio_rtf": 1.0, "audio_continuity": 1.0},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=[],
        max_concurrency=2,
        request_rate=float("inf"),
        benchmark_duration=2.0,
    )

    assert metrics.request_goodput == pytest.approx(0.5)


def test_audio_goodput_cli_validation():
    parsed = check_goodput_args(
        SimpleNamespace(goodput=["audio_ttfp:300", "audio_rtf:1", "audio_continuity:1"])
    )
    assert parsed == {"audio_ttfp": 300.0, "audio_rtf": 1.0, "audio_continuity": 1.0}
    with pytest.raises(ValueError, match="audio_continuity"):
        check_goodput_args(SimpleNamespace(goodput=["audio_continuity:0.95"]))


def test_audio_request_result_schema_timeline_is_opt_in():
    output = _make_tts_output(100)
    output.request_id = "seedtts-0001"
    output.audio_underrun_s = 0.02
    output.audio_underrun_event_count = 1
    output.audio_continuity_ok = True
    output.audio_timeline = [{"arrival_time_s": 0.1, "bytes": 4800, "audio_duration_s": 0.1}]

    compact = build_audio_request_results([output], save_timeline=False)[0]
    detailed = build_audio_request_results([output], save_timeline=True)[0]
    assert set(compact) == {
        "request_id",
        "success",
        "audio_ttfp_s",
        "e2e_latency_s",
        "audio_duration_s",
        "audio_rtf",
        "max_underrun_s",
        "underrun_event_count",
        "continuity_ok",
        "feedback_sent_count",
        "feedback_coalesced_count",
        "feedback_failure_count",
        "max_feedback_delay_s",
        "playback_buffer_timeline",
    }
    assert "audio_timeline" not in compact
    assert detailed["audio_timeline"] == output.audio_timeline


# ============================================================================
# TTFT suppression for pure-audio (TTS) benchmarks
# ============================================================================


class _EmptyAwareTokenizer:
    """Minimal tokenizer stub: token count == len(text), so '' -> 0 tokens.

    Mirrors production where a TTS speech endpoint returns empty generated_text,
    making total_output == 0 (the real CI path uses a real tokenizer, not None).
    """

    def __call__(self, text, add_special_tokens=False):
        class _R:
            pass

        r = _R()
        r.input_ids = [0] * len(text)
        return r


def _make_tts_output(prompt_len: int) -> MixRequestFuncOutput:
    """Pure-TTS output: no text tokens, only audio. ttft is left unset (0.0)."""
    output = MixRequestFuncOutput()
    output.success = True
    output.prompt_len = prompt_len
    output.output_tokens = 0
    output.generated_text = ""
    output.ttft = 0.0
    output.text_latency = 1.0
    output.latency = 1.0
    output.start_time = 0.0
    output.itl = []
    output.audio_ttfp = 0.05
    output.audio_rtf = 0.2
    output.audio_duration = 5.0
    output.audio_frames = 120000
    output.input_audio_duration = 0.0
    output.error = ""
    return output


_TTS_PERCENTILE_METRICS = ["ttft", "e2el", "audio_rtf", "audio_ttfp", "audio_duration"]


def test_tts_benchmark_omits_ttft(capsys):
    """Pure-TTS run (total_output == 0) must not print a Time to First Token section."""
    outputs = [_make_tts_output(100), _make_tts_output(120)]

    calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=10.0,
        tokenizer=_EmptyAwareTokenizer(),
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=_TTS_PERCENTILE_METRICS,
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
    )

    out = capsys.readouterr().out
    assert "Time to First Token" not in out, "TTS bench must not surface a meaningless TTFT"
    assert "Time to First Packet" in out, "audio TTFP must still be reported"
    assert "End-to-end Latency" in out, "e2el must still be reported"


def test_text_benchmark_still_reports_ttft(capsys):
    """Regression guard: real text generation (total_output > 0) keeps TTFT."""
    outputs = [_make_output(100, output_tokens=10), _make_output(120, output_tokens=10)]

    calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=10.0,
        tokenizer=_EmptyAwareTokenizer(),
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=_TTS_PERCENTILE_METRICS,
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
    )

    out = capsys.readouterr().out
    assert "Time to First Token" in out, "text benchmarks must keep TTFT"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
