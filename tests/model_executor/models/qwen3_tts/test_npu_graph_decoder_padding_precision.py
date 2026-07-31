# SPDX-License-Identifier: Apache-2.0
"""Ascend A2 production-gate evaluation for Code2Wav graph padding.

The production NPU wrapper remains exact-only.  This test module is the only
place that writes a short codec prefix into a larger captured graph input.
"""

from __future__ import annotations

import gc
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch

from tests.model_executor.models.qwen3_tts.code2wav_padding_eval import (
    PAD_MODES,
    PaddingThresholds,
    aggregate_rows,
    analyze_blind_results,
    assert_exact_metrics,
    assert_waveform_contract,
    eager_padded_decode,
    embedding_cosine_drop,
    export_blind_pairs,
    graph_mixed_padded_decode,
    graph_padded_decode,
    load_codec_tensor,
    load_manifest,
    load_scorer,
    metrics_row,
    next_bucket,
    optional_perceptual_metrics,
    production_gate,
    text_error_rate,
    validate_manifest_coverage,
    waveform_metrics,
    write_reports,
)
from vllm_omni.model_executor.models.qwen3_tts.npu_graph_decoder_wrapper import (
    NPUGraphDecoderWrapper,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Model,
)

BUCKETS = (1, 25, 26, 51, 53, 61, 72, 73, 74, 76, 97, 98, 123, 148, 169, 325)
BATCH_SIZES = (1, 2, 4, 6, 8, 16)
PAD_DISTANCES = (1, 2, 4, 8, 16, 24, 32)
CRITICAL_PAIRS = (
    (72, 73),
    (74, 76),
    (53, 61),
    (53, 72),
    (53, 73),
    (61, 72),
    (61, 73),
    (1, 25),
    (170, 325),
)
THRESHOLDS = PaddingThresholds()


def test_padding_bucket_overflow_is_not_selected():
    assert next_bucket(326, BUCKETS) is None


def _require_npu_and_model() -> Path:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    model_path = os.environ.get("QWEN3_TTS_MODEL_PATH")
    if not model_path:
        pytest.skip("QWEN3_TTS_MODEL_PATH is not set")
    path = Path(model_path)
    if not path.exists():
        pytest.skip(f"QWEN3_TTS_MODEL_PATH does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def real_decoder():
    model_path = _require_npu_and_model()
    tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(
        str(model_path),
        subfolder="speech_tokenizer",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    decoder = tokenizer.decoder.eval().to(device=torch.device("npu"), dtype=torch.float32)
    del tokenizer.encoder
    del tokenizer
    gc.collect()
    if hasattr(decoder, "precompute_snake_caches"):
        decoder.precompute_snake_caches()
    yield decoder
    del decoder
    gc.collect()
    torch.npu.empty_cache()


def _capture_pair(decoder: torch.nn.Module, batch_size: int, actual_frames: int, padded_frames: int):
    shapes = sorted({(batch_size, actual_frames), (batch_size, padded_frames)})
    wrapper = NPUGraphDecoderWrapper(
        decoder,
        capture_sizes=[size for batch, size in shapes if batch == 1],
        extra_capture_shapes=[shape for shape in shapes if shape[0] != 1],
        num_quantizers=int(decoder.config.num_quantizers),
        stats_log_every=0,
    )
    wrapper.warmup(torch.device("npu"))
    missing = set(shapes) - set(wrapper.graphs)
    if missing:
        pytest.fail(f"Failed to capture required padding precision graphs: {sorted(missing)}")
    return wrapper


def _release_wrapper(wrapper: NPUGraphDecoderWrapper) -> None:
    wrapper.graphs.clear()
    wrapper.static_inputs.clear()
    wrapper.static_outputs.clear()
    del wrapper
    gc.collect()
    torch.npu.empty_cache()


def _synthetic_codes(decoder: Any, batch_size: int, frames: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(
        0,
        int(decoder.config.codebook_size),
        (batch_size, int(decoder.config.num_quantizers), frames),
        generator=generator,
        dtype=torch.long,
    )
    return codes.to(device="npu", non_blocking=False)


@pytest.fixture(scope="module")
def quick_codec_fixture(real_decoder):
    fixture_value = os.environ.get("QWEN3_TTS_PRECISION_CODES")
    if not fixture_value:
        return None
    return load_codec_tensor(
        Path(fixture_value),
        num_quantizers=int(real_decoder.config.num_quantizers),
        codebook_size=int(real_decoder.config.codebook_size),
    )[:1]


def _quick_eager_codes(
    decoder: Any,
    frames: int,
    codec_fixture: torch.Tensor | None,
) -> torch.Tensor:
    if codec_fixture is not None:
        value = codec_fixture
        repeats = math.ceil(frames / int(value.shape[-1]))
        value = value.repeat(1, 1, repeats)[..., :frames]
        return value.to(device="npu", non_blocking=False)
    return _synthetic_codes(decoder, 1, frames, seed=20260731)


def _quick_eager_metrics(candidate: torch.Tensor, eager: torch.Tensor) -> dict[str, float]:
    candidate_cpu = candidate.detach().cpu().to(torch.float64).reshape(-1)
    eager_cpu = eager.detach().cpu().to(torch.float64).reshape(-1)
    error = candidate_cpu - eager_cpu
    absolute_error = error.abs()
    signal = float(torch.sum(eager_cpu.square()))
    noise = float(torch.sum(error.square()))
    if noise == 0:
        snr_db = math.inf
    elif signal == 0:
        snr_db = -math.inf
    else:
        snr_db = 10.0 * math.log10(signal / noise)
    denominator = float(torch.linalg.vector_norm(eager_cpu) * torch.linalg.vector_norm(candidate_cpu))
    cosine = float(torch.dot(eager_cpu, candidate_cpu)) / denominator if denominator > 0 else 1.0
    return {
        "max_abs": float(absolute_error.max()) if absolute_error.numel() else 0.0,
        "mean_abs": float(absolute_error.mean()) if absolute_error.numel() else 0.0,
        "p99_abs": float(torch.quantile(absolute_error, 0.99)) if absolute_error.numel() else 0.0,
        "cosine": cosine,
        "snr_db": snr_db,
    }


@pytest.mark.npu
@pytest.mark.A2
@pytest.mark.tts
@pytest.mark.full_model
@pytest.mark.slow
def test_eager_zero_padding_quick_b1(real_decoder, quick_codec_fixture):
    """Quickly measure eager-only right-padding error without graph capture."""
    results: list[dict[str, float | int]] = []
    threshold_failures: list[str] = []
    assert_thresholds = os.environ.get("QWEN3_TTS_PADDING_EAGER_QUICK_ASSERT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    for actual_frames, padded_frames in CRITICAL_PAIRS:
        codes = _quick_eager_codes(real_decoder, actual_frames, quick_codec_fixture)
        padded_codes = codes.new_zeros((1, int(real_decoder.config.num_quantizers), padded_frames))
        padded_codes[..., :actual_frames].copy_(codes)
        with torch.inference_mode():
            eager_exact = real_decoder(codes).clone()
            eager_padded_full = real_decoder(padded_codes)
        torch.npu.synchronize()

        expected_length = actual_frames * int(real_decoder.total_upsample)
        assert int(eager_exact.shape[-1]) == expected_length
        assert int(eager_padded_full.shape[-1]) >= expected_length
        eager_padded = eager_padded_full[..., :expected_length].clone()
        assert eager_padded.shape == eager_exact.shape
        assert eager_padded.dtype == eager_exact.dtype == torch.float32
        assert bool(torch.isfinite(eager_exact).all())
        assert bool(torch.isfinite(eager_padded).all())
        assert float(eager_exact.min()) >= -1.0
        assert float(eager_exact.max()) <= 1.0
        assert float(eager_padded.min()) >= -1.0
        assert float(eager_padded.max()) <= 1.0

        metrics = _quick_eager_metrics(eager_padded, eager_exact)
        result: dict[str, float | int] = {
            "actual_frames": actual_frames,
            "padded_frames": padded_frames,
            "padding_frames": padded_frames - actual_frames,
            **metrics,
        }
        results.append(result)
        print(
            "[Qwen3-TTS][Code2Wav padding eager quick]\n"
            f"F={actual_frames} P={padded_frames} pad={padded_frames - actual_frames}\n"
            f"max_abs={metrics['max_abs']:.9g} "
            f"mean_abs={metrics['mean_abs']:.9g} "
            f"p99_abs={metrics['p99_abs']:.9g} "
            f"cosine={metrics['cosine']:.9g} "
            f"snr_db={metrics['snr_db']:.6g}",
            flush=True,
        )
        if assert_thresholds and (
            metrics["max_abs"] > 5e-3
            or metrics["cosine"] < 0.9999
            or metrics["snr_db"] < 60.0
        ):
            threshold_failures.append(
                f"F={actual_frames},P={padded_frames},metrics={json.dumps(metrics, allow_nan=True)}"
            )

    sorted_results = sorted(results, key=lambda item: float(item["max_abs"]), reverse=True)
    print(
        "[Qwen3-TTS][Code2Wav padding eager quick] summary_by_max_abs\n"
        + "\n".join(
            f"F={item['actual_frames']} P={item['padded_frames']} "
            f"pad={item['padding_frames']} max_abs={float(item['max_abs']):.9g} "
            f"mean_abs={float(item['mean_abs']):.9g} p99_abs={float(item['p99_abs']):.9g} "
            f"cosine={float(item['cosine']):.9g} snr_db={float(item['snr_db']):.6g}"
            for item in sorted_results
        ),
        flush=True,
    )
    if threshold_failures:
        pytest.fail("Padding eager quick thresholds failed:\n" + "\n".join(threshold_failures))


@pytest.mark.npu
@pytest.mark.A2
@pytest.mark.tts
@pytest.mark.full_model
@pytest.mark.slow
def test_graph_zero_padding_quick_b1(real_decoder, quick_codec_fixture):
    """Quickly compare zero-padded graph replay directly with exact eager."""
    capture_sizes = sorted({padded_frames for _, padded_frames in CRITICAL_PAIRS})
    wrapper = NPUGraphDecoderWrapper(
        real_decoder,
        capture_sizes=capture_sizes,
        num_quantizers=int(real_decoder.config.num_quantizers),
        stats_log_every=0,
    )
    wrapper.warmup(torch.device("npu"))
    missing = {(1, size) for size in capture_sizes} - set(wrapper.graphs)
    if missing:
        pytest.fail(f"Failed to capture quick padding graphs: {sorted(missing)}")

    results: list[dict[str, float | int]] = []
    threshold_failures: list[str] = []
    assert_thresholds = os.environ.get("QWEN3_TTS_PADDING_GRAPH_QUICK_ASSERT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        for actual_frames, padded_frames in CRITICAL_PAIRS:
            codes = _quick_eager_codes(real_decoder, actual_frames, quick_codec_fixture)
            with torch.inference_mode():
                eager_exact = real_decoder(codes).clone()
                graph_padded = graph_padded_decode(wrapper, codes, padded_frames, "zero")
            torch.npu.synchronize()

            expected_length = actual_frames * int(real_decoder.total_upsample)
            assert int(eager_exact.shape[-1]) == expected_length
            assert int(graph_padded.shape[-1]) == expected_length
            assert graph_padded.shape == eager_exact.shape
            assert graph_padded.dtype == eager_exact.dtype == torch.float32
            assert bool(torch.isfinite(eager_exact).all())
            assert bool(torch.isfinite(graph_padded).all())
            assert float(eager_exact.min()) >= -1.0
            assert float(eager_exact.max()) <= 1.0
            assert float(graph_padded.min()) >= -1.0
            assert float(graph_padded.max()) <= 1.0

            metrics = _quick_eager_metrics(graph_padded, eager_exact)
            result: dict[str, float | int] = {
                "actual_frames": actual_frames,
                "padded_frames": padded_frames,
                "padding_frames": padded_frames - actual_frames,
                **metrics,
            }
            results.append(result)
            print(
                "[Qwen3-TTS][Code2Wav graph padding quick]\n"
                f"F={actual_frames} P={padded_frames} pad={padded_frames - actual_frames}\n"
                f"max_abs={metrics['max_abs']:.9g} "
                f"mean_abs={metrics['mean_abs']:.9g} "
                f"p99_abs={metrics['p99_abs']:.9g} "
                f"cosine={metrics['cosine']:.9g} "
                f"snr_db={metrics['snr_db']:.6g}",
                flush=True,
            )
            if assert_thresholds and (
                metrics["max_abs"] > 5e-3
                or metrics["cosine"] < 0.9999
                or metrics["snr_db"] < 60.0
            ):
                threshold_failures.append(
                    f"F={actual_frames},P={padded_frames},metrics={json.dumps(metrics, allow_nan=True)}"
                )
    finally:
        _release_wrapper(wrapper)

    sorted_results = sorted(results, key=lambda item: float(item["max_abs"]), reverse=True)
    print(
        "[Qwen3-TTS][Code2Wav graph padding quick] summary_by_max_abs\n"
        + "\n".join(
            f"F={item['actual_frames']} P={item['padded_frames']} "
            f"pad={item['padding_frames']} max_abs={float(item['max_abs']):.9g} "
            f"mean_abs={float(item['mean_abs']):.9g} p99_abs={float(item['p99_abs']):.9g} "
            f"cosine={float(item['cosine']):.9g} snr_db={float(item['snr_db']):.6g}"
            for item in sorted_results
        ),
        flush=True,
    )
    if threshold_failures:
        pytest.fail("Graph padding quick thresholds failed:\n" + "\n".join(threshold_failures))


def _all_shape_pairs() -> list[tuple[int, int]]:
    pairs = set(CRITICAL_PAIRS)
    for padded_frames in BUCKETS:
        for distance in PAD_DISTANCES:
            actual_frames = padded_frames - distance
            if actual_frames > 0:
                pairs.add((actual_frames, padded_frames))
    return sorted(pairs)


def _run_synthetic_pair(
    decoder: Any,
    *,
    batch_size: int,
    actual_frames: int,
    padded_frames: int,
    seeds: int,
) -> list[dict[str, Any]]:
    wrapper = _capture_pair(decoder, batch_size, actual_frames, padded_frames)
    rows: list[dict[str, Any]] = []
    try:
        for seed_index in range(seeds):
            seed = batch_size * 1_000_000 + actual_frames * 1_000 + padded_frames * 10 + seed_index
            codes = _synthetic_codes(decoder, batch_size, actual_frames, seed)
            with torch.inference_mode():
                eager_exact = decoder(codes).clone()
                if int(eager_exact.shape[-1]) != actual_frames * int(decoder.total_upsample):
                    pytest.fail(
                        f"Unexpected eager length for B={batch_size},F={actual_frames}: "
                        f"{eager_exact.shape[-1]}"
                    )
                graph_exact = wrapper.decode(codes)
            torch.npu.synchronize()
            assert_waveform_contract(graph_exact, eager_exact, "graph_exact")
            torch.testing.assert_close(graph_exact, eager_exact, atol=1e-5, rtol=1e-5)
            exact_metrics = waveform_metrics(graph_exact, eager_exact)
            assert_exact_metrics(exact_metrics, THRESHOLDS)

            for pad_mode in PAD_MODES:
                with torch.inference_mode():
                    eager_padded = eager_padded_decode(decoder, codes, padded_frames, pad_mode)
                    graph_padded = graph_padded_decode(wrapper, codes, padded_frames, pad_mode)
                    graph_repeat = graph_padded_decode(wrapper, codes, padded_frames, pad_mode)
                torch.npu.synchronize()
                assert_waveform_contract(eager_padded, eager_exact, f"eager_padded:{pad_mode}")
                assert_waveform_contract(graph_padded, eager_exact, f"graph_padded:{pad_mode}")
                torch.testing.assert_close(graph_repeat, graph_padded, atol=0, rtol=0)

                # The two comparisons isolate semantic padding from graph replay.
                eager_padding_metrics = waveform_metrics(eager_padded, eager_exact)
                graph_replay_metrics = waveform_metrics(graph_padded, eager_padded)
                candidate_metrics = waveform_metrics(graph_padded, eager_exact)
                row = metrics_row(
                    case_id=f"synthetic-{seed}",
                    batch_size=batch_size,
                    actual_frames=actual_frames,
                    padded_frames=padded_frames,
                    pad_mode=pad_mode,
                    metrics=candidate_metrics,
                    seed=seed,
                )
                row.update(
                    {
                        "scope": "synthetic",
                        "exact_graph_max_abs": exact_metrics.max_abs,
                        "eager_padding_max_abs": eager_padding_metrics.max_abs,
                        "graph_replay_max_abs": graph_replay_metrics.max_abs,
                    }
                )
                rows.append(row)
    finally:
        _release_wrapper(wrapper)
    return rows


@pytest.mark.npu
@pytest.mark.A2
@pytest.mark.tts
@pytest.mark.full_model
@pytest.mark.slow
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_critical_padding_matrix(real_decoder, batch_size):
    seeds = int(os.environ.get("QWEN3_TTS_PADDING_SYNTHETIC_SEEDS", "10"))
    assert seeds >= 10, "Production-gate evaluation requires at least 10 deterministic seeds"
    rows: list[dict[str, Any]] = []
    for actual_frames, padded_frames in CRITICAL_PAIRS:
        rows.extend(
            _run_synthetic_pair(
                real_decoder,
                batch_size=batch_size,
                actual_frames=actual_frames,
                padded_frames=padded_frames,
                seeds=seeds,
            )
        )
    decision = production_gate(
        rows,
        thresholds=THRESHOLDS,
        optional_metrics_complete=False,
        blind_complete=False,
    )
    assert decision["quality_status"] != "fail", json.dumps(
        decision["mode_failures"], ensure_ascii=False, indent=2
    )


@pytest.mark.npu
@pytest.mark.A2
@pytest.mark.tts
@pytest.mark.full_model
@pytest.mark.slow
def test_full_padding_distance_matrix(real_decoder):
    if os.environ.get("QWEN3_TTS_PADDING_FULL_MATRIX", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("Set QWEN3_TTS_PADDING_FULL_MATRIX=1 for the exhaustive distance sweep")
    seeds = int(os.environ.get("QWEN3_TTS_PADDING_SYNTHETIC_SEEDS", "10"))
    rows: list[dict[str, Any]] = []
    for batch_size in BATCH_SIZES:
        for actual_frames, padded_frames in _all_shape_pairs():
            rows.extend(
                _run_synthetic_pair(
                    real_decoder,
                    batch_size=batch_size,
                    actual_frames=actual_frames,
                    padded_frames=padded_frames,
                    seeds=seeds,
                )
            )
    decision = production_gate(
        rows,
        thresholds=THRESHOLDS,
        optional_metrics_complete=False,
        blind_complete=False,
    )
    assert decision["quality_status"] != "fail", json.dumps(
        decision["mode_failures"], ensure_ascii=False, indent=2
    )


@pytest.mark.npu
@pytest.mark.A2
@pytest.mark.tts
@pytest.mark.full_model
@pytest.mark.slow
@pytest.mark.parametrize("batch_size", BATCH_SIZES[1:])
def test_mixed_length_batch_rows_match_independent_eager(real_decoder, batch_size):
    lengths = (1, 26, 51, 72, 74, 97)
    row_lengths = [lengths[index % len(lengths)] for index in range(batch_size)]
    padded_frames = 98
    wrapper = _capture_pair(real_decoder, batch_size, padded_frames, padded_frames)
    metric_rows: list[dict[str, Any]] = []
    try:
        rows = [
            _synthetic_codes(real_decoder, 1, frames, seed=9_000_000 + batch_size * 100 + index)
            for index, frames in enumerate(row_lengths)
        ]
        with torch.inference_mode():
            eager = [real_decoder(codes).clone() for codes in rows]
            for pad_mode in PAD_MODES:
                candidate = graph_mixed_padded_decode(wrapper, rows, padded_frames, pad_mode)
                torch.npu.synchronize()
                for row_index, (candidate_row, eager_row) in enumerate(zip(candidate, eager, strict=True)):
                    assert_waveform_contract(candidate_row, eager_row, f"mixed-row-{row_index}")
                    metric_rows.append(
                        metrics_row(
                            case_id=f"mixed-B{batch_size}-row{row_index}",
                            batch_size=batch_size,
                            actual_frames=row_lengths[row_index],
                            padded_frames=padded_frames,
                            pad_mode=pad_mode,
                            metrics=waveform_metrics(candidate_row, eager_row),
                        )
                    )
                    metric_rows[-1]["scope"] = "mixed_batch"
    finally:
        _release_wrapper(wrapper)
    decision = production_gate(
        metric_rows,
        thresholds=THRESHOLDS,
        optional_metrics_complete=False,
        blind_complete=False,
    )
    assert decision["quality_status"] != "fail", json.dumps(
        decision["mode_failures"], ensure_ascii=False, indent=2
    )


@pytest.mark.npu
@pytest.mark.A2
@pytest.mark.tts
@pytest.mark.full_model
@pytest.mark.slow
def test_padding_static_buffer_long_short_long_has_no_tail_leak(real_decoder):
    wrapper = _capture_pair(real_decoder, 1, 53, 73)
    try:
        long_codes = _synthetic_codes(real_decoder, 1, 72, seed=7_200_001)
        short_codes = _synthetic_codes(real_decoder, 1, 53, seed=5_300_001)
        for pad_mode in PAD_MODES:
            first = graph_padded_decode(wrapper, long_codes, 73, pad_mode)
            short = graph_padded_decode(wrapper, short_codes, 73, pad_mode)
            second = graph_padded_decode(wrapper, long_codes, 73, pad_mode)
            torch.npu.synchronize()
            torch.testing.assert_close(second, first, atol=0, rtol=0)
            assert int(short.shape[-1]) == 53 * int(real_decoder.total_upsample)
    finally:
        _release_wrapper(wrapper)


def _streaming_windows(codes: torch.Tensor, ref_frames: int) -> list[dict[str, Any]]:
    if ref_frames >= int(codes.shape[-1]):
        raise ValueError(f"ref_context_frames={ref_frames} leaves no generated codec frames")
    reference = codes[..., :ref_frames]
    generated = codes[..., ref_frames:]
    windows: list[dict[str, Any]] = []
    emitted = 0
    next_frames = 1
    total_frames = int(generated.shape[-1])
    while emitted < total_frames:
        end = min(total_frames, emitted + next_frames)
        left_start = max(0, emitted - 72)
        left_and_new = generated[..., left_start:end]
        window = torch.cat((reference, left_and_new), dim=-1) if ref_frames else left_and_new
        left_frames = emitted - left_start
        windows.append(
            {
                "codes": window,
                "context_frames": ref_frames + left_frames,
                "left_frames": left_frames,
                "new_frames": end - emitted,
                "is_final": end == total_frames,
            }
        )
        emitted = end
        next_frames = 25
    return windows


def _overlap_safe_waveform(
    chunks: Sequence[dict[str, Any]],
    *,
    total_upsample: int,
    margin_frames: int,
) -> tuple[torch.Tensor, int]:
    emitted = [chunk["candidate_emitted"].clone() for chunk in chunks]
    wait_frames = 0
    for index in range(1, len(chunks)):
        current = chunks[index]
        previous = chunks[index - 1]
        available_left = int(current["left_frames"])
        margin = min(margin_frames, available_left, int(previous["new_frames"]))
        if margin <= 0:
            continue
        source = current["eager_full"] if current["is_final"] else current["candidate_full"]
        context_frames = int(current["context_frames"])
        source_start = (context_frames - margin) * total_upsample
        source_end = context_frames * total_upsample
        emitted[index - 1][..., -margin * total_upsample :] = source[..., source_start:source_end]
        wait_frames += int(current["new_frames"])
    # The final tail is correctness-critical and always uses exact eager output.
    emitted[-1] = chunks[-1]["eager_emitted"].clone()
    return torch.cat(emitted, dim=-1), wait_frames


def _score_optional(
    metrics: Any,
    *,
    candidate: torch.Tensor,
    eager: torch.Tensor,
    case: Any,
    asr_scorer: Any,
    speaker_scorer: Any,
    eager_asr_text: str | None,
    eager_speaker_embedding: Any,
) -> bool:
    metrics.pesq_drop, metrics.stoi_drop = optional_perceptual_metrics(candidate, eager)
    complete = metrics.pesq_drop is not None and metrics.stoi_drop is not None
    if asr_scorer is not None:
        padded_text = str(asr_scorer(candidate, 24_000, case.language))
        metrics.asr_error_rate_delta = text_error_rate(case.text, padded_text, case.language) - text_error_rate(
            case.text, str(eager_asr_text), case.language
        )
    else:
        complete = False
    if speaker_scorer is not None:
        metrics.speaker_cosine_drop = embedding_cosine_drop(
            eager_speaker_embedding,
            speaker_scorer(candidate, 24_000),
        )
    else:
        complete = False
    return complete


def _append_blind_candidates(
    output: list[dict[str, Any]],
    *,
    sample_id: str,
    pad_mode: str,
    eager: torch.Tensor,
    padded: torch.Tensor,
    boundaries: Sequence[int],
) -> None:
    def append(category: str, suffix: str, eager_clip: torch.Tensor, padded_clip: torch.Tensor) -> None:
        output.append(
            {
                "sample_id": f"{sample_id}-{suffix}",
                "eager": eager_clip,
                "padded": padded_clip,
                "metrics": waveform_metrics(padded_clip, eager_clip),
                "category": category,
                "pad_mode": pad_mode,
            }
        )

    append("full", "full", eager, padded)
    three_seconds = 3 * 24_000
    append("first", "first", eager[..., :three_seconds], padded[..., :three_seconds])
    append("tail", "tail", eager[..., -three_seconds:], padded[..., -three_seconds:])
    if boundaries:
        radius = 24_000
        error = (padded - eager).abs().reshape(-1)
        boundary = max(
            boundaries,
            key=lambda offset: float(error[max(0, offset - radius) : min(error.numel(), offset + radius)].max()),
        )
        start = max(0, boundary - radius)
        end = min(int(eager.shape[-1]), boundary + radius)
        append("boundary", "boundary", eager[..., start:end], padded[..., start:end])


@pytest.mark.npu
@pytest.mark.A2
@pytest.mark.tts
@pytest.mark.full_model
@pytest.mark.slow
@pytest.mark.padding_corpus
def test_padding_production_corpus(real_decoder):
    manifest_value = os.environ.get("QWEN3_TTS_PADDING_PRECISION_MANIFEST")
    if not manifest_value:
        pytest.skip("QWEN3_TTS_PADDING_PRECISION_MANIFEST is not set")
    manifest_path = Path(manifest_value)
    cases = load_manifest(manifest_path)
    coverage_problems = validate_manifest_coverage(cases)
    if coverage_problems:
        pytest.fail("Invalid production corpus:\n" + "\n".join(coverage_problems))

    output_dir = Path(
        os.environ.get(
            "QWEN3_TTS_PADDING_PRECISION_OUTPUT_DIR",
            str(manifest_path.parent / "padding_precision_results"),
        )
    )
    asr_scorer = load_scorer(os.environ.get("QWEN3_TTS_PADDING_ASR_SCORER"))
    speaker_scorer = load_scorer(os.environ.get("QWEN3_TTS_PADDING_SPEAKER_SCORER"))
    num_quantizers = int(real_decoder.config.num_quantizers)
    codebook_size = int(real_decoder.config.codebook_size)
    total_upsample = int(real_decoder.total_upsample)

    all_jobs: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    case_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_by_id = {case.case_id: case for case in cases}
    duration_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        codes = load_codec_tensor(
            case.codes_path,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
        )
        for row_index in range(int(codes.shape[0])):
            generated_frames = int(codes.shape[-1]) - case.ref_context_frames
            duration_seconds = generated_frames / 12.0
            duration_bucket = "short" if duration_seconds < 5.0 else "medium" if duration_seconds < 15.0 else "long"
            duration_counts[duration_bucket] += 1
            row_id = case.case_id if codes.shape[0] == 1 else f"{case.case_id}-row{row_index}"
            if row_id != case.case_id:
                case_by_id[row_id] = case
            windows = _streaming_windows(codes[row_index : row_index + 1], case.ref_context_frames)
            for chunk_index, window in enumerate(windows):
                actual_frames = int(window["codes"].shape[-1])
                padded_frames = next_bucket(actual_frames, BUCKETS)
                if padded_frames is None:
                    continue
                job = {
                    **window,
                    "row_id": row_id,
                    "chunk_index": chunk_index,
                    "actual_frames": actual_frames,
                    "padded_frames": padded_frames,
                }
                all_jobs[(actual_frames, padded_frames)].append(job)
                case_chunks[row_id].append(job)
    missing_durations = [name for name in ("short", "medium", "long") if duration_counts[name] < 20]
    if missing_durations:
        pytest.fail(
            "Production corpus needs >=20 short/medium/long rows; "
            f"counts={dict(duration_counts)}, missing={missing_durations}"
        )

    rows: list[dict[str, Any]] = []
    optional_complete = True
    blind_candidates: list[dict[str, Any]] = []
    for (actual_frames, padded_frames), jobs in sorted(all_jobs.items()):
        wrapper = _capture_pair(real_decoder, 1, actual_frames, padded_frames)
        try:
            for job in jobs:
                codes = job["codes"].to(device="npu", non_blocking=False)
                with torch.inference_mode():
                    eager_full = real_decoder(codes).clone()
                    graph_exact = wrapper.decode(codes)
                torch.npu.synchronize()
                expected_length = actual_frames * total_upsample
                if int(eager_full.shape[-1]) != expected_length:
                    pytest.fail(
                        f"Unexpected eager length for corpus shape F={actual_frames}: "
                        f"{eager_full.shape[-1]} != {expected_length}"
                    )
                torch.testing.assert_close(graph_exact, eager_full, atol=1e-5, rtol=1e-5)
                exact_metrics = waveform_metrics(graph_exact, eager_full)
                assert_exact_metrics(exact_metrics, THRESHOLDS)
                context_samples = int(job["context_frames"]) * total_upsample
                eager_emitted = eager_full[..., context_samples:].clone()
                job["eager_full"] = eager_full.cpu()
                job["eager_emitted"] = eager_emitted.cpu()
                for pad_mode in PAD_MODES:
                    with torch.inference_mode():
                        eager_padded_full = eager_padded_decode(
                            real_decoder,
                            codes,
                            padded_frames,
                            pad_mode,
                        )
                        candidate_full = graph_padded_decode(wrapper, codes, padded_frames, pad_mode)
                    torch.npu.synchronize()
                    candidate_emitted = candidate_full[..., context_samples:].clone()
                    eager_padded_emitted = eager_padded_full[..., context_samples:].clone()
                    metrics = waveform_metrics(candidate_emitted, eager_emitted)
                    rows.append(
                        metrics_row(
                            case_id=f"{job['row_id']}-chunk{job['chunk_index']}",
                            batch_size=1,
                            actual_frames=actual_frames,
                            padded_frames=padded_frames,
                            pad_mode=pad_mode,
                            metrics=metrics,
                        )
                    )
                    rows[-1]["scope"] = "chunk"
                    rows[-1]["exact_graph_max_abs"] = exact_metrics.max_abs
                    rows[-1]["eager_padding_max_abs"] = float(
                        (eager_padded_emitted - eager_emitted).abs().max()
                    )
                    rows[-1]["graph_replay_max_abs"] = float(
                        (candidate_emitted - eager_padded_emitted).abs().max()
                    )
                    job[f"{pad_mode}_full"] = candidate_full.cpu()
                    job[f"{pad_mode}_emitted"] = candidate_emitted.cpu()
        finally:
            _release_wrapper(wrapper)

    for row_id, chunks in case_chunks.items():
        chunks.sort(key=lambda item: int(item["chunk_index"]))
        case = case_by_id[row_id]
        eager_wave = torch.cat([chunk["eager_emitted"] for chunk in chunks], dim=-1)
        eager_asr_text = str(asr_scorer(eager_wave, 24_000, case.language)) if asr_scorer else None
        eager_speaker_embedding = speaker_scorer(eager_wave, 24_000) if speaker_scorer else None
        boundaries: list[int] = []
        offset = 0
        for chunk in chunks[:-1]:
            offset += int(chunk["eager_emitted"].shape[-1])
            boundaries.append(offset)
        for pad_mode in PAD_MODES:
            candidate_wave = torch.cat([chunk[f"{pad_mode}_emitted"] for chunk in chunks], dim=-1)
            metrics = waveform_metrics(
                candidate_wave,
                eager_wave,
                boundary_offsets=boundaries,
                boundary_radius=total_upsample,
            )
            optional_complete &= _score_optional(
                metrics,
                candidate=candidate_wave,
                eager=eager_wave,
                case=case,
                asr_scorer=asr_scorer,
                speaker_scorer=speaker_scorer,
                eager_asr_text=eager_asr_text,
                eager_speaker_embedding=eager_speaker_embedding,
            )
            rows.append(
                metrics_row(
                    case_id=f"{row_id}-full",
                    batch_size=1,
                    actual_frames=sum(int(chunk["new_frames"]) for chunk in chunks),
                    padded_frames=sum(int(chunk["padded_frames"]) for chunk in chunks),
                    pad_mode=pad_mode,
                    metrics=metrics,
                )
            )
            rows[-1]["language"] = case.language
            rows[-1]["mode"] = case.mode
            rows[-1]["speaker_id"] = case.speaker_id
            rows[-1]["scope"] = "full"
            _append_blind_candidates(
                blind_candidates,
                sample_id=row_id,
                pad_mode=pad_mode,
                eager=eager_wave,
                padded=candidate_wave,
                boundaries=boundaries,
            )

        # Evaluate the overlap-safe remediation using zero-padded graph chunks.
        for chunk in chunks:
            chunk["candidate_full"] = chunk["zero_full"]
            chunk["candidate_emitted"] = chunk["zero_emitted"]
        for margin_frames in (1, 2, 4):
            candidate_wave, wait_frames = _overlap_safe_waveform(
                chunks,
                total_upsample=total_upsample,
                margin_frames=margin_frames,
            )
            metrics = waveform_metrics(
                candidate_wave,
                eager_wave,
                boundary_offsets=boundaries,
                boundary_radius=total_upsample,
            )
            optional_complete &= _score_optional(
                metrics,
                candidate=candidate_wave,
                eager=eager_wave,
                case=case,
                asr_scorer=asr_scorer,
                speaker_scorer=speaker_scorer,
                eager_asr_text=eager_asr_text,
                eager_speaker_embedding=eager_speaker_embedding,
            )
            row = metrics_row(
                case_id=f"{row_id}-full",
                batch_size=1,
                actual_frames=sum(int(chunk["new_frames"]) for chunk in chunks),
                padded_frames=sum(int(chunk["padded_frames"]) for chunk in chunks),
                pad_mode=f"overlap_safe_{margin_frames}",
                metrics=metrics,
            )
            row["held_margin_frames"] = margin_frames
            row["accumulated_wait_frames"] = wait_frames
            row["minimum_margin_latency_ms"] = 1000.0 * margin_frames / 12.0
            row["maximum_next_chunk_wait_frames"] = max(int(chunk["new_frames"]) for chunk in chunks)
            row["maximum_next_chunk_wait_ms"] = 1000.0 * row["maximum_next_chunk_wait_frames"] / 12.0
            row["language"] = case.language
            row["mode"] = case.mode
            row["speaker_id"] = case.speaker_id
            row["scope"] = "full"
            rows.append(row)
            _append_blind_candidates(
                blind_candidates,
                sample_id=f"{row_id}-overlap-safe-{margin_frames}",
                pad_mode=f"overlap_safe_{margin_frames}",
                eager=eager_wave,
                padded=candidate_wave,
                boundaries=boundaries,
            )

    preliminary = production_gate(
        rows,
        thresholds=THRESHOLDS,
        optional_metrics_complete=optional_complete,
        blind_complete=False,
    )
    passing_modes = preliminary["passing_modes"]
    blind_mode = (
        "zero"
        if "zero" in passing_modes
        else "repeat_last"
        if "repeat_last" in passing_modes
        else passing_modes[0]
        if passing_modes
        else "zero"
    )
    blind_candidates = [item for item in blind_candidates if item["pad_mode"] == blind_mode]
    blind_manifest = export_blind_pairs(output_dir, blind_candidates, limit=60)
    blind_results_value = os.environ.get("QWEN3_TTS_PADDING_BLIND_RESULTS")
    blind_complete = False
    blind_result: dict[str, Any] | None = None
    if blind_results_value:
        blind_result = analyze_blind_results(
            Path(blind_results_value),
            blind_manifest.parent / "blind_key.json",
        )
        blind_complete = True

    decision = production_gate(
        rows,
        thresholds=THRESHOLDS,
        optional_metrics_complete=optional_complete,
        blind_complete=blind_complete,
    )
    decision["manifest_coverage"] = {"cases": len(cases), "problems": coverage_problems}
    decision["duration_coverage"] = dict(duration_counts)
    decision["blind_result"] = blind_result
    decision["blind_pad_mode"] = blind_mode
    decision["blind_manifest"] = str(blind_manifest)
    decision["thresholds"] = THRESHOLDS.__dict__
    decision["aggregates"] = aggregate_rows(rows)
    if blind_result is not None and not blind_result["pass"]:
        decision["failures"].append("blind listening gate")
        decision["mode_failures"].setdefault(blind_mode, []).append("blind listening gate")
        decision["status"] = "fail"
    write_reports(output_dir, rows, decision)
    print(
        "[Qwen3-TTS][NPU Code2Wav padding precision] "
        f"status={decision['status']} rows={len(rows)} output_dir={output_dir}",
        flush=True,
    )
    if decision["status"] == "fail":
        pytest.fail(json.dumps(decision["mode_failures"], ensure_ascii=False, indent=2))
