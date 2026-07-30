# SPDX-License-Identifier: Apache-2.0
"""Ascend A2 numerical parity tests for the Qwen3-TTS Code2Wav NPU graph."""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.npu_graph_decoder_wrapper import (
    NPUGraphDecoderWrapper,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Model,
)

pytestmark = [
    pytest.mark.npu,
    pytest.mark.A2,
    pytest.mark.tts,
    pytest.mark.full_model,
    pytest.mark.slow,
]


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
def real_decoder_and_graph():
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

    wrapper = NPUGraphDecoderWrapper(
        decoder,
        capture_sizes=[25, 73, 97, 169],
        extra_capture_shapes=[(2, 97), (2, 169)],
        num_quantizers=int(decoder.config.num_quantizers),
    )
    wrapper.warmup(torch.device("npu"))
    if not wrapper.graphs:
        pytest.fail("No Code2Wav NPU graph was captured")

    yield decoder, wrapper

    del wrapper
    del decoder
    gc.collect()
    torch.npu.empty_cache()


@pytest.fixture(scope="module")
def codec_fixture():
    fixture_path = os.environ.get("QWEN3_TTS_PRECISION_CODES")
    if not fixture_path:
        return None
    value = torch.load(fixture_path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("codes", "codec_codes", "audio_codes"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        raise TypeError("QWEN3_TTS_PRECISION_CODES must contain a tensor")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"Expected codec fixture [Q,F] or [B,Q,F], got {tuple(value.shape)}")
    return value.to(dtype=torch.long, device="cpu")


def _codes(
    decoder,
    codec_fixture: torch.Tensor | None,
    batch_size: int,
    frames: int,
) -> torch.Tensor:
    num_quantizers = int(decoder.config.num_quantizers)
    codebook_size = int(decoder.config.codebook_size)
    if codec_fixture is None:
        generator = torch.Generator(device="cpu").manual_seed(batch_size * 10000 + frames)
        value = torch.randint(
            0,
            codebook_size,
            (batch_size, num_quantizers, frames),
            generator=generator,
            dtype=torch.long,
        )
    else:
        value = codec_fixture
        if int(value.shape[1]) != num_quantizers:
            raise ValueError(
                f"Codec fixture has {value.shape[1]} quantizers; decoder expects {num_quantizers}"
            )
        batch_repeats = math.ceil(batch_size / int(value.shape[0]))
        frame_repeats = math.ceil(frames / int(value.shape[-1]))
        value = value.repeat(batch_repeats, 1, frame_repeats)[:batch_size, :, :frames]
        value = value.remainder(codebook_size)
    return value.to(device="npu", non_blocking=False)


def _precision_metrics(graph: torch.Tensor, eager: torch.Tensor) -> tuple[float, float, float, float]:
    graph_cpu = graph.detach().to(device="cpu", dtype=torch.float64)
    eager_cpu = eager.detach().to(device="cpu", dtype=torch.float64)
    error = graph_cpu - eager_cpu
    max_abs = float(error.abs().max()) if error.numel() else 0.0
    mean_abs = float(error.abs().mean()) if error.numel() else 0.0
    eager_flat = eager_cpu.reshape(-1)
    graph_flat = graph_cpu.reshape(-1)
    denom = float(torch.linalg.vector_norm(eager_flat) * torch.linalg.vector_norm(graph_flat))
    cosine = float(torch.dot(eager_flat, graph_flat)) / denom if denom > 0 else 1.0
    signal = float(torch.sum(eager_flat.square()))
    noise = float(torch.sum(error.reshape(-1).square()))
    snr = math.inf if noise == 0 else 10.0 * math.log10(signal / max(noise, torch.finfo(torch.float64).tiny))
    return max_abs, mean_abs, cosine, snr


def _assert_precision(graph: torch.Tensor, eager: torch.Tensor, label: str) -> None:
    assert graph.shape == eager.shape, label
    assert graph.dtype == eager.dtype == torch.float32, label
    assert torch.isfinite(graph).all(), label
    assert graph.min() >= -1 and graph.max() <= 1, label
    torch.testing.assert_close(graph, eager, atol=1e-5, rtol=1e-5, msg=label)
    max_abs, mean_abs, cosine, snr = _precision_metrics(graph, eager)
    assert max_abs <= 1e-4, f"{label}: max_abs={max_abs}"
    assert mean_abs <= 1e-6, f"{label}: mean_abs={mean_abs}"
    assert cosine >= 0.99999, f"{label}: cosine={cosine}"
    assert snr >= 80.0, f"{label}: snr={snr} dB"


@pytest.mark.parametrize(
    ("batch_size", "frames"),
    [
        (1, 25),
        (1, 73),
        (1, 97),
        (1, 169),
        (1, 26),
        (1, 74),
        (1, 98),
        (2, 97),
        (2, 169),
    ],
)
def test_real_decoder_npugraph_matches_eager(
    real_decoder_and_graph,
    codec_fixture,
    batch_size,
    frames,
):
    decoder, wrapper = real_decoder_and_graph
    codes = _codes(decoder, codec_fixture, batch_size, frames)
    with torch.inference_mode():
        eager = decoder(codes).clone()
        graph = wrapper.decode(codes)
    torch.npu.synchronize()
    _assert_precision(graph, eager, f"batch={batch_size},frames={frames}")


def _streaming_decode(
    decode_fn,
    codes: torch.Tensor,
    *,
    total_upsample: int,
    reference: torch.Tensor | None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    chunks: list[torch.Tensor] = []
    emitted_frames = 0
    total_frames = int(codes.shape[-1])
    next_frames = 1
    while emitted_frames < total_frames:
        end = min(total_frames, emitted_frames + next_frames)
        left_start = max(0, emitted_frames - 72)
        window = codes[..., left_start:end]
        context_frames = emitted_frames - left_start
        if reference is not None:
            window = torch.cat((reference, window), dim=-1)
            context_frames += int(reference.shape[-1])
        wav = decode_fn(window)
        chunk = wav[..., context_frames * total_upsample :].clone()
        chunks.append(chunk)
        emitted_frames = end
        next_frames = 25
    return chunks, torch.cat(chunks, dim=-1)


@pytest.mark.parametrize("with_reference", [False, True])
def test_real_streaming_waveform_and_boundaries_match_eager(
    real_decoder_and_graph,
    codec_fixture,
    with_reference,
):
    decoder, wrapper = real_decoder_and_graph
    codes = _codes(decoder, codec_fixture, 1, 300)
    reference = _codes(decoder, codec_fixture, 1, 72) if with_reference else None
    upsample = int(decoder.total_upsample)

    with torch.inference_mode():
        eager_chunks, eager = _streaming_decode(
            decoder,
            codes,
            total_upsample=upsample,
            reference=reference,
        )
        graph_chunks, graph = _streaming_decode(
            wrapper.decode,
            codes,
            total_upsample=upsample,
            reference=reference,
        )
    torch.npu.synchronize()

    assert len(graph_chunks) == len(eager_chunks)
    offset = 0
    boundary_width = upsample
    for index, (graph_chunk, eager_chunk) in enumerate(zip(graph_chunks, eager_chunks, strict=True)):
        _assert_precision(graph_chunk, eager_chunk, f"stream_chunk={index},reference={with_reference}")
        offset += int(graph_chunk.shape[-1])
        start = max(0, offset - boundary_width)
        end = min(int(graph.shape[-1]), offset + boundary_width)
        _assert_precision(
            graph[..., start:end],
            eager[..., start:end],
            f"stream_boundary={index},reference={with_reference}",
        )

    _assert_precision(graph, eager, f"stream_full,reference={with_reference}")

    with torch.inference_mode():
        _, graph_repeat = _streaming_decode(
            wrapper.decode,
            codes,
            total_upsample=upsample,
            reference=reference,
        )
    torch.npu.synchronize()
    torch.testing.assert_close(graph_repeat, graph, atol=0, rtol=0)
