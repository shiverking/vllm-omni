# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_tts.npu_graph_decoder_wrapper import (
    NPUGraphDecoderWrapper,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_QUANTIZERS = 4
TOTAL_UPSAMPLE = 3


class _TinyCausalDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.total_upsample = TOTAL_UPSAMPLE

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        values = codes.to(torch.float32).sum(dim=1, keepdim=True) / 100.0
        values = torch.cumsum(values, dim=-1)
        return values.repeat_interleave(self.total_upsample, dim=-1).clamp(-1, 1)


class _CapturedGraph:
    def __init__(self, replay_fn=None) -> None:
        self._replay_fn = replay_fn

    def replay(self) -> None:
        if self._replay_fn is not None:
            self._replay_fn()


class _FakeNPU:
    def __init__(self) -> None:
        self.pool = object()
        self.graph_pools: list[object] = []
        self.sync_calls = 0

    def graph_pool_handle(self):
        return self.pool

    def NPUGraph(self):
        return _CapturedGraph()

    def graph(self, _graph, *, pool):
        self.graph_pools.append(pool)
        return nullcontext()

    def synchronize(self):
        self.sync_calls += 1


class _FailingFakeNPU(_FakeNPU):
    def __init__(self, fail_capture_index: int) -> None:
        super().__init__()
        self.fail_capture_index = fail_capture_index

    def graph(self, _graph, *, pool):
        self.graph_pools.append(pool)
        if len(self.graph_pools) == self.fail_capture_index:
            raise RuntimeError("injected capture failure")
        return nullcontext()


@pytest.fixture
def wrapper(monkeypatch):
    fake_npu = _FakeNPU()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    decoder = _TinyCausalDecoder().eval()
    graph_wrapper = NPUGraphDecoderWrapper(
        decoder,
        capture_sizes=[25, 73, 97, 169],
        extra_capture_shapes=[(2, 25), (2, 97), (2, 169), (4, 97), (4, 169)],
        num_quantizers=NUM_QUANTIZERS,
    )
    graph_wrapper.warmup(torch.device("cpu"))

    # The fake capture context cannot record PyTorch operations. Attach replay
    # callbacks that preserve the real static-input/static-output semantics.
    for key in graph_wrapper.graphs:
        static_input = graph_wrapper.static_inputs[key]
        static_output = graph_wrapper.static_outputs[key]

        def replay(inp=static_input, out=static_output):
            out.copy_(decoder(inp))

        graph_wrapper.graphs[key] = _CapturedGraph(replay)
    return decoder, graph_wrapper, fake_npu


@pytest.fixture
def padding_wrapper(monkeypatch):
    fake_npu = _FakeNPU()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    decoder = _TinyCausalDecoder().eval()
    graph_wrapper = NPUGraphDecoderWrapper(
        decoder,
        capture_sizes=[25, 51, 73],
        extra_capture_shapes=[(2, 51), (2, 73)],
        num_quantizers=NUM_QUANTIZERS,
        padding_enabled=True,
        max_pad_frames=0,
    )
    graph_wrapper.warmup(torch.device("cpu"))
    for key in graph_wrapper.graphs:
        static_input = graph_wrapper.static_inputs[key]
        static_output = graph_wrapper.static_outputs[key]

        def replay(inp=static_input, out=static_output):
            out.copy_(decoder(inp))

        graph_wrapper.graphs[key] = _CapturedGraph(replay)
    return decoder, graph_wrapper, fake_npu


def _codes(batch_size: int, frames: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(batch_size * 1000 + frames)
    return torch.randint(0, 8, (batch_size, NUM_QUANTIZERS, frames), generator=generator)


def _assert_waveform_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert actual.min() >= -1 and actual.max() <= 1
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_capture_uses_one_pool_and_expected_sparse_shapes(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    fake_npu = _FakeNPU()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    graph_wrapper = NPUGraphDecoderWrapper(
        _TinyCausalDecoder().eval(),
        capture_sizes=[25, 73, 97, 169],
        extra_capture_shapes=[(2, 25), (2, 97), (2, 169), (4, 97), (4, 169)],
        num_quantizers=NUM_QUANTIZERS,
    )
    graph_wrapper.warmup(torch.device("cpu"))
    expected = {
        (1, 25),
        (1, 73),
        (1, 97),
        (1, 169),
        (2, 25),
        (2, 97),
        (2, 169),
        (4, 97),
        (4, 169),
    }
    assert set(graph_wrapper.graphs) == expected
    assert fake_npu.graph_pools == [fake_npu.pool] * len(expected)
    output = caplog.text
    assert "[Qwen3-TTS][NPU Code2Wav graph] enabled" in output
    assert "[Qwen3-TTS][NPU Code2Wav graph] capture complete" in output


def test_one_capture_failure_keeps_other_graphs_and_reports_failure(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    fake_npu = _FailingFakeNPU(fail_capture_index=2)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    graph_wrapper = NPUGraphDecoderWrapper(
        _TinyCausalDecoder().eval(),
        capture_sizes=[25, 73, 97],
        num_quantizers=NUM_QUANTIZERS,
    )
    graph_wrapper.warmup(torch.device("cpu"))
    assert set(graph_wrapper.graphs) == {(1, 25), (1, 97)}
    output = caplog.text
    assert "failed_keys=[(1, 73)]" in output


@pytest.mark.parametrize("frames", [25, 73, 97, 169])
def test_exact_shape_matches_eager_bitwise(wrapper, frames):
    decoder, graph_wrapper, _ = wrapper
    codes = _codes(1, frames)
    eager = decoder(codes)
    graph = graph_wrapper.decode(codes)
    torch.testing.assert_close(graph, eager, atol=0, rtol=0)


@pytest.mark.parametrize(
    "frames",
    [1, 26, 74, 98],
)
def test_non_exact_shape_falls_back_to_eager(wrapper, frames):
    decoder, graph_wrapper, _ = wrapper
    codes = _codes(1, frames)
    torch.testing.assert_close(graph_wrapper.decode(codes), decoder(codes), atol=0, rtol=0)
    assert graph_wrapper._get_graph_key(1, frames) is None


@pytest.mark.parametrize(("batch_size", "frames"), [(2, 97), (2, 169), (4, 97)])
def test_sparse_batch_shapes_match_eager(wrapper, batch_size, frames):
    decoder, graph_wrapper, _ = wrapper
    codes = _codes(batch_size, frames)
    _assert_waveform_close(graph_wrapper.decode(codes), decoder(codes))


def test_uncaptured_shape_falls_back_to_eager(wrapper):
    decoder, graph_wrapper, _ = wrapper
    codes = _codes(3, 200)
    torch.testing.assert_close(graph_wrapper.decode(codes), decoder(codes), atol=0, rtol=0)


def test_exact_graph_output_survives_later_replay(wrapper):
    _, graph_wrapper, _ = wrapper
    codes = _codes(1, 169)
    first = graph_wrapper.decode(codes)
    expected = first.clone()

    _ = graph_wrapper.decode(torch.flip(codes, dims=[-1]))

    torch.testing.assert_close(first, expected, atol=0, rtol=0)


def test_chunked_decode_matches_eager(wrapper):
    decoder, graph_wrapper, _ = wrapper
    codes = _codes(1, 80)
    eager = _eager_chunked_decode(decoder, codes, chunk_size=25, left_context_size=20)
    graph = graph_wrapper.chunked_decode_with_npugraph(codes, chunk_size=25, left_context_size=20)
    _assert_waveform_close(graph, eager)


def _eager_chunked_decode(
    decoder: nn.Module,
    codes: torch.Tensor,
    *,
    chunk_size: int,
    left_context_size: int,
) -> torch.Tensor:
    eager_chunks: list[torch.Tensor] = []
    start = 0
    while start < codes.shape[-1]:
        end = min(start + chunk_size, codes.shape[-1])
        context = left_context_size if start - left_context_size > 0 else start
        eager_chunks.append(decoder(codes[..., start - context : end])[..., context * TOTAL_UPSAMPLE :])
        start = end
    return torch.cat(eager_chunks, dim=-1)


def test_variable_length_batched_decode_matches_per_request_eager(wrapper):
    decoder, graph_wrapper, _ = wrapper
    long_codes = _codes(1, 50)
    short_codes = _codes(1, 25)
    padded = torch.zeros(2, NUM_QUANTIZERS, 50, dtype=torch.long)
    padded[0] = long_codes[0]
    padded[1, :, :25] = short_codes[0]
    graph = graph_wrapper.batched_chunked_decode_with_npugraph(
        padded,
        [50, 25],
        chunk_size=25,
        left_context_size=0,
        max_batch_size=2,
    )
    eager_long = _eager_chunked_decode(decoder, long_codes, chunk_size=25, left_context_size=0)
    eager_short = _eager_chunked_decode(decoder, short_codes, chunk_size=25, left_context_size=0)
    _assert_waveform_close(graph[0:1, :, : eager_long.shape[-1]], eager_long)
    _assert_waveform_close(graph[1:2, :, : eager_short.shape[-1]], eager_short)


def test_only_fallback_reason_is_logged_once(wrapper, caplog):
    caplog.set_level(logging.INFO)
    _, graph_wrapper, _ = wrapper
    caplog.clear()
    graph_wrapper.decode(_codes(1, 25))
    graph_wrapper.decode(_codes(1, 20))
    graph_wrapper.decode(_codes(1, 25))
    graph_wrapper.decode(_codes(1, 20))
    output = caplog.text
    assert "[Qwen3-TTS][NPU Code2Wav graph] exact hit" not in output
    assert output.count("[Qwen3-TTS][NPU Code2Wav graph] eager fallback") == 1
    assert "eager fallback batch_size=1 frames=20 reason=padding_disabled" in output


def test_stats_log_records_exact_hit_probability(wrapper, caplog):
    caplog.set_level(logging.INFO)
    _, graph_wrapper, _ = wrapper
    graph_wrapper.stats_log_every = 3
    caplog.clear()
    graph_wrapper.decode(_codes(1, 25))
    graph_wrapper.decode(_codes(1, 20))
    graph_wrapper.decode(_codes(1, 25))
    output = caplog.text
    assert (
        "stats total=3 exact_hits=2 padded_hits=0 graph_hits=2 "
        "fallbacks=1 exact_hit_rate=66.67% graph_hit_rate=66.67%"
    ) in output
    assert graph_wrapper._stats_total == 3
    assert graph_wrapper._stats_exact_hits == 2
    assert graph_wrapper._stats_padded_hits == 0
    assert graph_wrapper._stats_fallbacks == 1


def test_padding_exact_priority_and_smallest_same_batch_bucket(padding_wrapper):
    decoder, graph_wrapper, _ = padding_wrapper
    exact_codes = _codes(1, 25)
    padded_codes = _codes(1, 26)

    assert graph_wrapper._get_graph_key(1, 25) == (1, 25)
    assert graph_wrapper._get_graph_key(1, 26) == (1, 51)
    torch.testing.assert_close(graph_wrapper.decode(exact_codes), decoder(exact_codes), atol=0, rtol=0)
    _assert_waveform_close(graph_wrapper.decode(padded_codes), decoder(padded_codes))


def test_padding_never_selects_a_different_batch(padding_wrapper):
    decoder, graph_wrapper, _ = padding_wrapper
    codes = _codes(3, 26)
    assert graph_wrapper._get_graph_key(3, 26) is None
    torch.testing.assert_close(graph_wrapper.decode(codes), decoder(codes), atol=0, rtol=0)


def test_padding_clears_static_tail_trims_output_and_avoids_replay_pollution(padding_wrapper):
    decoder, graph_wrapper, _ = padding_wrapper
    graph_key = (1, 25)
    graph_wrapper.static_inputs[graph_key].fill_(7)

    short_codes = _codes(1, 20)
    short_output = graph_wrapper.decode(short_codes)
    assert short_output.shape[-1] == 20 * TOTAL_UPSAMPLE
    assert torch.count_nonzero(graph_wrapper.static_inputs[graph_key][..., 20:]) == 0
    _assert_waveform_close(short_output, decoder(short_codes))

    longer_codes = _codes(1, 24)
    _assert_waveform_close(graph_wrapper.decode(longer_codes), decoder(longer_codes))
    _assert_waveform_close(graph_wrapper.decode(short_codes), decoder(short_codes))
    assert torch.count_nonzero(graph_wrapper.static_inputs[graph_key][..., 20:]) == 0


@pytest.mark.parametrize(
    ("max_pad_frames", "expected_key"),
    [(0, (1, 25)), (5, (1, 25)), (4, None)],
)
def test_padding_limit_controls_routing(monkeypatch, max_pad_frames, expected_key):
    fake_npu = _FakeNPU()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    graph_wrapper = NPUGraphDecoderWrapper(
        _TinyCausalDecoder().eval(),
        capture_sizes=[25],
        num_quantizers=NUM_QUANTIZERS,
        padding_enabled=True,
        max_pad_frames=max_pad_frames,
    )
    graph_wrapper.warmup(torch.device("cpu"))
    assert graph_wrapper._get_graph_key(1, 20) == expected_key


def test_padding_limit_fallback_logs_specific_reason(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    fake_npu = _FakeNPU()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    graph_wrapper = NPUGraphDecoderWrapper(
        _TinyCausalDecoder().eval(),
        capture_sizes=[25],
        num_quantizers=NUM_QUANTIZERS,
        padding_enabled=True,
        max_pad_frames=4,
    )
    graph_wrapper.warmup(torch.device("cpu"))
    caplog.clear()
    graph_wrapper.decode(_codes(1, 20))
    assert "eager fallback batch_size=1 frames=20 reason=padding_limit" in caplog.text


def test_failed_capture_is_not_a_padding_bucket(monkeypatch):
    fake_npu = _FailingFakeNPU(fail_capture_index=2)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    graph_wrapper = NPUGraphDecoderWrapper(
        _TinyCausalDecoder().eval(),
        capture_sizes=[25, 51, 73],
        num_quantizers=NUM_QUANTIZERS,
        padding_enabled=True,
    )
    graph_wrapper.warmup(torch.device("cpu"))
    assert (1, 51) not in graph_wrapper.graphs
    assert graph_wrapper._get_graph_key(1, 26) == (1, 73)


def test_padding_stats_are_split_by_route_without_per_hit_logs(padding_wrapper, caplog):
    caplog.set_level(logging.INFO)
    _, graph_wrapper, _ = padding_wrapper
    graph_wrapper.stats_log_every = 3
    caplog.clear()

    graph_wrapper.decode(_codes(1, 25))
    graph_wrapper.decode(_codes(1, 26))
    graph_wrapper.decode(_codes(3, 26))
    graph_wrapper.decode(_codes(1, 26))
    output = caplog.text

    assert "exact hit" not in output
    assert "padded hit" not in output
    assert output.count("eager fallback batch_size=3 frames=26 reason=no_graph_bucket") == 1
    assert (
        "stats total=3 exact_hits=1 padded_hits=1 graph_hits=2 "
        "fallbacks=1 exact_hit_rate=33.33% graph_hit_rate=66.67%"
    ) in output
    assert graph_wrapper._stats_exact_hits == 1
    assert graph_wrapper._stats_padded_hits == 2
    assert graph_wrapper._stats_fallbacks == 1


def test_route_log_writes_one_compact_aggregate_line_per_stats_window(monkeypatch, tmp_path):
    fake_npu = _FakeNPU()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    decoder = _TinyCausalDecoder().eval()
    route_log_file = tmp_path / "code2wav-routes.jsonl"
    graph_wrapper = NPUGraphDecoderWrapper(
        decoder,
        capture_sizes=[25, 51],
        num_quantizers=NUM_QUANTIZERS,
        padding_enabled=True,
        stats_log_every=3,
        route_log_file=str(route_log_file),
    )
    graph_wrapper.warmup(torch.device("cpu"))
    for key in graph_wrapper.graphs:
        static_input = graph_wrapper.static_inputs[key]
        static_output = graph_wrapper.static_outputs[key]

        def replay(inp=static_input, out=static_output):
            out.copy_(decoder(inp))

        graph_wrapper.graphs[key] = _CapturedGraph(replay)

    graph_wrapper.decode(_codes(1, 25))
    graph_wrapper.decode(_codes(1, 26))
    assert route_log_file.read_text(encoding="utf-8") == ""
    graph_wrapper.decode(_codes(3, 26))

    lines = route_log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert {key: record[key] for key in ("total", "exact", "padded", "fallback", "window")} == {
        "total": 3,
        "exact": 1,
        "padded": 1,
        "fallback": 1,
        "window": 3,
    }
    assert record["routes"] == [
        ["E", 1, 25, 25, 1],
        ["F:no_graph_bucket", 3, 26, 0, 1],
        ["P", 1, 26, 51, 1],
    ]

    graph_wrapper.decode(_codes(1, 26))
    graph_wrapper.log_decode_stats()
    lines = route_log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["routes"] == [["P", 1, 26, 51, 1]]
