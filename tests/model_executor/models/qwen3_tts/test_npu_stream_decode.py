# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_tts.npu_stream_decode import NPUStreamDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeCode2Wav(nn.Module):
    config = SimpleNamespace(sliding_window=72)
    total_upsample = 2

    def batched_chunked_decode(self, codes, lengths, *, caches, chunk_size, left_context_size, max_batch_size):
        assert (chunk_size, left_context_size, max_batch_size) == (8, 72, 4)
        assert codes.shape[-1] == max(lengths)
        for cache, length in zip(caches, lengths, strict=True):
            cache["frames"] = cache.get("frames", 0) + length
        return [codes[row, :1, :length].float().repeat_interleave(2) for row, length in enumerate(lengths)]


def make_decoder():
    return NPUStreamDecoder(
        FakeCode2Wav(),
        first_chunk_frames=4,
        chunk_frames=8,
        left_context_frames=72,
        sample_rate=24000,
        max_batch_size=4,
    )


def test_eager_chunks_keep_request_cache_and_sample_counts():
    decoder = make_decoder()
    assert (decoder.decoder._initial_codec_chunk_frames, decoder.decoder._incremental_chunk_frames) == (4, 8)
    caches = [decoder.new_cache(), decoder.new_cache()]
    codes = [torch.arange(4).repeat(1, 16, 1), torch.arange(2).repeat(1, 16, 1) + 4]
    first, tail = decoder.decode_chunks(codes, caches)
    assert first.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert tail.tolist() == [4, 4, 5, 5]
    _, second = decoder.decode_chunks(list(reversed(codes)), list(reversed(caches)))
    assert second.tolist() == first.tolist()
    assert caches == [{"prefix_frames": 0, "frames": 8}, {"prefix_frames": 0, "frames": 4}]


def test_graph_dispatch_forwards_request_inputs_without_repacking():
    decoder = make_decoder()
    codes = [torch.ones((1, 16, 8), dtype=torch.long)]
    caches = [decoder.new_cache()]

    def decode(input_codes, input_caches):
        assert input_codes is codes
        assert input_caches is caches
        return [torch.ones((1, 1, 16))]

    decoder.graphs = SimpleNamespace(_batched_request_decode=decode)
    assert decoder.decode_chunks(codes, caches)[0].shape == (16,)


def test_detects_incorrect_pcm_length():
    decoder = make_decoder()
    decoder.graphs = SimpleNamespace(_batched_request_decode=lambda *args: [torch.empty(1)])
    codes = [torch.ones((1, 16, 4), dtype=torch.long)]
    with pytest.raises(RuntimeError, match="PCM length"):
        decoder.decode_chunks(codes, [decoder.new_cache()])
