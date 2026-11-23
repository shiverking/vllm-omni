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


@torch.inference_mode()
def test_icl_incremental_stream_matches_generated_pcm_and_releases_slots(monkeypatch):
    from tests.model_executor.models.qwen3_tts.test_qwen3_tts_incremental_decode import (
        _make_cpu_incremental_codec_fixture,
    )
    from vllm_omni.platforms.npu.layers.incremental_codec import NPUIncrementalCodec

    decoder, generator, replays = _make_cpu_incremental_codec_fixture(monkeypatch)
    stream = NPUStreamDecoder(
        decoder,
        first_chunk_frames=4,
        chunk_frames=8,
        left_context_frames=72,
        sample_rate=24000,
        max_batch_size=2,
    )
    arena = NPUIncrementalCodec(decoder, 2, 2, 8, initial_chunk_frames=4, reference_cache_capacity=2)
    decoder._npu_incremental = arena

    lengths = (53, 72)
    references = [torch.randint(0, 32, (length, 2), generator=generator) for length in lengths]
    generated = torch.randint(0, 32, (2, 2, 15), generator=generator)
    caches = [stream.new_cache(), stream.new_cache()]
    keys = ("alice:53:2:digest", "bob:72:2:digest")
    expected = []
    for row, (cache, reference, key) in enumerate(zip(caches, references, keys, strict=True)):
        stream.prepare_reference(cache, reference, key)
        all_codes = torch.cat((reference.T, generated[row]), dim=-1).unsqueeze(0)
        expected.append(decoder(all_codes)[0, 0, lengths[row] * stream.samples_per_frame :])

    def check_phase(lo, hi, order):
        inputs = [generated[row : row + 1, :, lo:hi] for row in order]
        waves = stream.decode_chunks(inputs, [caches[row] for row in order])
        for row, wave in zip(order, waves, strict=True):
            pcm_lo = lo * stream.samples_per_frame
            pcm_hi = hi * stream.samples_per_frame
            assert wave.numel() == pcm_hi - pcm_lo
            torch.testing.assert_close(wave, expected[row][pcm_lo:pcm_hi], atol=1e-5, rtol=1e-4)

    try:
        check_phase(0, 4, (0, 1))
        assert set(arena._reference_states) == set(keys)
        assert (2, 4) in replays
        replays.clear()
        check_phase(4, 12, (1, 0))
        check_phase(12, 15, (0, 1))
        assert replays == [(2, 8)]

        first_slot, second_slot = [cache["npu_slot"] for cache in caches]
        assert arena.frames[first_slot].item() == lengths[0] + 15
        assert arena.frames[second_slot].item() == lengths[1] + 15
        stream.release_cache(caches[0])
        stream.release_cache(caches[0])
        assert arena.free == [first_slot]
        assert arena.frames[first_slot].item() == 0
        assert arena.frames[second_slot].item() == lengths[1] + 15

        fresh = stream.new_cache()
        stream.prepare_reference(fresh, references[0], keys[0])
        replays.clear()
        (warm_pcm,) = stream.decode_chunks([generated[0:1, :, :4]], [fresh])
        assert replays == [(1, 4)]
        assert fresh["npu_slot"] == first_slot
        torch.testing.assert_close(warm_pcm, expected[0][: 4 * stream.samples_per_frame], atol=1e-5, rtol=1e-4)
        stream.release_cache(fresh)
        stream.release_cache(caches[1])
        assert set(arena.free) == {first_slot, second_slot}
    finally:
        arena.close()
