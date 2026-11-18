# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState, FrameBatch
from vllm_omni.worker_v2.streaming_audio import ChunkedAudioBuffer, ChunkedAudioOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeDecoder:
    first_chunk_frames = 4
    chunk_frames = 8
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def new_cache(self):
        return {"position": 0}

    def decode_chunks(self, codes, caches):
        self.calls.append([(int(cache["position"]), code.shape[-1]) for code, cache in zip(codes, caches, strict=True)])
        waves = []
        for code, cache in zip(codes, caches, strict=True):
            assert code.shape[1] == 16 and code.dtype == torch.long
            cache["position"] += code.shape[-1]
            waves.append(code[0, 0].float())
        return waves


def _frame(token: int) -> torch.Tensor:
    return torch.full((16,), token, dtype=torch.long)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, []),
        (1, [1]),
        (3, [3]),
        (4, [4]),
        (5, [4, 1]),
        (11, [4, 7]),
        (12, [4, 8]),
        (13, [4, 8, 1]),
        (99, [4] + [8] * 11 + [7]),
    ],
)
def test_npu_chunk_sizes_and_eos_without_frame(count, expected):
    decoder = FakeDecoder()
    state = ChunkedAudioBuffer()
    output = []
    for token in range(count):
        output.extend(state.decode(decoder, {"r": _frame(token)}, set()).values())
    output.extend(state.decode(decoder, {}, {"r"}).values())
    assert [len(chunk) for chunk in output] == expected
    assert not output or torch.cat(output).tolist() == list(range(count))
    assert state.decode(decoder, {}, {"r"}) == {}
    assert state.requests == {}


def test_npu_chunked_audio_request_order_cancel_and_reuse():
    state, decoder = ChunkedAudioBuffer(), FakeDecoder()
    outputs = {}
    for offset in range(4):
        rows = [("a", _frame(10 + offset)), ("b", _frame(20 + offset)), ("c", _frame(30 + offset))]
        if offset % 2:
            rows.reverse()
        outputs.update(state.decode(decoder, dict(rows), set()))
    assert {req_id: pcm.tolist() for req_id, pcm in outputs.items()} == {
        "a": [10, 11, 12, 13],
        "b": [20, 21, 22, 23],
        "c": [30, 31, 32, 33],
    }
    assert state.decode(decoder, {"a": _frame(14), "b": _frame(24)}, set()) == {}
    state.discard(("a",))
    output = state.decode(decoder, {"a": _frame(90)}, {"a", "b"})
    assert output["a"].tolist() == [90]
    assert output["b"].tolist() == [24]
    assert decoder.calls[-1] == [(0, 1), (4, 1)]
    assert state.requests["c"].cache["position"] == 4


def test_npu_chunk_output_snapshot():
    first, second = torch.arange(4), torch.arange(8)
    pending = ChunkedAudioOutput([first, None, second], torch.tensor(24000))
    snapshot = pending.to_cpu(None, torch.clone)
    first.zero_()
    second.zero_()
    output = snapshot.get_output()
    assert output[0]["model_outputs"].tolist() == list(range(4))
    assert output[1] is None
    assert output[2]["model_outputs"].tolist() == list(range(8))
    assert output[2]["sr"].item() == 24000


def test_npu_worker_flushes_eos_and_length_before_request_completion():
    decoder = FakeDecoder()
    scopes = []
    graph_codes = torch.empty(2, 16, dtype=torch.long)

    def mtp(_req_indices, ids, emb, _hidden, _step, _dispatcher, *, req_ids):
        scopes.append(tuple(req_ids))
        graph_codes[: len(ids)].copy_(ids[:, None])
        return emb, graph_codes

    model = SimpleNamespace(
        npu_stream_decoder=decoder,
        _codec_eos_token_id=2048,
        mtp_frame_valid=lambda ids: (ids >= 0) & (ids < 2048),
        embed_input_ids=lambda ids: ids.float(),
    )
    owner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(max_model_len=20),
        ),
        _eager_mtp=True,
        _mtp_input_ids=torch.empty(2, dtype=torch.long),
        _mtp_input_embeds=torch.empty(2, 1),
        _mtp_hidden=torch.empty(2, 1),
        _mtp_text_step=torch.empty(2, 1),
        _mtp_forward=mtp,
    )
    state = EagerMTPState(owner)
    state.embeds = torch.zeros(2, 1)
    req_states = SimpleNamespace(max_seq_len=np.array([20, 4]))

    def step(req_ids, slots, computed, sampled, num_sampled):
        batch = SimpleNamespace(
            num_reqs=2,
            req_ids=req_ids,
            idx_mapping_np=np.array(slots),
            num_computed_tokens_np=np.array(computed),
            num_scheduled_tokens=np.ones(2, dtype=int),
            query_start_loc=torch.arange(3),
        )
        state.recorded = (
            batch,
            FrameBatch([(i, slot, req_id, False) for i, (req_id, slot) in enumerate(zip(req_ids, slots, strict=True))]),
            torch.tensor([1, 2]),
        )
        payload = {
            "codes": {"audio": torch.zeros(2, 16, dtype=torch.long)},
            "meta": {"codec_frame_valid": torch.zeros(2)},
        }
        return state.run_eager_mtp(
            batch,
            torch.zeros(2, 1),
            torch.tensor(sampled),
            payload,
            chunk_sampling=(req_states, torch.tensor(num_sampled)),
        ).get_output()

    assert step(["a", "b"], [0, 1], [1, 1], [7, 12], [1, 1]) == [None, None]
    # Length capping retains b's last sampled frame, while a emits only its
    # already buffered frame on codec EOS. Graph-owned rows are reused.
    result = step(["b", "a"], [1, 0], [2, 2], [13, 2048], [1, 1])
    assert [item["model_outputs"].tolist() for item in result] == [[12, 13], [7]]
    assert scopes == [("a", "b"), ("b",)]
    assert step(["a", "b"], [0, 1], [3, 3], [99, 99], [0, 1]) == [None, None]
    assert state._chunked_audio.requests == {}
    assert scopes == [("a", "b"), ("b",)]
