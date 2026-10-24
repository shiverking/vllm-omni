# SPDX-License-Identifier: Apache-2.0
"""Per-decoder opt-in frozen constants; no global CUDA or baseline patches."""

from contextlib import contextmanager

import torch
from torch import nn

from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import EuclideanCodebook

VARIANTS = ("current", "constants", "pooled", "fused")


class FrozenCodebook(EuclideanCodebook):
    def _apply(self, fn, recurse=True):
        self._normalized_embedding = None
        return super()._apply(fn, recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._normalized_embedding = None
        return super()._load_from_state_dict(*args, **kwargs)

    def prepare(self):
        self._normalized_embedding = (
            (self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]).detach().contiguous()
        )

    def decode(self, codes):
        if self._normalized_embedding is None:
            self.prepare()
        return nn.functional.embedding(codes, self._normalized_embedding)


def prepare_constants(decoder):
    for module in decoder.modules():
        if isinstance(module, EuclideanCodebook):
            if not isinstance(module, FrozenCodebook):
                module.__class__ = FrozenCodebook
                module.register_buffer("_normalized_embedding", None, persistent=False)
            module.prepare()


def derived_bytes(decoder):
    return sum(
        v.numel() * v.element_size()
        for n, v in decoder.named_buffers()
        if "_normalized_embedding" in n or "_packed_" in n
    )


@contextmanager
def fixed_stage(wrapper, batch, length):
    """Use explicit immutable positions/mask only during graph construction."""
    if wrapper.optimization == "current":
        yield
        return
    transformer = wrapper.decoder.pre_transformer
    key = (batch, length)
    if key not in wrapper._stage_constants:
        parameter = next(transformer.parameters())
        positions = torch.arange(length, device=parameter.device)
        dummy = torch.empty(
            batch, length, transformer.config.hidden_size, device=parameter.device, dtype=parameter.dtype
        )
        # Explicit mask preserves the existing sliding window, including its
        # exact boundary convention. Build it outside the capture context.
        import inspect

        from transformers.masking_utils import create_sliding_window_causal_mask

        kwargs = dict(
            config=transformer.config,
            attention_mask=torch.ones(batch, length, device=parameter.device, dtype=torch.bool),
            past_key_values=None,
            position_ids=positions.unsqueeze(0),
        )
        if "input_embeds" in inspect.signature(create_sliding_window_causal_mask).parameters:
            kwargs.update(input_embeds=dummy, cache_position=positions)
        else:
            kwargs["inputs_embeds"] = dummy
        from .npu_segmented_graph_wrapper import capture_mask_context

        with capture_mask_context():
            mask = create_sliding_window_causal_mask(**kwargs)
        cos, sin = transformer.rotary_emb(dummy, positions.unsqueeze(0))
        wrapper._stage_constants[key] = (positions, mask, (cos, sin))
    previous = getattr(transformer, "_code2wav_fixed_stage", None)
    transformer._code2wav_fixed_stage = wrapper._stage_constants[key]
    try:
        yield
    finally:
        transformer._code2wav_fixed_stage = previous


class PackedProjection(nn.Module):
    """Derived buffers, preserving original checkpoint parameter ownership."""

    def __init__(self, sources):
        super().__init__()
        object.__setattr__(self, "sources", tuple(sources))
        self.sizes = tuple(m.out_features for m in sources)
        self.register_buffer("_packed_weight", None, persistent=False)
        self.register_buffer("_packed_bias", None, persistent=False)
        self.prepare()

    def prepare(self):
        self._packed_weight = torch.cat([m.weight.detach() for m in self.sources], 0).contiguous()
        if any(m.bias is not None for m in self.sources):
            self._packed_bias = torch.cat(
                [m.bias.detach() if m.bias is not None else m.weight.new_zeros(m.out_features) for m in self.sources], 0
            ).contiguous()

    def _apply(self, fn, recurse=True):
        self._packed_weight = self._packed_bias = None
        return super()._apply(fn, recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._packed_weight = self._packed_bias = None
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, hidden):
        if self._packed_weight is None:
            self.prepare()
        return nn.functional.linear(hidden, self._packed_weight, self._packed_bias).split(self.sizes, -1)


def prepare_fused(decoder):
    for layer in decoder.pre_transformer.layers:
        attn = layer.self_attn
        attn._code2wav_qkv = PackedProjection((attn.q_proj, attn.k_proj, attn.v_proj))
        mlp = layer.mlp
        mlp._code2wav_gate_up = PackedProjection((mlp.gate_proj, mlp.up_proj))
