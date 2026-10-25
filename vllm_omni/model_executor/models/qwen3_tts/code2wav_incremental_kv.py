# SPDX-License-Identifier: Apache-2.0
"""Opt-in prefixless Code2Wav experiments. Never changes the stock decoder."""

from __future__ import annotations

import torch


def causal_window_mask(query, history, window, device, dtype):
    """Window includes the query itself: retain at most window-1 old keys."""
    q = torch.arange(history, history + query, device=device)[:, None]
    k = torch.arange(history + query, device=device)[None, :]
    visible = (k <= q) & (k > q - window)
    return torch.zeros((query, history + query), device=device, dtype=dtype).masked_fill(
        ~visible, torch.finfo(dtype).min
    )[None, None]


def waveform(decoder, hidden, frames):
    x = hidden.transpose(1, 2)
    for blocks in decoder.upsample:
        for block in blocks:
            x = block(x)
    for block in decoder.decoder:
        x = block(x)
    return x[..., -frames * decoder.total_upsample :].clamp(-1, 1)


@torch.no_grad()
def continuous_reference(decoder, codes, new_frames):
    """Full continuous convolution and sliding-attention recomputation.

    Diagnostic only: codes contain the complete request history, not a crop.
    Downstream context matches the production decoder (12 historical frames).
    """
    conv = decoder.pre_conv(decoder.quantizer.decode(codes)).transpose(1, 2)
    n = conv.shape[1]
    positions = torch.arange(n, device=conv.device)
    mask = causal_window_mask(n, 0, decoder.config.sliding_window, conv.device, conv.dtype)
    hidden = decoder.pre_transformer(
        inputs_embeds=conv,
        attention_mask={"sliding_attention": mask},
        position_ids=positions[None],
        cache_position=positions,
        use_cache=False,
    ).last_hidden_state
    return waveform(decoder, hidden[:, -(new_frames + 12) :], new_frames), hidden
