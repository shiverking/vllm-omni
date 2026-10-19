# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in, zero-training target-sampling tree coverage experiment.

No KV cache, learned parameters, rejection/residual resampling, or RNG-dependent
drafts. The explicit attention backend is deliberately a correctness reference.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache

import torch


def validate_tree(paths):
    paths = [tuple(p) for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate rank paths")
    for p in paths:
        if not 1 <= len(p) <= 3 or any(type(r) is not int or not 0 <= r < 4 for r in p):
            raise ValueError("rank paths must have depth 1..3 and ranks 0..3")
        if len(p) > 1 and p[:-1] not in paths:
            raise ValueError("tree must be ancestor closed")
    return sorted(paths, key=lambda p: (len(p), p))


def fit_tree(rank_paths, budget):
    """Greedy marginal covered prefix length; each extension requires ancestors.

    Each calibration observation is a full future rank path (including -1 for
    out-of-top4). Adding p gains one accepted position iff that complete prefix
    is covered. Unlike independent per-depth hit rates this retains correlation.
    """
    if budget not in (8, 16, 32):
        raise ValueError("budget must be 8, 16 or 32")
    counts = Counter(tuple(p[:d]) for p in rank_paths for d in range(1, len(p) + 1))
    universe = [p for d in range(1, 4) for p in itertools.product(range(4), repeat=d)]
    chosen = []
    while len(chosen) < budget:
        frontier = [p for p in universe if p not in chosen and (len(p) == 1 or p[:-1] in chosen)]
        chosen.append(min(frontier, key=lambda p: (-counts[p], len(p), p)))
    return validate_tree(chosen)


def layout(prefix_len, paths, device):
    """True=visible, prefix causal, candidates see only prefix/self/ancestors."""
    return _layout(prefix_len, tuple(validate_tree(paths)), str(device))


@lru_cache(maxsize=512)
def _layout(prefix_len, paths, device):
    # Static metadata is constructed on the host once, never with hundreds of
    # scalar device writes in the measured loop. Returned tensors are read-only.
    mask = [[j <= i for j in range(prefix_len)] + [False] * len(paths) for i in range(prefix_len)]
    positions = list(range(prefix_len))
    for p in paths:
        positions.append(prefix_len + len(p) - 1)
        mask.append([True] * prefix_len + [p[: len(a)] == a for a in paths])
    return torch.tensor([positions], device=device), torch.tensor(mask, dtype=torch.bool, device=device)


def frame_uniforms(seed, request_id, frame_id, groups, vocab, device):
    """CPU RNG is portable across replay devices; every codebook has its own key."""
    rows = []
    for position in range(1, groups):
        key = f"{seed}:{request_id}:{frame_id}:{position}".encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)
        rows.append(torch.rand(vocab, generator=torch.Generator().manual_seed(value)))
    return torch.stack(rows).unsqueeze(0).to(device) if rows else torch.empty(1, 0, vocab, device=device)


class PhaseTimer:
    """Optional diagnostic timers. Resolve events only after the entire frame."""

    def __init__(self, device, enabled=False):
        self.enabled = enabled
        self.api = getattr(torch, device.type, None) if device.type in ("cuda", "npu") else None
        self.records = []

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        start = self.api.Event(enable_timing=True) if self.api else None
        end = self.api.Event(enable_timing=True) if self.api else None
        wall = time.perf_counter()
        if start:
            start.record()
        yield
        if end:
            end.record()
        self.records.append((name, time.perf_counter() - wall, start, end))

    def resolve(self):
        if self.api and self.records:
            self.api.synchronize()
        return [
            dict(phase=n, wall_ms=w * 1000, device_ms=s.elapsed_time(e) if s else None) for n, w, s, e in self.records
        ]


def reference_forward(predictor, prefix):
    positions, mask = layout(prefix.shape[1], [], prefix.device)
    return predictor.model(prefix, positions, mask)


def initial_prefix(predictor, layer0_embed, last_talker_hidden):
    dtype = next(predictor.model.parameters()).dtype
    project = predictor.small_to_mtp_projection
    return torch.cat(
        (project(last_talker_hidden.reshape(1, 1, -1).to(dtype)), project(layer0_embed.reshape(1, 1, -1).to(dtype))),
        dim=1,
    )


def sample(predictor, logits, position, uniforms, do_sample, temperature, top_k):
    if not do_sample or temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    scaled = logits * (1.0 / max(temperature, 1e-6))
    if top_k > 0:
        values = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1).values
        scaled = scaled.masked_fill(scaled < values[..., -1:], float("-inf"))
    return predictor._sample_codes_gumbel(scaled, uniforms=uniforms[:, position - 1])


def select_covered(predictions, parents, tokens, max_steps):
    """Tensor-only traversal; predictions[0] is the true-prefix target draw."""
    device = predictions.device
    node_ids = torch.arange(1, len(tokens) + 1, device=device)
    current = torch.zeros((), dtype=torch.long, device=device)
    live = torch.ones((), dtype=torch.bool, device=device)
    count = torch.zeros((), dtype=torch.long, device=device)
    accepted = count.clone()
    selected = []
    for _ in range(max_steps):
        token = predictions[current]
        selected.append(token)
        count += live.long()
        matches = (parents == current) & (tokens == token) & live
        child = torch.where(matches, node_ids, 0).max()
        hit = child > 0
        accepted += hit.long()
        live = live & hit
        current = child
    return torch.stack(selected), count, accepted


def matmul_flops(predictor, positions, attention_pairs, head_rows, projection_rows):
    """Analytical dense matmul work, excluding pointwise ops and hardware padding."""
    costs = getattr(predictor, "_research_matmul_costs", None)
    if costs is None:
        linear = 2 * sum(
            module.weight.numel() for module in predictor.model.modules() if isinstance(module, torch.nn.Linear)
        )
        attention = 4 * sum(layer.self_attn.num_heads * layer.self_attn.head_dim for layer in predictor.model.layers)
        head = 2 * predictor.config.hidden_size * predictor.config.vocab_size
        project = predictor.small_to_mtp_projection
        projection = 2 * project.weight.numel() if isinstance(project, torch.nn.Linear) else 0
        costs = predictor._research_matmul_costs = (linear, attention, head, projection)
    return sum(a * b for a, b in zip(costs, (positions, attention_pairs, head_rows, projection_rows)))


@torch.inference_mode()
def decode(
    predictor,
    layer0_code,
    layer0_embed,
    last_talker_hidden,
    *,
    paths=(),
    sample_uniforms=None,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
    detailed=False,
    profile=False,
    audit=False,
    **unused,
):
    """Return codes plus buffered diagnostics; one host decision per round.

    The round boundary transfers only (count, accepted) to the host, never each
    token. This synchronization is included in timing and explicitly reported.
    """
    if layer0_code.shape[0] != 1:
        raise ValueError("zero-training experiment supports exactly one request")
    if top_p != 1.0:
        raise ValueError("experimental sampling requires top_p=1.0")
    groups = predictor._num_groups
    device = layer0_code.device
    if do_sample and temperature > 0:
        if sample_uniforms is None or sample_uniforms.shape != (1, groups - 1, predictor.config.vocab_size):
            raise ValueError("provide position-bound sample_uniforms for experimental sampling")
    paths = validate_tree(paths)
    prefix = initial_prefix(predictor, layer0_embed, last_talker_hidden)
    codes = [layer0_code.reshape(1, 1)]
    timer = PhaseTimer(device, profile)
    events, calibration = [], []
    calls, work, attention_pairs = 0, 0, 0
    head_rows, projection_rows = 0, 2
    audit_errors = []
    position = 1
    while position < groups:
        start_position = position
        with timer.phase("draft"):
            root = reference_forward(predictor, prefix)[:, -1]
            calls += 1
            work += prefix.shape[1]
            attention_pairs += prefix.shape[1] ** 2
            active = [p for p in paths if len(p) <= groups - position]
            depth = max(map(len, active), default=0)
            head_rows += depth
            candidates = [predictor.lm_head[position + d - 1](root).topk(4, dim=-1).indices for d in range(depth)]
            if detailed:
                calibration.append((position, root.detach().clone()))
        if not active:
            head_rows += 1
            with timer.phase("select"):
                code = sample(
                    predictor,
                    predictor.lm_head[position - 1](root),
                    position,
                    sample_uniforms,
                    do_sample,
                    temperature,
                    top_k,
                )
                committed = code
            count, accepted = 1, 0
        else:
            head_rows += 1 + sum(position + len(p) < groups for p in active)
            projection_rows += len(active)
            with timer.phase("verify"):
                node_codes = [candidates[len(p) - 1][:, p[-1] : p[-1] + 1] for p in active]
                node_embeds = [
                    predictor.small_to_mtp_projection(predictor.model.codec_embedding[position + len(p) - 2](code))
                    for p, code in zip(active, node_codes)
                ]
                packed = torch.cat([prefix, *node_embeds], dim=1)
                pos, mask = layout(prefix.shape[1], active, device)
                hidden = predictor.model(packed, pos, mask)
                calls += 1
                work += packed.shape[1]
                attention_pairs += packed.shape[1] ** 2
                if audit:
                    for i, path in enumerate(active):
                        branch = [node_embeds[active.index(path[:d])] for d in range(1, len(path) + 1)]
                        serial = reference_forward(predictor, torch.cat([prefix, *branch], dim=1))[:, -1]
                        next_position = position + len(path)
                        if next_position < groups:
                            head = predictor.lm_head[next_position - 1]
                            left, right = head(hidden[:, prefix.shape[1] + i]), head(serial)
                            tolerance = 1e-5 if left.dtype == torch.float32 else 2e-2
                            torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)
                            audit_errors.append((left - right).abs().max().detach())
                # Row zero is the real prefix. Each other row predicts after
                # a complete candidate path, using its real codebook head.
                predictions = [
                    sample(
                        predictor,
                        predictor.lm_head[position - 1](hidden[:, prefix.shape[1] - 1]),
                        position,
                        sample_uniforms,
                        do_sample,
                        temperature,
                        top_k,
                    )
                ]
                for i, p in enumerate(active):
                    next_position = position + len(p)
                    predictions.append(
                        sample(
                            predictor,
                            predictor.lm_head[next_position - 1](hidden[:, prefix.shape[1] + i]),
                            next_position,
                            sample_uniforms,
                            do_sample,
                            temperature,
                            top_k,
                        )
                        if next_position < groups
                        else torch.zeros_like(predictions[0])
                    )
            with timer.phase("select"):
                predictions = torch.cat(predictions, dim=1).reshape(-1)
                parents = torch.tensor([0 if len(p) == 1 else active.index(p[:-1]) + 1 for p in active], device=device)
                tokens = torch.cat(node_codes, dim=1).reshape(-1)
                selected, count_t, accepted_t = select_covered(
                    predictions, parents, tokens, min(depth + 1, groups - position)
                )
                # One synchronization at the variable-length round boundary.
                count, accepted = torch.stack((count_t, accepted_t)).cpu().tolist()
                committed = selected[:count].reshape(1, -1)
        with timer.phase("prefix_update"):
            codes.append(committed)
            for d in range(count):
                if position + d < groups - 1:
                    projection_rows += 1
                    embed = predictor.model.codec_embedding[position + d - 1](committed[:, d : d + 1])
                    prefix = torch.cat((prefix, predictor.small_to_mtp_projection(embed)), dim=1)
            position += count
        if detailed:
            events.append(
                dict(
                    position=start_position,
                    committed=count,
                    accepted=accepted,
                    nodes=len(active),
                    backbone_calls=2 if active else 1,
                    fallback_reason=(
                        "frame_end"
                        if position == groups
                        else "reference_ar"
                        if not active
                        else "leaf_bonus"
                        if accepted == depth
                        else "target_outside_tree"
                    ),
                )
            )
    return torch.cat(codes, dim=1), dict(
        events=events,
        states=calibration,
        backbone_calls=calls,
        processed_positions=work,
        attention_pairs=attention_pairs,
        estimated_matmul_flops=matmul_flops(predictor, work, attention_pairs, head_rows, projection_rows),
        audit_max_logit_error=torch.stack(audit_errors).max().cpu().tolist() if audit_errors else None,
        timings=timer.resolve(),
        attention_backend="explicit_fp32_matmul_softmax",
        graph="eager",
        host_sync="one_count_transfer_per_tree_round",
    )
