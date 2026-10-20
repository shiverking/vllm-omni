# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-request RVQ pipeline. A sampled tail is processed in the next round.

Experimental, frozen weights, target-sampling coverage (not residual sampling).
No host decision or cache inspection occurs inside a captured step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from . import qwen3_code_predictor_spec as spec

CANDIDATE_LAYOUT = "known_tail_plus_next3_v1"


@dataclass
class LayerKV:
    """K is already normalized and rotated. Storage stays in KV-head layout."""

    keys: torch.Tensor
    values: torch.Tensor
    capacity: int

    def stage(self, keys, values):
        length = keys.shape[2]
        self.keys[:, :, self.capacity : self.capacity + length].copy_(keys)
        self.values[:, :, self.capacity : self.capacity + length].copy_(values)
        return self.keys, self.values


def select_path(predictions, parents, tokens, steps):
    """Return new output tokens, hit path, and the parent of the last draw.

    Row zero is the known tail. Path entries index rows of this round's
    hidden/KV tensors. The final miss/bonus draw has no KV yet.
    """
    node_ids = torch.arange(1, len(tokens) + 1, device=tokens.device)
    current = predictions.new_zeros(())
    live = torch.ones((), dtype=torch.bool, device=tokens.device)
    count = current.clone()
    accepted = current.clone()
    parent_of_tail = current.clone()
    values, hit_path = [], [current]
    for _ in range(steps):
        # A scalar tensor used in Python indexing may dispatch .item() on NPU.
        # index_select keeps the dynamic branch choice entirely on device.
        value = predictions.index_select(0, current.reshape(1)).squeeze(0)
        values.append(value)
        parent_of_tail = torch.where(live, current, parent_of_tail)
        count = count + live.long()
        matches = (parents == current) & (tokens == value) & live
        child = torch.where(matches, node_ids, 0).max() if len(tokens) else current.new_zeros(())
        hit = child > 0
        hit_path.append(child)
        accepted = accepted + hit.long()
        live = live & hit
        current = child
    return torch.stack(values), count, accepted, torch.stack(hit_path), parent_of_tail


class PipelineExecutor:
    """Reusable buffers and optional NPUGraphs; each decode starts a fresh frame.

    An executor has fixed weights, tree, sampling parameters, and cache policy.
    Graph mode is strict: unsupported operators/capture errors propagate.
    """

    def __init__(
        self, predictor, paths=(), *, cache=True, execution="eager", do_sample=True, temperature=0.9, top_k=50
    ):
        if execution not in ("eager", "graph"):
            raise ValueError("execution must be eager or graph")
        self.predictor = predictor
        self.paths = tuple(spec.validate_tree(paths))
        self.cache = cache
        self.execution = execution
        self.do_sample, self.temperature, self.top_k = do_sample, temperature, top_k
        parameter = next(predictor.model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        if execution == "graph" and (self.device.type != "npu" or not cache):
            raise ValueError("pipeline graph requires NPU and KV cache")
        self.groups = predictor._num_groups
        if self.groups < 2:
            raise ValueError("pipeline requires at least two codebooks")
        self.capacity = self.groups + 1
        self.query_size = 1 + len(self.paths)
        hidden = predictor.config.hidden_size
        self.codes = torch.zeros(1, self.groups, dtype=torch.long, device=self.device)
        self.pending = self.codes.new_zeros(1, 1)
        self.draft_hidden = parameter.new_zeros(1, hidden)
        self.uniforms = parameter.new_zeros(1, self.groups - 1, predictor.config.vocab_size, dtype=torch.float32)
        self.prefix = parameter.new_zeros(1, self.capacity, hidden)
        self.query = parameter.new_zeros(1, self.query_size, hidden)
        self.last_hidden = torch.zeros_like(self.query)
        self.packed = parameter.new_zeros(1, self.capacity + self.query_size, hidden)
        self.state = self.codes.new_zeros(2)
        self.path_buffer = self.codes.new_zeros(4)
        self.layers = []
        for layer in predictor.model.layers:
            attn = layer.self_attn
            shape = (1, attn.num_kv_heads, self.capacity + max(2, self.query_size), attn.head_dim)
            self.layers.append(LayerKV(parameter.new_zeros(shape), parameter.new_zeros(shape), self.capacity))
        # Metadata is built before timing, never by reading a device mask.
        self.layouts = {p: self._layout(p) for p in range(1, self.groups - 1)}
        self.prefill_positions = torch.arange(2, device=self.device)[None]
        prefill_mask = torch.ones(2, 2, dtype=torch.bool, device=self.device).tril()
        self.prefill_mask = spec.canonical_mask(prefill_mask, self.capacity)
        # Cached prefill writes to the temporary area, whose first row is capacity.
        self.cached_prefill_mask = type(self.prefill_mask)(
            self.prefill_mask.visibility,
            self.prefill_mask.ancestor_indices + self.capacity,
            self.prefill_mask.ancestor_valid,
        )
        self.graphs = {}
        self.capture_seconds = 0.0
        self.capture_error = None
        self.capture_progress = None
        self.prepared = False
        self.timer = spec.PhaseTimer(self.device, False)
        self.last_debug = {}
        self.last_position = None

    def _layout(self, position):
        active = tuple(p for p in self.paths if position + len(p) < self.groups)
        # Physical keys = fixed prefix capacity + known root + candidate nodes.
        rows, positions = [], []
        for index in range(self.query_size):
            if index == 0:
                branch = [0]
                pos = position + 1
            elif index <= len(active):
                path = active[index - 1]
                branch = [0] + [1 + active.index(path[:d]) for d in range(1, len(path) + 1)]
                pos = position + 1 + len(path)
            else:
                branch, pos = [index], 0
            indices = list(range(position + 1)) + [self.capacity + i for i in branch]
            rows.append(indices)
            positions.append(pos)
        valid = [[True] * len(row) + [False] * (self.capacity - len(row)) for row in rows]
        indices = [row + [row[0]] * (self.capacity - len(row)) for row in rows]
        mask_type = type(self.prefill_mask) if hasattr(self, "prefill_mask") else None
        if mask_type is None:
            from .qwen3_code_predictor import CanonicalAttentionMask

            mask_type = CanonicalAttentionMask
        visible = torch.zeros(1, self.query_size, self.capacity + self.query_size, dtype=torch.bool)
        for i, row in enumerate(rows):
            visible[0, i, row] = True
        mask = mask_type(
            visible.to(self.device),
            torch.tensor(indices, device=self.device),
            torch.tensor(valid, device=self.device),
        )
        # No-cache control recomputes fixed prefix slots followed by the same tree.
        dense_rows = []
        for i in range(self.capacity):
            dense_rows.append(list(range(i + 1)) if i < position + 1 else [0])
        dense_rows.extend(rows)
        full_mask = torch.zeros(self.capacity + self.query_size, self.capacity + self.query_size, dtype=torch.bool)
        for i, row in enumerate(dense_rows):
            full_mask[i, row] = True
        dense_mask = spec.canonical_mask(full_mask.to(self.device), self.capacity)
        full_positions = list(range(position + 1)) + [0] * (self.capacity - position - 1) + positions
        parents = [0 if len(p) == 1 else active.index(p[:-1]) + 1 for p in active]
        return dict(
            active=active,
            positions=torch.tensor([positions], device=self.device),
            mask=mask,
            full_positions=torch.tensor([full_positions], device=self.device),
            full_mask=dense_mask,
            parents=torch.tensor(parents, device=self.device, dtype=torch.long),
        )

    def _sample(self, logits, position):
        return spec.sample(
            self.predictor, logits, position, self.uniforms, self.do_sample, self.temperature, self.top_k
        )

    def _prefill(self):
        self.codes.zero_()
        self.codes[:, :1].copy_(self.input_code)
        self.prefix.zero_()
        self.prefix[:, :2].copy_(spec.initial_prefix(self.predictor, self.input_embed, self.input_hidden))
        if self.cache:
            for layer in self.layers:
                layer.keys.zero_()
                layer.values.zero_()
        hidden = self.predictor.model(
            self.prefix[:, :2],
            self.prefill_positions,
            self.cached_prefill_mask if self.cache else self.prefill_mask,
            self.layers if self.cache else None,
        )
        if self.cache:
            for layer in self.layers:
                layer.keys[:, :, :2].copy_(layer.keys[:, :, self.capacity : self.capacity + 2])
                layer.values[:, :, :2].copy_(layer.values[:, :, self.capacity : self.capacity + 2])
        self.draft_hidden.copy_(hidden[:, -1])
        self.pending.copy_(self._sample(self.predictor.lm_head[0](self.draft_hidden), 1))
        self.codes[:, 1:2].copy_(self.pending)

    def _step(self, position):
        predictor = self.predictor
        meta = self.layouts[position]
        active = meta["active"]
        with self.timer.phase("draft"):
            depth = max(map(len, active), default=0)
            candidates = [
                predictor.lm_head[position + d](self.draft_hidden).topk(4, dim=-1).indices for d in range(depth)
            ]
            self.query.zero_()
            embed = predictor.model.codec_embedding[position - 1](self.pending)
            self.query[:, :1].copy_(predictor.small_to_mtp_projection(embed))
            node_codes = []
            for i, path in enumerate(active):
                code = candidates[len(path) - 1][:, path[-1] : path[-1] + 1]
                node_codes.append(code)
                embed = predictor.model.codec_embedding[position + len(path) - 1](code)
                self.query[:, i + 1 : i + 2].copy_(predictor.small_to_mtp_projection(embed))
        with self.timer.phase("backbone_inclusive"):
            if self.cache:
                hidden = predictor.model(self.query, meta["positions"], meta["mask"], self.layers)
            else:
                self.packed[:, : self.capacity].copy_(self.prefix)
                self.packed[:, self.capacity :].copy_(self.query)
                hidden = predictor.model(self.packed, meta["full_positions"], meta["full_mask"])
                hidden = hidden[:, self.capacity :]
        with self.timer.phase("sample_select"):
            predictions = [self._sample(predictor.lm_head[position](hidden[:, 0]), position + 1)]
            for i, path in enumerate(active):
                next_position = position + len(path) + 1
                predictions.append(
                    self._sample(predictor.lm_head[next_position - 1](hidden[:, i + 1]), next_position)
                    if next_position < self.groups
                    else self.pending.new_zeros(1, 1)
                )
            predictions = torch.cat(predictions, dim=1).flatten()
            tokens = torch.cat(node_codes, dim=1).flatten() if node_codes else self.codes.new_empty(0)
            selected, count, accepted, hit_path, parent = select_path(
                predictions, meta["parents"], tokens, min(depth + 1, self.groups - position - 1)
            )
            self.state[0].copy_(count)
            self.state[1].copy_(accepted)
            self.path_buffer.zero_()
            self.path_buffer[: min(4, len(hit_path))].copy_(hit_path[:4])
            self.pending.copy_(selected.gather(0, (count - 1).reshape(1)).reshape(1, 1))
            self.draft_hidden.copy_(hidden.index_select(1, parent.reshape(1))[:, 0])
            for d in range(len(selected)):
                dest = position + d + 1
                if dest < self.groups:
                    old = self.codes[:, dest : dest + 1]
                    old.copy_(torch.where(count > d, selected[d], old))
        with self.timer.phase("kv_commit"):
            # The known root and accepted candidate nodes have real KV. The new
            # tail usually does not; never commit it using a parent's state.
            for d in range(min(depth + 1, self.capacity - position - 1)):
                source = self.path_buffer[d : d + 1] + self.capacity
                dest = position + 1 + d
                if self.cache:
                    for layer in self.layers:
                        for storage in (layer.keys, layer.values):
                            old = storage[:, :, dest : dest + 1]
                            value = storage.index_select(2, source)
                            old.copy_(torch.where(accepted >= d, value, old))
                else:
                    value = self.query.index_select(1, self.path_buffer[d : d + 1])
                    old = self.prefix[:, dest : dest + 1]
                    old.copy_(torch.where(accepted >= d, value, old))
        self.last_hidden.copy_(hidden)

    def _stage_inputs(self, inputs):
        if inputs["layer0_code"].shape[0] != 1:
            raise ValueError("pipeline supports exactly one request")
        if inputs.get("top_p", 1.0) != 1.0:
            raise ValueError("pipeline requires top_p=1.0")
        for name, expected in (("do_sample", self.do_sample), ("temperature", self.temperature), ("top_k", self.top_k)):
            if inputs.get(name, expected) != expected:
                raise ValueError(f"executor sampling parameter changed: {name}")
        if not hasattr(self, "input_code"):
            self.input_code = self.codes.new_zeros(1, 1)
            self.input_embed = torch.zeros_like(inputs["layer0_embed"].reshape(1, 1, -1), dtype=self.dtype)
            self.input_hidden = torch.zeros_like(inputs["last_talker_hidden"].reshape(1, 1, -1), dtype=self.dtype)
        self.input_code.copy_(inputs["layer0_code"].reshape(1, 1))
        self.input_embed.copy_(inputs["layer0_embed"].reshape(1, 1, -1))
        self.input_hidden.copy_(inputs["last_talker_hidden"].reshape(1, 1, -1))
        if self.do_sample and self.temperature > 0:
            uniforms = inputs.get("sample_uniforms")
            if uniforms is None or uniforms.shape != self.uniforms.shape:
                raise ValueError("pipeline requires position-bound sample_uniforms")
            self.uniforms.copy_(uniforms)

    @torch.inference_mode()
    def prepare(self, inputs):
        self._stage_inputs(inputs)
        if self.prepared:
            return
        if self.execution == "graph":
            started = time.perf_counter()
            pool = torch.npu.graph_pool_handle()
            try:
                for position in (0, *self.layouts):
                    fn = self._prefill if position == 0 else lambda p=position: self._step(p)
                    for _ in range(2):
                        self._prefill()
                        fn()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, pool=pool):
                        fn()
                    self.graphs[position] = graph
                    if self.capture_progress is not None:
                        self.capture_progress(position, len(self.graphs), len(self.layouts) + 1)
                torch.npu.synchronize()
            except Exception as exc:
                self.capture_error = repr(exc)
                raise RuntimeError(f"pipeline NPU graph capture failed at position {position}") from exc
            finally:
                self.capture_seconds = time.perf_counter() - started
        self.prepared = True

    def snapshot(self):
        return dict(
            codes=self.codes.detach().cpu(),
            state=self.state.detach().cpu(),
            hit_path=self.path_buffer.detach().cpu(),
            draft_hidden=self.draft_hidden.detach().cpu(),
            pending=self.pending.detach().cpu(),
            keys=[layer.keys.detach().cpu() for layer in self.layers],
            values=[layer.values.detach().cpu() for layer in self.layers],
            graph=dict(count=len(self.graphs), capture_seconds=self.capture_seconds, error=self.capture_error),
            last_debug=self.last_debug,
            last_position=self.last_position,
            query=self.query.detach().cpu(),
            last_hidden=self.last_hidden.detach().cpu(),
            layout={
                key: value.detach().cpu()
                for key, value in self.layouts[self.last_position].items()
                if torch.is_tensor(value)
            }
            if self.last_position is not None
            else None,
        )

    def _audit(self, position, hidden):
        """Compare every live node to its exact serial logical prefix."""
        predictor = self.predictor
        prefix = spec.initial_prefix(predictor, self.input_embed, self.input_hidden)
        for p in range(1, position):
            embed = predictor.model.codec_embedding[p - 1](self.codes[:, p : p + 1])
            prefix = torch.cat((prefix, predictor.small_to_mtp_projection(embed)), dim=1)
        root = self.query[:, :1]
        errors = []
        active = self.layouts[position]["active"]
        for i, path in enumerate(((), *active)):
            branch = [root] + [
                self.query[:, active.index(path[:d]) + 1 : active.index(path[:d]) + 2] for d in range(1, len(path) + 1)
            ]
            next_position = position + len(path) + 1
            if next_position >= self.groups:
                continue
            serial = spec.reference_forward(predictor, torch.cat([prefix, *branch], dim=1))[:, -1]
            head = predictor.lm_head[next_position - 1]
            left, right = head(hidden[:, i]), head(serial)
            tolerance = 1e-5 if left.dtype == torch.float32 else 2e-2
            self.last_debug = dict(
                position=position, rank_path=list(path), logits=left.detach().cpu(), serial_logits=right.detach().cpu()
            )
            torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)
            errors.append((left - right).abs().max().detach())
        return torch.stack(errors).max().item() if errors else 0.0

    @torch.inference_mode()
    def decode(self, *, audit=False, record_events=False, profile=False, **inputs):
        self.prepare(inputs)
        self.timer = spec.PhaseTimer(self.device, profile)
        for layer in self.layers:
            layer.timer = self.timer if profile else None
        if profile and self.execution == "graph":
            raise ValueError("profile eager separately; graph timings must remain uninstrumented")
        with self.timer.phase("prefill"):
            self.graphs[0].replay() if self.execution == "graph" else self._prefill()
        position, calls, audit_calls = 1, 1, 0
        events, audit_max = [], 0.0
        while position < self.groups - 1:
            self.last_position = position
            self.graphs[position].replay() if self.execution == "graph" else self._step(position)
            calls += 1
            count, accepted = self.state.cpu().tolist()  # one round-boundary synchronization
            if audit:
                audit_max = max(audit_max, self._audit(position, self.last_hidden))
                audit_calls += sum(position + len(p) + 1 < self.groups for p in ((), *self.layouts[position]["active"]))
            if record_events:
                events.append(
                    dict(
                        position=position + 1,
                        committed=count,
                        accepted=accepted,
                        backbone_calls=1,
                        cached_prefix_length=position + 1,
                    )
                )
            position += count
        if record_events:
            codes = self.codes.cpu().tolist()[0]
            for event in events:
                start = event["position"]
                event["committed_codes"] = codes[start : start + event["committed"]]
                event["accepted_codes"] = event["committed_codes"][: event["accepted"]]
        return self.codes.clone(), dict(
            backbone_calls=calls,
            prefill_backbone_calls=1,
            verification_backbone_calls=calls - 1,
            supplemental_backbone_calls=0,
            audit_backbone_calls=audit_calls,
            graph_replays=calls if self.execution == "graph" else 0,
            bootstrap_committed=1,
            events=events,
            audit_max_logit_error=audit_max if audit else None,
            timings=self.timer.resolve(),
            attention_backend=spec.attention_backend(self.device),
            candidate_layout=CANDIDATE_LAYOUT,
        )
