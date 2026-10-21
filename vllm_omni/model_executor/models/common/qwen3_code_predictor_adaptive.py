# SPDX-License-Identifier: Apache-2.0
"""Frozen-weight RVQ executor with shared state and position-specific queries.

Unlike PipelineExecutor, an action owns only temporary query/layout buffers.
Committed KV, the pending tail and its parent hidden are shared by all actions.
Requests may diverge in position; real homogeneous sub-batches are dispatched.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter
from dataclasses import replace

import torch

from . import qwen3_code_predictor_spec as spec
from .qwen3_code_predictor_pipeline import LayerKV, PipelineCaptureError, PipelineExecutor


class SharedFrameState:
    def __init__(self, predictor, batch, max_nodes):
        weight = next(predictor.model.parameters())
        self.batch, self.groups = batch, predictor._num_groups
        self.capacity = self.groups + 1
        h, v = predictor.config.hidden_size, predictor.config.vocab_size
        self.codes = torch.zeros(batch, self.groups, dtype=torch.long, device=weight.device)
        self.pending = self.codes.new_zeros(batch, 1)
        self.draft_hidden = weight.new_zeros(batch, h)
        self.uniforms = weight.new_zeros(batch, self.groups - 1, v, dtype=torch.float32)
        self.prefix = weight.new_zeros(batch, self.capacity, h)
        self.state = self.codes.new_zeros(batch, 2)
        self.path_buffer = self.codes.new_zeros(batch, 4)
        self.layers = []
        for layer in predictor.model.layers:
            a = layer.self_attn
            shape = (batch, a.num_kv_heads, self.capacity + max(2, max_nodes + 1), a.head_dim)
            self.layers.append(LayerKV(weight.new_zeros(shape), weight.new_zeros(shape), self.capacity))

    def tensors(self):
        names = ("codes", "pending", "draft_hidden", "uniforms", "prefix", "state", "path_buffer")
        return [getattr(self, n) for n in names] + [t for layer in self.layers for t in (layer.keys, layer.values)]


class ActionStep(PipelineExecutor):
    """One exact-position step. All request operations remain batched."""

    def __init__(self, predictor, paths, position, shared, *, do_sample, execution, query_capacity=None):
        # A candidate at the final codebook needs no hidden/KV: the parent's
        # target draw already completes the frame. Omit it from physical query.
        paths = [p for p in paths if position + len(p) < predictor._num_groups - 1]
        super().__init__(predictor, paths, do_sample=do_sample, execution="eager", optimization="optimized")
        if position and query_capacity is not None and query_capacity > self.query_size:
            self.query_size = query_capacity
            self.query = self.query.new_zeros(1, query_capacity, self.query.shape[-1])
            self.embedding_pack = self.embedding_pack.new_zeros(1, query_capacity, self.embedding_pack.shape[-1])
            self.layouts[position] = self._layout(position)
            indices = self.layouts[position]["query_map"]
            self.layouts[position]["query_map"] = torch.cat((indices, indices.new_zeros(query_capacity - len(indices))))
        self.execution, self.position, self.shared = execution, position, shared
        for name in ("codes", "pending", "draft_hidden", "uniforms", "prefix", "state", "path_buffer", "layers"):
            setattr(self, name, getattr(shared, name))
        batch = shared.batch
        self.query = self.query.repeat(batch, 1, 1)
        self.embedding_pack = self.embedding_pack.repeat(batch, 1, 1)
        self.last_hidden = torch.zeros_like(self.query)
        self.layouts = {position: self.layouts[position]} if position else {}
        if position:
            mask = self.layouts[position]["mask"]
            self.layouts[position]["mask"] = replace(mask, blocked=mask.blocked.repeat(batch, 1, 1, 1))
        self.cached_prefill_mask = replace(
            self.cached_prefill_mask, blocked=self.cached_prefill_mask.blocked.repeat(batch, 1, 1, 1)
        )
        self.graph = None
        self.prefill_positions = self.prefill_positions.expand(batch, -1)

    def _sample(self, logits, position):
        if not self.do_sample or self.temperature <= 0:
            return logits.argmax(-1)
        scaled = logits * (1.0 / max(self.temperature, 1e-6))
        if self.top_k > 0:
            values = scaled.topk(min(self.top_k, scaled.shape[-1]), dim=-1).values
            scaled = scaled.masked_fill(scaled < values[..., -1:], float("-inf"))
        noise = self.uniforms[:, position - 1]
        if logits.ndim == 3:
            noise = noise[:, None, :].expand_as(logits)
        return self.predictor._sample_codes_gumbel(scaled, uniforms=noise).reshape(logits.shape[:-1])

    def _prefill(self):
        self.codes[:, 1:].zero_()
        self.prefix.zero_()
        project = self.predictor.small_to_mtp_projection
        self.prefix[:, :1].copy_(project(self.input_hidden))
        self.prefix[:, 1:2].copy_(project(self.input_embed))
        self.codes[:, :1].copy_(self.input_code)
        for layer in self.layers:
            layer.keys.zero_()
            layer.values.zero_()
        hidden = self.predictor.model(self.prefix[:, :2], self.prefill_positions, self.cached_prefill_mask, self.layers)
        for layer in self.layers:
            for storage in (layer.keys, layer.values):
                storage[:, :, :2].copy_(storage[:, :, self.capacity : self.capacity + 2])
        self.draft_hidden.copy_(hidden[:, -1])
        self.pending.copy_(self._sample(self.predictor.lm_head[0](self.draft_hidden), 1).reshape(-1, 1))
        self.codes[:, 1:2].copy_(self.pending)

    def _step(self, position):
        m, meta = self.predictor, self.layouts[position]
        active, batch = meta["active"], self.shared.batch
        depth = max(map(len, active), default=0)
        candidates = [m.lm_head[position + d](self.draft_hidden).topk(4, dim=-1).indices for d in range(depth)]
        self.embedding_pack[:, :1].copy_(m.model.codec_embedding[position - 1](self.pending))
        for d, ranks, offset in meta["embedding_groups"]:
            values = m.model.codec_embedding[position + d - 1](candidates[d - 1].index_select(1, ranks))
            self.embedding_pack[:, offset : offset + len(ranks)].copy_(values)
        projected = m.small_to_mtp_projection(self.embedding_pack[:, : meta["embedding_count"]])
        self.query.copy_(projected.index_select(1, meta["query_map"]))
        hidden = m.model(self.query, meta["positions"].expand(batch, -1), meta["mask"], self.layers)
        logits, draws = [], []
        for target, nodes in meta["head_groups"]:
            values = m.lm_head[target - 1](hidden.index_select(1, nodes))
            logits.append(values)
            draws.append(self._sample(values, target))
        predictions = torch.cat(draws, dim=1).index_select(1, meta["prediction_restore"])
        tokens = (
            torch.cat([candidates[len(p) - 1][:, p[-1] : p[-1] + 1] for p in active], 1)
            if active
            else self.codes[:, :0]
        )
        current = self.codes.new_zeros(batch, 1)
        live = torch.ones(batch, 1, dtype=torch.bool, device=self.device)
        count, accepted, parent = current.clone(), current.clone(), current.clone()
        values, path = [], [current]
        # Static node IDs and commit offsets were allocated with the layout.
        ids = meta["node_ids"]
        for _ in range(min(depth + 1, self.groups - position - 1)):
            value = predictions.gather(1, current)
            values.append(value)
            parent = torch.where(live, current, parent)
            count = count + live.long()
            matches = (meta["parents"][None] == current) & (tokens == value) & live
            child = torch.where(matches, ids, 0).amax(1, keepdim=True) if active else current * 0
            hit = child > 0
            accepted = accepted + hit.long()
            path.append(child)
            live = live & hit
            current = child
        selected = torch.cat(values, 1)
        self.state[:, :1].copy_(count)
        self.state[:, 1:2].copy_(accepted)
        self.pending.copy_(selected.gather(1, count - 1))
        self.draft_hidden.copy_(hidden.gather(1, parent[:, :, None].expand(-1, -1, hidden.shape[-1]))[:, 0])
        self.path_buffer.zero_()
        # The final (bonus/mismatch) draw has no processed KV. Only root plus
        # up to three accepted candidate nodes can enter the committed path.
        processed = path[: self.path_buffer.shape[1]]
        self.path_buffer[:, : len(processed)].copy_(torch.cat(processed, 1))
        for d in range(selected.shape[1]):
            old = self.codes[:, position + d + 1 : position + d + 2]
            old.copy_(torch.where(count > d, selected[:, d : d + 1], old))
        offsets = meta["commit_offsets"]
        length = len(offsets)
        source = self.path_buffer[:, :length] + self.capacity
        valid = (accepted >= offsets[None])[:, None, :, None]
        for layer in self.layers:
            for storage in (layer.keys, layer.values):
                indices = source[:, None, :, None].expand(-1, storage.shape[1], -1, storage.shape[-1])
                old = storage[:, :, position + 1 : position + 1 + length]
                old.copy_(torch.where(valid, storage.gather(2, indices), old))
        self.last_hidden.copy_(hidden)
        self.debug_tensors[position] = dict(candidates=candidates, logits=logits, predictions=predictions)

    def run(self):
        if self.graph is not None:
            self.graph.replay()
        elif self.execution == "graph":
            raise RuntimeError("graph was not prepared; eager fallback is forbidden")
        elif self.position:
            self._step(self.position)
        else:
            self._prefill()

    def audit(self):
        p, m = self.position, self.predictor
        active = self.layouts[p]["active"]
        tolerance = 1e-5 if self.dtype == torch.float32 else 2e-2
        maximum, calls = 0.0, 0
        for row in range(self.shared.batch):
            prefix = spec.initial_prefix(m, self.input_embed[row : row + 1], self.input_hidden[row : row + 1])
            for k in range(1, p):
                prefix = torch.cat(
                    (
                        prefix,
                        m.small_to_mtp_projection(m.model.codec_embedding[k - 1](self.codes[row : row + 1, k : k + 1])),
                    ),
                    1,
                )
            for i, path in enumerate(((), *active)):
                target = p + len(path) + 1
                branch = [0] + [active.index(path[:d]) + 1 for d in range(1, len(path) + 1)]
                inputs = torch.cat([prefix, *[self.query[row : row + 1, n : n + 1] for n in branch]], 1)
                right = m.lm_head[target - 1](spec.reference_forward(m, inputs)[:, -1])
                group, slot = next(
                    (g, nodes.tolist().index(i))
                    for g, (_, nodes) in enumerate(self.layouts[p]["head_groups"])
                    if i in nodes.tolist()
                )
                left = self.debug_tensors[p]["logits"][group][row : row + 1, slot]
                torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)
                maximum = max(maximum, float((left - right).abs().max()))
                calls += 1
        return maximum, calls


class AdaptiveExecutor:
    """Real batched execution, no serial-per-row backbone implementation.

    At B=1 action switching aliases the same state directly. At larger B only
    regrouping copies request slots; changing a tree never rebuilds the prefix.
    """

    def __init__(self, predictor, actions, *, batch_size=1, execution="eager", do_sample=True, trim_tail=True):
        if batch_size not in (1, 2, 4, 8) or execution not in ("eager", "graph"):
            raise ValueError("unsupported batch/execution")
        self.predictor, self.batch_size, self.execution, self.do_sample = predictor, batch_size, execution, do_sample
        self.trim_tail = trim_tail
        self.actions = {a: spec.validate_tree(p) for a, p in actions.items()}
        if self.actions.get("serial") != []:
            raise ValueError("serial action must have no candidate nodes")
        maximum = max(map(len, self.actions.values()))
        self.bank = SharedFrameState(predictor, batch_size, maximum)
        self.work = {
            b: (self.bank if b == batch_size else SharedFrameState(predictor, b, maximum))
            for b in (1, 2, 4, 8)
            if b <= batch_size
        }
        self.steps, self.aliases, self.prefills = {}, {}, {}
        self.device = next(predictor.parameters()).device
        self.indices = {
            rows: torch.tensor(rows, device=self.device)
            for n in self.work
            for rows in itertools.combinations(range(batch_size), n)
        }
        self.prepared = False
        self.capture_seconds = 0.0
        self.last_context = {}

    @torch.inference_mode()
    def prepare(self, inputs):
        if self.prepared:
            return
        if self.execution == "graph" and self.device.type != "npu":
            raise ValueError("graph mode requires NPU")
        start = time.perf_counter()
        for batch, shared in self.work.items():
            prefill = ActionStep(self.predictor, [], 0, shared, do_sample=self.do_sample, execution=self.execution)
            prefill.input_code = self.bank.codes.new_zeros(batch, 1)
            dtype = shared.prefix.dtype
            prefill.input_embed = inputs["layer0_embed"].reshape(self.batch_size, 1, -1)[:batch].to(dtype).clone()
            prefill.input_hidden = (
                inputs["last_talker_hidden"].reshape(self.batch_size, 1, -1)[:batch].to(dtype).clone()
            )
            self.prefills[batch] = prefill
            for p in range(1, shared.groups - 1):
                for action, paths in self.actions.items():
                    clipped = tuple(path for path in paths if p + len(path) < shared.groups - 1)
                    capacity = 1 + len(clipped if self.trim_tail else paths)
                    key = (batch, p, clipped, capacity)
                    self.aliases[batch, p, action] = key
                    if key not in self.steps:
                        step = ActionStep(
                            self.predictor,
                            clipped,
                            p,
                            shared,
                            do_sample=self.do_sample,
                            execution=self.execution,
                            query_capacity=capacity,
                        )
                        step.input_embed, step.input_hidden = prefill.input_embed, prefill.input_hidden
                        step.layouts[p]["node_ids"] = torch.arange(1, len(clipped) + 1, device=self.device)[None]
                        self.steps[key] = step
            if self.execution == "graph":
                pool = torch.npu.graph_pool_handle()
                try:
                    for step in (prefill, *(s for (b, _, _, _), s in self.steps.items() if b == batch)):
                        self.last_context = dict(
                            capture=True, batch=batch, position=step.position, query_size=step.query_size
                        )
                        fn = step._prefill if not step.position else lambda s=step: s._step(s.position)
                        for _ in range(2):
                            prefill._prefill()
                            fn()
                        torch.npu.synchronize()
                        step.graph = torch.npu.NPUGraph()
                        with torch.npu.graph(step.graph, pool=pool):
                            fn()
                        if step.position in (0, 7, 14):
                            print(f"[adaptive capture] B={batch} p={step.position} Q={step.query_size}", flush=True)
                except Exception as exc:
                    raise PipelineCaptureError(
                        f"adaptive capture failed: batch={batch}, p={step.position}: {exc}"
                    ) from exc
        self.capture_seconds = time.perf_counter() - start
        self.prepared = True

    def _gather(self, rows):
        work = self.work[len(rows)]
        if work is not self.bank:
            for source, target in zip(self.bank.tensors(), work.tensors()):
                torch.index_select(source, 0, self.indices[rows], out=target)
        return work

    def _scatter(self, rows):
        work = self.work[len(rows)]
        if work is not self.bank:
            for target, source in zip(self.bank.tensors(), work.tensors()):
                target.index_copy_(0, self.indices[rows], source)

    @torch.inference_mode()
    def start(self, inputs):
        if inputs["layer0_code"].shape[0] != self.batch_size:
            raise ValueError("frame batch size changed")
        if inputs.get("top_p", 1.0) != 1.0 or inputs.get("top_k", 50) != 50 or inputs.get("temperature", 0.9) != 0.9:
            raise ValueError("adaptive experiment fixes top_p=1, top_k=50, temperature=.9")
        if inputs.get("do_sample", self.do_sample) != self.do_sample:
            raise ValueError("sampling mode changed")
        self.prepare(inputs)
        prefill = self.prefills[self.batch_size]
        prefill.input_code.copy_(inputs["layer0_code"].reshape(self.batch_size, 1))
        prefill.input_embed.copy_(inputs["layer0_embed"].reshape(self.batch_size, 1, -1))
        prefill.input_hidden.copy_(inputs["last_talker_hidden"].reshape(self.batch_size, 1, -1))
        if self.do_sample:
            noise = inputs.get("sample_uniforms")
            if noise is None or noise.shape != self.bank.uniforms.shape:
                raise ValueError("request/frame/position-bound uniforms required")
            self.bank.uniforms.copy_(noise)
        prefill.run()

    @torch.inference_mode()
    def advance(self, position, action, rows, *, audit=False):
        rows = tuple(rows)
        before = time.perf_counter()
        work = self._gather(rows)
        self.last_regroup_ms = (time.perf_counter() - before) * 1000
        step = self.steps[self.aliases[len(rows), position, action]]
        if audit and len(rows) != self.batch_size:
            root = self.prefills[self.batch_size]
            torch.index_select(root.input_embed, 0, self.indices[rows], out=step.input_embed)
            torch.index_select(root.input_hidden, 0, self.indices[rows], out=step.input_hidden)
        self.last_context = dict(position=position, action=action, rows=rows, query_size=step.query_size)
        step.run()
        states = work.state.cpu().tolist()  # one transfer per homogeneous round
        error, audit_calls = step.audit() if audit else (0.0, 0)
        before = time.perf_counter()
        self._scatter(rows)
        self.last_regroup_ms += (time.perf_counter() - before) * 1000
        return states, error, audit_calls

    @torch.inference_mode()
    def decode(self, *, choose=None, audit=False, record_events=False, **inputs):
        choose = choose or (lambda p, b: "serial")
        self.start(inputs)
        positions = [1] * self.batch_size
        events, calls, audit_calls, max_error = [], 1, 0, 0.0
        lookup_ms, regroup_ms = 0.0, 0.0
        batches, deltas, accepted_counts = Counter(), Counter(), Counter()
        while min(positions) < self.bank.groups - 1:
            p = min(positions)
            ready = [i for i, position in enumerate(positions) if position == p]
            while ready:
                batch = max(b for b in self.work if b <= len(ready))
                rows, ready = ready[:batch], ready[batch:]
                before = time.perf_counter()
                action = choose(p, batch)
                lookup_ms += (time.perf_counter() - before) * 1000
                states, error, extra = self.advance(p, action, rows, audit=audit)
                regroup_ms += self.last_regroup_ms
                batches[batch] += 1
                calls += 1
                audit_calls += extra
                max_error = max(max_error, error)
                for row, (count, accepted) in zip(rows, states):
                    deltas[count] += 1
                    accepted_counts[accepted] += 1
                    positions[row] += count
                    if record_events:
                        events.append(
                            dict(
                                row=row,
                                position=p,
                                batch=batch,
                                action=action,
                                delta=count,
                                accepted=accepted,
                                query_size=self.last_context["query_size"],
                            )
                        )
        return self.bank.codes.clone(), dict(
            backbone_calls=calls,
            prefill_calls=1,
            audit_calls=audit_calls,
            audit_max_logit_error=max_error,
            events=events,
            capture_seconds=self.capture_seconds,
            graphs=sum(s.graph is not None for s in (*self.steps.values(), *self.prefills.values())),
            lookup_ms=lookup_ms,
            regroup_host_ms=regroup_ms,
            subbatch_counts=dict(batches),
            delta_counts=dict(deltas),
            accepted_counts=dict(accepted_counts),
        )

    def snapshot(self):
        result = dict(
            context=self.last_context,
            codes=self.bank.codes.cpu(),
            pending=self.bank.pending.cpu(),
            draft_hidden=self.bank.draft_hidden.cpu(),
            uniforms=self.bank.uniforms.cpu(),
            kv=[(layer.keys.cpu(), layer.values.cpu()) for layer in self.bank.layers],
        )
        if "action" in self.last_context:
            context = self.last_context
            step = self.steps[self.aliases[len(context["rows"]), context["position"], context["action"]]]
            result["step"] = dict(
                query=step.query.cpu(),
                positions=step.layouts[step.position]["positions"].cpu(),
                logits=[v.cpu() for v in step.debug_tensors[step.position]["logits"]],
                candidates=[v.cpu() for v in step.debug_tensors[step.position]["candidates"]],
                tensors=[t.cpu() for t in step.shared.tensors()],
            )
        return result

    def close(self):
        if self.execution == "graph" and self.prepared:
            torch.npu.synchronize()
            for step in (*self.prefills.values(), *self.steps.values()):
                if step.graph is not None:
                    step.graph.reset()
                    step.graph = None
        self.prepared = False
