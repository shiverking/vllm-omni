# SPDX-License-Identifier: Apache-2.0
"""Opt-in prefixless Code2Wav experiments. Never changes the stock decoder."""

from __future__ import annotations

import torch

from collections import Counter, defaultdict
import threading
import time


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


class LayerKV:
    """Fixed stage-local scratch; only update() is used by decoder attention."""

    def __init__(self, old_k, old_v, query):
        self.old_k, self.old_v = old_k, old_v
        shape = list(old_k.shape)
        shape[-2] += query
        self.k, self.v = old_k.new_empty(shape), old_v.new_empty(shape)

    def update(self, k, v, layer_idx, cache_kwargs=None):
        n = self.old_k.shape[-2]
        self.k[layer_idx, ..., :n, :].copy_(self.old_k[layer_idx])
        self.v[layer_idx, ..., :n, :].copy_(self.old_v[layer_idx])
        self.k[layer_idx, ..., n:, :].copy_(k)
        self.v[layer_idx, ..., n:, :].copy_(v)
        return self.k[layer_idx], self.v[layer_idx]


class IncrementalStage:
    """One static shape. Tensor inputs/outputs are stable during graph replay."""

    def __init__(self, decoder, batch, previous, query):
        self.decoder, self.previous, self.query = decoder, previous, query
        cfg = decoder.config
        self.window = cfg.sliding_window
        p = next(decoder.parameters())
        valid = min(previous, self.window - 1)
        head = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        kvshape = (cfg.num_hidden_layers, batch, cfg.num_key_value_heads, valid, head)
        self.inputs = dict(
            codes=torch.zeros(batch, cfg.num_quantizers, query, device=p.device, dtype=torch.long),
            k=p.new_zeros(kvshape), v=p.new_zeros(kvshape),
            quantized=p.new_zeros(batch, cfg.codebook_dim, min(previous, self.window)),
            conv=p.new_zeros(batch, min(previous, self.window - 2), cfg.latent_dim),
            hidden=p.new_zeros(batch, min(previous, 12), cfg.latent_dim),
            positions=torch.arange(previous, previous + query, device=p.device)[None].repeat(batch, 1),
        )
        self.cache = LayerKV(self.inputs["k"], self.inputs["v"], query)
        self.mask = causal_window_mask(query, valid, self.window, p.device, p.dtype)
        self.graph = None

    def forward(self):
        d, x, n = self.decoder, self.inputs, self.query
        quantized = d.quantizer.decode(x["codes"])
        context = x["quantized"][..., -2:]
        conv = d.pre_conv(torch.cat((context, quantized), -1))[..., -n:].transpose(1, 2)
        hidden = d.pre_transformer(
            inputs_embeds=conv,
            attention_mask={"sliding_attention": self.mask},
            position_ids=x["positions"],
            cache_position=x["positions"][0],
            past_key_values=self.cache,
            use_cache=True,
        ).last_hidden_state
        context_hidden = torch.cat((x["hidden"], hidden), 1)
        self.outputs = dict(
            wav=waveform(d, context_hidden, n),
            k=self.cache.k[..., -(self.window - 1):, :],
            v=self.cache.v[..., -(self.window - 1):, :],
            quantized=torch.cat((x["quantized"], quantized), -1)[..., -self.window:],
            conv=torch.cat((x["conv"], conv), 1)[:, -(self.window - 2):],
            hidden=context_hidden[:, -12:],
        )
        return self.outputs


class IncrementalKVExecutor:
    """Request-local bounded state; growth mode reuses original rolling decode."""

    optimization = "incremental_kv"  # direct request-list service dispatch

    def __init__(self, decoder, mode="growth_kv", execution="eager", current=None, telemetry=None):
        if mode not in ("growth_kv", "stream_kv"):
            raise ValueError(f"Unsupported incremental mode: {mode}")
        if execution not in ("eager", "graph"):
            raise ValueError(execution)
        self.decoder, self.mode, self.execution = decoder, mode, execution
        self.current, self.telemetry = current, telemetry
        self.window = decoder.config.sliding_window
        self.stages, self.owners = {}, {}
        self.routes = Counter()
        self.capture_seconds = 0.0
        self._lock = threading.Lock()
        self._stream = None

    def stage(self, batch, previous, query):
        # Once all state lengths saturate, absolute position is a live input.
        key = (batch, min(previous, self.window), query)
        if key not in self.stages:
            self.stages[key] = IncrementalStage(self.decoder, *key)
        return self.stages[key]

    def warmup(self, device, dtype=torch.long):
        if self.execution == "graph":
            raise NotImplementedError("NPU graph capture is not installed yet")

    def release_requests(self, caches):
        for cache in caches:
            self.owners.pop(id(cache), None)
            cache.clear()
            cache["prefix_frames"] = 0

    def _legacy_state(self, cache, row, outputs, total):
        from transformers.cache_utils import DynamicCache

        if "past_key_values" not in cache:
            empty = DynamicCache(config=self.decoder.config)
            cfg = self.decoder.config
            empty.early_initialization(
                batch_size=1, num_heads=cfg.num_key_value_heads,
                head_dim=getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
                dtype=outputs["hidden"].dtype, device=outputs["hidden"].device,
            )
            cache.update(past_key_values=empty, decoder_prefix_frames=0)
        cache["suffix_quantized"] = outputs["quantized"][row:row+1].clone()
        cache["suffix_conv"] = outputs["conv"][row:row+1].clone()
        cache["ref_hidden"] = cache["suffix_quantized"][..., :0]
        cache["ref_conv"] = cache["suffix_conv"][:, :0]
        cache["prefix_hidden"] = outputs["hidden"][row:row+1, :0].clone()
        cache["suffix_frames"] = min(total, self.window)

    def _commit(self, cache, row, outputs):
        """Allocate request buffers once; only copy valid state on later calls."""
        if "_ikv_storage" not in cache:
            storage = {}
            for name, value in outputs.items():
                if name == "wav":
                    continue
                shape = list(value.shape)
                if name in ("k", "v"):
                    shape[1], shape[-2] = 1, self.window - 1
                else:
                    shape[0] = 1
                    shape[-1 if name == "quantized" else 1] = {
                        "quantized": self.window, "conv": self.window - 2, "hidden": 12
                    }[name]
                storage[name] = value.new_empty(shape)
            cache["_ikv_storage"] = storage
        state = {}
        for name, target in cache["_ikv_storage"].items():
            source = outputs[name]
            if name in ("k", "v"):
                source = source[:, row:row+1]
                view = target[..., :source.shape[-2], :]
            elif name == "quantized":
                source = source[row:row+1]
                view = target[..., :source.shape[-1]]
            else:
                source = source[row:row+1]
                view = target[:, :source.shape[1]]
            view.copy_(source)
            state[name] = view
            self.routes["state_commit_bytes"] += source.numel() * source.element_size()
        cache["_ikv"] = state

    @torch.no_grad()
    def _batched_request_decode(self, codes_list, caches):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent Code2Wav buffer reuse is forbidden")
        try:
            return self._decode(codes_list, caches)
        except Exception:
            self.release_requests(caches)
            raise
        finally:
            self._lock.release()

    def _decode(self, codes_list, caches):
        if len(codes_list) != len(caches) or len({id(c) for c in caches}) != len(caches):
            raise ValueError("Request/cache rows must be distinct and aligned")
        fresh = {id(c): c for c in caches if id(c) not in self.owners}
        if len(self.owners) + len(fresh) > 16:
            raise RuntimeError("Incremental Code2Wav supports at most 16 active requests")
        self.owners.update(fresh)
        groups = defaultdict(list)
        for i, (codes, cache) in enumerate(zip(codes_list, caches)):
            if int(cache.get("prefix_frames", 0)):
                raise ValueError("Incremental Code2Wav only supports prefixless CustomVoice")
            n = int(codes.shape[-1])
            if not 0 < n <= 25:
                raise ValueError("Expected 1..25 new codec frames")
            previous = cache.get("_ikv_total", 0)
            groups[(previous, n)].append(i)
        result = [None] * len(caches)
        for (previous, query), indices in groups.items():
            for start in range(0, len(indices), 4):
                rows = indices[start:start+4]
                if self.mode == "growth_kv" and previous >= self.window:
                    cs = [caches[i] for i in rows]
                    for c in cs:
                        c.pop("_ikv", None)
                        c.pop("_ikv_storage", None)
                    if self.current is None:
                        out = [self.decoder.chunked_decode(codes_list[i], caches[i]) for i in rows]
                    else:
                        out = self.current._batched_request_decode([codes_list[i] for i in rows], cs)
                    self.routes["rolling_recompute"] += 1
                    self.routes["transformer_positions"] += sum(min(previous, self.window) + query for _ in rows)
                    for i, wav in zip(rows, out):
                        result[i] = wav.clone()
                        caches[i]["_ikv_total"] = previous + query
                    continue
                batch = next(b for b in (1, 2, 4) if b >= len(rows))
                stage = self.stage(batch, previous, query)
                x = stage.inputs
                for t in x.values():
                    t.zero_()
                for row, i in enumerate(rows):
                    x["codes"][row].copy_(codes_list[i].reshape_as(x["codes"][row]))
                    x["positions"][row].copy_(torch.arange(previous, previous+query, device=x["positions"].device))
                    if previous:
                        state = caches[i]["_ikv"]
                        for name in ("k", "v"):
                            x[name][:, row:row+1].copy_(state[name])
                        for name in ("quantized", "conv", "hidden"):
                            x[name][row:row+1].copy_(state[name])
                outputs = stage.forward()
                self.routes["eager"] += 1
                self.routes["transformer_positions"] += query * len(rows)
                self.routes["padding_positions"] += query * (batch - len(rows))
                wave = outputs["wav"][:len(rows)].clone()
                for row, i in enumerate(rows):
                    cache = caches[i]
                    self._commit(cache, row, outputs)
                    cache["_ikv_total"] = previous + query
                    self._legacy_state(cache, row, outputs, previous + query)
                    result[i] = wave[row:row+1]
        return result

    def statistics(self):
        return dict(mode=self.mode, execution=self.execution, routes=dict(self.routes),
                    graphs=sum(s.graph is not None for s in self.stages.values()),
                    capture_seconds=self.capture_seconds, active_requests=len(self.owners),
                    attention_backend=self.decoder.config._attn_implementation)
