# SPDX-License-Identifier: Apache-2.0
"""Opt-in frame-local cached AR. Shared weights; no speculative decoding."""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

KV_IMPLEMENTATION = "fia_gqa_short_projection_v1"
FRAME_OPTIMIZATION = "batched_gumbel_projected_embeddings_v1"


class CachedARExecutor:
    """One owner stream per executor; independent buffers per batch bucket.

    ``eager`` is a diagnostic path, never a fallback for failed graph capture.
    Hooks see 15 actual backbone forwards during warmup/capture, not replay.
    """

    def __init__(self, wrapper, mode="kv_frame", buckets=(1, 2, 4, 8, 16)):
        if mode not in ("eager", "kv_step", "kv_frame"):
            raise ValueError(mode)
        self.wrapper, self.mode = wrapper, mode
        self.buckets = tuple(buckets)
        self.device = next(wrapper.parameters()).device
        self.dtype = next(wrapper.parameters()).dtype
        if mode != "eager" and self.device.type != "npu":
            raise ValueError("NPU graph mode requires an NPU; use eager explicitly for CPU diagnostics")
        self.groups = wrapper._num_groups
        self.vocab = int(wrapper.config.vocab_size)
        self.buffers = {}
        self.graphs = {}
        self.graph_logits = {}
        self.capture_seconds = 0.0
        self.batch_counts = Counter()
        self._lock = threading.Lock()
        self._stream = None
        self.last_stats = {}
        self._projected_embeddings = None

    def _allocate(self, batch):
        if batch in self.buffers:
            return self.buffers[batch]
        w = self.wrapper
        hidden = int(w.config.hidden_size)
        embed_dim = w.model.codec_embedding[0].weight.shape[1]
        opts = dict(device=self.device, dtype=self.dtype)
        shape = (
            len(w.model.layers),
            batch,
            int(w.config.num_key_value_heads),
            self.groups,
            w.model.layers[0].self_attn.head_dim,
        )
        b = dict(
            hidden=torch.zeros(batch, 1, embed_dim, **opts),
            embed=torch.zeros(batch, 1, embed_dim, **opts),
            codes=torch.zeros(batch, self.groups, device=self.device, dtype=torch.long),
            inputs=torch.zeros(batch, self.groups, hidden, **opts),
            cache=(torch.zeros(shape, **opts), torch.zeros(shape, **opts)),
            uniforms=torch.full((batch, self.groups - 1, self.vocab), 0.5, device=self.device, dtype=torch.float32),
            positions=[
                torch.arange(2 if step == 0 else 1, device=self.device)
                .add_(0 if step == 0 else step + 1)
                .unsqueeze(0)
                .expand(batch, -1)
                .contiguous()
                for step in range(self.groups - 1)
            ],
        )
        if self.mode == "kv_frame":
            b["log_noise"] = torch.empty_like(b["uniforms"])
        b["last_logits"] = [None] * (self.groups - 1)
        self.buffers[batch] = b
        return b

    @torch.inference_mode()
    def _prepare_frame_tables(self):
        """Build once after weight loading, outside capture; weights stay frozen.

        Nonpersistent model buffer: shared across executors, buckets and sampling
        graphs, never serialized in checkpoints. Recreate executors/model after
        changing weights, as required for their captured graphs as well.
        """
        if self.mode != "kv_frame" or self._projected_embeddings is not None:
            return
        w = self.wrapper
        name = "_kv_frame_projected_embeddings"
        table = getattr(w, name, None)
        if table is None:
            started = time.perf_counter()
            table = torch.empty(
                self.groups - 2, self.vocab, int(w.config.hidden_size), device=self.device, dtype=self.dtype
            )
            # c15 is output only. Keep all original embeddings for Talker use.
            # Bound build temporaries; do not stack another copy of the table.
            for step in range(self.groups - 2):
                weight = w.model.codec_embedding[step].weight
                for start in range(0, self.vocab, 256):
                    table[step, start : start + 256].copy_(w.small_to_mtp_projection(weight[start : start + 256]))
            if self.device.type == "npu":
                torch.npu.synchronize(self.device)
            w.register_buffer(name, table, persistent=False)
            logger.info(
                "kv_frame projected embeddings ready: tables=%d bytes=%d seconds=%.3f; shared across buckets",
                self.groups - 2,
                table.numel() * table.element_size(),
                time.perf_counter() - started,
            )
        self._projected_embeddings = table

    def optimization_metadata(self):
        table = self._projected_embeddings
        return dict(
            optimization=FRAME_OPTIMIZATION if self.mode == "kv_frame" else "none",
            projected_embedding_bytes=0 if table is None else table.numel() * table.element_size(),
            projected_embedding_tables=0 if table is None else self.groups - 2,
            gumbel_buffer_bytes=sum(
                b["log_noise"].numel() * b["log_noise"].element_size()
                for b in self.buffers.values()
                if "log_noise" in b
            ),
            projected_embedding_scope="shared model buffer, frozen weights; excludes original embeddings",
        )

    def _project(self, b):
        project = self.wrapper.small_to_mtp_projection
        b["inputs"][:, 0:1].copy_(project(b["hidden"]))
        b["inputs"][:, 1:2].copy_(project(b["embed"]))

    def _backbone(self, b, step):
        start, length = (0, 2) if step == 0 else (step + 1, 1)
        return self.wrapper.model(b["inputs"][:, start : start + length], b["positions"][step], b["cache"], start)

    def _head(self, b, step, hidden, sampling, audit=None):
        w = self.wrapper
        logits = w.lm_head[step](hidden[:, -1])
        b["last_logits"][step] = logits
        if audit is not None:
            audit.append(logits.detach().clone())
        if sampling:
            scaled = logits * (1.0 / 0.9)
            if self.vocab >= 50:
                vals = scaled.topk(50, dim=-1).values
                scaled = scaled.masked_fill(scaled < vals[:, -1:], float("-inf"))
            if self.mode == "kv_frame":
                # Same subtraction and FP32 arithmetic as _sample_codes_gumbel.
                code = (scaled.float() - b["log_noise"][:, step]).argmax(dim=-1, keepdim=True)
            else:
                code = w._sample_codes_gumbel(scaled, uniforms=b["uniforms"][:, step])
        else:
            code = logits.argmax(dim=-1, keepdim=True)
        b["codes"][:, step + 1].copy_(code.reshape(-1))
        if step + 1 < self.groups - 1:
            if self.mode == "kv_frame":
                projected = torch.nn.functional.embedding(code.reshape(-1), self._projected_embeddings[step]).unsqueeze(
                    1
                )
            else:
                embedded = w.model.codec_embedding[step](code.reshape(-1)).unsqueeze(1)
                projected = w.small_to_mtp_projection(embedded)
            b["inputs"][:, step + 2 : step + 3].copy_(projected)

    def _frame(self, b, sampling, audit=None):
        if self.mode == "kv_frame" and sampling:
            # One transform for [batch, codebook, vocabulary]. Keep generation
            # of uniforms unchanged: step then real row, excluding padding.
            torch.log(b["uniforms"], out=b["log_noise"])
            b["log_noise"].neg_().log_()
        self._project(b)
        for step in range(self.groups - 1):
            self._head(b, step, self._backbone(b, step), sampling, audit)

    def graph_key(self, batch, sampling):
        # Backbone computation does not depend on how its logits are sampled.
        return (batch, bool(sampling) if self.mode == "kv_frame" else None)

    @property
    def graph_count(self):
        return len(self.graphs) * (1 if self.mode == "kv_frame" else self.groups - 1)

    def prepare_all(self):
        samples = (False, True) if self.mode == "kv_frame" else (False,)
        total = len(self.buckets) * len(samples)
        for index, (bucket, sampling) in enumerate(((b, s) for b in self.buckets for s in samples), 1):
            logger.info(
                "Cached AR capture %d/%d: mode=%s bucket=%d sampling=%s",
                index,
                total,
                self.mode,
                bucket,
                sampling if self.mode == "kv_frame" else "shared",
            )
            self.prepare(bucket, sampling)
        logger.info(
            "Cached AR ready: implementation=%s mode=%s buckets=%s graphs=%d capture_seconds=%.3f",
            KV_IMPLEMENTATION,
            self.mode,
            self.buckets,
            self.graph_count,
            self.capture_seconds,
        )

    @torch.inference_mode()
    def prepare(self, batch, sampling):
        """Capture outside benchmark timing; graph errors propagate."""
        b = self._allocate(batch)
        key = self.graph_key(batch, sampling)
        if self.mode == "eager" or key in self.graphs:
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Cached AR owns its graphs; outer MTP capture must be disabled")
        self._prepare_frame_tables()
        started = time.perf_counter()
        for _ in range(3):
            self._frame(b, sampling)
        torch.npu.synchronize(self.device)
        pool = torch.npu.graph_pool_handle()
        if self.mode == "kv_frame":
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=pool):
                self._frame(b, sampling)
            entry = graph
        else:
            entry = []
            for step in range(self.groups - 1):
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, pool=pool):
                    output = self._backbone(b, step)
                entry.append((graph, output))
        torch.npu.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.capture_seconds += elapsed
        self.graphs[key] = entry
        logger.info(
            "Cached AR captured: mode=%s bucket=%d sampling=%s graphs=%d seconds=%.3f total_graphs=%d",
            self.mode,
            batch,
            sampling if self.mode == "kv_frame" else "shared",
            1 if self.mode == "kv_frame" else self.groups - 1,
            elapsed,
            self.graph_count,
        )
        if self.mode == "kv_frame":
            self.graph_logits[key] = tuple(b["last_logits"])

    def _fill_uniforms(self, b, batch, uniforms, generator, generators):
        if uniforms is not None:
            if tuple(uniforms.shape) != (batch, self.groups - 1, self.vocab):
                raise ValueError("Invalid sample_uniforms shape")
            b["uniforms"][:batch].copy_(uniforms)
            return
        if generators is not None and len(generators) != batch:
            raise ValueError("generators must have one entry per real row")
        # Match stock RNG consumption order: codebook first, then real rows.
        for step in range(self.groups - 1):
            target = b["uniforms"][:batch, step]
            if generators is None:
                target.copy_(
                    torch.rand((batch, self.vocab), device=self.device, dtype=torch.float32, generator=generator)
                )
            else:
                for row, gen in enumerate(generators):
                    target[row].copy_(torch.rand((self.vocab,), device=self.device, dtype=torch.float32, generator=gen))

    @torch.inference_mode()
    def __call__(
        self,
        layer0_code,
        layer0_embed,
        last_talker_hidden,
        *,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        generator=None,
        generators=None,
        sample_uniforms=None,
        audit=None,
    ):
        sampling = do_sample and temperature > 0
        if top_p != 1.0 or (sampling and (temperature != 0.9 or top_k != 50)):
            raise ValueError("Cached AR experiment supports greedy or temperature=.9/top_k=50/top_p=1 only")
        batch = int(layer0_code.shape[0])
        bucket = next((n for n in self.buckets if n >= batch), None)
        if batch < 1 or bucket is None:
            raise ValueError(f"Unsupported batch {batch}; buckets={self.buckets}")
        if audit is not None and self.mode != "eager":
            raise ValueError("Per-position audit requires explicit eager mode")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent calls cannot share cached AR buffers")
        try:
            if self.device.type == "npu":
                stream = torch.npu.current_stream(self.device)
                if self._stream is not None and stream != self._stream:
                    raise RuntimeError("Cached AR executor cannot change execution stream")
                self._stream = stream
            self.prepare(bucket, sampling)
            b = self.buffers[bucket]
            b["hidden"].zero_()
            b["embed"].zero_()
            b["codes"][:, 0].zero_()
            b["hidden"][:batch].copy_(last_talker_hidden.reshape(batch, 1, -1))
            b["embed"][:batch].copy_(layer0_embed.reshape(batch, 1, -1))
            b["codes"][:batch, 0].copy_(layer0_code.reshape(batch))
            if sampling:
                self._fill_uniforms(b, batch, sample_uniforms, generator, generators)
            if self.mode == "kv_frame":
                self.graphs[self.graph_key(bucket, sampling)].replay()
            elif self.mode == "kv_step":
                self._project(b)
                for step, (graph, output) in enumerate(self.graphs[self.graph_key(bucket, sampling)]):
                    graph.replay()
                    self._head(b, step, output, sampling)
            else:
                self._frame(b, sampling, audit)
            self.batch_counts[(batch, bucket)] += 1
            self.last_stats = dict(
                backbone_calls=self.groups - 1,
                graph_replays=1 if self.mode == "kv_frame" else self.groups - 1 if self.mode == "kv_step" else 0,
                batch=batch,
                bucket=bucket,
                mode=self.mode,
            )
            return b["codes"][:batch].clone()
        finally:
            self._lock.release()

    def last_logits(self, batch, sampling):
        bucket = next(n for n in self.buckets if n >= batch)
        if self.mode == "kv_frame":
            return self.graph_logits[self.graph_key(bucket, sampling)]
        return self.buffers[bucket]["last_logits"]

    def snapshot(self):
        return {
            str(batch): {
                "codes": b["codes"].cpu(),
                "inputs": b["inputs"].cpu(),
                "key_cache": b["cache"][0].cpu(),
                "value_cache": b["cache"][1].cpu(),
                "uniforms": b["uniforms"].cpu(),
            }
            for batch, b in self.buffers.items()
        }


class BatchTelemetry:
    """Host counters only; background flush, no device sync or hot-path I/O."""

    def __init__(self, directory, execution):
        self.path = Path(directory) / f"batches-{os.getpid()}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.execution = execution
        self.counts = Counter()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def record(self, real, padded, outer_graph):
        with self.lock:
            self.counts[(real, padded, bool(outer_graph))] += 1

    def flush(self):
        with self.lock:
            rows = [
                dict(real_batch=k[0], execution_rows=k[1], outer_mtp_graph=k[2], calls=v)
                for k, v in self.counts.items()
            ]
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dict(execution=self.execution, timestamp=time.time(), rows=rows)), encoding="utf-8"
        )
        temporary.replace(self.path)

    def _run(self):
        while not self.stop.wait(1):
            self.flush()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.flush()
