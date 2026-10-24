# SPDX-License-Identifier: Apache-2.0
"""Prefixless, request-state-aware Code2Wav NPU graphs.

The CUDA wrapper supplies phase grouping, static state and ping-pong ownership.
Only backend operations and exact-length routing are specialized here.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from vllm.logger import init_logger

from .code2wav_state_pool import PooledGraphMixin
from .segmented_graph_wrapper import CUDAGraphDecoderWrapper

logger = init_logger(__name__)


@contextmanager
def capture_mask_context():
    """Materialize masks without device-to-host skip probes during capture."""
    # The SDPA causal-mask shortcut calls fast_all(padding_mask), which reads
    # a device scalar. An explicit mask preserves padding and sliding-window
    # semantics while avoiding the unsupported capture-time synchronization.
    with (
        patch("transformers.masking_utils.find_packed_sequence_indices", return_value=None),
        patch("transformers.masking_utils._ignore_causal_mask_sdpa", return_value=False),
    ):
        yield


class NPUGraphDecoderWrapper(PooledGraphMixin, CUDAGraphDecoderWrapper):
    def __init__(self, decoder, capture_batch_sizes=(1, 2, 4), telemetry=None, optimization="current"):
        from .code2wav_optimization import VARIANTS, prepare_constants

        if optimization not in VARIANTS:
            raise ValueError(f"Unknown Code2Wav optimization: {optimization}")
        self.optimization = optimization
        self._stage_constants = {}
        self._execution_lock = threading.Lock()
        self._execution_stream = None
        if optimization != "current":
            prepare_constants(decoder)
        if optimization == "fused":
            from .code2wav_optimization import prepare_fused

            prepare_fused(decoder)
        super().__init__(
            decoder,
            capture_modes=("xvec",),
            num_quantizers=decoder.config.num_quantizers,
            capture_batch_sizes=list(capture_batch_sizes),
            initial_chunk_frames=getattr(decoder, "_initial_codec_chunk_frames", 1),
            codec_chunk_frames=getattr(decoder, "_incremental_chunk_frames", 25),
            codec_chunk_ramp=getattr(decoder, "_incremental_chunk_ramp", None),
        )
        if self.codec_chunk_ramp:
            raise ValueError("Code2Wav NPU graph v1 requires fixed chunks, without ramp")
        if self.capture_batch_sizes != [1, 2, 4]:
            raise ValueError("Code2Wav NPU graph v1 supports buckets [1, 2, 4]")
        self.telemetry = Path(telemetry) if telemetry else None
        self.routes = Counter()
        self.capture_seconds = 0.0
        self._pool = None
        self._stats_enabled = False  # use device-independent, structured counters below
        self._route_calls = 0

    def _capture_xvec_prefix(self, batch_size, device, dtype):
        if self.pooled:
            return self._capture_pool(batch_size, None, device, dtype)
        from .code2wav_optimization import fixed_stage

        with fixed_stage(self, batch_size, self.initial_chunk_frames):
            return super()._capture_xvec_prefix(batch_size, device, dtype)

    def _capture_combined_suffix(self, mode, batch_size, target_frames, caches, device, code_dtype):
        if self.pooled:
            return self._capture_pool(batch_size, target_frames, device, code_dtype)
        from .code2wav_optimization import fixed_stage

        with fixed_stage(self, batch_size, target_frames):
            return super()._capture_combined_suffix(mode, batch_size, target_frames, caches, device, code_dtype)

    def _copy_request_caches_to_batch(self, request_caches, static_caches):
        if self.optimization == "current":
            return super()._copy_request_caches_to_batch(request_caches, static_caches)
        # Prefixless CustomVoice has no reference tensors or retained prefix KV.
        if any(int(c.get("prefix_frames", 0)) for c in request_caches):
            raise ValueError("Optimized Code2Wav only supports prefixless requests")

    def _is_capturing(self):
        return torch.npu.is_current_stream_capturing()

    def _new_graph(self):
        return torch.npu.NPUGraph()

    def _graph_context(self, graph):
        return torch.npu.graph(graph, pool=self._pool)

    def _synchronize(self, device):
        torch.npu.synchronize(device)

    def warmup(self, device, dtype=torch.long):
        if self._warmed_up:
            return
        if device.type != "npu":
            raise ValueError("NPU graph capture requires an Ascend device")
        self.decoder.eval()
        self._device = device
        self._pool = torch.npu.graph_pool_handle()
        started = time.perf_counter()
        total = len(self.capture_batch_sizes) * (1 + len(self._xvec_previous_frames_by_target))
        count = 0
        # Positions in these captures are contiguous and non-packed. Restrict
        # this workaround to initialization; never patch live request execution.
        with capture_mask_context():
            for batch in self.capture_batch_sizes:
                phases = [("first", None), *[("suffix", n) for n in self._xvec_previous_frames_by_target]]
                for phase, target in phases:
                    tick = time.perf_counter()
                    logger.info(
                        "Code2Wav NPU capture %d/%d: batch=%d phase=%s target=%s",
                        count + 1,
                        total,
                        batch,
                        phase,
                        target,
                    )
                    try:
                        if phase == "first":
                            self._capture_xvec_prefix(batch, device, dtype)
                        else:
                            cache = self._make_dummy_xvec_cache(batch, device, next(self.decoder.parameters()).dtype)
                            self._capture_combined_suffix("xvec", batch, target, cache, device, dtype)
                    except Exception as exc:
                        self.capture_seconds = time.perf_counter() - started
                        self.flush_stats(error=dict(batch=batch, phase=phase, target=target, error=repr(exc)))
                        raise RuntimeError(
                            f"Code2Wav NPU capture failed: batch={batch}, phase={phase}, target={target}"
                        ) from exc
                    count += 1
                    logger.info("Code2Wav NPU captured %d/%d in %.3fs", count, total, time.perf_counter() - tick)
        self.capture_seconds = time.perf_counter() - started
        self._warmed_up = True
        self.flush_stats()

    def _record_graph_hit(self, phase, batch_size, request_count):
        self.routes[("graph", phase, batch_size, request_count)] += 1
        self._maybe_flush()

    def _record_graph_fallback(self, phase, request_count):
        # The fallback executor records each actual invocation, so a declined
        # group is not double-counted as both a group and individual requests.
        pass

    def _maybe_flush(self):
        self._route_calls += 1
        # The Stage-1 owner flushes periodically and on request completion;
        # standalone timing keeps counters in memory only.

    def statistics(self):
        from .code2wav_optimization import derived_bytes

        return dict(
            optimization=self.optimization,
            state_pool_bytes=self.state_pool.nbytes if getattr(self, "state_pool", None) else 0,
            attention_backend=self.decoder.pre_transformer.config._attn_implementation,
            derived_buffer_bytes=derived_bytes(self.decoder),
            stage_constant_bytes=sum(
                t.numel() * t.element_size()
                for pos, mask, rope in self._stage_constants.values()
                for t in (pos, mask, *rope)
                if t is not None
            ),
            backend="torch.npu.NPUGraph",
            graph_owner="code2wav",
            outer_graph=False,
            graph_count=len(self.xvec_prefix_states) + len(self.combined_states),
            capture_seconds=self.capture_seconds,
            buckets=self.capture_batch_sizes,
            transitions=self._xvec_previous_frames_by_target,
            routes=[
                dict(route=k[0], phase=k[1], bucket=k[2], requests=k[3], calls=v)
                for k, v in sorted(self.routes.items())
            ],
        )

    def flush_stats(self, error=None):
        data = self.statistics()
        if error:
            data["capture_failure"] = error
        if self.telemetry:
            self.telemetry.mkdir(parents=True, exist_ok=True)
            path = self.telemetry / f"code2wav-{os.getpid()}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        return data

    def log_decode_stats(self):
        logger.info("Code2Wav NPU graph statistics: %s", self.flush_stats())

    def _decode_xvec_prefix_batch(self, codes_list, request_caches):
        if self.pooled and all(x.shape[-1] == self.initial_chunk_frames for x in codes_list):
            batch = self._select_batch_bucket(len(codes_list), {1, 2, 4})
            return self._pool_replay(self.xvec_prefix_states[batch], codes_list, request_caches, None)
        return super()._decode_xvec_prefix_batch(codes_list, request_caches)

    def _decode_suffix_batch(self, mode, target_frames, codes_list, request_caches, new_frames_list):
        if self.pooled:
            return self._pool_suffix(target_frames, codes_list, request_caches, new_frames_list)
        expected = target_frames - self._transitions_for_mode(mode)[target_frames]
        if mode != "xvec":
            return None
        exact = [i for i, n in enumerate(new_frames_list) if n == expected]
        if len(exact) != len(codes_list):
            # Short rows must not force unrelated full rows in the same phase
            # to eager. Restore original request order after the two routes.
            outputs = [None] * len(codes_list)
            for i, n in enumerate(new_frames_list):
                if n != expected:
                    outputs[i] = self._decode_request_fallback(codes_list[i], request_caches[i])
            if exact:
                full = super()._decode_suffix_batch(
                    mode,
                    target_frames,
                    [codes_list[i] for i in exact],
                    [request_caches[i] for i in exact],
                    [new_frames_list[i] for i in exact],
                )
                if full is None:
                    full = [self._decode_request_fallback(codes_list[i], request_caches[i]) for i in exact]
                for i, output in zip(exact, full, strict=True):
                    outputs[i] = output
            return outputs
        with torch.profiler.record_function("code2wav.graph_suffix"):
            return super()._decode_suffix_batch(mode, target_frames, codes_list, request_caches, new_frames_list)

    def _decode_request_fallback(self, codes, cache):
        if cache.get("_is_dummy_run"):
            return self.decoder._forward_exact(codes)
        reason = (
            "first_shape"
            if "suffix_quantized" not in cache
            else ("short_chunk" if codes.shape[-1] != self.codec_chunk_frames else "unreachable_state")
        )
        self.routes[("eager", reason, 1, 1)] += 1
        self._maybe_flush()
        with torch.profiler.record_function("code2wav.eager_fallback"):
            if not self.pooled:
                return super()._decode_request_fallback(codes, cache)
            self.ensure_pool(codes.device)
            self.state_pool.acquire(cache)
            output = super()._decode_request_fallback(codes, cache)
            self.state_pool.store(cache, cache["suffix_quantized"], cache["suffix_conv"], 1 - cache["_c2w_bank"])
            return output

    def _batched_request_decode(self, codes_list, request_caches):
        if self.optimization == "current":
            return self._decode_requests_impl(codes_list, request_caches)
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent Code2Wav graph submission would overwrite static buffers")
        try:
            if codes_list and codes_list[0].device.type == "npu":
                stream = torch.npu.current_stream(codes_list[0].device).npu_stream
                if self._execution_stream is None:
                    self._execution_stream = stream
                elif self._execution_stream != stream:
                    raise RuntimeError("Code2Wav graph buffers require a single ordered execution stream")
            return self._decode_requests_impl(codes_list, request_caches)
        finally:
            self._execution_lock.release()

    def _decode_requests_impl(self, codes_list, request_caches):
        if self._is_capturing():
            raise RuntimeError("Code2Wav inner NPU graphs cannot run inside an outer capture")
        if any(int(cache.get("prefix_frames", 0)) != 0 for cache in request_caches):
            raise ValueError("Code2Wav NPU graph v1 only supports prefixless CustomVoice")
        try:
            return self.decoder.batched_request_decode(
                codes_list,
                request_caches,
                prefix_length=self.prefix_length,
                initial_chunk_frames=self.initial_chunk_frames,
                codec_chunk_frames=self.codec_chunk_frames,
                transitions_by_mode={"icl": {}, "xvec": self._xvec_previous_frames_by_target},
                decode_icl_prefix_batch=None,
                decode_xvec_prefix_batch=self._decode_xvec_prefix_batch,
                decode_suffix_batch=self._decode_suffix_batch,
                decode_fallback=self._decode_request_fallback,
                backend_max_batch_size=4,
            )
        except Exception:
            self.release_requests(request_caches)
            raise
