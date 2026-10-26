# SPDX-License-Identifier: Apache-2.0
"""Prefixless Code2Wav NPU graphs with independent request caches."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from unittest.mock import patch

import torch
from vllm.logger import init_logger

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


class NPUGraphDecoderWrapper(CUDAGraphDecoderWrapper):
    def __init__(self, decoder, capture_batch_sizes=(1, *range(2, 33, 2))):
        sizes = sorted(set(capture_batch_sizes))
        if not sizes or sizes[0] != 1 or any(not isinstance(n, int) or n < 1 for n in sizes):
            raise ValueError("Code2Wav graph buckets must be positive integers including 1")
        super().__init__(
            decoder,
            capture_modes=("xvec",),
            num_quantizers=decoder.config.num_quantizers,
            capture_batch_sizes=sizes,
            initial_chunk_frames=getattr(decoder, "_initial_codec_chunk_frames", 1),
            codec_chunk_frames=getattr(decoder, "_incremental_chunk_frames", 25),
            codec_chunk_ramp=getattr(decoder, "_incremental_chunk_ramp", None),
        )
        if self.codec_chunk_ramp:
            raise ValueError("Code2Wav NPU graphs require fixed chunks without ramp")
        self._pool = None
        self._execution_lock = threading.Lock()
        self._execution_stream = None

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
                        raise RuntimeError(
                            f"Code2Wav NPU capture failed: batch={batch}, phase={phase}, target={target}"
                        ) from exc
                    count += 1
                    logger.info("Code2Wav NPU captured %d/%d in %.3fs", count, total, time.perf_counter() - tick)
        self.capture_seconds = time.perf_counter() - started
        self._warmed_up = True
        logger.info(
            "Code2Wav NPU ready: buckets=%s graphs=%d capture_seconds=%.3f",
            self.capture_batch_sizes,
            count,
            self.capture_seconds,
        )

    def _decode_suffix_batch(self, mode, target_frames, codes_list, request_caches, new_frames_list):
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
        return super()._decode_suffix_batch(mode, target_frames, codes_list, request_caches, new_frames_list)

    def _batched_request_decode(self, codes_list, request_caches):
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
            backend_max_batch_size=max(self.capture_batch_sizes),
        )

    def _decode_request_fallback(self, codes, cache):
        if cache.get("_is_dummy_run"):
            return self.decoder._forward_exact(codes)
        return super()._decode_request_fallback(codes, cache)
