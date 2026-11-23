# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stateful Code2Wav NPU graphs for prefixless or ICL requests."""

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
    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_batch_sizes: list[int] | tuple[int, ...] = (1, *range(2, 33, 2)),
        *,
        capture_modes: tuple[str, ...] = ("xvec",),
    ):
        sizes = sorted(set(capture_batch_sizes))
        if not sizes or sizes[0] != 1 or any(not isinstance(n, int) or n < 1 for n in sizes):
            raise ValueError("Code2Wav graph buckets must be positive integers including 1")
        super().__init__(
            decoder,
            capture_modes=capture_modes,
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
        total = len(self.capture_batch_sizes) * sum(
            1 + len(self._transitions_for_mode(mode)) for mode in self.capture_modes
        )
        count = 0
        model_dtype = next(self.decoder.parameters()).dtype
        # Positions in these captures are contiguous and non-packed. Restrict
        # this workaround to initialization; never patch live request execution.
        with capture_mask_context():
            for batch in self.capture_batch_sizes:
                for mode in self.capture_modes:
                    capture_prefix = self._capture_icl_prefix if mode == "icl" else self._capture_xvec_prefix
                    make_cache = self._make_dummy_icl_cache if mode == "icl" else self._make_dummy_xvec_cache
                    suffix_cache = None
                    for target in (None, *self._transitions_for_mode(mode)):
                        tick = time.perf_counter()
                        phase = "first" if target is None else "suffix"
                        logger.info(
                            "Code2Wav NPU capture %d/%d: mode=%s batch=%d phase=%s target=%s",
                            count + 1,
                            total,
                            mode,
                            batch,
                            phase,
                            target,
                        )
                        try:
                            if target is None:
                                capture_prefix(batch, device, dtype)
                            else:
                                if suffix_cache is None or mode == "xvec":
                                    # Suffix graphs only read reference state. Reuse its batched
                                    # staging tensors across phases on the ordered replay stream.
                                    suffix_cache = make_cache(batch, device, model_dtype)
                                self._capture_combined_suffix(mode, batch, target, suffix_cache, device, dtype)
                        except Exception as exc:
                            self.capture_seconds = time.perf_counter() - started
                            raise RuntimeError(
                                f"Code2Wav NPU capture failed: mode={mode}, batch={batch}, "
                                f"phase={phase}, target={target}"
                            ) from exc
                        count += 1
                        logger.info("Code2Wav NPU captured %d/%d in %.3fs", count, total, time.perf_counter() - tick)
        self.capture_seconds = time.perf_counter() - started
        self._warmed_up = True
        logger.info(
            "Code2Wav NPU ready: modes=%s buckets=%s graphs=%d capture_seconds=%.3f",
            self.capture_modes,
            self.capture_batch_sizes,
            count,
            self.capture_seconds,
        )

    def _decode_icl_prefix_batch(self, codes_list, request_caches):
        if "icl" in self.capture_modes:
            outputs = super()._decode_icl_prefix_batch(codes_list, request_caches)
            if outputs is not None:
                return outputs
        return self.decoder._decode_icl_prefix_eager_batch(
            codes_list,
            request_caches,
            prefix_length=self.prefix_length,
            initial_chunk_frames=self.initial_chunk_frames,
        )

    def _decode_xvec_prefix_batch(self, codes_list, request_caches):
        if "xvec" in self.capture_modes:
            return super()._decode_xvec_prefix_batch(codes_list, request_caches)
        return self.decoder._decode_xvec_prefix_eager_batch(
            codes_list, request_caches, initial_chunk_frames=self.initial_chunk_frames
        )

    def _decode_suffix_batch(self, mode, target_frames, codes_list, request_caches, new_frames_list):
        transitions = self._transitions_for_mode(mode)
        if mode not in self.capture_modes:
            return self.decoder._decode_suffix_eager_batch(
                mode,
                target_frames,
                codes_list,
                request_caches,
                new_frames_list,
                prefix_length=self.prefix_length,
                codec_chunk_frames=self.codec_chunk_frames,
                transitions_by_mode={mode: transitions},
            )
        expected = target_frames - transitions[target_frames]
        exact = [i for i, n in enumerate(new_frames_list) if n == expected]
        if len(exact) != len(codes_list):
            # Short rows must not force unrelated full rows in the same phase
            # to eager. Restore original request order after the two routes.
            outputs = [None] * len(codes_list)
            self._record_graph_fallback(f"{mode}_suffix_{target_frames}:short", len(codes_list) - len(exact))
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
        return self.decoder.batched_request_decode(
            codes_list,
            request_caches,
            prefix_length=self.prefix_length,
            initial_chunk_frames=self.initial_chunk_frames,
            codec_chunk_frames=self.codec_chunk_frames,
            transitions_by_mode={
                "icl": self._icl_previous_frames_by_target,
                "xvec": self._xvec_previous_frames_by_target,
            },
            decode_icl_prefix_batch=self._decode_icl_prefix_batch,
            decode_xvec_prefix_batch=self._decode_xvec_prefix_batch,
            decode_suffix_batch=self._decode_suffix_batch,
            decode_fallback=self._decode_request_fallback,
            backend_max_batch_size=max(self.capture_batch_sizes),
        )

    def _decode_request_fallback(self, codes, cache):
        if cache.get("_is_dummy_run"):
            return self.decoder._forward_exact(codes)
        return super()._decode_request_fallback(codes, cache)
