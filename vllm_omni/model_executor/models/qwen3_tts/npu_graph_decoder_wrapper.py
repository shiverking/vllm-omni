# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""NPU Graph wrapper for the Qwen3-TTS speech tokenizer decoder."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import torch
from vllm.logger import init_logger

from .cuda_graph_decoder_wrapper import _batched_chunked_decode

logger = init_logger(__name__)


class NPUGraphDecoderWrapper:
    """Capture fixed Code2Wav decoder shapes and replay them on Ascend NPU."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        extra_capture_shapes: list[tuple[int, int]] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
        stats_log_every: int = 100,
        padding_enabled: bool = False,
        max_pad_frames: int = 0,
    ) -> None:
        self.decoder = decoder
        self._explicit_sizes = capture_sizes is not None
        self.capture_sizes = sorted({int(size) for size in capture_sizes or [] if int(size) > 0})
        self.extra_capture_shapes = sorted(
            {
                (int(batch_size), int(size))
                for batch_size, size in extra_capture_shapes or []
                if int(batch_size) > 0 and int(size) > 0
            }
        )
        self.num_quantizers = int(num_quantizers)
        self.enabled = bool(enabled)
        self.stats_log_every = max(0, int(stats_log_every))
        self.padding_enabled = bool(padding_enabled)
        self.max_pad_frames = int(max_pad_frames)
        if self.max_pad_frames < 0:
            raise ValueError(f"max_pad_frames must be non-negative, got {max_pad_frames}")

        self.graphs: dict[tuple[int, int], object] = {}
        self.static_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self._frame_buckets_by_batch: dict[int, list[int]] = {}
        self._printed_active_graph_keys: set[tuple[int, int]] = set()
        self._printed_padded_routes: set[tuple[int, int, int]] = set()
        self._printed_fallback_shapes: set[tuple[int, int, str]] = set()
        self._stats_total = 0
        self._stats_exact_hits = 0
        self._stats_padded_hits = 0
        self._stats_fallbacks = 0
        self._warmed_up = False
        self._device: torch.device | None = None

    @staticmethod
    def compute_capture_sizes(
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ) -> list[int]:
        sizes: set[int] = set()
        if codec_chunk_frames > 0:
            sizes.add(int(codec_chunk_frames))
            if codec_left_context_frames > 0:
                sizes.add(int(codec_chunk_frames + codec_left_context_frames))
        sizes.add(int(decode_chunk_size + decode_left_context))
        return sorted(size for size in sizes if size > 0)

    def _get_capture_shapes(self) -> list[tuple[int, int]]:
        shapes = {(1, size) for size in self.capture_sizes}
        shapes.update(self.extra_capture_shapes)
        return sorted(shapes)

    def _capture_one(
        self,
        *,
        batch_size: int,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
        pool: object,
    ) -> tuple[object, torch.Tensor, torch.Tensor]:
        static_input = torch.zeros(
            batch_size,
            self.num_quantizers,
            size,
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            _ = self.decoder(static_input)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.no_grad():
            with torch.npu.graph(graph, pool=pool):
                static_output = self.decoder(static_input)
        return graph, static_input, static_output

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.long,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ) -> None:
        if not self.enabled or self._warmed_up:
            return

        self._device = device
        self.decoder.eval()
        if not self._explicit_sizes:
            self.capture_sizes = self.compute_capture_sizes(
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
                decode_chunk_size=decode_chunk_size,
                decode_left_context=decode_left_context,
            )

        capture_shapes = self._get_capture_shapes()
        print(
            "[Qwen3-TTS][NPU Code2Wav graph] enabled "
            f"capture_shapes={capture_shapes} "
            f"padding_enabled={self.padding_enabled} "
            f"max_pad_frames={'unlimited' if self.max_pad_frames == 0 else self.max_pad_frames} "
            f"non_packed_position_ids=True "
            f"stats_log_every={self.stats_log_every}",
            flush=True,
        )

        pool = torch.npu.graph_pool_handle()
        graph_keys: list[tuple[int, int]] = []
        failed_keys: list[tuple[int, int]] = []
        # Code2Wav position ids are always one monotonically increasing,
        # non-packed sequence. Transformers otherwise probes this with an NPU
        # ``.all()`` consumed by Python, which synchronizes the captured stream.
        with patch(
            "transformers.masking_utils.find_packed_sequence_indices",
            return_value=None,
        ):
            for batch_size, size in capture_shapes:
                key = (batch_size, size)
                try:
                    graph, static_input, static_output = self._capture_one(
                        batch_size=batch_size,
                        size=size,
                        device=device,
                        dtype=dtype,
                        pool=pool,
                    )
                    self.graphs[key] = graph
                    self.static_inputs[key] = static_input
                    self.static_outputs[key] = static_output
                    graph_keys.append(key)
                except Exception:
                    failed_keys.append(key)
                    logger.warning(
                        "Failed to capture Qwen3-TTS Code2Wav NPU graph for batch=%d frames=%d",
                        batch_size,
                        size,
                        exc_info=True,
                    )

        buckets_by_batch: dict[int, list[int]] = {}
        for batch_size, size in self.graphs:
            buckets_by_batch.setdefault(batch_size, []).append(size)
        self._frame_buckets_by_batch = {
            batch_size: sorted(sizes) for batch_size, sizes in buckets_by_batch.items()
        }
        self._warmed_up = True
        print(
            "[Qwen3-TTS][NPU Code2Wav graph] capture complete "
            f"graph_keys={sorted(graph_keys)} "
            f"failed_keys={sorted(failed_keys)}",
            flush=True,
        )

    def _get_graph_key(self, batch_size: int, actual_size: int) -> tuple[int, int] | None:
        key = (batch_size, actual_size)
        if key in self.graphs:
            return key
        if not self.padding_enabled:
            return None
        for graph_size in self._frame_buckets_by_batch.get(batch_size, []):
            padding_frames = graph_size - actual_size
            if padding_frames < 0:
                continue
            if self.max_pad_frames > 0 and padding_frames > self.max_pad_frames:
                return None
            return (batch_size, graph_size)
        return None

    def _fallback_reason(self, batch_size: int, actual_size: int) -> str:
        if not self.padding_enabled:
            return "padding_disabled"
        upper_buckets = [
            size for size in self._frame_buckets_by_batch.get(batch_size, []) if size >= actual_size
        ]
        if not upper_buckets:
            return "no_graph_bucket"
        if self.max_pad_frames > 0 and upper_buckets[0] - actual_size > self.max_pad_frames:
            return "padding_limit"
        return "no_graph_bucket"

    def _print_stats(self) -> None:
        exact_hit_rate = 100.0 * self._stats_exact_hits / self._stats_total if self._stats_total else 0.0
        graph_hits = self._stats_exact_hits + self._stats_padded_hits
        graph_hit_rate = 100.0 * graph_hits / self._stats_total if self._stats_total else 0.0
        print(
            "[Qwen3-TTS][NPU Code2Wav graph] stats "
            f"total={self._stats_total} "
            f"exact_hits={self._stats_exact_hits} "
            f"padded_hits={self._stats_padded_hits} "
            f"graph_hits={graph_hits} "
            f"fallbacks={self._stats_fallbacks} "
            f"exact_hit_rate={exact_hit_rate:.2f}% "
            f"graph_hit_rate={graph_hit_rate:.2f}%",
            flush=True,
        )

    def _record_route(
        self,
        *,
        request_key: tuple[int, int],
        graph_key: tuple[int, int] | None,
        fallback_reason: str | None = None,
    ) -> None:
        self._stats_total += 1
        if graph_key == request_key:
            self._stats_exact_hits += 1
            if graph_key not in self._printed_active_graph_keys:
                self._printed_active_graph_keys.add(graph_key)
                print(
                    "[Qwen3-TTS][NPU Code2Wav graph] exact hit "
                    f"batch_size={graph_key[0]} frames={graph_key[1]}",
                    flush=True,
                )
        elif graph_key is not None:
            self._stats_padded_hits += 1
            route = (request_key[0], request_key[1], graph_key[1])
            if route not in self._printed_padded_routes:
                self._printed_padded_routes.add(route)
                print(
                    "[Qwen3-TTS][NPU Code2Wav graph] padded hit "
                    f"batch_size={request_key[0]} "
                    f"actual_frames={request_key[1]} "
                    f"graph_frames={graph_key[1]} "
                    f"padding_frames={graph_key[1] - request_key[1]}",
                    flush=True,
                )
        else:
            self._stats_fallbacks += 1
            reason = fallback_reason or "no_graph_bucket"
            fallback_key = (request_key[0], request_key[1], reason)
            if fallback_key not in self._printed_fallback_shapes:
                self._printed_fallback_shapes.add(fallback_key)
                print(
                    "[Qwen3-TTS][NPU Code2Wav graph] eager fallback "
                    f"batch_size={request_key[0]} frames={request_key[1]} "
                    f"reason={reason}",
                    flush=True,
                )
        if self.stats_log_every > 0 and self._stats_total % self.stats_log_every == 0:
            self._print_stats()

    def log_decode_stats(self) -> None:
        if self._stats_total > 0:
            self._print_stats()

    def _decode(self, codes: torch.Tensor, *, clone_graph_output: bool) -> torch.Tensor:
        if not self.enabled or not self._warmed_up:
            return self.decoder(codes)

        batch_size = int(codes.shape[0])
        actual_size = int(codes.shape[-1])
        request_key = (batch_size, actual_size)
        graph_key = self._get_graph_key(batch_size, actual_size)
        if graph_key is None:
            self._record_route(
                request_key=request_key,
                graph_key=None,
                fallback_reason=self._fallback_reason(batch_size, actual_size),
            )
            return self.decoder(codes)

        self._record_route(request_key=request_key, graph_key=graph_key)
        static_input = self.static_inputs[graph_key]
        graph_size = graph_key[1]
        if graph_size == actual_size:
            static_input.copy_(codes)
        else:
            static_input.zero_()
            static_input[..., :actual_size].copy_(codes)

        self.graphs[graph_key].replay()
        output = self.static_outputs[graph_key]
        if graph_size != actual_size:
            padding_samples = (graph_size - actual_size) * int(self.decoder.total_upsample)
            output = output[..., : output.shape[-1] - padding_samples]
        return output.clone() if clone_graph_output else output

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self._decode(codes, clone_graph_output=True)

    def chunked_decode_with_npugraph(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        wavs: list[torch.Tensor] = []
        start_index = 0
        total_len = int(codes.shape[-1])
        total_upsample = int(self.decoder.total_upsample)

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self._decode(codes_chunk, clone_graph_output=False)
            wavs.append(wav_chunk[..., context_size * total_upsample :].clone())
            start_index = end_index

        if not wavs:
            return self.decoder(codes)
        return torch.cat(wavs, dim=-1)

    def batched_chunked_decode_with_npugraph(
        self,
        codes: torch.Tensor,
        lengths: Sequence[int],
        chunk_size: int = 300,
        left_context_size: int = 25,
        max_batch_size: int = 0,
    ) -> torch.Tensor:
        return _batched_chunked_decode(
            codes,
            lengths,
            decode_fn=lambda codes_chunk: self._decode(codes_chunk, clone_graph_output=False),
            total_upsample=int(self.decoder.total_upsample),
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            max_batch_size=max_batch_size,
        )
