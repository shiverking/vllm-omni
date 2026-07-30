# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""NPU Graph wrapper for the Qwen3-TTS speech tokenizer decoder."""

from __future__ import annotations

import bisect
from collections.abc import Sequence

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

        self.graphs: dict[tuple[int, int], object] = {}
        self.static_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self._bucket_sizes_by_batch: dict[int, list[int]] = {}
        self._printed_active_graph_keys: set[tuple[int, int]] = set()
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

    def _refresh_bucket_sizes(self) -> None:
        buckets: dict[int, list[int]] = {}
        for batch_size, size in self.graphs:
            buckets.setdefault(batch_size, []).append(size)
        self._bucket_sizes_by_batch = {
            batch_size: sorted(set(sizes)) for batch_size, sizes in buckets.items()
        }

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
            f"capture_shapes={capture_shapes}",
            flush=True,
        )

        pool = torch.npu.graph_pool_handle()
        graph_keys: list[tuple[int, int]] = []
        failed_keys: list[tuple[int, int]] = []
        for batch_size, size in capture_shapes:
            key = (batch_size, size)
            try:
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

        self._refresh_bucket_sizes()
        self._warmed_up = True
        print(
            "[Qwen3-TTS][NPU Code2Wav graph] capture complete "
            f"graph_keys={sorted(graph_keys)} "
            f"failed_keys={sorted(failed_keys)}",
            flush=True,
        )

    def _get_graph_key(self, batch_size: int, actual_size: int) -> tuple[int, int] | None:
        sizes = self._bucket_sizes_by_batch.get(batch_size)
        if not sizes:
            return None
        index = bisect.bisect_left(sizes, actual_size)
        if index >= len(sizes):
            return None
        return batch_size, sizes[index]

    def _trim_replay_output(
        self,
        static_output: torch.Tensor,
        actual_size: int,
        graph_size: int,
    ) -> torch.Tensor:
        drop = (graph_size - actual_size) * int(self.decoder.total_upsample)
        actual_output_size = max(0, int(static_output.shape[-1]) - drop)
        return static_output[..., :actual_output_size]

    def _decode(self, codes: torch.Tensor, *, clone_graph_output: bool) -> torch.Tensor:
        if not self.enabled or not self._warmed_up:
            return self.decoder(codes)

        batch_size = int(codes.shape[0])
        actual_size = int(codes.shape[-1])
        graph_key = self._get_graph_key(batch_size, actual_size)
        if graph_key is None:
            return self.decoder(codes)

        graph_size = graph_key[1]
        static_input = self.static_inputs[graph_key]
        if actual_size == graph_size:
            static_input.copy_(codes)
        else:
            static_input.zero_()
            static_input[..., :actual_size].copy_(codes)

        self.graphs[graph_key].replay()
        output = self._trim_replay_output(
            self.static_outputs[graph_key],
            actual_size,
            graph_size,
        )
        if graph_key not in self._printed_active_graph_keys:
            self._printed_active_graph_keys.add(graph_key)
            print(
                "[Qwen3-TTS][NPU Code2Wav graph] active "
                f"batch_size={batch_size} "
                f"actual_frames={actual_size} "
                f"graph_frames={graph_size}",
                flush=True,
            )
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
