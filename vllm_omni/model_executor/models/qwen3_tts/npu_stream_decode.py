# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunked Code2Wav inside the Ascend Qwen3-TTS Talker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .npu_segmented_graph_wrapper import NPUGraphDecoderWrapper
    from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder

logger = init_logger(__name__)


class NPUStreamDecoder(nn.Module):
    """Decode complete chunks against the request caches owned by MRv2."""

    def __init__(
        self,
        decoder: Qwen3TTSTokenizerV2Decoder,
        *,
        first_chunk_frames: int,
        chunk_frames: int,
        left_context_frames: int,
        sample_rate: int,
        max_batch_size: int,
    ) -> None:
        super().__init__()
        if min(first_chunk_frames, chunk_frames, left_context_frames, max_batch_size) <= 0:
            raise ValueError("Code2Wav chunk sizes, context and batch size must be positive")
        if left_context_frames != decoder.config.sliding_window:
            raise ValueError("codec_left_context_frames must match the loaded Code2Wav sliding window")
        self.decoder = decoder
        self.first_chunk_frames = first_chunk_frames
        self.chunk_frames = chunk_frames
        self.left_context_frames = left_context_frames
        self.sample_rate = sample_rate
        self.samples_per_frame = int(decoder.total_upsample)
        self.max_batch_size = max_batch_size
        self.graphs: NPUGraphDecoderWrapper | None = None
        decoder._initial_codec_chunk_frames = first_chunk_frames
        decoder._incremental_chunk_frames = chunk_frames

    @classmethod
    def load(cls, model_path: str, vllm_config: VllmConfig) -> tuple[NPUStreamDecoder, set[str]]:
        from vllm_omni.platforms.npu.models.qwen3_tts import (
            _prepare_npu_code2wav_runtime,
            _prepare_npu_decoder_weights,
        )

        from .first_frame_decoder import Qwen3TTSFirstFrameDecoder
        from .npu_segmented_graph_wrapper import NPUGraphDecoderWrapper
        from .qwen3_tts_code_predictor_vllm import Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor

        extra = Predictor._stage_connector_extra_config(vllm_config)
        execution = extra.get("code2wav_execution", "npu_graph")
        if execution not in ("baseline", "npu_graph"):
            raise ValueError("Ascend single-stage Code2Wav supports baseline or npu_graph")
        if extra.get("codec_chunk_ramp") or extra.get("codec_chunk_adaptive"):
            raise ValueError("Ascend single-stage Code2Wav requires fixed chunk sizes")

        _prepare_npu_code2wav_runtime()
        holder = Qwen3TTSFirstFrameDecoder(model_path)
        loaded = holder.load(vllm_config)
        _prepare_npu_decoder_weights(holder.decoder)
        context = int(holder.decoder.config.sliding_window)
        buckets = sorted(Predictor._parse_positive_int_set(extra.get("code2wav_graph_batch_sizes") or [1]))
        max_batch_size = int(extra.get("decode_batch_max_size", max(buckets, default=1)))
        capacity = vllm_config.scheduler_config.max_num_seqs
        if max_batch_size > capacity:
            raise ValueError("decode_batch_max_size cannot exceed max_num_seqs")

        stream = cls(
            holder.decoder,
            first_chunk_frames=int(extra.get("initial_codec_chunk_frames", 4)),
            chunk_frames=int(extra.get("codec_chunk_frames", 8)),
            left_context_frames=int(extra.get("codec_left_context_frames", context)),
            sample_rate=holder.sample_rate,
            max_batch_size=max_batch_size,
        )
        if execution == "npu_graph":
            if not buckets or max_batch_size != buckets[-1]:
                raise ValueError("decode_batch_max_size must match the largest Code2Wav graph bucket")
            stream.graphs = NPUGraphDecoderWrapper(stream.decoder, buckets)
            device = torch.device(vllm_config.device_config.device)
            stream.graphs.warmup(device)
            logger.info(
                "Talker Code2Wav capture process high-water: allocated=%.1f MiB reserved=%.1f MiB",
                torch.npu.max_memory_allocated(device) / (1024**2),
                torch.npu.max_memory_reserved(device) / (1024**2),
            )
        logger.info(
            "Talker Code2Wav ready: execution=%s first_frames=%d chunk_frames=%d context=%d batch=%d buckets=%s",
            execution,
            stream.first_chunk_frames,
            stream.chunk_frames,
            stream.left_context_frames,
            stream.max_batch_size,
            buckets if stream.graphs is not None else [],
        )
        return stream, loaded

    @staticmethod
    def new_cache() -> dict:
        return {"prefix_frames": 0}

    def decode_chunks(self, codes: list[torch.Tensor], caches: list[dict]) -> list[torch.Tensor]:
        """Receive new codes as one [1, codebooks, frames] tensor per request."""
        if len(codes) != len(caches):
            raise ValueError("Code2Wav requires one cache per request chunk")
        if not codes:
            return []
        lengths = [code.shape[-1] for code in codes]
        if self.graphs is not None:
            # Input is already request-packed and the graph wrapper owns the
            # static-batch staging. A second padded tensor would copy every row.
            pcm = self.graphs._batched_request_decode(codes, caches)
        else:
            width = max(lengths)
            packed = torch.cat([F.pad(code, (0, width - length)) for code, length in zip(codes, lengths, strict=True)])
            pcm = self.decoder.batched_chunked_decode(
                packed,
                lengths,
                caches=caches,
                chunk_size=self.chunk_frames,
                left_context_size=self.left_context_frames,
                max_batch_size=self.max_batch_size,
            )
        chunks = [wave.reshape(-1) for wave in pcm]
        for wave, length in zip(chunks, lengths, strict=True):
            if wave.numel() != length * self.samples_per_frame:
                raise RuntimeError("Code2Wav returned a PCM length different from its input chunk")
        return chunks
