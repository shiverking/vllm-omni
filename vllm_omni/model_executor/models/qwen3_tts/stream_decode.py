# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eligibility and model options for the single-stage PCM pipeline."""

import torch
from vllm.config import VllmConfig

from vllm_omni.platforms import current_omni_platform


def talker_stream_decode_enabled(vllm_config: VllmConfig) -> bool:
    """The Talker decodes every frame itself and emits PCM as its final output."""
    from .qwen3_tts_code_predictor_vllm import Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor

    extra = Predictor._stage_connector_extra_config(vllm_config)
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    requested = Predictor._parse_bool_config(extra.get("talker_stream_decode"))
    if not requested:
        if getattr(model, "engine_output_type", None) == "audio":
            raise ValueError("The single-stage Qwen3-TTS pipeline requires talker_stream_decode=True")
        if Predictor._parse_bool_config(extra.get("talker_stream_first_audio")):
            raise ValueError("talker_stream_first_audio requires the single-stage talker_stream_decode path")
        return False
    if getattr(model, "engine_output_type", None) != "audio" or not getattr(model, "final_output", False):
        raise ValueError(
            "talker_stream_decode requires a final audio-output Talker; "
            "select qwen3_tts_fused_single_gpu.yaml instead of enabling it on the two-stage pipeline"
        )
    if getattr(model, "supports_running_prefix_cache_reset", True):
        raise ValueError(
            "The single-stage PCM pipeline must disable running prefix-cache reset; "
            "select qwen3_tts_fused_single_gpu.yaml"
        )
    device_type = torch.device(vllm_config.device_config.device).type
    is_cuda = current_omni_platform.is_cuda() and device_type == "cuda"
    is_npu = current_omni_platform.is_npu() and device_type == "npu"
    if not (
        (is_cuda or is_npu)
        and bool(getattr(model, "use_v2_model_runner", False))
        and bool(getattr(model, "async_chunk", False))
        and parallel.tensor_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and parallel.distributed_executor_backend in (None, "uni")
        and not vllm_config.cache_config.enable_prefix_caching
    ):
        raise ValueError(
            "talker_stream_decode requires CUDA or Ascend NPU, MRv2, async_chunk=True, "
            "TP=PP=1, an in-process worker and enable_prefix_caching=False"
        )
    if Predictor._parse_bool_config(extra.get("talker_first_audio")):
        raise ValueError("talker_stream_decode and talker_first_audio are mutually exclusive")
    if is_npu:
        if model.dtype != torch.bfloat16:
            raise ValueError("Ascend single-stage Qwen3-TTS requires BF16")
        if Predictor._parse_bool_config(extra.get("talker_stream_first_audio")):
            raise ValueError("Ascend single-stage Qwen3-TTS delivers its first chunk through MRv2 output")
    return True


def stream_ref_context_frames(vllm_config: VllmConfig) -> int:
    """Reference-code context frames for voice-clone priming in the Talker."""
    from .qwen3_tts_code_predictor_vllm import Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor

    extra = Predictor._stage_connector_extra_config(vllm_config)
    return int(extra.get("ref_code_context_frames") or extra.get("codec_left_context_frames", 25))
