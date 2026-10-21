"""Qwen3-TTS Code Predictor -- thin wrapper over CodePredictorWrapper."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config

from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
)
from vllm_omni.platforms import current_omni_platform

from .configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig

# Backward-compat alias used by tests
Qwen3TTSTalkerCodePredictorModelVLLM = CodePredictorBaseModel


class Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(CodePredictorWrapper):
    """Qwen3-TTS code predictor (per-call sampling, projection)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
        prefix: str = "code_predictor",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            cp_config=config,
            wrapper_config=CodePredictorWrapperConfig(
                use_cuda_graphs=current_omni_platform.is_npu(),
                use_parallel_embedding=False,
                use_projection=(config.hidden_size != talker_config.hidden_size),
                return_proj_buf=False,
                sampling_mode="per_call",
            ),
            talker_hidden_size=int(talker_config.hidden_size),
            prefix=prefix,
        )
        # Store talker_config for backward compat (accessed by some callers)
        self.talker_config = talker_config
        self._vllm_config = vllm_config
        self.research = None
        self.playback = None
        additional = getattr(vllm_config, "additional_config", None)
        playback_config = additional.get("tts_playback_experiment") if isinstance(additional, dict) else None
        if playback_config:
            from vllm_omni.model_executor.models.common.qwen3_tts_playback_session import PlaybackSession

            self.playback = PlaybackSession(playback_config)
        if os.environ.get("VLLM_OMNI_TTS_SPEC_CONFIG"):
            from vllm_omni.model_executor.models.common.qwen3_code_predictor_research import ResearchSession

            self.research = ResearchSession(os.environ["VLLM_OMNI_TTS_SPEC_CONFIG"])

    def forward(self, *args, **kwargs):
        if getattr(self, "playback", None) is not None:
            bound = inspect.signature(super().forward).bind(*args, **kwargs)
            bound.apply_defaults()
            return self.playback.forward(self, super().forward, dict(bound.arguments))
        if getattr(self, "research", None) is None:
            return super().forward(*args, **kwargs)
        bound = inspect.signature(super().forward).bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        inputs = [values.pop(key) for key in ("layer0_code", "layer0_embed", "last_talker_hidden")]
        return self.research.forward(self, super().forward, *inputs, values)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with vllm config context (required for VocabParallelEmbedding)."""
        with set_current_vllm_config(self._vllm_config):
            return super().load_weights(weights)
