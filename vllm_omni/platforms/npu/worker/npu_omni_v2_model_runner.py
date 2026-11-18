# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni Model Runner V2 with the Ascend input batch and attention backend."""

from typing import Any

from vllm.logger import init_logger
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

logger = init_logger(__name__)


class NPUQwen3TTSModelState(OmniModelState, AscendModelState):
    """Keep Omni's audio lifecycle and build metadata from Ascend batches."""


class NPUOmniV2RunnerMixin:
    """Use Ascend's batch preparation and forward context with Omni V2."""

    def _make_dummy_input_batch(
        self,
        num_reqs: int,
        num_tokens: int,
        *,
        max_query_len: int | None,
        is_padding: bool,
    ) -> AscendInputBatch:
        return AscendInputBatch.make_dummy(
            num_reqs,
            num_tokens,
            self.input_buffers,
            max_query_len=max_query_len,
            is_padding=is_padding,
        )

    def _forward_context(self, attn_metadata: Any, vllm_config: Any, **kwargs: Any):
        kwargs["aclgraph_runtime_mode"] = kwargs.pop("cudagraph_runtime_mode")
        # Ascend's attention metadata already holds the slot mapping.
        kwargs.pop("slot_mapping", None)
        kwargs.pop("is_padding", None)
        return set_ascend_forward_context(attn_metadata, vllm_config, model_instance=self.model, **kwargs)

    def _close_omni_aux_output(self) -> None:
        """Omni V2 does not allocate the V1 auxiliary-output streams."""


class NPUOmniARModelRunner(NPUOmniV2RunnerMixin, OmniARModelRunner, NPUModelRunner):
    """Run the Qwen3-TTS Talker with Ascend V2 attention metadata."""

    def _init_omni_model_state(self, vllm_config: Any, model: Any, encoder_cache: Any, device: Any):
        if getattr(vllm_config.model_config, "model_stage", None) != "qwen3_tts":
            raise NotImplementedError("This Ascend AR Model Runner V2 backport supports the Qwen3-TTS Talker")
        logger.info("Using Ascend attention metadata and Omni V2 request state for Qwen3-TTS")
        return NPUQwen3TTSModelState(vllm_config, model, encoder_cache, device)
