# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for code predictor dtype alignment (fix for #2385).

Verifies that the code predictor handles dtype mismatches between input
tensors and model parameters without raising RuntimeError. This can happen
when model weights are loaded in float16/bfloat16 but upstream modules
produce float32 hidden states.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from contextlib import nullcontext

import pytest
import torch
from pytest_mock import MockerFixture

# Direct file import to avoid vllm_omni.__init__ patch dependencies.
_MODELS = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    os.pardir,
    os.pardir,
    "vllm_omni",
    "model_executor",
    "models",
)
_BASE = os.path.join(_MODELS, "qwen3_tts")
_COMMON = os.path.join(_MODELS, "common")


def _load_module(name: str, filename: str):
    path = os.path.abspath(os.path.join(_BASE, filename))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec (needed for dataclasses etc.)
    spec.loader.exec_module(mod)
    return mod


def _build_mock_modules(mocker: MockerFixture) -> dict[str, object]:
    """Build the dict of modules to inject into sys.modules."""
    platforms_mock = mocker.MagicMock()
    platforms_mock.current_omni_platform.supports_torch_inductor.return_value = False
    platforms_mock.current_omni_platform.is_npu.return_value = False

    logger_mock = mocker.MagicMock()
    logger_mock.init_logger = logging.getLogger

    vllm_config_mod = mocker.MagicMock()
    vllm_config_mod.set_current_vllm_config = lambda cfg: mocker.MagicMock(
        __enter__=mocker.MagicMock(),
        __exit__=mocker.MagicMock(),
    )

    weight_utils_mock = mocker.MagicMock()
    weight_utils_mock.default_weight_loader = lambda p, w: None

    tts_pkg = types.ModuleType("vllm_omni.model_executor.models.qwen3_tts")
    tts_pkg.__path__ = [os.path.abspath(_BASE)]

    common_pkg = types.ModuleType("vllm_omni.model_executor.models.common")
    common_pkg.__path__ = [os.path.abspath(_COMMON)]

    models_pkg = types.ModuleType("vllm_omni.model_executor.models")
    models_pkg.__path__ = [os.path.abspath(_MODELS)]

    vllm_parallel_mock = mocker.MagicMock()
    vllm_parallel_mock.VocabParallelEmbedding = torch.nn.Embedding

    return {
        "vllm_omni": mocker.MagicMock(),
        "vllm_omni.platforms": platforms_mock,
        "vllm.logger": logger_mock,
        "vllm.config": mocker.MagicMock(),
        "vllm.config.vllm": vllm_config_mod,
        "vllm.model_executor.model_loader.weight_utils": weight_utils_mock,
        "vllm.model_executor.layers.vocab_parallel_embedding": vllm_parallel_mock,
        "vllm_omni.model_executor": types.ModuleType("vllm_omni.model_executor"),
        "vllm_omni.model_executor.models": models_pkg,
        "vllm_omni.model_executor.models.common": common_pkg,
        "vllm_omni.model_executor.models.qwen3_tts": tts_pkg,
    }


def _load_target_classes(mocker: MockerFixture):
    """Load config and code predictor modules with mocked dependencies.

    Uses mocker.patch.dict to ensure sys.modules is always restored, even on failure.
    """
    mocks = _build_mock_modules(mocker)
    mocker.patch.dict(sys.modules, mocks)
    config_mod = _load_module(
        "vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts",
        "configuration_qwen3_tts.py",
    )
    sys.modules["vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts"] = config_mod

    # Load the shared common module (thin wrappers import from it)
    common_cp_path = os.path.abspath(os.path.join(_COMMON, "qwen3_code_predictor.py"))
    common_spec = importlib.util.spec_from_file_location(
        "vllm_omni.model_executor.models.common.qwen3_code_predictor", common_cp_path
    )
    common_cp_mod = importlib.util.module_from_spec(common_spec)
    sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"] = common_cp_mod
    common_spec.loader.exec_module(common_cp_mod)

    cp_mod = _load_module(
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm",
        "qwen3_tts_code_predictor_vllm.py",
    )

    return config_mod, cp_mod


@pytest.fixture
def loaded_target_classes(mocker: MockerFixture):
    config_mod, cp_mod = _load_target_classes(mocker)
    return (
        config_mod.Qwen3TTSTalkerCodePredictorConfig,
        config_mod.Qwen3TTSTalkerConfig,
        cp_mod.Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM,
        cp_mod.Qwen3TTSTalkerCodePredictorModelVLLM,
        cp_mod.CodePredictorWrapperConfig,
    )


def _make_tiny_config(loaded_target_classes) -> tuple:
    """Create minimal configs for a tiny code predictor model."""
    (
        qwen3_tts_talker_code_predictor_config,
        qwen3_tts_talker_config,
        _,
        _,
        _,
    ) = loaded_target_classes
    cp_config = qwen3_tts_talker_code_predictor_config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_code_groups=4,
        rms_norm_eps=1e-6,
    )
    talker_config = qwen3_tts_talker_config(
        hidden_size=32,
        num_code_groups=4,
    )
    return cp_config, talker_config


def _make_vllm_config(mocker: MockerFixture, max_num_seqs: int = 4):
    """Create a mock VllmConfig with scheduler_config."""
    vllm_config = mocker.MagicMock()
    vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    return vllm_config


class _FakeNPUGraph:
    def __init__(self) -> None:
        self.replay_count = 0

    def replay(self) -> None:
        self.replay_count += 1


class _FakeNPU:
    def __init__(self) -> None:
        self.graphs: list[_FakeNPUGraph] = []

    @staticmethod
    def graph_pool_handle():
        return object()

    def NPUGraph(self) -> _FakeNPUGraph:
        graph = _FakeNPUGraph()
        self.graphs.append(graph)
        return graph

    @staticmethod
    def graph(_graph, *, pool):
        assert pool is not None
        return nullcontext()


def _make_npu_prefix_wrapper(
    mocker: MockerFixture,
    loaded_target_classes,
    *,
    prefix_graphs: bool = True,
    prefix_buckets: list[int] | None = None,
    prefix_seq_lens: list[int] | None = None,
):
    common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
    mocker.patch.object(common_mod.current_omni_platform, "is_npu", return_value=True)

    cp_config, _ = _make_tiny_config(loaded_target_classes)
    vllm_config = _make_vllm_config(mocker, max_num_seqs=2)
    vllm_config.model_config.stage_connector_config = {
        "extra": {
            "code_predictor_prefix_graphs": prefix_graphs,
            "code_predictor_prefix_graph_buckets": prefix_buckets or [],
            "code_predictor_prefix_graph_seq_lens": prefix_seq_lens or [],
        }
    }
    wrapper = common_mod.CodePredictorWrapper(
        vllm_config=vllm_config,
        cp_config=cp_config,
        wrapper_config=common_mod.CodePredictorWrapperConfig(use_cuda_graphs=True),
    )
    wrapper._model_dtype = next(wrapper.model.parameters()).dtype
    wrapper._test_forward_seq_lens = []

    def _recording_forward(hidden_states, _position_ids):
        wrapper._test_forward_seq_lens.append(int(hidden_states.shape[1]))
        return hidden_states.clone()

    wrapper._compiled_model_fwd = _recording_forward
    wrapper._lm_heads_list = list(wrapper.lm_head)
    wrapper._codec_embeds_list = list(wrapper.model.codec_embedding)
    return wrapper


def _make_npu_kv_wrapper(
    mocker: MockerFixture,
    loaded_target_classes,
    *,
    kv_cache: bool = True,
    kv_buckets: list[int] | None = None,
    prefix_graphs: bool = False,
    fia_gqa: bool = False,
):
    common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
    mocker.patch.object(common_mod.current_omni_platform, "is_npu", return_value=True)

    cp_config, _ = _make_tiny_config(loaded_target_classes)
    vllm_config = _make_vllm_config(mocker, max_num_seqs=2)
    vllm_config.model_config.stage_connector_config = {
        "extra": {
            "code_predictor_kv_cache": kv_cache,
            "code_predictor_kv_cache_buckets": kv_buckets or [],
            "code_predictor_prefix_graphs": prefix_graphs,
            "code_predictor_fia_gqa": fia_gqa,
        }
    }
    wrapper = common_mod.CodePredictorWrapper(
        vllm_config=vllm_config,
        cp_config=cp_config,
        wrapper_config=common_mod.CodePredictorWrapperConfig(use_cuda_graphs=True),
    )
    wrapper._model_dtype = next(wrapper.model.parameters()).dtype
    wrapper._test_kv_calls = []

    def _recording_forward(
        hidden_states,
        _position_ids,
        key_cache=None,
        value_cache=None,
        cache_position=0,
    ):
        wrapper._test_kv_calls.append(
            (
                int(hidden_states.shape[1]),
                int(cache_position),
                None if key_cache is None else id(key_cache),
                None if value_cache is None else id(value_cache),
            )
        )
        return hidden_states.clone()

    wrapper._compiled_model_fwd = _recording_forward
    wrapper._lm_heads_list = list(wrapper.lm_head)
    wrapper._codec_embeds_list = list(wrapper.model.codec_embedding)
    return wrapper


class TestCodePredictorDtypeAlignment:
    """Test that code predictor buffers match model parameter dtype."""

    def test_ensure_buffers_uses_given_dtype(self, mocker: MockerFixture, loaded_target_classes) -> None:
        """_ensure_buffers should create proj_buf with the given dtype."""
        _, _, code_predictor_wrapper, _, _ = loaded_target_classes
        cp_config, talker_config = _make_tiny_config(loaded_target_classes)
        vllm_config = _make_vllm_config(mocker)

        predictor = code_predictor_wrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )

        # Create buffer in float16
        predictor._ensure_buffers(torch.device("cpu"), torch.float16, 4)
        assert predictor._proj_buf is not None
        assert predictor._proj_buf.dtype == torch.float16

        # Re-create buffer in float32 (different dtype triggers re-allocation)
        predictor._ensure_buffers(torch.device("cpu"), torch.float32, 4)
        assert predictor._proj_buf.dtype == torch.float32

    def test_warmup_aligns_buffer_to_model_params(self, mocker: MockerFixture, loaded_target_classes) -> None:
        """_warmup_buckets should align proj_buf dtype to model parameters."""
        _, _, code_predictor_wrapper, _, _ = loaded_target_classes
        cp_config, talker_config = _make_tiny_config(loaded_target_classes)
        vllm_config = _make_vllm_config(mocker, max_num_seqs=2)

        predictor = code_predictor_wrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )

        # Cast model to float16 (simulating vLLM loading weights in half precision)
        predictor = predictor.to(torch.float16)

        # Pre-create proj_buf with WRONG dtype (float32) — simulating the bug
        predictor._ensure_buffers(torch.device("cpu"), torch.float32, 2)
        assert predictor._proj_buf.dtype == torch.float32

        # Simulate _setup_compile having cached model dtype and compiled forward
        predictor._model_dtype = torch.float16
        predictor._compiled_model_fwd = predictor.model.forward

        # Ensure NPU path is not taken on non-NPU hardware
        common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
        mocker.patch.object(common_mod.current_omni_platform, "is_npu", return_value=False)

        # _warmup_buckets should fix the dtype mismatch
        predictor._warmup_buckets()

        assert predictor._proj_buf.dtype == torch.float16

    def test_setup_compile_caches_model_dtype(self, mocker: MockerFixture, loaded_target_classes) -> None:
        """_setup_compile should cache model parameter dtype."""
        _, _, code_predictor_wrapper, _, _ = loaded_target_classes
        cp_config, talker_config = _make_tiny_config(loaded_target_classes)
        vllm_config = _make_vllm_config(mocker, max_num_seqs=2)

        predictor = code_predictor_wrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )
        predictor = predictor.to(torch.float16)

        assert predictor._model_dtype is None
        # Ensure NPU path is not taken on non-NPU hardware
        common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
        mocker.patch.object(common_mod.current_omni_platform, "is_npu", return_value=False)
        predictor._setup_compile()
        assert predictor._model_dtype == torch.float16

    def test_forward_with_mismatched_input_dtype(self, mocker: MockerFixture, loaded_target_classes) -> None:
        """forward() should not crash when inputs are float32 but model is float16."""
        _, _, code_predictor_wrapper, _, _ = loaded_target_classes
        cp_config, talker_config = _make_tiny_config(loaded_target_classes)
        vllm_config = _make_vllm_config(mocker, max_num_seqs=2)

        predictor = code_predictor_wrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )

        # Model in float16
        predictor = predictor.to(torch.float16)

        bsz = 1
        num_groups = cp_config.num_code_groups
        hidden = talker_config.hidden_size

        # Inputs in float32 (simulating the dtype mismatch from #2385)
        layer0_code = torch.zeros(bsz, dtype=torch.long)
        layer0_embed = torch.randn(bsz, hidden, dtype=torch.float32)
        last_talker_hidden = torch.randn(bsz, hidden, dtype=torch.float32)

        # This should NOT raise RuntimeError about dtype mismatch
        result = predictor(
            layer0_code=layer0_code,
            layer0_embed=layer0_embed,
            last_talker_hidden=last_talker_hidden,
            do_sample=False,
        )

        assert result.shape == (bsz, num_groups)
        assert result.dtype == torch.long

    def test_forward_generator_controls_sampling(self, mocker: MockerFixture, loaded_target_classes) -> None:
        _, _, code_predictor_wrapper, _, _ = loaded_target_classes
        common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
        mocker.patch.object(common_mod.current_omni_platform, "is_npu", return_value=False)
        cp_config, talker_config = _make_tiny_config(loaded_target_classes)
        vllm_config = _make_vllm_config(mocker, max_num_seqs=2)

        predictor = code_predictor_wrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )
        predictor._wrapper_config.use_cuda_graphs = False

        bsz = 1
        hidden = talker_config.hidden_size
        torch.manual_seed(123)
        layer0_code = torch.zeros(bsz, dtype=torch.long)
        layer0_embed = torch.randn(bsz, hidden)
        last_talker_hidden = torch.randn(bsz, hidden)

        first_generator = torch.Generator(device=layer0_code.device)
        first_generator.manual_seed(1234)
        second_generator = torch.Generator(device=layer0_code.device)
        second_generator.manual_seed(1234)
        different_generator = torch.Generator(device=layer0_code.device)
        different_generator.manual_seed(4321)

        first = predictor(
            layer0_code=layer0_code,
            layer0_embed=layer0_embed,
            last_talker_hidden=last_talker_hidden,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            generator=first_generator,
        )
        second = predictor(
            layer0_code=layer0_code,
            layer0_embed=layer0_embed,
            last_talker_hidden=last_talker_hidden,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            generator=second_generator,
        )
        different = predictor(
            layer0_code=layer0_code,
            layer0_embed=layer0_embed,
            last_talker_hidden=last_talker_hidden,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            generator=different_generator,
        )

        assert torch.equal(first, second)
        assert not torch.equal(first[:, 1:], different[:, 1:])


class TestCodePredictorModelDtype:
    """Test the inner model forward with different dtypes."""

    def test_model_forward_float16(self, loaded_target_classes) -> None:
        """Inner model forward should work in float16."""
        _, _, _, code_predictor_model, _ = loaded_target_classes
        cp_config, _ = _make_tiny_config(loaded_target_classes)
        model = code_predictor_model(cp_config, embedding_dim=32).to(torch.float16)

        bsz, seq_len = 1, 4
        inputs = torch.randn(bsz, seq_len, 32, dtype=torch.float16)
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

        output = model(inputs, pos_ids)
        assert output.dtype == torch.float16
        assert output.shape == (bsz, seq_len, 32)

    def test_model_forward_float32(self, loaded_target_classes) -> None:
        """Inner model forward should work in float32."""
        _, _, _, code_predictor_model, _ = loaded_target_classes
        cp_config, _ = _make_tiny_config(loaded_target_classes)
        model = code_predictor_model(cp_config, embedding_dim=32).to(torch.float32)

        bsz, seq_len = 1, 4
        inputs = torch.randn(bsz, seq_len, 32, dtype=torch.float32)
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

        output = model(inputs, pos_ids)
        assert output.dtype == torch.float32
        assert output.shape == (bsz, seq_len, 32)


class TestCodePredictorWrapperConfig:
    """Test wrapper configuration for different models."""

    def test_omni_config(self, loaded_target_classes) -> None:
        """Qwen3-Omni uses correct wrapper config."""
        _, _, _, _, code_predictor_wrapper_config = loaded_target_classes
        config = code_predictor_wrapper_config(
            use_cuda_graphs=False,
            use_parallel_embedding=True,
            use_projection=False,
            return_proj_buf=True,
            sampling_mode="stored",
        )
        assert config.use_cuda_graphs is False
        assert config.use_parallel_embedding is True
        assert config.return_proj_buf is True
        assert config.sampling_mode == "stored"

    def test_tts_config(self, loaded_target_classes) -> None:
        """Qwen3-TTS uses correct wrapper config."""
        _, _, _, _, code_predictor_wrapper_config = loaded_target_classes
        config = code_predictor_wrapper_config(
            use_cuda_graphs=True,
            use_parallel_embedding=False,
            use_projection=True,
            return_proj_buf=False,
            sampling_mode="per_call",
        )
        assert config.use_cuda_graphs is True
        assert config.use_parallel_embedding is False
        assert config.return_proj_buf is False
        assert config.sampling_mode == "per_call"

    def test_prefix_graph_config_helpers(self, loaded_target_classes) -> None:
        """Prefix graph helpers parse deploy config values and keep valid seq lens only."""
        _ = loaded_target_classes
        common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
        wrapper_cls = common_mod.CodePredictorWrapper

        assert wrapper_cls._parse_positive_int_set("64; 128,0,-1") == {
            64,
            128,
        }
        assert wrapper_cls._parse_positive_int_set([2, "4", 0]) == {2, 4}
        with pytest.raises(ValueError, match="Invalid positive int config value 'bad'"):
            wrapper_cls._parse_positive_int_set("2,bad")

        wrapper = object.__new__(wrapper_cls)
        wrapper._prefix_graph_seq_lens = {1, 2, 4, 8, 99}
        assert wrapper._prefix_seq_lens(6) == [2, 4]

    def test_prefix_graph_env_requires_cuda_graphs(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
    ) -> None:
        """Avoid prefix warmup on shared code-predictor users that disable CUDA graphs."""
        _ = loaded_target_classes
        common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
        mocker.patch.object(common_mod.current_omni_platform, "is_npu", return_value=False)

        cp_config, _ = _make_tiny_config(loaded_target_classes)
        vllm_config = _make_vllm_config(mocker, max_num_seqs=2)
        vllm_config.model_config.stage_connector_config = {
            "extra": {
                "code_predictor_prefix_graphs": True,
                "code_predictor_prefix_graph_buckets": [2],
                "code_predictor_prefix_graph_seq_lens": "2,3",
            }
        }

        no_graph_wrapper = common_mod.CodePredictorWrapper(
            vllm_config=vllm_config,
            cp_config=cp_config,
            wrapper_config=common_mod.CodePredictorWrapperConfig(use_cuda_graphs=False),
        )
        assert no_graph_wrapper._prefix_graphs_enabled is False
        assert no_graph_wrapper._prefix_graph_buckets == {2}
        assert no_graph_wrapper._prefix_graph_seq_lens == {2, 3}

        graph_wrapper = common_mod.CodePredictorWrapper(
            vllm_config=vllm_config,
            cp_config=cp_config,
            wrapper_config=common_mod.CodePredictorWrapperConfig(use_cuda_graphs=True),
        )
        assert graph_wrapper._prefix_graphs_enabled is True

    def test_npu_prefix_graph_config_can_be_enabled(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        wrapper = _make_npu_prefix_wrapper(
            mocker,
            loaded_target_classes,
            prefix_buckets=[2],
            prefix_seq_lens=[2, 3, 4],
        )

        assert wrapper._prefix_graphs_enabled is True
        assert wrapper._prefix_graph_buckets == {2}
        assert wrapper._prefix_graph_seq_lens == {2, 3, 4}
        assert (
            "[Qwen3-TTS][NPU prefix graph] enabled buckets=[2] seq_lens=[2, 3, 4]" in caplog.text
        )

        disabled_wrapper = _make_npu_prefix_wrapper(
            mocker,
            loaded_target_classes,
            prefix_graphs=False,
            prefix_buckets=[2],
            prefix_seq_lens=[2, 3, 4],
        )
        assert disabled_wrapper._prefix_graphs_enabled is False

    def test_npu_prefix_graph_capture_and_short_prefix_routing(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        wrapper = _make_npu_prefix_wrapper(
            mocker,
            loaded_target_classes,
            prefix_buckets=[2],
            prefix_seq_lens=[2, 3, 4],
        )
        fake_npu = _FakeNPU()
        mocker.patch.object(torch, "npu", fake_npu, create=True)

        wrapper._warmup_buckets()
        wrapper._test_forward_seq_lens.clear()
        wrapper._capture_npu_graphs()

        assert wrapper._test_forward_seq_lens == [5, 2, 3, 4]
        assert set(wrapper._device_graphs) == {
            1,
            (2, 2),
            (2, 3),
            (2, 4),
        }

        bsz = 2
        hidden_size = wrapper.config.hidden_size
        inputs = {
            "layer0_code": torch.zeros(bsz, dtype=torch.long),
            "layer0_embed": torch.randn(bsz, hidden_size),
            "last_talker_hidden": torch.randn(bsz, hidden_size),
            "do_sample": False,
        }
        first = wrapper(**inputs)
        second = wrapper(**inputs)

        assert first.shape == (bsz, wrapper.config.num_code_groups)
        assert second.shape == first.shape
        for graph_key in ((2, 2), (2, 3), (2, 4)):
            graph = wrapper._device_graphs[graph_key][0]
            assert graph.replay_count == 2

        output = caplog.text
        assert (
            "[Qwen3-TTS][NPU prefix graph] capture complete "
            "prefix_keys=[(2, 2), (2, 3), (2, 4)] full_fallback_keys=[1]" in output
        )
        marker = (
            "[Qwen3-TTS][NPU prefix graph] short prefix active "
            "batch_bucket=2 seq_len=2 full_seq_len=5"
        )
        assert output.count(marker) == 1

    def test_npu_prefix_graph_falls_back_to_full_graph(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
    ) -> None:
        wrapper = _make_npu_prefix_wrapper(
            mocker,
            loaded_target_classes,
            prefix_buckets=[2],
            prefix_seq_lens=[2, 3],
        )
        fake_npu = _FakeNPU()
        mocker.patch.object(torch, "npu", fake_npu, create=True)

        wrapper._warmup_buckets()
        wrapper._test_forward_seq_lens.clear()
        wrapper._capture_npu_graphs()

        assert wrapper._test_forward_seq_lens == [5, 5, 2, 3]
        assert set(wrapper._device_graphs) == {
            1,
            2,
            (2, 2),
            (2, 3),
        }

        bsz = 2
        hidden_size = wrapper.config.hidden_size
        wrapper(
            layer0_code=torch.zeros(bsz, dtype=torch.long),
            layer0_embed=torch.randn(bsz, hidden_size),
            last_talker_hidden=torch.randn(bsz, hidden_size),
            do_sample=False,
        )

        assert wrapper._device_graphs[(2, 2)][0].replay_count == 1
        assert wrapper._device_graphs[(2, 3)][0].replay_count == 1
        assert wrapper._device_graphs[2][0].replay_count == 1


class TestCodePredictorKVCache:
    def test_config_enable_disable_and_conflict(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        wrapper = _make_npu_kv_wrapper(
            mocker,
            loaded_target_classes,
            kv_buckets=[2],
        )
        assert wrapper._kv_cache_enabled is True
        assert wrapper._kv_cache_buckets == {2}
        assert (
            "[Qwen3-TTS][NPU KV cache] enabled buckets=[2] cache_lens=[2, 3, 4]"
            in caplog.text
        )

        disabled = _make_npu_kv_wrapper(
            mocker,
            loaded_target_classes,
            kv_cache=False,
            kv_buckets=[2],
        )
        assert disabled._kv_cache_enabled is False

        with pytest.raises(
            ValueError,
            match="code_predictor_kv_cache and code_predictor_prefix_graphs",
        ):
            _make_npu_kv_wrapper(
                mocker,
                loaded_target_classes,
                kv_buckets=[2],
                prefix_graphs=True,
            )

        with pytest.raises(
            ValueError,
            match="code_predictor_fia_gqa requires code_predictor_kv_cache",
        ):
            _make_npu_kv_wrapper(
                mocker,
                loaded_target_classes,
                kv_cache=False,
                fia_gqa=True,
            )

    def test_npu_kv_graph_capture_and_routing(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        wrapper = _make_npu_kv_wrapper(
            mocker,
            loaded_target_classes,
            kv_buckets=[2],
        )
        fake_npu = _FakeNPU()
        mocker.patch.object(torch, "npu", fake_npu, create=True)

        wrapper._warmup_buckets()
        wrapper._test_kv_calls.clear()
        wrapper._capture_npu_graphs()

        assert [(input_len, cache_pos) for input_len, cache_pos, _, _ in wrapper._test_kv_calls] == [
            (5, 0),
            (2, 0),
            (1, 2),
            (1, 3),
        ]
        assert set(wrapper._kv_device_graphs) == {(2, 2), (2, 3), (2, 4)}
        assert set(wrapper._device_graphs) == {1}
        cache_ids = {(key_id, value_id) for _, _, key_id, value_id in wrapper._test_kv_calls[1:]}
        assert len(cache_ids) == 1
        key_cache, value_cache = wrapper._kv_cache_by_bucket[2]
        assert key_cache.shape == (1, 2, 2, 4, 8)
        assert value_cache.shape == key_cache.shape

        inputs = {
            "layer0_code": torch.zeros(2, dtype=torch.long),
            "layer0_embed": torch.randn(2, wrapper.config.hidden_size),
            "last_talker_hidden": torch.randn(2, wrapper.config.hidden_size),
            "do_sample": False,
        }
        wrapper(**inputs)
        wrapper(**inputs)

        for graph_key in ((2, 2), (2, 3), (2, 4)):
            assert wrapper._kv_device_graphs[graph_key][0].replay_count == 2
        assert wrapper._device_graphs[1][0].replay_count == 0

        output = caplog.text
        assert (
            "[Qwen3-TTS][NPU KV cache] capture complete "
            "graph_count=3 buckets=[2] "
            "cache_shapes={2: (1, 2, 2, 4, 8)} full_fallback_keys=[1]"
            in output
        )
        marker = (
            "[Qwen3-TTS][NPU KV cache] active batch_bucket=2 "
            "prefill_input_len=2 decode_input_len=1 max_cache_len=4"
        )
        assert output.count(marker) == 1

    def test_npu_kv_bucket_miss_uses_full_graph(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
    ) -> None:
        wrapper = _make_npu_kv_wrapper(
            mocker,
            loaded_target_classes,
            kv_buckets=[1],
        )
        fake_npu = _FakeNPU()
        mocker.patch.object(torch, "npu", fake_npu, create=True)
        wrapper._warmup_buckets()
        wrapper._capture_npu_graphs()

        wrapper(
            layer0_code=torch.zeros(2, dtype=torch.long),
            layer0_embed=torch.randn(2, wrapper.config.hidden_size),
            last_talker_hidden=torch.randn(2, wrapper.config.hidden_size),
            do_sample=False,
        )
        assert wrapper._device_graphs[2][0].replay_count == 3
        assert all(graph.replay_count == 0 for graph, _ in wrapper._kv_device_graphs.values())

    def test_cached_decode_matches_full_causal_model(
        self,
        loaded_target_classes,
    ) -> None:
        common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
        cp_config, _ = _make_tiny_config(loaded_target_classes)
        torch.manual_seed(0)
        model = common_mod.CodePredictorBaseModel(cp_config).eval()
        inputs = torch.randn(2, 4, cp_config.hidden_size)
        pos_ids = torch.arange(4).unsqueeze(0).expand(2, -1)

        with torch.inference_mode():
            full_output = model(inputs, pos_ids)
            cache_shape = (1, 2, 2, 4, 8)
            key_cache = torch.empty(cache_shape)
            value_cache = torch.empty(cache_shape)
            cached_outputs = [
                model(inputs[:, :2], pos_ids[:, :2], key_cache, value_cache, 0),
                model(inputs[:, 2:3], pos_ids[:, 2:3], key_cache, value_cache, 2),
                model(inputs[:, 3:4], pos_ids[:, 3:4], key_cache, value_cache, 3),
            ]
        cached_output = torch.cat(cached_outputs, dim=1)
        torch.testing.assert_close(cached_output, full_output, rtol=1e-5, atol=1e-5)

        # Reuse the same buffers for a second request. Prefill/decode must
        # overwrite every readable position instead of leaking prior K/V.
        second_inputs = torch.randn_like(inputs)
        with torch.inference_mode():
            second_full = model(second_inputs, pos_ids)
            second_cached = torch.cat(
                [
                    model(second_inputs[:, :2], pos_ids[:, :2], key_cache, value_cache, 0),
                    model(second_inputs[:, 2:3], pos_ids[:, 2:3], key_cache, value_cache, 2),
                    model(second_inputs[:, 3:4], pos_ids[:, 3:4], key_cache, value_cache, 3),
                ],
                dim=1,
            )
        torch.testing.assert_close(second_cached, second_full, rtol=1e-5, atol=1e-5)

    def test_npu_prefill_and_decode_attention_modes(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
    ) -> None:
        common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
        mocker.patch.object(common_mod.current_omni_platform, "is_npu", return_value=True)
        calls = []
        fake_torch_npu = types.ModuleType("torch_npu")

        def _fake_fusion_attention(*args, **kwargs):
            calls.append((args, kwargs))
            return args[0], None

        fake_torch_npu.npu_fusion_attention = _fake_fusion_attention
        mocker.patch.dict(sys.modules, {"torch_npu": fake_torch_npu})
        cp_config, _ = _make_tiny_config(loaded_target_classes)
        attention = common_mod.CodePredictorAttention(cp_config).eval()
        key_cache = torch.empty(1, 2, 4, 8)
        value_cache = torch.empty_like(key_cache)

        prefill = torch.randn(1, 2, cp_config.hidden_size)
        prefill_rope = (torch.ones(1, 2, 8), torch.zeros(1, 2, 8))
        attention(prefill, prefill_rope, key_cache, value_cache, 0)
        decode = torch.randn(1, 1, cp_config.hidden_size)
        decode_rope = (torch.ones(1, 1, 8), torch.zeros(1, 1, 8))
        attention(decode, decode_rope, key_cache, value_cache, 2)

        assert calls[0][1]["atten_mask"] is not None
        assert calls[0][1]["sparse_mode"] == 2
        assert calls[1][1]["atten_mask"] is None
        assert calls[1][1]["sparse_mode"] == 0
        assert calls[1][0][1].shape[2] == 3

    def test_npu_fia_gqa_uses_full_static_kv_cache(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
    ) -> None:
        common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
        mocker.patch.object(common_mod.current_omni_platform, "is_npu", return_value=True)
        fia_calls = []
        legacy_calls = []
        fake_torch_npu = types.ModuleType("torch_npu")

        def _fake_fia(**kwargs):
            fia_calls.append(kwargs)
            return kwargs["query"], None

        def _fake_legacy(*args, **kwargs):
            legacy_calls.append((args, kwargs))
            return args[0], None

        fake_torch_npu.npu_fused_infer_attention_score = _fake_fia
        fake_torch_npu.npu_fusion_attention = _fake_legacy
        mocker.patch.dict(sys.modules, {"torch_npu": fake_torch_npu})

        cp_config, _ = _make_tiny_config(loaded_target_classes)
        attention = common_mod.CodePredictorAttention(cp_config).to(torch.bfloat16).eval()
        attention._npu_fia_gqa_enabled = True
        key_cache = torch.empty(2, 2, 4, 8, dtype=torch.bfloat16)
        value_cache = torch.empty_like(key_cache)
        key_ptr = key_cache.data_ptr()
        value_ptr = value_cache.data_ptr()

        for cache_position, query_len in ((0, 2), (2, 1), (3, 1)):
            hidden = torch.randn(2, query_len, cp_config.hidden_size, dtype=torch.bfloat16)
            rope = (
                torch.ones(2, query_len, 8, dtype=torch.bfloat16),
                torch.zeros(2, query_len, 8, dtype=torch.bfloat16),
            )
            attention(hidden, rope, key_cache, value_cache, cache_position)

        assert not legacy_calls
        assert len(fia_calls) == 3
        assert [tuple(call["query"].shape) for call in fia_calls] == [
            (2, 4, 2, 8),
            (2, 4, 1, 8),
            (2, 4, 1, 8),
        ]
        assert all(tuple(call["key"].shape) == (2, 2, 4, 8) for call in fia_calls)
        assert all(tuple(call["value"].shape) == (2, 2, 4, 8) for call in fia_calls)
        assert all(call["key"].data_ptr() == key_ptr for call in fia_calls)
        assert all(call["value"].data_ptr() == value_ptr for call in fia_calls)
        assert [call["actual_seq_lengths_kv"] for call in fia_calls] == [[2, 2], [3, 3], [4, 4]]
        assert [call["actual_seq_lengths"] for call in fia_calls] == [[2, 2], [1, 1], [1, 1]]
        assert all(call["input_layout"] == "BNSD" for call in fia_calls)
        assert all(call["num_heads"] == 4 for call in fia_calls)
        assert all(call["num_key_value_heads"] == 2 for call in fia_calls)
        assert fia_calls[0]["atten_mask"] is not None
        assert fia_calls[0]["sparse_mode"] == 2
        assert fia_calls[1]["atten_mask"] is None
        assert fia_calls[1]["sparse_mode"] == 0

    def test_npu_fia_gqa_float32_falls_back_to_legacy_attention(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
    ) -> None:
        common_mod = sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"]
        mocker.patch.object(common_mod.current_omni_platform, "is_npu", return_value=True)
        fia_calls = []
        legacy_calls = []
        fake_torch_npu = types.ModuleType("torch_npu")

        def _fake_fia(**kwargs):
            fia_calls.append(kwargs)
            return kwargs["query"], None

        def _fake_legacy(*args, **kwargs):
            legacy_calls.append((args, kwargs))
            return args[0], None

        fake_torch_npu.npu_fused_infer_attention_score = _fake_fia
        fake_torch_npu.npu_fusion_attention = _fake_legacy
        mocker.patch.dict(sys.modules, {"torch_npu": fake_torch_npu})

        cp_config, _ = _make_tiny_config(loaded_target_classes)
        attention = common_mod.CodePredictorAttention(cp_config).eval()
        attention._npu_fia_gqa_enabled = True
        key_cache = torch.empty(1, 2, 4, 8)
        value_cache = torch.empty_like(key_cache)
        hidden = torch.randn(1, 1, cp_config.hidden_size)
        rope = (torch.ones(1, 1, 8), torch.zeros(1, 1, 8))
        attention(hidden, rope, key_cache, value_cache, 2)

        assert not fia_calls
        assert len(legacy_calls) == 1
        assert legacy_calls[0][0][1].shape == (1, 4, 3, 8)

    def test_npu_fia_gqa_config_capture_and_logs(
        self,
        mocker: MockerFixture,
        loaded_target_classes,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        wrapper = _make_npu_kv_wrapper(
            mocker,
            loaded_target_classes,
            kv_buckets=[2],
            fia_gqa=True,
        )
        wrapper.to(torch.bfloat16)
        wrapper._model_dtype = torch.bfloat16
        wrapper._configure_fia_gqa()
        assert wrapper._fia_gqa_enabled is True
        assert all(layer.self_attn._npu_fia_gqa_enabled for layer in wrapper.model.layers)

        fake_npu = _FakeNPU()
        mocker.patch.object(torch, "npu", fake_npu, create=True)
        wrapper._warmup_buckets()
        wrapper._capture_npu_graphs()
        inputs = {
            "layer0_code": torch.zeros(2, dtype=torch.long),
            "layer0_embed": torch.randn(2, wrapper.config.hidden_size),
            "last_talker_hidden": torch.randn(2, wrapper.config.hidden_size),
            "do_sample": False,
        }
        wrapper(**inputs)
        wrapper(**inputs)

        output = caplog.text
        assert (
            "[Qwen3-TTS][NPU FIA GQA] enabled "
            "dtype=torch.bfloat16 query_heads=4 kv_heads=2 layout=BNSD"
            in output
        )
        assert (
            "[Qwen3-TTS][NPU FIA GQA] capture complete "
            "graph_count=3 query_heads=4 kv_heads=2"
            in output
        )
        marker = (
            "[Qwen3-TTS][NPU FIA GQA] active batch_bucket=2 "
            "query_heads=4 kv_heads=2 cache_backed=true"
        )
        assert output.count(marker) == 1
