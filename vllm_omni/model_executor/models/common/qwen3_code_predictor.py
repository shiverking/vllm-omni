"""Qwen3 Code Predictor -- optimized re-prefill, no KV cache.

Shared by Qwen3-Omni and Qwen3-TTS talker models.

* SDPA attention (F.scaled_dot_product_attention) with native GQA support
* HF-compatible numerics (float32 RMSNorm, float32 RoPE, separate linear layers)
* Per-call embedding buffer to avoid cross-request aliasing
* Pre-allocated position_ids (read-only, safe to persist)
* torch.compile (epilogue_fusion=False) on inner transformer by default
* Optional manual CUDA graph capture per batch-size bucket
* Inline sampling (top-k + top-p) -- no custom op overhead
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_GeneratorLike = torch.Generator | Sequence[torch.Generator | None] | None
_UNIFORM_EPS = 1e-20


# ===================================================================
# HF-numerics-compatible layers for code predictor
# ===================================================================
#
# These use plain PyTorch ops (nn.Linear, manual RMSNorm in float32,
# rotate_half RoPE) to produce outputs numerically identical to the
# HuggingFace reference. vLLM's fused kernels (RMSNorm, QKVParallel,
# get_rope) introduce small precision differences that compound across
# the autoregressive steps of the code predictor, causing severe
# audio quality degradation.
#
# See: https://github.com/vllm-project/vllm-omni/issues/2274


class _RMSNorm(nn.Module):
    """RMSNorm matching HuggingFace's implementation exactly.

    Computes variance in float32 to avoid bfloat16 precision loss.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _RotaryEmbedding(nn.Module):
    """RoPE matching HuggingFace's implementation exactly.

    Forces float32 computation for cos/sin, matching HF's torch.autocast(enabled=False).
    """

    def __init__(self, config) -> None:
        super().__init__()
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        rope_theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 (matching HF)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ===================================================================
#  Attention
# ===================================================================


class CodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor.

    Uses ``F.scaled_dot_product_attention`` with HF-compatible RoPE and RMSNorm.
    Supports both the legacy re-prefill path and an optional static KV cache.

    Input : [B, seq_len, hidden_size]
    Output: [B, seq_len, hidden_size]
    """

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.is_gqa = self.num_kv_heads != self.num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        self.max_seq = int(config.num_code_groups) + 1
        self._npu_fia_gqa_enabled = False

        # Separate q/k/v projections matching HF (no fused packing)
        bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if current_omni_platform.is_npu():
            if self.max_seq > 2048:
                raise ValueError(
                    "Qwen3-TTS code predictor NPU fusion attention uses a fixed 2048x2048 "
                    f"causal mask, but max_seq={self.max_seq} exceeds the mask size."
                )
            # Ascend SDPA is_causal migration example uses a fixed 2048x2048
            # compressed causal mask with sparse_mode=2.
            fusion_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.bool),
                diagonal=1,
            )
            self.register_buffer("_fusion_causal_mask", fusion_mask, persistent=False)

    def _forward_npu_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        kv_len: int,
        *,
        causal: bool,
    ) -> torch.Tensor:
        import torch_npu

        q_f, k_f, v_f = q, k, v
        if self.is_gqa:
            k_f = (
                k[:, :, None, :, :]
                .expand(bsz, self.num_kv_heads, self.num_queries_per_kv, kv_len, self.head_dim)
                .reshape(bsz, self.num_heads, kv_len, self.head_dim)
            )
            v_f = (
                v[:, :, None, :, :]
                .expand(bsz, self.num_kv_heads, self.num_queries_per_kv, kv_len, self.head_dim)
                .reshape(bsz, self.num_heads, kv_len, self.head_dim)
            )

        mask = self._fusion_causal_mask.contiguous() if causal else None
        q_f = q_f.contiguous()
        k_f = k_f.contiguous()
        v_f = v_f.contiguous()
        return torch_npu.npu_fusion_attention(
            q_f,
            k_f,
            v_f,
            self.num_heads,
            "BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=mask,
            scale=float(self.scaling),
            keep_prob=1.0,
            # Keep torch_npu's API spelling.
            pre_tockens=2147483647,
            next_tockens=2147483647,
            inner_precise=0,
            prefix=None,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
            # Decode has one newest query and may attend every cached key.
            sparse_mode=2 if causal else 0,
            gen_mask_parallel=True,
            # Keep sync=True for the NPU fused attention path.
            sync=True,
        )[0]

    def _forward_npu_fia_gqa(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        bsz: int,
        query_len: int,
        kv_len: int,
        *,
        causal: bool,
    ) -> torch.Tensor:
        """Run native FIA GQA directly against the full static KV cache."""
        import torch_npu

        if not key_cache.is_contiguous() or not value_cache.is_contiguous():
            raise ValueError("Qwen3-TTS FIA GQA requires contiguous full KV cache tensors")

        mask = self._fusion_causal_mask.contiguous() if causal else None
        return torch_npu.npu_fused_infer_attention_score(
            query=q.contiguous(),
            key=key_cache,
            value=value_cache,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            atten_mask=mask,
            actual_seq_lengths=[query_len] * bsz,
            actual_seq_lengths_kv=[kv_len] * bsz,
            scale=float(self.scaling),
            pre_tokens=2147483647,
            next_tokens=2147483647,
            sparse_mode=2 if causal else 0,
        )[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor | None = None,
        value_cache: torch.Tensor | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape_q = (bsz, seq_len, self.num_heads, self.head_dim)
        hidden_shape_kv = (bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape_q)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape_kv)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape_kv).transpose(1, 2)

        cos, sin = position_embeddings
        # cos/sin are [batch, seq_len, head_dim], need unsqueeze at dim=1 for heads
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

        use_cache = key_cache is not None or value_cache is not None
        use_native_fia_gqa = False
        if use_cache:
            if key_cache is None or value_cache is None:
                raise ValueError("key_cache and value_cache must be provided together")
            cache_end = cache_position + seq_len
            if cache_position < 0 or cache_end > key_cache.shape[2]:
                raise ValueError(
                    f"Invalid code predictor cache range [{cache_position}, {cache_end}) "
                    f"for capacity {key_cache.shape[2]}"
                )
            key_cache[:, :, cache_position:cache_end, :].copy_(k)
            value_cache[:, :, cache_position:cache_end, :].copy_(v)
            kv_len = cache_end
            causal = cache_position == 0 and seq_len > 1
            use_native_fia_gqa = (
                current_omni_platform.is_npu()
                and self._npu_fia_gqa_enabled
                and self.is_gqa
                and q.dtype == torch.bfloat16
                and key_cache.is_contiguous()
                and value_cache.is_contiguous()
            )
            if not use_native_fia_gqa:
                k = key_cache[:, :, :cache_end, :]
                v = value_cache[:, :, :cache_end, :]
        else:
            kv_len = seq_len
            causal = True

        if not current_omni_platform.is_npu():
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scaling,
                is_causal=causal,
                enable_gqa=self.is_gqa,
            )
        elif use_native_fia_gqa:
            assert key_cache is not None and value_cache is not None
            attn_out = self._forward_npu_fia_gqa(
                q,
                key_cache,
                value_cache,
                bsz,
                seq_len,
                kv_len,
                causal=causal,
            )
        else:
            attn_out = self._forward_npu_attention(
                q,
                k,
                v,
                bsz,
                kv_len,
                causal=causal,
            )

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out)


# ===================================================================
#  MLP
# ===================================================================


class CodePredictorMLP(nn.Module):
    """SiLU-gated MLP for code predictor, matching HF's implementation."""

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# ===================================================================
#  Decoder Layer
# ===================================================================


class CodePredictorDecoderLayer(nn.Module):
    """Transformer decoder layer with optional static KV cache."""

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = CodePredictorAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = CodePredictorMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor | None = None,
        value_cache: torch.Tensor | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ===================================================================
#  Base Transformer Model
# ===================================================================


class CodePredictorBaseModel(nn.Module):
    """Inner transformer for code predictor.

    ``key_cache`` and ``value_cache`` are optional static per-layer buffers.
    Omitting them preserves the legacy full causal re-prefill behavior.
    """

    def __init__(
        self,
        config,
        *,
        embedding_dim: int | None = None,
        use_parallel_embedding: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        emb_dim = int(embedding_dim) if embedding_dim is not None else int(config.hidden_size)
        if use_parallel_embedding:
            self.codec_embedding = nn.ModuleList(
                [VocabParallelEmbedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
            )
        else:
            self.codec_embedding = nn.ModuleList(
                [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
            )

        self.layers = nn.ModuleList(
            [
                CodePredictorDecoderLayer(config, prefix=f"{prefix}.layers.{idx}")
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _RotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor | None = None,
        value_cache: torch.Tensor | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:
        # Run the transformer body in float32 when the model is in fp16.
        # fp16 lacks the dynamic range for stable attention scores and
        # SiLU-gated MLP intermediates, producing NaN on GPUs without
        # native bf16 support (Turing, Volta).  The RMSNorm and RoPE
        # layers already upcast internally; this extends the same
        # treatment to attention and MLP.
        # autocast to float32 is unsupported on CPU; skip fp32 upcast there
        # (CPU uses full-precision intermediates internally).
        input_dtype = inputs_embeds.dtype
        use_fp32 = input_dtype == torch.float16 and inputs_embeds.device.type != "cpu"
        if use_fp32:
            inputs_embeds = inputs_embeds.float()
        hidden_states = inputs_embeds
        with torch.amp.autocast(inputs_embeds.device.type, enabled=use_fp32, dtype=torch.float32):
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            if (key_cache is None) != (value_cache is None):
                raise ValueError("key_cache and value_cache must be provided together")
            if key_cache is not None and key_cache.shape[0] != len(self.layers):
                raise ValueError(
                    f"Expected {len(self.layers)} code predictor cache layers, got {key_cache.shape[0]}"
                )
            for layer_idx, layer in enumerate(self.layers):
                layer_key_cache = None if key_cache is None else key_cache[layer_idx]
                layer_value_cache = None if value_cache is None else value_cache[layer_idx]
                hidden_states = layer(
                    hidden_states,
                    position_embeddings,
                    key_cache=layer_key_cache,
                    value_cache=layer_value_cache,
                    cache_position=cache_position,
                )
            hidden_states = self.norm(hidden_states)
        return hidden_states.to(input_dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ===================================================================
#  Wrapper Configuration
# ===================================================================


@dataclasses.dataclass
class CodePredictorWrapperConfig:
    """Controls behavioral differences between model-specific code predictors."""

    use_cuda_graphs: bool = False
    use_parallel_embedding: bool = False
    use_projection: bool = False
    return_proj_buf: bool = False
    sampling_mode: str = "stored"


# ===================================================================
#  Code Predictor Wrapper (optimized re-prefill, persistent buffers)
# ===================================================================


class CodePredictorWrapper(nn.Module):
    """Optimized code predictor with re-prefill and optional NPU KV cache.

    The default path re-prefills the growing sequence. Ascend deployments may
    opt into a static per-bucket KV cache: prefill two tokens once, then replay
    one-token decode graphs for the remaining code groups.

    Optimizations:
      1. Per-call embedding buffer -- avoids cross-request aliasing.
      2. Pre-allocated position_ids -- no torch.arange per step.
      3. Cached module references -- bypass ModuleList indexing.
      4. torch.compile on inner transformer.
      5. Inline sampling (top-k + top-p) -- no custom op overhead.
      6. Optional manual CUDA graph capture per batch-size bucket.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        cp_config,
        wrapper_config: CodePredictorWrapperConfig,
        talker_hidden_size: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = cp_config
        self._wrapper_config = wrapper_config
        self.prefix = prefix

        self._num_groups = int(cp_config.num_code_groups)
        self._cp_hidden = int(cp_config.hidden_size)

        # For Omni backward compat (accessed by the talker)
        self.num_code_groups = self._num_groups

        # Determine embedding dimension
        _talker_hidden = int(talker_hidden_size) if talker_hidden_size is not None else self._cp_hidden

        self.model = CodePredictorBaseModel(
            cp_config,
            embedding_dim=_talker_hidden,
            use_parallel_embedding=wrapper_config.use_parallel_embedding,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        self.lm_head = nn.ModuleList(
            [nn.Linear(cp_config.hidden_size, cp_config.vocab_size, bias=False) for _ in range(self._num_groups - 1)]
        )

        # Projection: Identity when hidden sizes match or not needed
        if wrapper_config.use_projection and _talker_hidden != self._cp_hidden:
            self.small_to_mtp_projection = nn.Linear(_talker_hidden, self._cp_hidden, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Sampling defaults for "stored" mode
        self._top_k: int = 50
        self._top_p: float = 0.8

        # Lazily initialised state
        self._proj_buf: torch.Tensor | None = None
        self._model_dtype: torch.dtype | None = None
        self._compiled_model_fwd = None
        self._bucket_sizes: list[int] = []
        self._bucket_pos_ids: dict[int | tuple[int, int], torch.Tensor] = {}
        self._lm_heads_list: list[nn.Module] | None = None
        self._codec_embeds_list: list[nn.Module] | None = None
        self._device_graphs: dict[int | tuple[int, int], tuple] = {}  # (graph, static_output) per bucket
        self._kv_device_graphs: dict[tuple[int, int], tuple] = {}
        # All graphs in one bucket share these address-stable buffers. Replays
        # must stay serialized per wrapper; concurrent execution needs one
        # wrapper/cache set per execution slot.
        self._kv_cache_by_bucket: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        graph_cfg = self._stage_connector_extra_config(vllm_config)
        prefix_graphs_requested = self._parse_bool_config(graph_cfg.get("code_predictor_prefix_graphs"))
        kv_cache_requested = self._parse_bool_config(graph_cfg.get("code_predictor_kv_cache"))
        fia_gqa_requested = self._parse_bool_config(graph_cfg.get("code_predictor_fia_gqa"))
        if prefix_graphs_requested and kv_cache_requested:
            raise ValueError(
                "code_predictor_kv_cache and code_predictor_prefix_graphs cannot both be enabled"
            )
        is_npu = current_omni_platform.is_npu()
        if is_npu and fia_gqa_requested and not kv_cache_requested:
            raise ValueError("code_predictor_fia_gqa requires code_predictor_kv_cache on NPU")
        self._is_npu = is_npu
        self._prefix_graphs_enabled = prefix_graphs_requested and wrapper_config.use_cuda_graphs
        self._kv_cache_enabled = kv_cache_requested and is_npu and wrapper_config.use_cuda_graphs
        self._fia_gqa_requested = fia_gqa_requested
        self._fia_gqa_enabled = False
        self._fia_gqa_configured = False
        if prefix_graphs_requested and not self._prefix_graphs_enabled:
            logger.warning(
                "[Qwen3-TTS][prefix graph] requested but disabled "
                "use_device_graphs=%s is_npu=%s",
                wrapper_config.use_cuda_graphs,
                is_npu,
            )
        self._prefix_graph_buckets = self._parse_positive_int_set(
            graph_cfg.get("code_predictor_prefix_graph_buckets")
        )
        self._prefix_graph_seq_lens = self._parse_positive_int_set(
            graph_cfg.get("code_predictor_prefix_graph_seq_lens")
        )
        self._kv_cache_buckets = self._parse_positive_int_set(
            graph_cfg.get("code_predictor_kv_cache_buckets")
        )
        self._printed_short_prefix_buckets: set[int] = set()
        self._printed_kv_cache_buckets: set[int] = set()
        self._printed_fia_gqa_buckets: set[int] = set()
        if is_npu and self._prefix_graphs_enabled:
            logger.info(
                "[Qwen3-TTS][NPU prefix graph] enabled buckets=%s seq_lens=%s",
                sorted(self._prefix_graph_buckets) if self._prefix_graph_buckets else "all",
                self._prefix_seq_lens(self._num_groups + 1),
            )
        if kv_cache_requested and not self._kv_cache_enabled:
            logger.warning(
                "[Qwen3-TTS][NPU KV cache] requested but disabled "
                "use_device_graphs=%s is_npu=%s",
                wrapper_config.use_cuda_graphs,
                is_npu,
            )
        if self._kv_cache_enabled:
            logger.info(
                "[Qwen3-TTS][NPU KV cache] enabled buckets=%s cache_lens=%s",
                sorted(self._kv_cache_buckets) if self._kv_cache_buckets else "all",
                list(range(2, self._num_groups + 1)),
            )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def set_sampling_params(self, top_k: int = 50, top_p: float = 0.8) -> None:
        """Configure sampling parameters to maintain consistency with previous implementation."""
        self._top_k = top_k
        self._top_p = top_p
        logger.debug("Sampling parameters updated: top_k=%d, top_p=%.2f", top_k, top_p)

    # ------------------------------------------------------------------
    #  Lazy-init helpers
    # ------------------------------------------------------------------

    def _ensure_buffers(self, device: torch.device, dtype: torch.dtype, bsz: int) -> None:
        """Ensure the projection buffer can hold at least *bsz* rows."""
        max_seq = self._num_groups + 1
        if (
            self._proj_buf is not None
            and self._proj_buf.device == device
            and self._proj_buf.dtype == dtype
            and self._proj_buf.shape[0] >= bsz
        ):
            return
        self._proj_buf = torch.zeros(bsz, max_seq, self._cp_hidden, dtype=dtype, device=device)

    def _kv_cache_dtype(self) -> torch.dtype:
        if self._model_dtype == torch.float16:
            return torch.float32
        return self._model_dtype

    def _ensure_kv_cache(
        self,
        device: torch.device,
        bsz: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        existing = self._kv_cache_by_bucket.get(bsz)
        cache_dtype = self._kv_cache_dtype()
        if (
            existing is not None
            and existing[0].device == device
            and existing[0].dtype == cache_dtype
        ):
            return existing
        cache_shape = (
            len(self.model.layers),
            bsz,
            int(self.config.num_key_value_heads),
            self._num_groups,
            int(self.model.layers[0].self_attn.head_dim),
        )
        caches = (
            torch.empty(cache_shape, dtype=cache_dtype, device=device),
            torch.empty(cache_shape, dtype=cache_dtype, device=device),
        )
        self._kv_cache_by_bucket[bsz] = caches
        return caches

    def _kv_bucket_enabled(self, bsz: int) -> bool:
        return self._kv_cache_enabled and (
            not self._kv_cache_buckets or bsz in self._kv_cache_buckets
        )

    def _configure_fia_gqa(self) -> None:
        if self._fia_gqa_configured:
            return
        self._fia_gqa_configured = True

        is_gqa = all(layer.self_attn.is_gqa for layer in self.model.layers)
        self._fia_gqa_enabled = (
            self._fia_gqa_requested
            and self._is_npu
            and self._kv_cache_enabled
            and is_gqa
            and self._model_dtype == torch.bfloat16
        )
        for layer in self.model.layers:
            layer.self_attn._npu_fia_gqa_enabled = self._fia_gqa_enabled

        if not self._fia_gqa_requested or not self._is_npu or not self._kv_cache_enabled:
            return
        query_heads = int(self.config.num_attention_heads)
        kv_heads = int(self.config.num_key_value_heads)
        if self._fia_gqa_enabled:
            logger.info(
                "[Qwen3-TTS][NPU FIA GQA] enabled "
                "dtype=%s query_heads=%d kv_heads=%d layout=BNSD",
                self._model_dtype,
                query_heads,
                kv_heads,
            )
        elif self._model_dtype != torch.bfloat16:
            logger.warning(
                "[Qwen3-TTS][NPU FIA GQA] dtype fallback "
                "dtype=%s backend=npu_fusion_attention",
                self._model_dtype,
            )

    def _setup_compile(self) -> None:
        """Lazily set up torch.compile with optional device graph capture."""
        if self._compiled_model_fwd is not None:
            return

        # Cache model parameter dtype so forward() doesn't need to query it
        # on every call.  Also ensures warmup buffers match model precision
        # even when upstream modules produce a different dtype (#2385).
        self._model_dtype = next(self.model.parameters()).dtype
        self._lm_heads_list = list(self.lm_head)
        self._codec_embeds_list = list(self.model.codec_embedding)
        self._configure_fia_gqa()

        if not current_omni_platform.supports_torch_inductor():
            # NPU or other platforms without Inductor support
            self._compiled_model_fwd = self.model.forward

            if current_omni_platform.is_npu() and self._wrapper_config.use_cuda_graphs:
                # For NPU, use eager + NPU graphs (no torch.compile)
                self._warmup_buckets()
                self._capture_npu_graphs()
                logger.info("code_predictor: eager mode + NPU graphs")
            else:
                logger.warning_once("code_predictor: torch.compile disabled")
            return

        # torch.compile fuses RMSNorm/RoPE in ways that lose float32
        # precision, compounding across AR steps. Use epilogue_fusion=False
        # to disable the problematic fusions while still getting kernel
        # fusion benefits for the linear layers and SDPA.
        self._compiled_model_fwd = torch.compile(
            self.model.forward,
            dynamic=False,
            options={"epilogue_fusion": False},
        )
        self._warmup_buckets()

        if self._wrapper_config.use_cuda_graphs:
            self._capture_cuda_graphs()
            logger.info("code_predictor: torch.compile (no epilogue fusion) + CUDA graphs")
        else:
            logger.info("code_predictor: torch.compile (dynamic=False, no epilogue fusion)")

    def _padded_bsz(self, bsz: int) -> int:
        """Round batch size up to nearest power-of-2 bucket."""
        for bucket in self._bucket_sizes:
            if bsz <= bucket:
                return bucket
        return bsz

    @staticmethod
    def _stage_connector_extra_config(vllm_config: VllmConfig) -> dict:
        model_cfg = getattr(vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra_cfg = connector_cfg.get("extra", connector_cfg)
        else:
            extra_cfg = getattr(connector_cfg, "extra", None)
        return extra_cfg if isinstance(extra_cfg, dict) else {}

    @staticmethod
    def _parse_bool_config(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(value, int):
            return bool(value)
        return False

    @staticmethod
    def _parse_positive_int_set(value: object) -> set[int]:
        if value is None:
            return set()
        if isinstance(value, str):
            raw_values = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        elif isinstance(value, int):
            raw_values = [value]
        else:
            try:
                raw_values = list(value)
            except TypeError as exc:
                raise ValueError(f"Invalid positive int config value {value!r}") from exc
        values: set[int] = set()
        for item in raw_values:
            try:
                parsed = int(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid positive int config value {item!r}") from exc
            if parsed > 0:
                values.add(parsed)
        return values

    @staticmethod
    def _normalize_generators(
        generator: _GeneratorLike, batch_size: int
    ) -> torch.Generator | list[torch.Generator | None] | None:
        if generator is None or isinstance(generator, torch.Generator):
            return generator

        row_generators = list(generator)
        if len(row_generators) != batch_size:
            raise ValueError(f"Expected {batch_size} per-row generators, but got {len(row_generators)}.")
        return row_generators

    @classmethod
    def _sample_codes_gumbel(cls, logits: torch.Tensor, generator: _GeneratorLike = None) -> torch.Tensor:
        """Sample ``logits`` via Gumbel-max with optional per-row generators."""
        row_generators = cls._normalize_generators(generator, int(logits.shape[0]))
        u = torch.empty_like(logits, dtype=torch.float32)
        if isinstance(row_generators, list):
            for row, row_generator in enumerate(row_generators):
                u[row : row + 1].uniform_(_UNIFORM_EPS, 1.0 - _UNIFORM_EPS, generator=row_generator)
        else:
            u.uniform_(_UNIFORM_EPS, 1.0 - _UNIFORM_EPS, generator=row_generators)
        return (logits.float() - torch.log(-torch.log(u))).argmax(dim=-1, keepdim=True)

    def _prefix_seq_lens(self, max_seq: int) -> list[int]:
        all_seq_lens = list(range(2, max_seq))
        if not self._prefix_graph_seq_lens:
            return all_seq_lens
        allowed = set(all_seq_lens)
        return sorted(seq_len for seq_len in self._prefix_graph_seq_lens if seq_len in allowed)

    def _warmup_buckets(self) -> None:
        """Warmup power-of-2 batch-size buckets to front-load Inductor compilation."""
        max_bsz = self._vllm_config.scheduler_config.max_num_seqs
        bucket_sizes = [1 << i for i in range(max_bsz.bit_length()) if (1 << i) <= max_bsz]
        if max_bsz not in bucket_sizes:
            bucket_sizes.append(max_bsz)
        self._bucket_sizes = sorted(bucket_sizes)

        max_seq = self._num_groups + 1
        device = next(self.model.parameters()).device

        # Ensure proj_buf matches model parameter dtype to avoid dtype
        # mismatch during warmup compilation (see #2385).
        self._ensure_buffers(device, self._model_dtype, max(self._bucket_sizes))
        proj_buf = self._proj_buf

        if self._kv_cache_enabled:
            for bsz in self._bucket_sizes:
                if not self._kv_bucket_enabled(bsz):
                    pos_ids = (
                        torch.arange(max_seq, device=device, dtype=torch.long)
                        .unsqueeze(0)
                        .expand(bsz, -1)
                        .contiguous()
                    )
                    self._bucket_pos_ids[bsz] = pos_ids
                    for _ in range(3):
                        self._compiled_model_fwd(proj_buf[:bsz, :max_seq, :], pos_ids)
                    continue

                key_cache, value_cache = self._ensure_kv_cache(device, bsz)
                for cache_len in range(2, self._num_groups + 1):
                    input_len = 2 if cache_len == 2 else 1
                    input_start = 0 if cache_len == 2 else cache_len - 1
                    pos_ids = (
                        torch.arange(input_start, input_start + input_len, device=device, dtype=torch.long)
                        .unsqueeze(0)
                        .expand(bsz, -1)
                        .contiguous()
                    )
                    self._bucket_pos_ids[(bsz, cache_len)] = pos_ids
                for _ in range(2):
                    for cache_len in range(2, self._num_groups + 1):
                        input_len = 2 if cache_len == 2 else 1
                        input_start = 0 if cache_len == 2 else cache_len - 1
                        self._compiled_model_fwd(
                            proj_buf[:bsz, input_start : input_start + input_len, :],
                            self._bucket_pos_ids[(bsz, cache_len)],
                            key_cache,
                            value_cache,
                            input_start,
                        )
            logger.info(
                "code_predictor: KV cache warmup done for buckets %s kv_buckets=%s",
                self._bucket_sizes,
                sorted(self._kv_cache_buckets) if self._kv_cache_buckets else "all",
            )
        elif self._prefix_graphs_enabled:
            prefix_seq_lens = self._prefix_seq_lens(max_seq)
            needs_full_graph = set(prefix_seq_lens) != set(range(2, max_seq))
            for bsz in self._bucket_sizes:
                capture_prefixes = not self._prefix_graph_buckets or bsz in self._prefix_graph_buckets
                if not capture_prefixes or needs_full_graph:
                    pos_ids = (
                        torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1).contiguous()
                    )
                    self._bucket_pos_ids[bsz] = pos_ids
                    for _ in range(3):
                        self._compiled_model_fwd(proj_buf[:bsz, :max_seq, :], pos_ids)
                if capture_prefixes:
                    for seq_len in prefix_seq_lens:
                        pos_ids = (
                            torch.arange(seq_len, device=device, dtype=torch.long)
                            .unsqueeze(0)
                            .expand(bsz, -1)
                            .contiguous()
                        )
                        self._bucket_pos_ids[(bsz, seq_len)] = pos_ids
                        for _ in range(2):
                            self._compiled_model_fwd(proj_buf[:bsz, :seq_len, :], pos_ids)
            logger.info(
                "code_predictor: prefix warmup done for buckets %s prefix_buckets=%s seq_lens=%s",
                self._bucket_sizes,
                sorted(self._prefix_graph_buckets) if self._prefix_graph_buckets else "all",
                prefix_seq_lens,
            )
        else:
            for bsz in self._bucket_sizes:
                pos_ids = (
                    torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1).contiguous()
                )
                self._bucket_pos_ids[bsz] = pos_ids
                for _ in range(3):
                    self._compiled_model_fwd(proj_buf[:bsz, :max_seq, :], pos_ids)
            logger.info("code_predictor: warmup done for buckets %s", self._bucket_sizes)

    def _capture_cuda_graphs(self) -> None:
        """Capture a CUDA graph per bucket using vLLM's global graph pool."""
        from vllm.platforms import current_platform

        pool = current_platform.get_global_graph_pool()
        max_seq = self._num_groups + 1
        proj_buf = self._proj_buf

        if self._prefix_graphs_enabled:
            prefix_seq_lens = self._prefix_seq_lens(max_seq)
            needs_full_graph = set(prefix_seq_lens) != set(range(2, max_seq))
            for bsz in self._bucket_sizes:
                capture_prefixes = not self._prefix_graph_buckets or bsz in self._prefix_graph_buckets
                if not capture_prefixes or needs_full_graph:
                    static_input = proj_buf[:bsz, :max_seq, :]
                    pos_ids = self._bucket_pos_ids[bsz]

                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool):
                        static_output = self._compiled_model_fwd(static_input, pos_ids)

                    self._device_graphs[bsz] = (g, static_output)

                if capture_prefixes:
                    for seq_len in prefix_seq_lens:
                        static_input = proj_buf[:bsz, :seq_len, :]
                        pos_ids = self._bucket_pos_ids[(bsz, seq_len)]

                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, pool=pool):
                            static_output = self._compiled_model_fwd(static_input, pos_ids)

                        self._device_graphs[(bsz, seq_len)] = (g, static_output)

            logger.info(
                "code_predictor: captured prefix CUDA graphs for buckets %s prefix_buckets=%s seq_lens=%s",
                self._bucket_sizes,
                sorted(self._prefix_graph_buckets) if self._prefix_graph_buckets else "all",
                prefix_seq_lens,
            )
        else:
            for bsz in self._bucket_sizes:
                static_input = proj_buf[:bsz, :max_seq, :]
                pos_ids = self._bucket_pos_ids[bsz]

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    static_output = self._compiled_model_fwd(static_input, pos_ids)

                self._device_graphs[bsz] = (g, static_output)

            logger.info("code_predictor: captured CUDA graphs for buckets %s", self._bucket_sizes)

    def _capture_npu_graphs(self) -> None:
        """Capture an NPU graph per bucket using torch_npu's NPUGraph."""
        max_seq = self._num_groups + 1
        proj_buf = self._proj_buf
        pool = torch.npu.graph_pool_handle()
        prefix_graph_keys: list[tuple[int, int]] = []
        full_graph_keys: list[int] = []

        if self._kv_cache_enabled:
            kv_graph_keys: list[tuple[int, int]] = []
            for bsz in self._bucket_sizes:
                if not self._kv_bucket_enabled(bsz):
                    static_input = proj_buf[:bsz, :max_seq, :]
                    pos_ids = self._bucket_pos_ids[bsz]
                    g = torch.npu.NPUGraph()
                    with torch.npu.graph(g, pool=pool):
                        static_output = self._compiled_model_fwd(static_input, pos_ids)
                    self._device_graphs[bsz] = (g, static_output)
                    full_graph_keys.append(bsz)
                    continue

                key_cache, value_cache = self._kv_cache_by_bucket[bsz]
                for cache_len in range(2, self._num_groups + 1):
                    input_len = 2 if cache_len == 2 else 1
                    input_start = 0 if cache_len == 2 else cache_len - 1
                    static_input = proj_buf[:bsz, input_start : input_start + input_len, :]
                    pos_ids = self._bucket_pos_ids[(bsz, cache_len)]
                    g = torch.npu.NPUGraph()
                    with torch.npu.graph(g, pool=pool):
                        static_output = self._compiled_model_fwd(
                            static_input,
                            pos_ids,
                            key_cache,
                            value_cache,
                            input_start,
                        )
                    graph_key = (bsz, cache_len)
                    self._kv_device_graphs[graph_key] = (g, static_output)
                    kv_graph_keys.append(graph_key)
            cache_shapes = {
                bsz: tuple(caches[0].shape)
                for bsz, caches in sorted(self._kv_cache_by_bucket.items())
            }
            logger.info(
                "[Qwen3-TTS][NPU KV cache] capture complete "
                "graph_count=%d buckets=%s cache_shapes=%s full_fallback_keys=%s",
                len(kv_graph_keys),
                sorted(self._kv_cache_by_bucket),
                cache_shapes,
                sorted(full_graph_keys),
            )
            if self._fia_gqa_enabled:
                logger.info(
                    "[Qwen3-TTS][NPU FIA GQA] capture complete "
                    "graph_count=%d query_heads=%d kv_heads=%d",
                    len(kv_graph_keys),
                    int(self.config.num_attention_heads),
                    int(self.config.num_key_value_heads),
                )
        elif self._prefix_graphs_enabled:
            prefix_seq_lens = self._prefix_seq_lens(max_seq)
            needs_full_graph = set(prefix_seq_lens) != set(range(2, max_seq))
            for bsz in self._bucket_sizes:
                capture_prefixes = not self._prefix_graph_buckets or bsz in self._prefix_graph_buckets
                if not capture_prefixes or needs_full_graph:
                    static_input = proj_buf[:bsz, :max_seq, :]
                    pos_ids = self._bucket_pos_ids[bsz]

                    g = torch.npu.NPUGraph()
                    with torch.npu.graph(g, pool=pool):
                        static_output = self._compiled_model_fwd(static_input, pos_ids)

                    self._device_graphs[bsz] = (g, static_output)
                    full_graph_keys.append(bsz)

                if capture_prefixes:
                    for seq_len in prefix_seq_lens:
                        static_input = proj_buf[:bsz, :seq_len, :]
                        pos_ids = self._bucket_pos_ids[(bsz, seq_len)]

                        g = torch.npu.NPUGraph()
                        with torch.npu.graph(g, pool=pool):
                            static_output = self._compiled_model_fwd(static_input, pos_ids)

                        graph_key = (bsz, seq_len)
                        self._device_graphs[graph_key] = (g, static_output)
                        prefix_graph_keys.append(graph_key)
        else:
            for bsz in self._bucket_sizes:
                static_input = proj_buf[:bsz, :max_seq, :]
                pos_ids = self._bucket_pos_ids[bsz]

                g = torch.npu.NPUGraph()
                with torch.npu.graph(g, pool=pool):
                    static_output = self._compiled_model_fwd(static_input, pos_ids)

                self._device_graphs[bsz] = (g, static_output)
                full_graph_keys.append(bsz)

        if self._kv_cache_enabled:
            return
        if self._prefix_graphs_enabled:
            logger.info(
                "[Qwen3-TTS][NPU prefix graph] capture complete "
                "prefix_keys=%s full_fallback_keys=%s",
                sorted(prefix_graph_keys),
                sorted(full_graph_keys),
            )
        else:
            logger.info("code_predictor: captured NPU graphs for buckets %s", self._bucket_sizes)

    # ------------------------------------------------------------------
    #  Forward -- re-prefill or static KV cache + inline sampling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
        generators: Sequence[torch.Generator | None] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict residual codebooks 1..G-1 autoregressively."""
        bsz = int(layer0_code.shape[0])
        if generators is not None and len(generators) != bsz:
            raise ValueError(f"generators must have one entry per row: got {len(generators)} for batch {bsz}")
        sample_generator: _GeneratorLike = generators if generators is not None else generator
        num_groups = self._num_groups
        device = layer0_code.device

        # _setup_compile caches _model_dtype on first call; use it for buffers
        # so they always match model weight precision (#2385).
        self._setup_compile()
        dtype = self._model_dtype

        padded_bsz = self._padded_bsz(bsz)
        self._ensure_buffers(device, dtype, padded_bsz)

        proj_buf = self._proj_buf
        max_seq = num_groups + 1
        projection = self.small_to_mtp_projection
        model_fwd = self._compiled_model_fwd
        lm_heads = self._lm_heads_list
        codec_embeds = self._codec_embeds_list

        # Zero the padded region of the buffer
        proj_buf[:padded_bsz].zero_()

        # Fill buffer positions 0 (talker hidden) & 1 (layer0 embed)
        proj_buf[:bsz, 0, :] = projection(last_talker_hidden.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)
        proj_buf[:bsz, 1, :] = projection(layer0_embed.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)

        # Prepare sampling parameters
        stored_mode = self._wrapper_config.sampling_mode == "stored"
        if stored_mode:
            s_top_k = self._top_k
            s_top_p = self._top_p
        else:
            use_sampling = do_sample and temperature > 0
            inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0
            if use_sampling and top_p != 1.0:
                raise NotImplementedError(
                    "top_p sampling is not implemented for the vLLM-native code predictor; please set top_p=1.0."
                )

        # Output codes -- shape depends on return mode
        if self._wrapper_config.return_proj_buf:
            all_codes = torch.empty(bsz, num_groups, 1, dtype=torch.int64, device=device)
            all_codes[:, 0] = layer0_code.reshape(bsz, -1)[:, :1]
        else:
            all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
            all_codes[:, 0] = layer0_code.reshape(bsz)

        required_kv_keys = [(padded_bsz, cache_len) for cache_len in range(2, num_groups + 1)]
        # Select the cache path only when the complete prefill/decode graph set
        # exists. Mid-request fallback would mix incompatible cache state.
        use_kv_cache = self._kv_bucket_enabled(padded_bsz) and all(
            graph_key in self._kv_device_graphs for graph_key in required_kv_keys
        )

        # Autoregressive loop: predict layers 1..G-1
        for step in range(1, num_groups):
            if use_kv_cache:
                cache_len = step + 1
                graph_entry = self._kv_device_graphs[(padded_bsz, cache_len)]
                if padded_bsz not in self._printed_kv_cache_buckets:
                    logger.info(
                        "[Qwen3-TTS][NPU KV cache] active "
                        "batch_bucket=%d prefill_input_len=2 decode_input_len=1 max_cache_len=%d",
                        padded_bsz,
                        num_groups,
                    )
                    self._printed_kv_cache_buckets.add(padded_bsz)
                if self._fia_gqa_enabled and padded_bsz not in self._printed_fia_gqa_buckets:
                    logger.info(
                        "[Qwen3-TTS][NPU FIA GQA] active "
                        "batch_bucket=%d query_heads=%d kv_heads=%d cache_backed=true",
                        padded_bsz,
                        int(self.config.num_attention_heads),
                        int(self.config.num_key_value_heads),
                    )
                    self._printed_fia_gqa_buckets.add(padded_bsz)
                graph_entry[0].replay()
                hidden_out = graph_entry[1]
                hidden_index = 1 if cache_len == 2 else 0
            else:
                graph_key: int | tuple[int, int] = padded_bsz
                seq_len = max_seq
                if self._prefix_graphs_enabled:
                    prefix_key = (padded_bsz, step + 1)
                    if prefix_key in self._device_graphs:
                        graph_key = prefix_key
                        seq_len = step + 1
                    if (
                        graph_key == prefix_key
                        and self._is_npu
                        and padded_bsz not in self._printed_short_prefix_buckets
                    ):
                        logger.info(
                            "[Qwen3-TTS][NPU prefix graph] short prefix active "
                            "batch_bucket=%d seq_len=%d full_seq_len=%d",
                            padded_bsz,
                            seq_len,
                            max_seq,
                        )
                        self._printed_short_prefix_buckets.add(padded_bsz)
                pos_ids = self._bucket_pos_ids.get(graph_key)
                if pos_ids is None:
                    pos_ids = (
                        torch.arange(seq_len, device=device, dtype=torch.long)
                        .unsqueeze(0)
                        .expand(padded_bsz, -1)
                        .contiguous()
                    )

                # Use captured device graph if available, otherwise call compiled fn.
                device_graph_entry = self._device_graphs.get(graph_key)

                # Run transformer (device graph replay or compiled forward)
                if device_graph_entry is not None:
                    device_graph_entry[0].replay()
                    hidden_out = device_graph_entry[1]
                else:
                    hidden_out = model_fwd(proj_buf[:padded_bsz, :seq_len, :], pos_ids)
                hidden_index = step

            logits = lm_heads[step - 1](hidden_out[:bsz, hidden_index, :])

            # Sample next code via Gumbel-max.
            #
            # ``argmax_i(logits_i + Gumbel_i)`` with
            # ``Gumbel_i = -log(-log(u_i)), u_i ~ Uniform(0, 1)`` is
            # distributionally identical to sampling from ``softmax(logits)``.
            # In this file the motivations are practical rather than graph
            # related: it is measurably cheaper than ``softmax + multinomial``
            # on the B x 2048 shapes used here, it stays well-defined for
            # degenerate masked rows with a surviving finite entry (and is more
            # defensive than ``multinomial`` around fully-masked/NaN inputs),
            # and the helper below can honor either one batch generator or one
            # generator per seeded row.
            if stored_mode:
                # "stored" mode: top-k -> top-p -> Gumbel-max
                if s_top_k > 0:
                    topk_vals, _ = logits.topk(s_top_k, dim=-1)
                    logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
                if s_top_p < 1.0:
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1, dtype=torch.float32)
                    cumulative_probs = sorted_probs.cumsum(dim=-1)
                    remove_mask = (cumulative_probs - sorted_probs) >= s_top_p
                    sorted_logits[remove_mask] = float("-inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                code = self._sample_codes_gumbel(logits, generator=sample_generator)
            else:
                # "per_call" mode: temperature-scaled + top-k -> Gumbel-max
                if use_sampling:
                    scaled = logits * inv_temperature
                    if top_k > 0:
                        topk_vals, _ = scaled.topk(top_k, dim=-1)
                        scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                    code = self._sample_codes_gumbel(scaled, generator=sample_generator)
                else:
                    code = logits.argmax(dim=-1, keepdim=True)

            # Store code
            if self._wrapper_config.return_proj_buf:
                all_codes[:, step] = code
            else:
                all_codes[:, step] = code.reshape(bsz)

            # Embed predicted code -> project -> next buffer position
            if step < num_groups - 1 or self._wrapper_config.return_proj_buf:
                new_embed = codec_embeds[step - 1](code)
                proj_buf[:bsz, step + 1, :] = projection(new_embed.reshape(bsz, 1, -1)).reshape(bsz, -1)

        if self._wrapper_config.return_proj_buf:
            return all_codes, proj_buf[:bsz].clone()
        return all_codes

    # ------------------------------------------------------------------
    #  Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights directly (no fused projection remapping needed)."""
        loaded: set[str] = set()
        model_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []

        for name, w in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("model."):
                model_weights.append((name[len("model.") :], w))
            else:
                other_weights.append((name, w))

        loaded_model = self.model.load_weights(model_weights)
        loaded |= {f"model.{n}" for n in loaded_model}

        params = dict(self.named_parameters(remove_duplicate=False))
        for name, w in other_weights:
            param = params.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w)
            loaded.add(name)

        return loaded
