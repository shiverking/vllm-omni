# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch

from .test_code_predictor_dtype import (  # noqa: F401
    _make_tiny_config,
    _make_vllm_config,
    loaded_target_classes,
)


@pytest.fixture
def models(mocker, loaded_target_classes):  # noqa: F811
    config, talker = _make_tiny_config(loaded_target_classes)
    config.num_code_groups = 16
    wrapper = loaded_target_classes[2](
        vllm_config=_make_vllm_config(mocker, 16),
        config=config,
        talker_config=talker,
    ).eval()
    executor = importlib.import_module(
        "vllm_omni.model_executor.models.common.qwen3_code_predictor_ar"
    ).CachedARExecutor(wrapper, "eager")
    return wrapper, executor


def inputs(batch):
    return dict(
        layer0_code=torch.zeros(batch, dtype=torch.long),
        layer0_embed=torch.randn(batch, 32),
        last_talker_hidden=torch.randn(batch, 32),
        sample_uniforms=torch.rand(batch, 15, 64).clamp_(1e-5, 1 - 1e-5),
    )


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 8, 9, 15, 16])
@pytest.mark.parametrize("sampling", [False, True])
@torch.inference_mode()
def test_cached_ar_matches_stock(models, batch, sampling):
    w, executor = models
    args = inputs(batch)
    calls = []
    hook = w.model.register_forward_hook(lambda *unused: calls.append(1))
    actual = executor(**args, do_sample=sampling)
    assert len(calls) == 15
    hook.remove()
    expected = w(**args, do_sample=sampling)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(executor(**args, do_sample=sampling), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_cache_isolation_reordering_and_output_lifetime(models):
    w, executor = models
    args = inputs(3)
    first = executor(**args)
    saved = first.clone()
    executor(**inputs(16))
    executor(**inputs(1))
    assert torch.equal(first, saved)
    assert torch.equal(executor(**args), saved)
    order = torch.tensor([2, 0, 1])
    assert torch.equal(executor(**{k: v[order] for k, v in args.items()}), saved[order])
    for row in range(3):
        assert torch.equal(executor(**{k: v[row : row + 1] for k, v in args.items()}), saved[row : row + 1])


@torch.inference_mode()
def test_rng_consumption_excludes_padding(models):
    w, executor = models
    args = inputs(3)
    args.pop("sample_uniforms")
    a = [torch.Generator().manual_seed(i + 123) for i in range(3)]
    b = [torch.Generator().manual_seed(i + 123) for i in range(3)]
    assert torch.equal(executor(**args, generators=a), w(**args, generators=b))
    for left, right in zip(a, b):
        assert torch.equal(left.get_state(), right.get_state())
    a = torch.Generator().manual_seed(42)
    b = torch.Generator().manual_seed(42)
    assert torch.equal(executor(**args, generator=a), w(**args, generator=b))
    assert torch.equal(a.get_state(), b.get_state())


@torch.inference_mode()
def test_every_position_logits_and_changed_uniforms(models):
    w, executor = models
    args = inputs(2)
    reference = []
    hooks = [h.register_forward_hook(lambda module, inp, out: reference.append(out.clone())) for h in w.lm_head]
    expected = w(**args)
    for h in hooks:
        h.remove()
    audit = []
    assert torch.equal(executor(**args, audit=audit), expected)
    for left, right in zip(audit, reference):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
    args["sample_uniforms"] = torch.rand_like(args["sample_uniforms"]).clamp_(1e-5, 1 - 1e-5)
    assert torch.equal(executor(**args), w(**args))
    assert not torch.equal(executor(**args), expected)


def test_invalid_options(models):
    _, executor = models
    with pytest.raises(ValueError, match="Unsupported batch"):
        executor(**inputs(17))
    with pytest.raises(ValueError, match="supports greedy"):
        executor(**inputs(1), top_p=0.8)


@pytest.mark.parametrize("position,length", [(0, 2), (2, 1), (15, 1)])
@torch.inference_mode()
def test_cached_mlp_uses_only_new_positions(models, position, length):
    wrapper, _ = models
    mlp = wrapper.model.layers[0].mlp
    x = torch.randn(8, length, wrapper.config.hidden_size)
    down_inputs, gate_shapes = [], []
    h1 = mlp.down_proj.register_forward_pre_hook(lambda mod, args: down_inputs.append(args[0].clone()))
    h2 = mlp.gate_up_proj.register_forward_pre_hook(lambda mod, args: gate_shapes.append(tuple(args[0].shape)))
    try:
        cached = mlp(x, cache_position=position)
        stock = mlp(x)
    finally:
        h1.remove()
        h2.remove()
    assert all(value.shape[1] == length for value in down_inputs)
    assert all(shape[1] == length for shape in gate_shapes)
    torch.testing.assert_close(down_inputs[0], down_inputs[1], rtol=0, atol=0)
    torch.testing.assert_close(cached, stock, rtol=2e-5, atol=2e-5)


@torch.inference_mode()
def test_cached_attention_projection_geometry_and_positions(models):
    w, executor = models
    b = executor._allocate(2)
    b["inputs"].normal_()
    attention = w.model.layers[0].self_attn
    projected = []
    qkv_lengths = []
    h1 = attention.o_proj.register_forward_pre_hook(lambda mod, args: projected.append(args[0].clone()))
    h2 = attention.qkv_proj.register_forward_pre_hook(lambda mod, args: qkv_lengths.append(args[0].shape[1]))
    try:
        for step in range(15):
            executor._backbone(b, step)
    finally:
        h1.remove()
        h2.remove()
    assert qkv_lengths == [2] + [1] * 14
    assert [value.shape[1] for value in projected] == [2] + [1] * 14
    assert all(torch.count_nonzero(value) > 0 for value in projected)


@pytest.mark.parametrize("mode,expected_graphs,expected_calls", [("kv_step", 15, 60), ("kv_frame", 2, 120)])
@torch.inference_mode()
def test_graph_capture_shared_only_for_step(models, monkeypatch, mode, expected_graphs, expected_calls):
    from contextlib import nullcontext
    from types import SimpleNamespace

    w, executor = models
    executor.mode = mode  # Exercise capture bookkeeping on CPU with a fake NPU API.
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            is_current_stream_capturing=lambda: False,
            synchronize=lambda *args: None,
            graph_pool_handle=lambda: object(),
            NPUGraph=lambda: object(),
            graph=lambda *args, **kw: nullcontext(),
        ),
        raising=False,
    )
    calls = []
    hook = w.model.register_forward_hook(lambda *args: calls.append(1))
    try:
        executor.prepare(2, False)
        executor.prepare(2, True)
        executor.prepare(2, False)
    finally:
        hook.remove()
    assert executor.graph_count == expected_graphs
    assert len(calls) == expected_calls
    if mode == "kv_step":
        assert executor.graph_key(2, False) == executor.graph_key(2, True)


@torch.inference_mode()
def test_cached_fia_reads_full_native_gqa_cache(models, monkeypatch):
    import sys
    from types import SimpleNamespace

    w, executor = models
    attn = w.model.layers[0].self_attn
    module = sys.modules[type(attn).__module__]
    monkeypatch.setattr(module.current_omni_platform, "is_npu", lambda: True)
    attn._fusion_causal_mask = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    batch, capacity, dim = 2, 16, attn.head_dim
    cache = tuple(torch.randn(batch, attn.num_kv_heads, capacity, dim, dtype=torch.bfloat16) for _ in range(2))
    seen = []

    def fia(**kw):
        seen.append(kw)
        end = kw["actual_seq_lengths_kv"][0]
        return (
            torch.nn.functional.scaled_dot_product_attention(
                kw["query"],
                kw["key"][:, :, :end],
                kw["value"][:, :, :end],
                is_causal=kw["sparse_mode"] == 2,
                enable_gqa=attn.is_gqa,
                scale=attn.scaling,
            ),
        )

    monkeypatch.setattr(module, "torch_npu", SimpleNamespace(npu_fused_infer_attention_score=fia), raising=False)
    for position, length in ((0, 2), (2, 1), (15, 1)):
        q = torch.randn(batch, attn.num_heads, length, dim, dtype=torch.bfloat16)
        k, v = (torch.randn(batch, attn.num_kv_heads, length, dim, dtype=torch.bfloat16) for _ in range(2))
        result = attn._cached_attention(q, k, v, cache, position)
        kw = seen[-1]
        assert kw["key"].data_ptr() == cache[0].data_ptr()
        assert kw["value"].data_ptr() == cache[1].data_ptr()
        assert kw["key"].shape == (batch, attn.num_kv_heads, capacity, dim)
        assert kw["actual_seq_lengths"] == [length] * batch
        assert kw["actual_seq_lengths_kv"] == [position + length] * batch
        assert kw["num_key_value_heads"] == attn.num_kv_heads
        assert (kw["atten_mask"] is not None) == (position == 0)
        assert result.shape == q.shape
        torch.testing.assert_close(cache[0][:, :, position : position + length], k, atol=0, rtol=0)
    monkeypatch.setattr(module, "torch_npu", SimpleNamespace(), raising=False)
    with pytest.raises(RuntimeError, match="no fallback"):
        attn._cached_attention(q, k, v, cache, 15)


@torch.inference_mode()
def test_graph_capture_failure_does_not_register_or_fallback(models, monkeypatch):
    from types import SimpleNamespace

    _, executor = models
    executor.mode = "kv_step"

    def failure(*args, **kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            is_current_stream_capturing=lambda: False,
            synchronize=lambda *args: None,
            graph_pool_handle=lambda: object(),
            NPUGraph=lambda: object(),
            graph=failure,
        ),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="capture failed"):
        executor.prepare(1, False)
    assert executor.graphs == {}
