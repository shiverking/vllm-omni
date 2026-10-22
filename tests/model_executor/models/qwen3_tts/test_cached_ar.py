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
def test_cached_mlp_preserves_stock_row_geometry(models, position, length):
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
    assert down_inputs[0].shape[1] == 17
    assert down_inputs[1].shape[1] == length
    assert all(shape[1] == length for shape in gate_shapes)
    torch.testing.assert_close(down_inputs[0][:, position : position + length], down_inputs[1], rtol=0, atol=0)
    remaining = down_inputs[0].clone()
    remaining[:, position : position + length].zero_()
    assert torch.count_nonzero(remaining) == 0
    torch.testing.assert_close(cached, stock, rtol=2e-5, atol=2e-5)
