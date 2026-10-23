# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only first-prefill comparison; never issues a correctness certificate."""

import argparse
from pathlib import Path

import experiment as exp
import torch


def compare(left, right):
    a, b = left.float(), right.float()
    return dict(
        exact_equal=torch.equal(left, right),
        max_abs=(a - b).abs().max().item(),
        outside_tolerance=(~torch.isclose(a, b, atol=0.02, rtol=0.02)).sum().item(),
    )


@torch.inference_mode()
def trace_prefill(wrapper, inputs, positions, cache=None, pad_down_proj=False):
    snapshots, handles = {}, []
    batch, length = inputs.shape[:2]
    restores = []
    if pad_down_proj:
        # Diagnostic only: preserve actual row positions inside the stock shape.
        for layer in wrapper.model.layers:
            module = layer.mlp.down_proj
            previous = module.__dict__.get("forward")
            original = module.forward

            def padded_forward(x, original=original):
                padded = x.new_zeros((x.shape[0], wrapper._num_groups + 1, x.shape[-1]))
                padded[:, : x.shape[1]].copy_(x)
                return original(padded)[:, : x.shape[1]]

            restores.append((module, previous))
            module.forward = padded_forward

    def save(name, value):
        if (
            torch.is_tensor(value)
            and value.ndim >= 3
            and value.shape[0] == batch
            and value.shape[1] in (length, wrapper._num_groups + 1)
        ):
            snapshots[name] = value[:, :2].detach().clone().cpu()

    for name, module in wrapper.model.named_modules():
        if name and not list(module.children()):
            handles.append(
                module.register_forward_pre_hook(lambda mod, args, name=name: save(name + ".input", args[0]))
            )
            handles.append(module.register_forward_hook(lambda mod, args, out, name=name: save(name + ".output", out)))
    try:
        hidden = wrapper.model(inputs, positions, cache, 0)
        snapshots["model.output"] = hidden[:, :2].detach().clone().cpu()
        snapshots["head1.logits"] = wrapper.lm_head[0](hidden[:, 1]).detach().clone().cpu()
    finally:
        for handle in handles:
            handle.remove()
        for module, previous in restores:
            if previous is None:
                del module.forward
            else:
                module.forward = previous
    return snapshots


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--failure", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--model", default=exp.MODEL)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=False)
    failure = torch.load(args.failure, map_location="cpu", weights_only=True)
    wrapper = exp.load_predictor(args, exp.capture_root(args.input))
    from vllm_omni.model_executor.models.common.qwen3_code_predictor_ar import CachedARExecutor

    executor = CachedARExecutor(wrapper, "eager")
    size = failure["inputs"]["layer0_code"].shape[0]
    bucket = next(b for b in exp.BUCKETS if b >= size)
    buffers = executor._allocate(bucket)
    for target, source in (("hidden", "last_talker_hidden"), ("embed", "layer0_embed")):
        buffers[target][:size].copy_(failure["inputs"][source].reshape(size, 1, -1))
    executor._project(buffers)
    short = buffers["inputs"][:, :2]
    full = torch.zeros(bucket, wrapper._num_groups + 1, short.shape[-1], device=short.device, dtype=short.dtype)
    full[:, :2].copy_(short)
    runs = {}
    for name, tensor, cache in (
        ("full_no_cache", full, None),
        ("short_no_cache", short, None),
        ("short_cached", short, buffers["cache"]),
    ):
        print(f"诊断前向: {name}, shape={tuple(tensor.shape)}", flush=True)
        positions = torch.arange(tensor.shape[1], device=tensor.device).unsqueeze(0).expand(bucket, -1).contiguous()
        runs[name] = trace_prefill(wrapper, tensor, positions, cache)
    print("诊断前向: short_cached_padded_down_proj（仅补齐 MLP down_proj）", flush=True)
    runs["short_cached_padded_down_proj"] = trace_prefill(
        wrapper, short, buffers["positions"][0], buffers["cache"], pad_down_proj=True
    )
    comparisons = []
    for left, right in (
        ("short_no_cache", "full_no_cache"),
        ("short_cached", "short_no_cache"),
        ("short_cached", "full_no_cache"),
        ("short_cached_padded_down_proj", "full_no_cache"),
    ):
        details = [
            dict(module=name, **compare(value[:size], runs[right][name][:size])) for name, value in runs[left].items()
        ]
        row = dict(
            comparison=f"{left} vs {right}",
            first_difference=next((v for v in details if not v["exact_equal"]), None),
            logits=compare(runs[left]["head1.logits"][:size], runs[right]["head1.logits"][:size]),
            modules=details,
        )
        comparisons.append(row)
        print({k: v for k, v in row.items() if k != "modules"}, flush=True)
    saved_stock = compare(runs["full_no_cache"]["head1.logits"][:size], failure["stock_logits"][0][:size])
    print("本次长序列 eager vs 已保存 stock c1:", saved_stock, flush=True)
    exp.write_json(
        args.results / "diagnostic.json",
        dict(
            context=failure["context"],
            comparisons=comparisons,
            saved_stock=saved_stock,
            note="仅定位首次 prefill，未修改容差，不代表整帧正确性通过。",
        ),
    )
    torch.save(runs, args.results / "module_tensors.pt")
    print(f"诊断已保存: {args.results}", flush=True)


if __name__ == "__main__":
    main()
