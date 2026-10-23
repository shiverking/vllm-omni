# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced, module-level diagnosis at the failed AR codebook position."""

import argparse
from pathlib import Path

import experiment as exp
import torch
from diagnose_prefill import compare


@torch.inference_mode()
def trace_position(wrapper, inputs, positions, codebook, target, cache=None, cache_position=0):
    snapshots, handles = {}, []
    batch, length = inputs.shape[:2]

    def save(name, value):
        if torch.is_tensor(value) and value.ndim >= 3 and value.shape[:2] == (batch, length):
            snapshots[name] = value[:, target : target + 1].detach().clone().cpu()

    for name, module in wrapper.model.named_modules():
        if name and not list(module.children()):
            handles.append(
                module.register_forward_pre_hook(lambda mod, args, name=name: save(name + ".input", args[0]))
            )
            handles.append(module.register_forward_hook(lambda mod, args, out, name=name: save(name + ".output", out)))
    try:
        hidden = wrapper.model(inputs, positions, cache, cache_position)
        snapshots["model.output"] = hidden[:, target : target + 1].detach().clone().cpu()
        snapshots["head.logits"] = wrapper.lm_head[codebook - 1](hidden[:, target]).detach().clone().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return snapshots


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", default=exp.MODEL)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=False)
    failure = torch.load(args.failure, map_location="cpu", weights_only=True)
    codebook = failure["context"]["position"]
    if not 2 <= codebook <= 15:
        raise ValueError("This tool requires a failed decode position c2..c15")
    wrapper = exp.load_predictor(args, exp.capture_root(args.input))
    from vllm_omni.model_executor.models.common.qwen3_code_predictor_ar import CachedARExecutor

    executor = CachedARExecutor(wrapper, "eager")
    size = failure["inputs"]["layer0_code"].shape[0]
    bucket = next(b for b in exp.BUCKETS if b >= size)
    b = executor._allocate(bucket)
    for target, source in (("hidden", "last_talker_hidden"), ("embed", "layer0_embed")):
        b[target][:size].copy_(failure["inputs"][source].reshape(size, 1, -1))
    executor._project(b)
    # Force all compared paths to use the saved stock tokens, never their own samples.
    for index in range(1, codebook):
        codes = torch.zeros(bucket, device=args.device, dtype=torch.long)
        codes[:size].copy_(failure["expected"][:, index])
        embedded = wrapper.model.codec_embedding[index - 1](codes).unsqueeze(1)
        b["inputs"][:, index + 1 : index + 2].copy_(wrapper.small_to_mtp_projection(embedded))
    for step in range(codebook - 1):
        executor._backbone(b, step)
    full = torch.zeros(
        bucket, wrapper._num_groups + 1, b["inputs"].shape[-1], device=args.device, dtype=b["inputs"].dtype
    )
    full[:, : codebook + 1].copy_(b["inputs"][:, : codebook + 1])
    runs = {}
    for name, tensor, target, cache, offset in (
        ("full_no_cache", full, codebook, None, 0),
        ("prefix_no_cache", full[:, : codebook + 1], codebook, None, 0),
        ("single_cached", b["inputs"][:, codebook : codebook + 1], 0, b["cache"], codebook),
    ):
        print(f"诊断 c{codebook}: {name}, shape={tuple(tensor.shape)}", flush=True)
        positions = torch.arange(offset, offset + tensor.shape[1], device=args.device)
        positions = positions.unsqueeze(0).expand(bucket, -1).contiguous()
        runs[name] = trace_position(wrapper, tensor, positions, codebook, target, cache, offset)
    comparisons = []
    for left, right in (
        ("prefix_no_cache", "full_no_cache"),
        ("single_cached", "prefix_no_cache"),
        ("single_cached", "full_no_cache"),
    ):
        details = [
            dict(module=name, **compare(value[:size], runs[right][name][:size]))
            for name, value in runs[left].items()
            if name in runs[right]
        ]
        row = dict(
            comparison=f"{left} vs {right}",
            first_difference=next((v for v in details if not v["exact_equal"]), None),
            logits=compare(runs[left]["head.logits"][:size], runs[right]["head.logits"][:size]),
            modules=details,
        )
        comparisons.append(row)
        print({k: v for k, v in row.items() if k != "modules"}, flush=True)
    saved_stock = compare(runs["full_no_cache"]["head.logits"][:size], failure["stock_logits"][codebook - 1][:size])
    print("本次长序列 eager vs 已保存 stock logits:", saved_stock, flush=True)
    exp.write_json(
        args.results / "diagnostic.json",
        dict(
            context=failure["context"],
            comparisons=comparisons,
            saved_stock=saved_stock,
            note="固定 stock 前缀；仅诊断，不授权性能测试。",
        ),
    )
    torch.save(dict(runs=runs, cache=tuple(t.cpu() for t in b["cache"])), args.results / "module_tensors.pt")
    print(f"诊断已保存: {args.results}", flush=True)


if __name__ == "__main__":
    main()
