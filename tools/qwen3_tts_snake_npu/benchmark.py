# SPDX-License-Identifier: Apache-2.0
"""Independent NPU microbenchmark; numerical differences never gate timing."""

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import time
from itertools import product
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "snake_npu", ROOT / "vllm_omni/model_executor/models/common/snake_activation_npu.py"
)
snake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snake)


def shapes_from_csv(path):
    shapes = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("Type") != "Sin":
                continue
            shape = tuple(int(v) for v in row["Input Shapes"].strip('"').split(","))
            if len(shape) == 3:
                shapes.add(shape)
    return sorted(shapes)


def difference(a, b):
    delta = (a.float() - b.float()).abs()
    return dict(
        finite=bool(torch.isfinite(b).all()),
        exact=bool(torch.equal(a, b)),
        max_abs=delta.max().item(),
        mean_abs=delta.mean().item(),
        outside_tolerance=int((~torch.isclose(a, b)).sum().item()),
    )


def measure(call, iterations):
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    start = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.npu.synchronize()
    return dict(
        wall_ms=(time.perf_counter() - start) * 1000 / iterations, device_ms=begin.elapsed_time(end) / iterations
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layouts", nargs="+", choices=["channel", "flat", "auto"], default=["channel"])
    parser.add_argument("--blocks", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--programs", nargs="+", type=int, default=[16, 32, 40])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--shapes", nargs="+", help="Explicit B,C,T triples, e.g. 4,1024,37")
    parser.add_argument("--kernel-csv", type=Path)
    parser.add_argument("--inputs", type=Path, help="Trusted local .pt: list of {x, alpha, inv_beta} real activations")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--executions", nargs="+", choices=["eager", "graph"], default=["eager", "graph"])
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("iterations and warmup must be positive")
    import torch_npu  # noqa: F401
    import triton

    torch.npu.set_device(args.device)
    args.results.mkdir(parents=True, exist_ok=True)
    versions = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": args.device,
        "device_name": torch.npu.get_device_name(),
        "args": vars(args),
    }
    for package in ("torch-npu", "triton-ascend"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    (args.results / "manifest.json").write_text(json.dumps(versions, default=str, indent=2), encoding="utf-8")
    shapes = (
        shapes_from_csv(args.kernel_csv)
        if args.kernel_csv
        else [(b, c, t) for b in args.batches for c, t in ((96, 1920), (1024, 37), (96, 71040))]
    )
    if args.shapes:
        shapes = [tuple(int(v) for v in shape.split(",")) for shape in args.shapes]
        if any(len(shape) != 3 or min(shape) < 1 for shape in shapes):
            parser.error("shapes must be positive B,C,T triples")
    if args.inputs and args.inputs.is_dir():
        samples = [args.inputs / row["file"] for row in json.loads((args.inputs / "index.json").read_text())]
    else:
        samples = torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else shapes
    rows = []
    for index, sample in enumerate(samples):
        if isinstance(sample, Path):
            sample = torch.load(sample, map_location="cpu", weights_only=True)
        if args.inputs:
            x, alpha, inv_beta = (sample[k].to(args.device) for k in ("x", "alpha", "inv_beta"))
        else:
            generator = torch.Generator().manual_seed(2026 + index)
            x = torch.randn(sample, generator=generator).to(device=args.device, dtype=torch.bfloat16)
            alpha = torch.randn(sample[1], generator=generator).exp().to(device=args.device, dtype=x.dtype)
            inv_beta = torch.randn(sample[1], generator=generator).exp().to(device=args.device, dtype=x.dtype)
        ref = snake.reference(x, alpha, inv_beta)
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        original = x.clone()
        configurations = [
            (layout, block, programs, execution)
            for layout in args.layouts
            for block, programs in ([(2048, 32)] if layout == "auto" else product(args.blocks, args.programs))
            for execution in args.executions
        ]
        for layout, block, programs, execution in configurations:
            row = dict(
                shape=list(x.shape),
                layout=layout,
                selected=snake.select_config(x.shape, layout, block, programs),
                block=block,
                programs=programs,
                execution=execution,
                input_source="real" if args.inputs else "synthetic",
            )
            try:
                x.copy_(original)

                def candidate():
                    return snake.fused_snake(x, alpha, inv_beta, out=out, block=block, programs=programs, layout=layout)

                def baseline():
                    return snake.reference(x, alpha, inv_beta)

                for _ in range(args.warmup):
                    candidate()
                    baseline()
                torch.npu.synchronize()
                graphs = []
                if execution == "graph":
                    for call in (baseline, candidate):
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            value = call()
                        graphs.append((graph, value))
                    baseline, candidate = graphs[0][0].replay, graphs[1][0].replay
                candidate()
                row["difference"] = difference(ref, out)
                if execution == "graph":
                    x.mul_(0.5)
                    candidate()
                    row["updated_input_difference"] = difference(snake.reference(x, alpha, inv_beta), out)
                    x.copy_(original)
                # ABBA, compilation/capture and correctness outside timing.
                row["timings"] = [
                    dict(side=side, **measure(call, args.iterations))
                    for side, call in (
                        ("baseline", baseline),
                        ("candidate", candidate),
                        ("candidate", candidate),
                        ("baseline", baseline),
                    )
                ]
                totals = {
                    side: sum(v["wall_ms"] for v in row["timings"] if v["side"] == side)
                    for side in ("baseline", "candidate")
                }
                row["speedup"] = totals["baseline"] / totals["candidate"]
            except Exception as error:
                row["error"] = repr(error)
            finally:
                # Captured output pointers remain alive until both graphs are released.
                graphs = []
            rows.append(row)
            (args.results / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in row.items() if k != "timings"}), flush=True)
    lines = [
        "# Triton-Ascend SnakeBeta 微基准",
        "",
        "独立算子实验；合成输入不代表真实波形正确性。",
        "设备事件包含提交间隙；捕获/JIT 不计时。数值差异不拦截其他测试。",
        "",
        "|形状|分块/programs|执行|加速比|最大误差|",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"|{row['shape']}|{row['selected']}|{row['execution']}|"
            f"{row.get('speedup', row.get('error'))}|{row.get('difference', {}).get('max_abs')}|"
        )
    (args.results / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
