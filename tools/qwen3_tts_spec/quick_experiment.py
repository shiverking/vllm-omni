# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sampled code correctness and paired timing; never a full replay acceptance."""

import argparse
import json
from pathlib import Path

import experiment as exp
import numpy as np
import torch
from diagnose_fusion import select_request_frames


def summarize(rows):
    results = {}
    for config, comparator in sorted({(r["config"], r["comparator"]) for r in rows}):
        paired = [r for r in rows if (r["config"], r["comparator"]) == (config, comparator)]
        baseline = [r for r in paired if r["kind"] == comparator]
        candidate = [r for r in paired if r["kind"] == config]

        def metrics(items):
            walls = [r["wall_ms"] for r in items]
            devices = [r["device_ms"] for r in items if r["device_ms"] is not None]
            return dict(
                executions=len(items),
                wall_p50_ms=float(np.median(walls)),
                wall_p95_ms=float(np.quantile(walls, 0.95)),
                wall_total_ms=sum(walls),
                device_p50_ms=float(np.median(devices)) if devices else None,
                device_p95_ms=float(np.quantile(devices, 0.95)) if devices else None,
                calls_per_frame=float(np.mean([r["backbone_calls"] for r in items])),
                effective_tokens_per_call=sum(r["residual_tokens"] for r in items)
                / sum(r["backbone_calls"] for r in items),
            )

        requests = sorted({r["request_id"] for r in paired})
        a = np.array([sum(r["wall_ms"] for r in baseline if r["request_id"] == rid) for rid in requests])
        b = np.array([sum(r["wall_ms"] for r in candidate if r["request_id"] == rid) for rid in requests])
        picks = np.random.default_rng(20261020).integers(0, len(requests), size=(2000, len(requests)))
        ratios = a[picks].sum(1) / b[picks].sum(1)
        results[f"{config}/{comparator}"] = dict(
            speedup=float(a.sum() / b.sum()),
            exploratory_request_bootstrap95=np.quantile(ratios, [0.025, 0.975]).tolist(),
            baseline=metrics(baseline),
            candidate=metrics(candidate),
        )
    return results


@torch.inference_mode()
def run(args):
    if min(args.frames_per_request, args.warmup, args.executions_per_round) <= 0:
        raise ValueError("frame count, warmup and executions must be positive")
    # A dedicated, new result directory prevents any mixing with full replay.
    args.results.mkdir(parents=True, exist_ok=False)
    progress = exp.ReplayProgress(args.results, args.progress_interval)
    status = dict(scope="sampled_exploration_not_full_acceptance", status="running", node_audit=False)
    exp.write_json(args.results / "status.json", status)
    try:
        execute(args, progress, status)
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        exp.write_json(args.results / "status.json", status)
        progress.message(f"Stopped: {exc}")
        raise


def execute(args, progress, status):
    frozen = json.loads((args.output / "trees.json").read_text(encoding="utf-8"))
    signature = frozen.pop("sha256")
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    if exp.digest(frozen) != signature or frozen["dataset_sha256"] != exp.digest(manifest["dataset"]):
        raise ValueError("frozen tree or dataset digest changed")
    configs = {"reference": [], "chain": frozen["configs"]["chain"], "tree16": frozen["configs"]["tree16"]}
    progress.message("Loading fixed sampled frames and predictor; using the integrated backend, no diagnostic patch.")
    selected = select_request_frames(exp.frames(args, "test"), args.frames_per_request)
    if not selected:
        raise ValueError("no sampled test frames")
    if args.executions_per_round < len(selected):
        raise ValueError("executions-per-round must cover every selected frame at least once")
    model, spec = exp.load_predictor(args), exp.spec_module()
    device = torch.device(args.device)
    api = exp.api_for(device)
    metadata = dict(
        **status,
        source_output=str(args.output.resolve()),
        tree_sha256=signature,
        environment=exp.environment(args),
        script_sha256=exp.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dtype=str(next(model.parameters()).dtype),
        attention_backend=spec.attention_backend(device),
        selected_frames=[dict(request_id=f["request_id"], frame_id=f["frame_id"]) for f in selected],
        unique_frames=len(selected),
        unique_requests=len({f["request_id"] for f in selected}),
        frames_per_request=args.frames_per_request,
        configs=configs,
        warmup_per_config=args.warmup,
        executions_per_round=args.executions_per_round,
        sampling="greedy" if args.greedy else "shared_uniforms",
        source_manifest=manifest,
    )
    exp.write_json(args.results / "manifest.json", metadata)
    checks = []
    progress.begin("sampled code correctness (no serial node audit)", len(selected) * 2 * len(configs))
    for frame in selected:
        for greedy in (True, False):
            inputs = exp.inputs_for(args, model, frame, greedy)
            stock = model(**inputs).clone()
            if not greedy and not torch.equal(stock.cpu(), frame["codes"]):
                failure = dict(
                    request_id=frame["request_id"],
                    frame_id=frame["frame_id"],
                    config="stock_export_replay",
                    greedy=False,
                    code_equal=False,
                    stock_codes=stock.cpu().tolist(),
                    captured_codes=frame["codes"].tolist(),
                )
                checks.append(failure)
                exp.write_json(args.results / "correctness.json", dict(passed=False, node_audit=False, checks=checks))
                raise RuntimeError("stock replay differs from captured codes; performance not started")
            for name, paths in configs.items():
                codes, _ = spec.decode(model, **inputs, paths=paths)
                row = dict(
                    request_id=frame["request_id"],
                    frame_id=frame["frame_id"],
                    greedy=greedy,
                    config=name,
                    code_equal=torch.equal(codes, stock),
                )
                checks.append(row)
                if not row["code_equal"]:
                    row.update(
                        positions=(codes != stock).nonzero().cpu().tolist(),
                        stock_codes=stock.cpu().tolist(),
                        candidate_codes=codes.cpu().tolist(),
                    )
                    exp.write_json(
                        args.results / "correctness.json", dict(passed=False, node_audit=False, checks=checks)
                    )
                    raise RuntimeError(f"code mismatch: {row}; performance not started")
                progress.advance(
                    f"request={frame['request_id']} frame={frame['frame_id']} config={name} greedy={greedy}"
                )
        exp.write_json(
            args.results / "correctness.json", dict(passed=False, partial=True, node_audit=False, checks=checks)
        )
    exp.write_json(args.results / "correctness.json", dict(passed=True, node_audit=False, checks=checks))
    progress.message("Sampled codes PASSED. Preparing inputs and warming all four configurations.")
    prepared = [exp.inputs_for(args, model, frame, args.greedy) for frame in selected]

    def decode(kind, inputs):
        return model(**inputs) if kind == "stock" else spec.decode(model, **inputs, paths=configs[kind])

    for kind in ("stock", *configs):
        progress.begin(f"warmup {kind}", args.warmup)
        for i in range(args.warmup):
            decode(kind, prepared[i % len(prepared)])
            progress.advance()
    if api:
        api.synchronize()
        api.reset_peak_memory_stats()
    rows = []
    groups = [
        (name, comparator)
        for name in configs
        for comparator in (("stock",) if name == "reference" else ("stock", "reference"))
    ]
    for group, (name, comparator) in enumerate(groups, 1):
        for round_id, kind in enumerate((comparator, name, name, comparator)):
            progress.begin(
                f"timing group={group}/{len(groups)} {name}/{comparator} round={round_id + 1}/4 {kind}",
                args.executions_per_round,
            )
            round_rows = []
            for execution in range(args.executions_per_round):
                index = execution % len(prepared)
                result, wall, event = exp.measured(lambda: decode(kind, prepared[index]), device)
                row = dict(
                    config=name,
                    comparator=comparator,
                    kind=kind,
                    round=round_id,
                    execution=execution,
                    request_id=selected[index]["request_id"],
                    frame_id=selected[index]["frame_id"],
                    wall_ms=wall,
                    device_ms=event,
                    residual_tokens=model._num_groups - 1,
                    backbone_calls=model._num_groups - 1 if kind == "stock" else result[1]["backbone_calls"],
                    attention_backend=("npu_fusion_attention_sparse_mode2" if device.type == "npu" else "torch_sdpa")
                    if kind == "stock"
                    else spec.attention_backend(device),
                )
                round_rows.append(row)
                progress.advance()
            # Flush only after each round, always outside frame timing.
            with (args.results / "timings.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("".join(json.dumps(r) + "\n" for r in round_rows))
            rows.extend(round_rows)
    summary = dict(
        scope=status["scope"],
        sampled_code_correctness=True,
        node_audit=False,
        unique_frames=len(selected),
        total_timed_executions=len(rows),
        performance=summarize(rows),
        attention_backend=spec.attention_backend(device),
        peak_memory_bytes=api.max_memory_allocated() if api else None,
        graph=dict(stock_buckets=list(model._device_graphs), reference="eager", tree="eager"),
        note="抽样探索结果；未执行全量节点审计或完整请求测试，不构成最终验收。",
    )
    exp.write_json(args.results / "summary.json", summary)
    lines = [
        "# Qwen3-TTS 抽样功能与性能实验",
        "",
        summary["note"],
        "",
        f"独立帧：{len(selected)}；计时帧执行：{len(rows)}；codes 检查：{len(checks)} 项通过。",
        "",
        "| 配置/对照 | 墙钟加速比 | 探索性配对请求 bootstrap 95% 区间 | 候选 p50/p95 ms | 候选主干调用/帧 |",
        "|---|---:|---|---|---:|",
    ]
    for name, item in summary["performance"].items():
        c = item["candidate"]
        lines.append(
            f"| {name} | {item['speedup']:.3f} | {item['exploratory_request_bootstrap95']} | "
            f"{c['wall_p50_ms']:.3f}/{c['wall_p95_ms']:.3f} | {c['calls_per_frame']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"实验 attention：`{summary['attention_backend']}`。加速比大于 1 表示候选更快。",
            "CPU 运行仅用于工具验证，不能证明 NPU 加速。性能轮关闭逐节点审计和详细事件采集。",
        ]
    )
    (args.results / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status.update(status="complete", sampled_code_correctness=True, total_timed_executions=len(rows))
    exp.write_json(args.results / "status.json", status)
    progress.message(f"Sampled experiment complete: {args.results / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing capture directory (read only)")
    parser.add_argument("--results", type=Path, required=True, help="new directory for sampled results")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--frames-per-request", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--executions-per-round", type=int, default=200)
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--greedy", action="store_true", help="time greedy instead of shared-uniform sampling")
    parser.add_argument("--vllm-repo", type=Path, default=exp.ROOT.parent / "vllm")
    parser.add_argument("--ascend-repo", type=Path, default=exp.ROOT.parent / "vllm-ascend")
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    run(args)
