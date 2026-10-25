# SPDX-License-Identifier: Apache-2.0
"""Same-codes growth_kv graph comparison; only SnakeBeta routing changes."""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "incremental_experiment", ROOT / "tools/qwen3_tts_code2wav_incremental_kv/experiment.py"
)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def configure(model, side):
    from vllm_omni.model_executor.models.common.snake_activation_npu import configure as apply

    apply(model, "eager" if side == "baseline" else "triton_ascend", layout="conservative")


def execute(model, wrapper, cohort, retain=False):
    caches = [dict(prefix_frames=0) for _ in cohort]
    audio = [[] for _ in cohort]
    calls = 0
    try:
        for position in range(max(len(r["chunks"]) for r in cohort)):
            ids = [i for i, r in enumerate(cohort) if position < len(r["chunks"])]
            waves = exp.run(model, wrapper, [cohort[i]["chunks"][position] for i in ids], [caches[i] for i in ids])
            if retain:
                for i, value in zip(ids, waves):
                    # Preserve graph outputs before the next replay reuses buffers.
                    audio[i].append(value.detach().clone())
            calls += 1
        return audio, calls
    finally:
        exp.release(wrapper, caches)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("check", "bench"))
    exp.common_args(parser)
    parser.add_argument("--input", type=Path, default=Path(exp.INPUT))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, choices=(1, 2, 3, 4), default=[1, 4])
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=2, help="Complete cohorts per side, outside timing")
    parser.add_argument("--repeats", type=int, default=1, help="ABBA repetitions of each complete cohort")
    args = parser.parse_args()
    if not args.device.startswith("npu") or args.warmup < 0 or args.repeats < 1:
        parser.error("Requires NPU, nonnegative warmup, positive repeats")
    if args.results.resolve() == args.input.resolve() or args.results.resolve() in args.input.resolve().parents:
        parser.error("Results must not overwrite capture")
    args.results.mkdir(parents=True, exist_ok=True)
    progress = exp.Progress(args.results)
    seq = exp.legacy.sequences(args)
    if not seq:
        parser.error("No completed requests")
    model, rate = exp.load_decoder(args)
    for request in seq:
        request["chunks"] = [c.to(args.device) for c in request["chunks"]]
    data = exp.manifest(args)
    data.update(
        baseline="growth_kv graph + eager SnakeBeta",
        candidate="growth_kv graph + conservative SnakeBeta",
        input_sha256=exp.input_identity(args),
        independent_requests=len(seq),
        policy="real-shape whitelist 2026-10-25",
        timing="complete cohort, includes request cache release",
    )
    for path in [
        Path(__file__),
        ROOT / "vllm_omni/model_executor/models/common/snake_activation_npu.py",
        ROOT / "vllm_omni/model_executor/models/common/snake_activation.py",
        ROOT / "vllm_omni/model_executor/models/qwen3_tts/code2wav_incremental_kv.py",
    ]:
        data["code"][str(path.relative_to(ROOT))] = exp.hashlib.sha256(path.read_bytes()).hexdigest()
    exp.write_json(args.results / "manifest.json", data)
    wrappers = {}
    for side in ("baseline", "candidate"):
        configure(model, side)
        progress.log(f"捕获 {side} growth_kv 图；JIT/捕获不计时")
        wrappers[side] = exp.make(args, model, "growth_kv", "graph")
    rows = []
    jobs = [(b, seq[start : start + b]) for b in args.batches for start in range(0, len(seq), b)]
    total = len(jobs)
    for job, (batch, cohort) in enumerate(jobs):
        progress.log(f"{args.command} {job + 1}/{total}: requested batch={batch}, actual={len(cohort)}")
        if args.command == "check":
            output = {}
            for side in wrappers:
                configure(model, side)  # Also selects the correct policy for short eager blocks.
                output[side], _ = execute(model, wrappers[side], cohort, retain=True)
            for i, request in enumerate(cohort):
                for pos, (a, b) in enumerate(zip(output["baseline"][i], output["candidate"][i])):
                    comparison = exp.compare(a, b, 1e-5, 0.016)
                    row = dict(batch=batch, actual_batch=len(cohort), request=request["id"], chunk=pos, **comparison)
                    rows.append(row)
                    if not comparison["passed"] and not (args.results / "first_difference.pt").exists():
                        torch.save(
                            dict(row=row, baseline=a.cpu(), candidate=b.cpu(), codes=request["chunks"][pos].cpu()),
                            args.results / "first_difference.pt",
                        )
                a, b = [torch.cat(output[side][i], -1) for side in ("baseline", "candidate")]
                rows.append(
                    dict(batch=batch, request=request["id"], chunk="concatenated", **exp.compare(a, b, 1e-5, 0.016))
                )
                for side, wave in (("baseline", a), ("candidate", b)):
                    exp.save_wav(args.results / "audio" / f"b{batch}-{request['id']}-{side}.wav", wave, rate)
        else:
            if job == 0 or jobs[job - 1][0] != batch:
                for side in wrappers:
                    configure(model, side)
                    for _ in range(args.warmup):
                        execute(model, wrappers[side], cohort)
            for repeat in range(args.repeats):
                for order, side in enumerate(("baseline", "candidate", "candidate", "baseline")):
                    configure(model, side)
                    torch.npu.synchronize()
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    start = time.perf_counter()
                    begin.record()
                    _, calls = execute(model, wrappers[side], cohort)
                    end.record()
                    torch.npu.synchronize()
                    rows.append(
                        dict(
                            job=job,
                            batch=batch,
                            actual_batch=len(cohort),
                            repeat=repeat,
                            order=order,
                            side=side,
                            calls=calls,
                            requests=[r["id"] for r in cohort],
                            wall_ms=(time.perf_counter() - start) * 1000,
                            device_ms=begin.elapsed_time(end),
                        )
                    )
        exp.write_json(args.results / "summary.json", rows)
        progress.update(job + 1, total, args.command)
    exp.write_json(args.results / "graphs.json", {s: w.statistics() for s, w in wrappers.items()})
    lines = [
        "# SnakeBeta 整段 Code2Wav 对照",
        "",
        "两组均为 growth_kv graph，仅切换 SnakeBeta。数值检查与性能独立；不是完整 TTS 服务计时。",
        "",
    ]
    if args.command == "check":
        lines.append(
            f"逐块与拼接波形检查：{sum(r['passed'] for r in rows)}/{len(rows)} 通过；BF16 atol=1e-5, rtol=0.016。"
        )
    else:
        lines += [
            "每个样本是完整请求组，含首块、增长、滚动、短块、状态交接及释放。codes 已在设备；不含 CPU→NPU 采集加载。",
            "预热单位为完整请求组；设备事件不等于纯 kernel 时间。尾组不复制请求，实际 batch 单独记录。",
            "",
            "|请求组 batch|baseline 总 ms|候选总 ms|加速比|候选组 p50/p95 ms|",
            "|---:|---:|---:|---:|---|",
        ]
        for batch in args.batches:
            a, b = [[r["wall_ms"] for r in rows if r["batch"] == batch and r["side"] == side] for side in wrappers]
            lines.append(
                f"|{batch}|{sum(a):.3f}|{sum(b):.3f}|{sum(a) / sum(b):.3f}|{np.percentile(b, [50, 95]).tolist()}|"
            )
    (args.results / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress.log(f"完成：{args.results / 'report.md'}")


if __name__ == "__main__":
    main()
