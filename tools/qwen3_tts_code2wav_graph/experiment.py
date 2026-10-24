# SPDX-License-Identifier: Apache-2.0
"""Independent Code2Wav cache/graph experiments with kv_frame fixed."""

from __future__ import annotations

import argparse
import io
import json
import time
import traceback
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from runtime import RESULTS, Progress, common_args, manifest, read_json, server, speech, texts, write_json


def collect(args, progress):
    raw = args.results / "chunks"
    raw.mkdir(parents=True, exist_ok=True)
    records = read_json(args.results / "requests.json") if (args.results / "requests.json").exists() else []
    done = {r["id"] for r in records if r.get("complete")}
    selected = texts()[: args.limit or 30]
    with server(args, "baseline", args.results / "server", progress, capture=raw) as url:
        for i, row in enumerate(selected):
            if row["id"] in done:
                progress.update(i + 1, len(selected), "采集（跳过已完成）")
                continue
            started = time.monotonic()
            before = set(raw.glob("*.pt"))
            item = dict(row, complete=False)
            try:
                audio = speech(url, args.model, row["text"])
                with wave.open(io.BytesIO(audio)) as wav:
                    rate, samples = wav.getframerate(), wav.getnframes()
                paths = sorted(set(raw.glob("*.pt")) - before)
                if not paths:
                    raise RuntimeError("服务未保存 Code2Wav 分块；检查部署采集开关")
                data = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
                ids = {r["request_id"] for d in data for r in d["rows"]}
                if len(ids) != 1 or any(len(d["rows"]) != 1 for d in data):
                    raise RuntimeError("单请求采集出现多个内部请求，拒绝错误关联")
                target = args.results / "audio" / (row["id"] + ".wav")
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(audio)
                item.update(
                    complete=True,
                    chunks=[str(p.relative_to(args.results)) for p in paths],
                    internal_request_id=next(iter(ids)),
                    audio=str(target.relative_to(args.results)),
                    sample_rate=rate,
                    duration_s=samples / rate,
                    final_finished_flag=bool(data[-1]["rows"][0]["finished"]),
                )
            except Exception as exc:
                item["error"] = repr(exc)
                progress.log(f"{row['id']} 采集失败：{exc}")
            item["wall_s"] = time.monotonic() - started
            records = [r for r in records if r["id"] != row["id"]] + [item]
            write_json(args.results / "requests.json", records)
            progress.update(i + 1, len(selected), "真实音频采集")
    write_json(
        args.results / "collection.json",
        dict(
            completed=sum(r.get("complete", False) for r in records),
            attempted=len(records),
            note="采集期间 Stage 1 外层图关闭以执行记录钩子；正式基线配置保持原样",
        ),
    )


def sequences(args):
    rows = read_json(args.input / "requests.json")
    result = []
    for row in rows:
        if not row.get("complete"):
            continue
        chunks = []
        for name in row["chunks"]:
            artifact = torch.load(args.input / name, map_location="cpu", weights_only=True)
            for item in artifact["rows"]:
                if item["request_id"] != row["internal_request_id"]:
                    raise ValueError("采集请求关联不一致")
                chunks.append(item["codes"].unsqueeze(0))
        result.append(dict(id=row["id"], chunks=chunks))
    if not result:
        raise ValueError("没有完整采集请求")
    return result[:3] if args.probe_only else result


def setup(args, progress):
    from engine import eager_layout_class, load_decoder

    model, sr = load_decoder(args)
    if not args.device.startswith("npu"):
        raise ValueError("正式 check/bench 必须在 NPU 运行；CPU 使用 test_cpu.py")
    from vllm_omni.model_executor.models.qwen3_tts.npu_segmented_graph_wrapper import NPUGraphDecoderWrapper

    wrapper = NPUGraphDecoderWrapper(model)
    progress.log("捕获 Code2Wav 首块、增长和滚动图")
    try:
        wrapper.warmup(torch.device(args.device))
    except Exception as exc:
        write_json(
            args.results / "capture_failure.json",
            dict(error=repr(exc), traceback=traceback.format_exc(), graph=wrapper.statistics()),
        )
        raise
    write_json(
        args.results / "device.json",
        dict(
            device=args.device,
            name=torch.npu.get_device_name(torch.device(args.device)),
            dtype=str(next(model.parameters()).dtype),
        ),
    )
    if args.command == "check":
        reference = eager_layout_class()(model)
        reference.warmup(torch.device(args.device))
    else:
        reference = None
    return model, sr, wrapper, reference


def batches(seq, batch):
    # Repeated rows for a final partial cohort are explicitly identified.
    for start in range(0, len(seq), batch):
        cohort = [seq[(start + i) % len(seq)] for i in range(batch)]
        for position in range(max(len(r["chunks"]) for r in cohort)):
            active = [(i, r) for i, r in enumerate(cohort) if position < len(r["chunks"])]
            yield start, position, active


def compare_states(left, right, atol, rtol):
    from engine import compare, cpu_cache

    a, b = cpu_cache(left), cpu_cache(right)
    comparisons = {}
    for key in a.keys() | b.keys():
        x, y = a.get(key), b.get(key)
        comparisons[key] = (
            compare(x, y, atol, rtol)
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
            else {"passed": False if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) else x == y}
        )
    return comparisons


@torch.no_grad()
def check(args, progress):
    from engine import compare, cpu_cache, run

    seq = sequences(args)
    model, sr, wrapper, reference = setup(args, progress)
    # Probe always exercises rolling, even if real utterances are short.
    if args.probe_only:
        gen = torch.Generator().manual_seed(42)
        seq += [
            dict(
                id="synthetic-boundaries",
                chunks=[
                    torch.randint(model.config.codebook_size, (1, model.config.num_quantizers, n), generator=gen)
                    for n in (1, 25, 25, 25, 25, 3, 25, 2)
                ],
            )
        ]
    # Document the actual PyTorch BF16 defaults, rather than widening tolerances.
    atol, rtol = 1e-5, 1.6e-2
    checks, layout_differences, single_differences = [], [], []
    total = sum(sum(1 for _ in batches(seq, b)) for b in (1, 2, 3, 4))
    done = 0
    failure_written = False
    context = {}
    try:
        for batch in (1, 2, 3, 4):
            caches, refs, baseline, singles = {}, {}, {}, {}
            accumulated = defaultdict(lambda: ([], []))
            for start, position, active in batches(seq, batch):
                keys = [(start, i) for i, _ in active]
                codes = [r["chunks"][position].to(args.device) for _, r in active]
                for mapping in (caches, refs, baseline, singles):
                    for key in keys:
                        mapping.setdefault(key, dict(prefix_frames=0))
                row = dict(batch=batch, position=position, requests=[r["id"] for _, r in active])
                context = dict(row=row, codes=[x.cpu() for x in codes], caches=[cpu_cache(caches[k]) for k in keys])
                # Compare the same starting histories; each executor retains its
                # own state afterwards, so state drift cannot be reset away.
                actual = run(model, wrapper, codes, [caches[k] for k in keys])
                expected = run(model, reference, codes, [refs[k] for k in keys])
                stock = run(model, None, codes, [baseline[k] for k in keys])
                one = [model.chunked_decode(x, singles[k])[0] for x, k in zip(codes, keys)]
                outputs = [compare(a, b, atol, rtol) for a, b in zip(actual, expected)]
                states = [compare_states(caches[k], refs[k], atol, rtol) for k in keys]
                passed = all(x["passed"] for x in outputs) and all(x["passed"] for s in states for x in s.values())
                boundary = [
                    dict(
                        head=compare(a[..., : model.total_upsample], b[..., : model.total_upsample], atol, rtol),
                        tail=compare(a[..., -model.total_upsample :], b[..., -model.total_upsample :], atol, rtol),
                    )
                    for a, b in zip(actual, expected)
                ]
                row.update(passed=passed, waveform=outputs, boundary=boundary, state=states)
                checks.append(row)
                for k, a, b, s, independent in zip(keys, actual, expected, stock, one):
                    accumulated[k][0].append(a.cpu())
                    accumulated[k][1].append(b.cpu())
                    for target, x, y in ((layout_differences, a, s), (single_differences, s, independent)):
                        delta = compare(x, y, atol, rtol)
                        if not delta["exact"]:
                            target.append(dict(row, comparison=delta))
                if not passed:
                    torch.save(
                        dict(
                            context=row,
                            codes=[x.cpu() for x in codes],
                            actual=[x.cpu() for x in actual],
                            expected=[x.cpu() for x in expected],
                            caches=[cpu_cache(caches[k]) for k in keys],
                            reference=[cpu_cache(refs[k]) for k in keys],
                            graph=wrapper.statistics(),
                        ),
                        args.results / "failure.pt",
                    )
                    failure_written = True
                    raise AssertionError("同形状 eager 与 NPU graph 不一致，已保存 failure.pt")
                done += 1
                if done % 10 == 0 or done == total:
                    progress.update(done, total, "逐块波形与缓存检查")
            for key, (a, b) in accumulated.items():
                metric = compare(torch.cat(a, -1), torch.cat(b, -1), atol, rtol)
                if not metric["passed"]:
                    raise AssertionError(f"拼接波形不一致：{key}")
        passed, error = True, None
    except Exception as exc:
        passed, error = False, repr(exc)
        if not failure_written:
            torch.save(dict(context=context, error=error, graph=wrapper.statistics()), args.results / "failure.pt")
    write_json(
        args.results / "correctness.json",
        dict(
            passed=passed,
            error=error,
            atol=atol,
            rtol=rtol,
            sampling_rate=sr,
            independent_requests=len(seq),
            probe_only=args.probe_only,
            checks=checks,
            graph_vs_native_batch_differences=layout_differences,
            native_batch_vs_single_differences=single_differences,
            graph=wrapper.statistics(),
            note="同形状 graph/eager 检查；跨 batch/原生分组差异另列，未放宽容差",
        ),
    )
    progress.log(f"检查 {'通过' if passed else '失败'}；其他命令不依赖本结果。")
    if not passed:
        raise RuntimeError(error)


def phase(cache, codes, window):
    if "suffix_quantized" not in cache:
        return "first"
    if codes.shape[-1] != 25:
        return "short"
    return "rolling" if cache["suffix_quantized"].shape[-1] >= window else "growing"


def make_jobs(seq, batch, limit):
    jobs = []
    for start, position, active in batches(seq, batch):
        if len(active) != batch:
            continue
        jobs.append(dict(id=len(jobs), rows=[r for _, r in active], position=position))
    if not jobs:
        raise ValueError("没有可计时的分块")
    # Spread across the full corpus rather than tuning/selecting easy stages.
    indexes = np.linspace(0, len(jobs) - 1, min(limit, len(jobs)), dtype=int)
    return [jobs[i] for i in indexes]


def prepare_job(model, job, device):
    caches, codes = [], []
    for row in job["rows"]:
        cache = dict(prefix_frames=0)
        for x in row["chunks"][: job["position"]]:
            model.chunked_decode(x.to(device), cache)
        caches.append(cache)
        codes.append(row["chunks"][job["position"]].to(device))
    return codes, caches


def measured(model, wrapper, codes, caches):
    from engine import run

    torch.npu.synchronize()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    tick = time.perf_counter()
    begin.record()
    output = run(model, wrapper, codes, caches)
    # Graph paths already detach output from static buffers. Eager outputs
    # independently own their storage. Keep outputs alive through completion.
    end.record()
    torch.npu.synchronize()
    return (time.perf_counter() - tick) * 1000, begin.elapsed_time(end), output


@torch.no_grad()
def bench(args, progress):
    seq = sequences(args)
    model, sr, wrapper, _ = setup(args, progress)
    rows = []
    graph_meta = wrapper.statistics()
    torch.npu.reset_peak_memory_stats()
    total = len(args.batches) * 4 * args.executions_per_round
    done = 0
    for batch in args.batches:
        jobs = make_jobs(seq, batch, args.executions_per_round)
        for mode, executor in (("baseline", None), ("code2wav_graph", wrapper)):
            progress.log(f"batch={batch} {mode} 预热 {args.warmup} 次")
            for i in range(args.warmup):
                codes, caches = prepare_job(model, jobs[i % len(jobs)], args.device)
                measured(model, executor, codes, caches)
        wrapper.routes.clear()
        for round_id, mode in enumerate(("baseline", "code2wav_graph", "code2wav_graph", "baseline")):
            for i in range(args.executions_per_round):
                job = jobs[i % len(jobs)]
                codes, caches = prepare_job(model, job, args.device)
                labels = {phase(c, x, model.config.sliding_window) for c, x in zip(caches, codes)}
                label = next(iter(labels)) if len(labels) == 1 else "mixed"
                wall, event, output = measured(model, wrapper if mode == "code2wav_graph" else None, codes, caches)
                rows.append(
                    dict(
                        batch=batch,
                        actual_batch=len(codes),
                        round=round_id,
                        execution=i,
                        job=job["id"],
                        mode=mode,
                        phase=label,
                        wall_ms=wall,
                        device_ms=event,
                        codec_frames=sum(x.shape[-1] for x in codes),
                        audio_seconds=sum(x.numel() for x in output) / sr,
                    )
                )
                done += 1
                if done % 20 == 0 or done == total:
                    progress.update(done, total, "Code2Wav ABBA 计时")
        graph_meta.setdefault("per_batch_routes", {})[str(batch)] = wrapper.statistics()["routes"]
        (args.results / "timings.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    graph_meta["peak_memory_bytes"] = torch.npu.max_memory_allocated()
    graph_meta["independent_requests"] = len(seq)
    write_json(args.results / "graph.json", graph_meta)
    report(args, progress)
    if args.profile:
        profiling(args, model, wrapper, seq, progress)


def profiling(args, model, wrapper, seq, progress):
    import torch_npu

    wrapper.profile_phases = True
    # A separate trace run, excluded from all timing summaries.
    for mode, executor in (("baseline", None), ("code2wav_graph", wrapper)):
        directory = args.results.parent / "profile" / mode
        directory.mkdir(parents=True, exist_ok=True)
        jobs = make_jobs(seq, 4, 6)
        prepared = [prepare_job(model, j, args.device) for j in jobs]
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        ) as profiler:
            for codes, caches in prepared:
                measured(model, executor, codes, caches)
                profiler.step()
        events = []
        for event in profiler.key_averages():
            events.append(
                dict(
                    operator=event.key,
                    count=event.count,
                    self_cpu_us=event.self_cpu_time_total,
                    self_device_us=getattr(event, "self_device_time_total", None),
                    total_cpu_us=event.cpu_time_total,
                )
            )
        write_json(
            directory / "operators.json",
            dict(
                batches=len(prepared),
                events=events,
                note=(
                    "独立诊断；input_staging 与 cache_and_output_handoff 是主机范围；"
                    "设备耗时看 trace，不可与整体重复相加"
                ),
            ),
        )
        progress.log(f"独立 profiling 已保存：{directory}")


def report(args, progress):
    path = args.results / "timings.jsonl"
    if not path.exists():
        raise ValueError("report 只需要当前结果目录中的 timings.jsonl")
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["batch"], row["phase"], row["mode"])].append(row)
        groups[(row["batch"], "all", row["mode"])].append(row)
    result = []
    lines = [
        "# kv_frame 基线：Code2Wav 缓存＋NPU graph",
        "",
        "独立性能测试，不读取正确性证书。设备事件区间不是纯 kernel 时间。",
        "单独计时的 baseline 为现有有状态 decoder 调用；服务基线另保留原外层图。",
        "历史状态恢复不计时；本次输入暂存、解码、状态更新和输出交接计时。",
        "",
        "| batch | 阶段 | baseline p50/p95 ms | graph p50/p95 ms | 加速比 | 配对95%区间 |",
        "|---:|---|---|---|---:|---|",
    ]
    rng = np.random.default_rng(42)
    for batch, stage in sorted({(b, s) for b, s, _ in groups}):
        sides = {}
        pairs = defaultdict(dict)
        for mode in ("baseline", "code2wav_graph"):
            group = groups[(batch, stage, mode)]
            if not group:
                continue
            values = [r["wall_ms"] for r in group]
            sides[mode] = dict(
                wall_p50_p95=np.percentile(values, [50, 95]).tolist(),
                device_p50_p95=np.percentile([r["device_ms"] for r in group], [50, 95]).tolist(),
                wall_total_ms=sum(values),
                executions=len(group),
                codec_frames_per_s=sum(r["codec_frames"] for r in group) / (sum(values) / 1000),
                audio_seconds_per_s=sum(r["audio_seconds"] for r in group) / (sum(values) / 1000),
            )
            for j in {r["job"] for r in group}:
                pairs[j][mode] = np.mean([r["wall_ms"] for r in group if r["job"] == j])
        if len(sides) != 2:
            continue
        paired = np.asarray([[v["baseline"], v["code2wav_graph"]] for v in pairs.values() if len(v) == 2])
        bootstrap = paired[rng.integers(0, len(paired), (1000, len(paired)))].mean(axis=1)
        ci = np.percentile(bootstrap[:, 0] / bootstrap[:, 1], [2.5, 97.5]).tolist()
        speed = sides["baseline"]["wall_total_ms"] / sides["code2wav_graph"]["wall_total_ms"]
        result.append(dict(batch=batch, phase=stage, sides=sides, speedup=speed, interval=ci))
        a, b = ["/".join(f"{v:.3f}" for v in sides[m]["wall_p50_p95"]) for m in ("baseline", "code2wav_graph")]
        lines.append(f"| {batch} | {stage} | {a} | {b} | {speed:.3f}× | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    graph = read_json(args.results / "graph.json") if (args.results / "graph.json").exists() else {}
    lines += [
        "",
        f"图与路由统计：{json.dumps(graph, ensure_ascii=False)}",
        "",
        "区间按配对解码批次重采样；重复使用有限请求，只用于探索。",
        "短块显式走 eager 并更新缓存；Code Predictor 两组始终为 kv_frame。",
    ]
    write_json(args.results / "summary.json", dict(comparisons=result, graph=graph, correctness_checked=False))
    (args.results / "report.md").write_text("\n".join(lines), encoding="utf-8")
    progress.log(f"报告：{args.results / 'report.md'}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("collect", "check", "bench", "report"))
    common_args(parser)
    parser.add_argument("--input", type=Path, default=Path(RESULTS) / "capture")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="collect请求数；0表示30条")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--executions-per-round", type=int, default=60)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--profile", action="store_true", help="正式计时后另采固定六批 trace")
    args = parser.parse_args()
    if any(b not in (1, 2, 3, 4) for b in args.batches) or args.executions_per_round < 1 or args.warmup < 0:
        parser.error("batch须为1/2/3/4，执行次数须为正，预热非负")
    if args.command != "collect" and args.results.resolve() == args.input.resolve():
        parser.error("结果目录不能覆盖采集输入")
    if args.limit < 0 or args.limit > 30:
        parser.error("--limit 范围为0..30")
    args.results.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.results)
    if args.command != "report":
        write_json(args.results / "manifest.json", manifest(args))
    try:
        globals()[args.command](args, progress)
    except Exception as exc:
        write_json(args.results / "status.json", dict(status="failed", error=repr(exc)))
        raise
    write_json(args.results / "status.json", dict(status="complete"))


if __name__ == "__main__":
    main()
