# SPDX-License-Identifier: Apache-2.0
"""Independent variant comparisons; correctness artifacts never gate timing."""

import gc
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch
from engine import compare, cpu_cache, load_decoder, run, snapshot_cache
from runtime import write_json

VARIANTS = ("baseline", "current", "constants", "pooled", "fused")
DEFAULT_VARIANTS = ("baseline", "current", "fused")


def make_executor(args, variant, progress):
    model, sr = load_decoder(args)
    from vllm_omni.model_executor.models.qwen3_tts.npu_segmented_graph_wrapper import NPUGraphDecoderWrapper

    if not args.device.startswith("npu"):
        raise ValueError("正式 check/bench 需要 NPU；CPU 使用 test_cpu.py")
    wrapper = None
    if variant != "baseline":
        wrapper = NPUGraphDecoderWrapper(model, optimization=variant)
        progress.log(f"{variant}: 捕获 15 张 Code2Wav 图")
        try:
            wrapper.warmup(torch.device(args.device))
        except Exception as exc:
            write_json(
                args.results / f"capture_failure_{variant}.json",
                dict(variant=variant, error=repr(exc), graph=wrapper.statistics()),
            )
            raise
    return model, sr, wrapper


def metadata(model, wrapper):
    result = wrapper.statistics() if wrapper else dict(optimization="baseline", graph_count=0)
    result.update(
        dtype=str(next(model.parameters()).dtype), attention_backend=model.pre_transformer.config._attn_implementation
    )
    return result


def release(wrapper, caches):
    if wrapper:
        wrapper.release_requests(list(caches))


def input_identity(args):
    digest = hashlib.sha256()
    manifest = args.input / "requests.json"
    digest.update(manifest.read_bytes())
    for row in json.loads(manifest.read_text(encoding="utf-8")):
        if row.get("complete"):
            for name in row["chunks"]:
                digest.update(name.encode())
                digest.update((args.input / name).read_bytes())
    return dict(path=str(args.input.resolve()), sha256=digest.hexdigest())


def diagnostic_modules(left_model, right_model, codes, left_history, right_history, atol, rtol):
    """Replay only after failure; never instrument formal timing or capture."""
    traces = []
    for model, history in ((left_model, left_history), (right_model, right_history)):
        trace, handles = {}, []

        def save(name):
            def hook(module, inputs, output):
                suffix = name.rsplit(".", 1)[-1]
                packed_names = {
                    "_code2wav_qkv": ("q_proj", "k_proj", "v_proj"),
                    "_code2wav_gate_up": ("gate_proj", "up_proj"),
                }
                if suffix in packed_names:
                    for original, value in zip(packed_names[suffix], output):
                        key = name.rsplit(".", 1)[0] + "." + original
                        trace.setdefault(key, []).append(value.detach().cpu().clone())
                else:
                    value = output[0] if isinstance(output, tuple) else output
                    if isinstance(value, torch.Tensor):
                        trace.setdefault(name, []).append(value.detach().cpu().clone())

            return hook

        for name, module in model.named_modules():
            if name.startswith("pre_transformer.layers"):
                handles.append(module.register_forward_hook(save(name)))
        try:
            model.chunked_decode(codes[0], snapshot_cache(history[0]))
        finally:
            for handle in handles:
                handle.remove()
        traces.append(trace)
    for name in traces[0]:
        if name in traces[1]:
            for index, (a, b) in enumerate(zip(traces[0][name], traces[1][name])):
                if not torch.equal(a, b):
                    return dict(
                        module=name,
                        invocation=index,
                        left=a,
                        right=b,
                        metrics=compare(a, b, atol, rtol),
                        note="eager diagnostic after graph failure",
                    )
    return dict(module=None, note="No difference in eager diagnostic; inspect graph state/layout")


@torch.no_grad()
def check(args, progress):
    from experiment import batches, compare_states, sequences

    seq = sequences(args)
    current_model, sr, current = make_executor(args, "current", progress)
    if args.probe_only:
        generator = torch.Generator().manual_seed(42)
        seq += [
            dict(
                id="synthetic-boundaries",
                chunks=[
                    torch.randint(
                        current_model.config.codebook_size,
                        (1, current_model.config.num_quantizers, n),
                        generator=generator,
                    )
                    for n in (1, 25, 25, 25, 25, 3, 25, 2)
                ],
            )
        ]
    atol, rtol = (1e-5, 1.6e-2) if args.device.startswith("npu") else (1e-5, 1e-4)
    results = dict(
        input=input_identity(args),
        variants={},
        atol=atol,
        rtol=rtol,
        sample_rate=sr,
        independent_requests=len(seq),
        probe_only=args.probe_only,
    )
    total = len(args.variants) * sum(sum(1 for _ in batches(seq, b)) for b in args.batches)
    done = 0
    for variant in args.variants:
        model = wrapper = None
        rows, accumulated, native_differences = [], defaultdict(lambda: ([], [])), []
        single_differences = []
        context = {}
        codes, before, ref_before = [], [], []
        (args.results / f"failure_{variant}.pt").unlink(missing_ok=True)
        try:
            model, _, wrapper = make_executor(args, variant, progress)
            for batch in args.batches:
                caches, refs, native, singles = {}, {}, {}, {}
                previous_start = None
                for start, position, active in batches(seq, batch):
                    if previous_start is not None and previous_start != start:
                        release(wrapper, caches.values())
                        release(current, refs.values())
                        caches, refs, native, singles = {}, {}, {}, {}
                    previous_start = start
                    keys = [i for i, _ in active]
                    for mapping in (caches, refs, native, singles):
                        for key in keys:
                            mapping.setdefault(key, dict(prefix_frames=0))
                    codes = [row["chunks"][position].to(args.device) for _, row in active]
                    cs, rs = [caches[k] for k in keys], [refs[k] for k in keys]
                    # Snapshot only in correctness runs to preserve first-failure histories.
                    before, ref_before = [snapshot_cache(c) for c in cs], [snapshot_cache(c) for c in rs]
                    context = dict(
                        variant=variant,
                        batch=batch,
                        start=start,
                        position=position,
                        requests=[r["id"] for _, r in active],
                    )
                    expected = run(current_model, current, codes, rs)
                    actual = run(model, wrapper, codes, cs)
                    baseline = run(current_model, None, codes, [native[k] for k in keys])
                    independent = [current_model.chunked_decode(x, singles[k])[0] for x, k in zip(codes, keys)]
                    for key, a, b in zip(keys, baseline, independent):
                        delta = compare(a, b, atol, rtol)
                        if not delta["exact"]:
                            single_differences.append(dict(context, row=key, metrics=delta))
                    metrics = [compare(a, b, atol, rtol) for a, b in zip(actual, expected)]
                    states = [compare_states(a, b, atol, rtol) for a, b in zip(cs, rs)]
                    boundaries = [
                        dict(
                            head=compare(a[..., : model.total_upsample], b[..., : model.total_upsample], atol, rtol),
                            tail=compare(a[..., -model.total_upsample :], b[..., -model.total_upsample :], atol, rtol),
                        )
                        for a, b in zip(actual, expected)
                    ]
                    passed = all(m["passed"] for m in metrics) and all(m["passed"] for s in states for m in s.values())
                    rows.append(dict(context, passed=passed, waveform=metrics, state=states, boundary=boundaries))
                    for k, a, b, eager in zip(keys, actual, expected, baseline):
                        accumulated[(batch, start, k)][0].append(a.cpu())
                        accumulated[(batch, start, k)][1].append(b.cpu())
                        delta = compare(a, eager, atol, rtol)
                        if not delta["exact"]:
                            native_differences.append(dict(context, row=k, metrics=delta))
                    if not passed:
                        diagnostic = dict(
                            context=context,
                            codes=[x.cpu() for x in codes],
                            before=[cpu_cache(c) for c in before],
                            reference_before=[cpu_cache(c) for c in ref_before],
                            actual=[x.cpu() for x in actual],
                            expected=[x.cpu() for x in expected],
                            caches=[cpu_cache(c) for c in cs],
                            reference=[cpu_cache(c) for c in rs],
                            slots=[{k: v for k, v in c.items() if k.startswith("_c2w_")} for c in cs],
                            graph=metadata(model, wrapper),
                        )
                        if variant == "fused":
                            try:
                                diagnostic["first_module_difference"] = diagnostic_modules(
                                    model, current_model, codes, before, ref_before, atol, rtol
                                )
                            except Exception as diagnostic_error:
                                diagnostic["module_diagnostic_error"] = repr(diagnostic_error)
                        torch.save(diagnostic, args.results / f"failure_{variant}.pt")
                        raise AssertionError(f"{variant} vs current 波形或缓存不同")
                    done += 1
                    if done % 10 == 0 or done == total:
                        progress.update(done, total, f"{variant} 波形/状态检查")
                    # Release completed requests immediately, including mixed-length cohorts.
                    for key, row in active:
                        if position + 1 == len(row["chunks"]):
                            release(wrapper, [caches[key]])
                            release(current, [refs[key]])
                release(wrapper, caches.values())
                release(current, refs.values())
            concatenated = [compare(torch.cat(a, -1), torch.cat(b, -1), atol, rtol) for a, b in accumulated.values()]
            if not all(m["passed"] for m in concatenated):
                raise AssertionError("拼接波形不同")
            results["variants"][variant] = dict(
                passed=True,
                checks=rows,
                concatenated=concatenated,
                native_batch_differences=native_differences,
                native_batch_vs_single_differences=single_differences,
                graph=metadata(model, wrapper),
            )
        except Exception as exc:
            if not (args.results / f"failure_{variant}.pt").exists():
                try:
                    torch.save(
                        dict(
                            context=context,
                            error=repr(exc),
                            codes=[x.cpu() for x in codes],
                            before=[cpu_cache(c) for c in before],
                            reference_before=[cpu_cache(c) for c in ref_before],
                        ),
                        args.results / f"failure_{variant}.pt",
                    )
                except Exception as save_error:
                    write_json(
                        args.results / f"failure_{variant}.json",
                        dict(context=context, error=repr(exc), artifact_error=repr(save_error)),
                    )
            results["variants"][variant] = dict(
                passed=False, error=repr(exc), context=context, checks=rows, native_batch_differences=native_differences
            )
            progress.log(f"{variant} 检查失败：{exc}；记录后继续其他独立档位")
        finally:
            if wrapper:
                wrapper.reset_pool()
            current.reset_pool()
            del model, wrapper
            gc.collect()
        write_json(args.results / "correctness.json", results)
    results["passed"] = all(v["passed"] for v in results["variants"].values())
    write_json(args.results / "correctness.json", results)
    lines = [
        "# Code2Wav 多档位正确性",
        "",
        "各命令独立；本结果不授权或阻止性能命令。",
        "",
        "| 档位 | 与 current 波形及状态 |",
        "|---|---|",
    ]
    lines += [
        f"| {name} | {'通过' if r['passed'] else '失败：' + r['error']} |" for name, r in results["variants"].items()
    ]
    (args.results / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if not results["passed"]:
        raise RuntimeError("部分档位检查失败，详见 correctness.json；其他命令独立运行")


def restore(wrapper, caches, device):
    if wrapper:
        wrapper.reset_pool()
        wrapper.import_histories([c for c in caches if "suffix_quantized" in c], torch.device(device))


@torch.no_grad()
def bench(args, progress):
    from experiment import make_jobs, measured, phase, prepare_job, sequences

    seq = sequences(args)
    base_model, sr, current = make_executor(args, "current", progress)
    # Current is always the anchor. It is not timed against itself.
    candidates = [v for v in args.variants if v != "current"]
    if not candidates:
        candidates = ["baseline"]
    rows, graphs = [], {"current": metadata(base_model, current)}
    done, total = 0, len(candidates) * len(args.batches) * 4 * args.executions_per_round
    for variant in candidates:
        model, _, wrapper = make_executor(args, variant, progress)
        for batch in args.batches:
            jobs = make_jobs(seq, batch, args.executions_per_round)
            for label, m, w in (("current", base_model, current), (variant, model, wrapper)):
                progress.log(f"{label} batch={batch}: 预热 {args.warmup}")
                for i in range(args.warmup):
                    codes, caches = prepare_job(base_model, jobs[i % len(jobs)], args.device)
                    restore(w, caches, args.device)
                    measured(m, w, codes, caches)
            for w in (current, wrapper):
                if w:
                    w.routes.clear()
            if args.device.startswith("npu"):
                torch.npu.reset_peak_memory_stats()
            for round_id, label in enumerate(("current", variant, variant, "current")):
                m, w = (base_model, current) if label == "current" else (model, wrapper)
                for i in range(args.executions_per_round):
                    job = jobs[i % len(jobs)]
                    codes, caches = prepare_job(base_model, job, args.device)
                    labels = {phase(c, x, base_model.config.sliding_window) for c, x in zip(caches, codes)}
                    stage = next(iter(labels)) if len(labels) == 1 else "mixed"
                    restore(w, caches, args.device)
                    before_routes = w.routes.copy() if w else {}
                    wall, event, output = measured(m, w, codes, caches)
                    route_delta = (
                        [
                            dict(route=k[0], phase=k[1], bucket=k[2], requests=k[3], calls=v - before_routes.get(k, 0))
                            for k, v in w.routes.items()
                            if v > before_routes.get(k, 0)
                        ]
                        if w
                        else []
                    )
                    rows.append(
                        dict(
                            pair=variant,
                            mode=label,
                            batch=batch,
                            actual_batch=len(codes),
                            padding_rows=sum(
                                (r["bucket"] - r["requests"]) * r["calls"] for r in route_delta if r["route"] == "graph"
                            )
                            if w
                            else None,
                            routes=route_delta,
                            phase=stage,
                            round=round_id,
                            job=job["id"],
                            execution=i,
                            wall_ms=wall,
                            device_ms=event,
                            codec_frames=sum(x.shape[-1] for x in codes),
                            audio_seconds=sum(x.numel() for x in output) / sr,
                        )
                    )
                    done += 1
                    if done % 20 == 0 or done == total:
                        progress.update(done, total, f"{variant}/current ABBA")
            graphs[f"{variant}/batch{batch}"] = dict(
                candidate=metadata(model, wrapper),
                anchor=metadata(base_model, current),
                peak_memory_bytes=torch.npu.max_memory_allocated() if args.device.startswith("npu") else None,
            )
            (args.results / "timings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            write_json(args.results / "graph.json", graphs)
        if args.profile:
            profile_one(args, variant, model, wrapper, base_model, seq, progress)
        if wrapper:
            wrapper.reset_pool()
        del model, wrapper
        gc.collect()
    if args.profile:
        profile_one(args, "current", base_model, current, base_model, seq, progress)
    write_json(args.results / "input.json", input_identity(args))
    report(args, progress)


def profile_one(args, variant, model, wrapper, history_model, seq, progress):
    import torch_npu
    from experiment import make_jobs, measured, prepare_job

    directory = args.results.parent / "profile" / variant
    directory.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(seq, 4, 6)
    if wrapper:
        wrapper.profile_phases = True
    # Prepare snapshots outside profiling as well as outside formal timing.
    snapshots = [prepare_job(history_model, job, args.device) for job in jobs]
    event_rows = []
    for i, (codes, caches) in enumerate(snapshots):
        restore(wrapper, caches, args.device)
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory / f"batch{i}")),
        ) as profiler:
            measured(model, wrapper, codes, caches)
        event_rows += [
            dict(
                batch=i,
                operator=e.key,
                count=e.count,
                self_cpu_us=e.self_cpu_time_total,
                self_device_us=getattr(e, "self_device_time_total", None),
            )
            for e in profiler.key_averages()
        ]
    write_json(
        directory / "operators.json",
        dict(
            variant=variant,
            batches=len(jobs),
            events=event_rows,
            note="独立诊断，不并入正式计时；图内 gather/scatter、matmul、SnakeBeta 见设备 trace",
        ),
    )
    progress.log(f"profiling 已保存：{directory}")


def report(args, progress):
    rows = [json.loads(line) for line in (args.results / "timings.jsonl").read_text().splitlines() if line.strip()]
    if rows and "pair" not in rows[0]:
        from experiment import report as legacy_report

        return legacy_report(args, progress)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["pair"], row["batch"], row["phase"])].append(row)
        groups[(row["pair"], row["batch"], "all")].append(row)
    result = []
    rng = np.random.default_rng(42)
    lines = [
        "# Code2Wav 分阶段优化性能",
        "",
        "独立性能测试，未读取正确性证书；耗时不代表正确性通过。",
        "历史状态/池快照恢复不计时；输入、图内状态搬运、计算、提交和输出交接计时。",
        "设备事件不是纯 kernel 时间；峰值内存包含同时驻留的 current 与候选。",
        "",
        "| 档位/current | batch | 阶段 | current p50/p95 ms | 候选 p50/p95 ms | current/候选 | 配对95%区间 |",
        "|---|---:|---|---|---|---:|---|",
    ]
    for (variant, batch, stage), group in sorted(groups.items()):
        sides, jobs = {}, defaultdict(dict)
        for mode in ("current", variant):
            selected = [r for r in group if r["mode"] == mode]
            if not selected:
                continue
            wall = [r["wall_ms"] for r in selected]
            sides[mode] = dict(
                wall_p50_p95=np.percentile(wall, [50, 95]).tolist(),
                device_p50_p95=np.percentile([r["device_ms"] for r in selected], [50, 95]).tolist(),
                wall_total_ms=sum(wall),
                executions=len(wall),
                codec_frames_per_s=sum(r["codec_frames"] for r in selected) * 1000 / sum(wall),
                audio_seconds_per_s=sum(r["audio_seconds"] for r in selected) * 1000 / sum(wall),
            )
            for j in {r["job"] for r in selected}:
                jobs[j][mode] = np.mean([r["wall_ms"] for r in selected if r["job"] == j])
        if len(sides) != 2:
            continue
        paired = np.asarray([[p["current"], p[variant]] for p in jobs.values() if len(p) == 2])
        samples = paired[rng.integers(len(paired), size=(1000, len(paired)))].mean(1)
        interval = np.percentile(samples[:, 0] / samples[:, 1], [2.5, 97.5]).tolist()
        speed = sides["current"]["wall_total_ms"] / sides[variant]["wall_total_ms"]
        result.append(dict(variant=variant, batch=batch, phase=stage, sides=sides, speedup=speed, interval=interval))
        a, b = ["/".join(f"{x:.3f}" for x in sides[m]["wall_p50_p95"]) for m in ("current", variant)]
        lines.append(f"| {variant} | {batch} | {stage} | {a} | {b} | {speed:.3f}× | {interval} |")
    lines += [
        "",
        "constants=常量与输入；pooled=追加状态池；fused=追加线性融合。",
        "消融均与 current 配对；相邻档位之比是独立测量的比较，不作为直接配对置信区间。",
        "短块保持 eager；图命中、内存、派生缓存及后端见 graph.json。服务收益单独测量。",
    ]
    write_json(args.results / "summary.json", dict(comparisons=result, correctness_checked=False))
    (args.results / "report.md").write_text("\n".join(lines), encoding="utf-8")
    progress.log(f"报告：{args.results / 'report.md'}")
