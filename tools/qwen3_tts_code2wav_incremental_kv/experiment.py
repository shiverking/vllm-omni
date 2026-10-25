# SPDX-License-Identifier: Apache-2.0
"""Independent incremental Code2Wav checks, timing and service experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "tools/qwen3_tts_code2wav_graph"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT))
from engine import compare, load_decoder, run  # noqa: E402
from runtime import Progress, common_args, manifest, read_json, write_json  # noqa: E402

RESULTS = "/softwarePlatform/y00899301/qwen3_tts_code2wav_incremental_kv_run01"
INPUT = "/softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/capture"


def load_script(name, file):
    spec = importlib.util.spec_from_file_location(name, OLD / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_script("code2wav_original_experiment", "experiment.py")


def input_identity(args):
    digest = hashlib.sha256((args.input / "requests.json").read_bytes())
    for r in read_json(args.input / "requests.json"):
        if r.get("complete"):
            for name in r["chunks"]:
                digest.update(name.encode())
                digest.update((args.input / name).read_bytes())
    return digest.hexdigest()


def make(args, model, mode, execution):
    from vllm_omni.model_executor.models.qwen3_tts.code2wav_incremental_kv import IncrementalKVExecutor
    from vllm_omni.model_executor.models.qwen3_tts.npu_segmented_graph_wrapper import NPUGraphDecoderWrapper

    if mode == "current":
        if execution == "eager":
            from engine import eager_layout_class

            wrapper = eager_layout_class()(model)
        else:
            wrapper = NPUGraphDecoderWrapper(model)
        wrapper.warmup(torch.device(args.device))
        return wrapper
    rolling = make(args, model, "current", execution) if mode == "growth_kv" else None
    wrapper = IncrementalKVExecutor(model, mode, execution, current=rolling)
    wrapper.warmup(torch.device(args.device))
    return wrapper


def release(wrapper, caches):
    wrapper.release_requests(caches)


def freeze(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: freeze(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [freeze(v) for v in value]
    if hasattr(value, "layers"):
        return [dict(k=freeze(getattr(x, "keys", None)), v=freeze(getattr(x, "values", None))) for x in value.layers]
    return value if isinstance(value, (int, float, bool, str, type(None))) else repr(value)


def save_wav(path, x, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = x.detach().float().cpu().reshape(-1).clamp(-1, 1).numpy()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((data * 32767).round().astype("<i2").tobytes())


def continuous_batch(model, histories, lengths):
    from vllm_omni.model_executor.models.qwen3_tts.code2wav_incremental_kv import continuous_reference

    groups = defaultdict(list)
    for i, (h, n) in enumerate(zip(histories, lengths)):
        groups[(h.shape[-1], n)].append(i)
    result = [None] * len(histories)
    for (_, n), rows in groups.items():
        for offset in range(0, len(rows), 4):
            group = rows[offset : offset + 4]
            b = next(b for b in (1, 2, 4) if b >= len(group))
            codes = torch.cat(
                [histories[i] for i in group] + [torch.zeros_like(histories[group[0]])] * (b - len(group))
            )
            wav, hidden, k, v = continuous_reference(model, codes, n, return_kv=True)
            for row, i in enumerate(group):
                result[i] = (
                    wav[row].clone(),
                    hidden[row : row + 1, -12:].clone(),
                    k[:, row : row + 1, ..., -(model.config.sliding_window - 1) :, :].clone(),
                    v[:, row : row + 1, ..., -(model.config.sliding_window - 1) :, :].clone(),
                )
    return result


def diagnostic(model, candidate, codes, caches, histories, lengths):
    """Save first module difference, including K/V projection and rotary outputs.

    Run before failed states are released. This is outside performance timing.
    """
    atol, rtol = (1e-5, 0.016) if next(model.parameters()).dtype == torch.bfloat16 else (1e-5, 1e-4)
    values = {}
    handles = []
    for name, module in model.named_modules():
        if name.startswith("pre_transformer") and (
            name.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "rotary_emb"))
        ):

            def hook(mod, inputs, output, name=name):
                values.setdefault(name, []).append(freeze(output))

            handles.append(module.register_forward_hook(hook))
    try:
        run(model, candidate, codes, caches)
        incremental = copy.deepcopy(values)
        values.clear()
        continuous_batch(model, histories, lengths)
        first = None
        for name, calls in incremental.items():
            if name not in values:
                continue
            left, right = calls[-1], values[name][-1]
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.ndim == right.ndim == 3:
                right = right[:, -left.shape[1] :]
                if left.shape == right.shape:
                    detail = compare(left, right, atol, rtol)
                    if not detail["exact"]:
                        first = dict(module=name, comparison=detail)
                        break
        return dict(
            incremental=incremental,
            continuous=values,
            first_different_module=first,
            stages={
                str(key): dict(inputs=freeze(s.inputs), mask=freeze(s.mask))
                for key, s in getattr(candidate, "stages", {}).items()
            },
        )
    finally:
        for h in handles:
            h.remove()
        release(candidate, caches)


@torch.no_grad()
def check(args, progress):
    model, rate = load_decoder(args)
    sequences = list(legacy.sequences(args))
    # Deterministic long input exercises several windows and short->full recovery.
    source = torch.cat(sequences[0]["chunks"], -1)
    sizes = [1, 25, 25, 25, 25, 7, 25, 25, 25, 25, 25, 25, 3]
    long = source.repeat(1, 1, (sum(sizes) + source.shape[-1] - 1) // source.shape[-1])
    offset, chunks = 0, []
    for n in sizes:
        chunks.append(long[..., offset : offset + n])
        offset += n
    sequences += [dict(id="synthetic-multi-window", chunks=chunks)]
    atol, rtol = (1e-5, 0.016) if args.device.startswith("npu") else (1e-5, 1e-4)
    anchor = make(args, model, "current", "graph" if args.device.startswith("npu") else "eager")
    checks, errors, listening = [], [], []
    total = (
        sum(1 for b in args.batches for _ in legacy.batches(sequences, b)) * len(args.variants) * len(args.executions)
    )
    done = 0
    for mode in args.variants:
        for execution in args.executions:
            label = f"{mode}/{execution}"
            candidate = eager = None
            caches, stocks, eager_caches = [], [], []
            try:
                (args.results / f"failure-{mode}-{execution}.pt").unlink(missing_ok=True)
                candidate = make(args, model, mode, execution)
                eager = make(args, model, mode, "eager") if execution == "graph" else None
                for batch in args.batches:
                    cohort = None
                    for start, position, active in legacy.batches(sequences, batch):
                        if start != cohort:
                            release(candidate, caches)
                            release(anchor, stocks)
                            if eager:
                                release(eager, eager_caches)
                            caches = [dict(prefix_frames=0) for _ in range(batch)]
                            stocks = [dict(prefix_frames=0) for _ in range(batch)]
                            eager_caches = [dict(prefix_frames=0) for _ in range(batch)]
                            singles = [dict(prefix_frames=0) for _ in range(batch)]
                            history = [None] * batch
                            audio = {i: dict(current=[], candidate=[]) for i in range(batch)}
                            cohort = start
                        ids = [i for i, _ in active]
                        codes = [r["chunks"][position].to(args.device) for _, r in active]
                        before = copy.deepcopy([caches[i] for i in ids])
                        for i, x in zip(ids, codes):
                            history[i] = x if history[i] is None else torch.cat((history[i], x), -1)
                        lengths = [x.shape[-1] for x in codes]
                        got = run(model, candidate, codes, [caches[i] for i in ids])
                        repeat_state = copy.deepcopy(before)
                        repeated = run(model, candidate, codes, repeat_state)
                        release(candidate, repeat_state)
                        stock = run(model, anchor, codes, [stocks[i] for i in ids])
                        single_stock = [model.chunked_decode(x, singles[i])[0] for i, x in zip(ids, codes)]
                        references = (
                            continuous_batch(model, [history[i] for i in ids], lengths) if mode == "stream_kv" else None
                        )
                        eager_wavs = run(model, eager, codes, [eager_caches[i] for i in ids]) if eager else None
                        for j, (i, request) in enumerate(active):
                            expected = references[j][0] if references else stock[j]
                            comparisons = dict(
                                wave=compare(got[j], expected, atol, rtol),
                                repeat=compare(got[j], repeated[j], 0, 0),
                                boundary=compare(
                                    got[j][..., : min(256, got[j].shape[-1])],
                                    expected[..., : min(256, expected.shape[-1])],
                                    atol,
                                    rtol,
                                ),
                            )
                            if eager:
                                comparisons["graph_vs_eager"] = compare(got[j], eager_wavs[j], atol, rtol)
                                for name, value in caches[i].get("_ikv", {}).items():
                                    comparisons[f"graph_state_{name}"] = compare(
                                        value, eager_caches[i]["_ikv"][name], atol, rtol
                                    )
                            if references:
                                comparisons["hidden_context"] = compare(
                                    caches[i]["_ikv"]["hidden"], references[j][1], atol, rtol
                                )
                                for index, name in ((2, "k"), (3, "v")):
                                    comparisons[f"continuous_{name}"] = compare(
                                        caches[i]["_ikv"][name], references[j][index], atol, rtol
                                    )
                            else:
                                for name in ("suffix_quantized", "suffix_conv"):
                                    comparisons[name] = compare(caches[i][name], stocks[i][name], atol, rtol)
                            row = dict(
                                mode=mode,
                                execution=execution,
                                batch=batch,
                                request=request["id"],
                                chunk=position,
                                passed=all(x["passed"] for x in comparisons.values()),
                                comparisons=comparisons,
                                vs_current=compare(got[j], stock[j], atol, rtol),
                                stock_batch_vs_single=compare(stock[j], single_stock[j], atol, rtol),
                            )
                            checks.append(row)
                            artifact_path = args.results / f"failure-{mode}-{execution}.pt"
                            if not row["passed"] and not artifact_path.exists():
                                artifact = dict(
                                    row=row,
                                    codes=freeze(codes),
                                    before=freeze(before),
                                    after=freeze(caches[i]),
                                    expected=freeze(expected),
                                    actual=freeze(got[j]),
                                    graph=candidate.statistics(),
                                )
                                try:
                                    artifact["modules"] = diagnostic(
                                        model,
                                        candidate,
                                        codes,
                                        copy.deepcopy(before),
                                        [history[k] for k in ids],
                                        lengths,
                                    )
                                except Exception as exc:
                                    artifact["diagnostic_error"] = repr(exc)
                                torch.save(artifact, artifact_path)
                                progress.log(
                                    f"{label} 首次数值差异：batch={batch} {request['id']} chunk={position}；"
                                    "已记录，继续全部请求和音频导出"
                                )
                            audio[i]["current"].append(stock[j].detach().cpu())
                            audio[i]["candidate"].append(got[j].detach().cpu())
                            if position + 1 == len(request["chunks"]):
                                a, b = [torch.cat(audio[i][key], -1) for key in ("current", "candidate")]
                                checks.append(
                                    dict(
                                        mode=mode,
                                        execution=execution,
                                        batch=batch,
                                        request=request["id"],
                                        kind="concatenated_vs_current",
                                        semantic_difference_allowed=mode == "stream_kv",
                                        comparison=compare(a, b, atol, rtol),
                                    )
                                )
                                # Audio pairs are generated once per actual request, batch=1 graph preferred.
                                if batch == 1 and execution == args.executions[-1]:
                                    stem = f"{mode}-{request['id']}"
                                    swap = int(hashlib.sha256(stem.encode()).hexdigest()[0], 16) % 2
                                    paths = [f"audio/{stem}-{s}.wav" for s in ("A", "B")]
                                    for path, wave_data in zip(paths, (b, a) if swap else (a, b)):
                                        save_wav(args.results / path, wave_data, rate)
                                    listening.append(
                                        dict(
                                            id=stem,
                                            files=paths,
                                            A=mode if swap else "current",
                                            B="current" if swap else mode,
                                            difference=compare(a, b, atol, rtol),
                                            seconds=a.numel() / rate,
                                            rms=[float(x.float().square().mean().sqrt()) for x in (a, b)],
                                            peak=[float(x.abs().max()) for x in (a, b)],
                                            boundary_jump=[
                                                max(
                                                    (
                                                        float((p[..., -1] - q[..., 0]).abs().max())
                                                        for p, q in zip(audio[i][key], audio[i][key][1:])
                                                    ),
                                                    default=0.0,
                                                )
                                                for key in ("current", "candidate")
                                            ],
                                        )
                                    )
                                release(candidate, [caches[i]])
                                release(anchor, [stocks[i]])
                                if eager:
                                    release(eager, [eager_caches[i]])
                        done += 1
                        if done % 10 == 0:
                            progress.update(done, total, f"增量正确性 {label}")
                write_json(args.results / f"graph-{mode}-{execution}.json", candidate.statistics())
            except Exception as exc:
                errors.append(dict(mode=mode, execution=execution, error=repr(exc)))
                path = args.results / f"failure-{mode}-{execution}.pt"
                if not path.exists():
                    try:
                        torch.save(
                            dict(
                                error=repr(exc),
                                before=freeze(locals().get("before", [])),
                                codes=freeze(locals().get("codes", [])),
                                graph=candidate.statistics() if candidate else None,
                            ),
                            path,
                        )
                    except Exception as diagnostic_error:
                        errors[-1]["diagnostic_error"] = repr(diagnostic_error)
                progress.log(f"{label} 失败；保留结果并继续独立配置：{exc}")
            finally:
                if candidate:
                    release(candidate, list(candidate.owners.values()) if hasattr(candidate, "owners") else caches)
                release(anchor, stocks)
                if eager:
                    release(eager, eager_caches)
                write_json(
                    args.results / "correctness.json",
                    check_summary(checks, errors, False, atol, rtol),
                )
    write_json(
        args.results / "correctness.json",
        check_summary(checks, errors, not errors, atol, rtol),
    )
    write_json(args.results / "listening-key.json", listening)
    write_json(args.results / "listening.json", [dict(id=x["id"], files=x["files"]) for x in listening])
    (args.results / "report.md").write_text(
        f"# Transformer 增量 KV 检查\n\n检查记录 {len(checks)}；运行异常配置 {len(errors)}；"
        f"数值差异记录 {sum(r.get('passed') is False for r in checks)}。\n\n"
        "数值对照仅记录，不中断请求或音频导出；运行完成不等于正确性通过。\n\n"
        "stream_kv 与连续重算对照；与 current 的差异单列，不代表音质通过。\n\n"
        "盲听文件见 listening.json，解盲和波形指标见 listening-key.json。\n",
        encoding="utf-8",
    )
    progress.update(done, total, "检查结束（数值差异不拦截）")


def check_summary(checks, errors, complete, atol, rtol):
    differences = sum(row.get("passed") is False for row in checks)
    return dict(
        passed=not errors and differences == 0,
        complete=complete,
        status="completed_with_runtime_errors"
        if errors
        else "completed_with_differences"
        if complete and differences
        else "complete"
        if complete
        else "running",
        nonblocking=True,
        mismatch_records=differences,
        checks=checks,
        errors=errors,
        atol=atol,
        rtol=rtol,
    )


def prepare(model, wrapper, job, device):
    caches = [dict(prefix_frames=0) for _ in job["rows"]]
    for position in range(job["position"]):
        codes = [row["chunks"][position].to(device) for row in job["rows"]]
        run(model, wrapper, codes, caches)
    return [r["chunks"][job["position"]].to(device) for r in job["rows"]], caches


def job_phase(job, window=72):
    phases = []
    for row in job["rows"]:
        n = row["chunks"][job["position"]].shape[-1]
        previous = sum(x.shape[-1] for x in row["chunks"][: job["position"]])
        phases.append(
            "first" if previous == 0 else "short" if n != 25 else "rolling" if previous >= window else "growing"
        )
    return phases[0] if len(set(phases)) == 1 else "mixed"


def weighted_jobs(seq, batch, count, window):
    """Cover every phase, including final partial batches; no best-case selection."""
    jobs = [
        dict(id=i, rows=[r for _, r in active], position=position)
        for i, (_, position, active) in enumerate(legacy.batches(seq, batch))
    ]
    groups = defaultdict(list)
    for job in jobs:
        groups[job_phase(job, window)].append(job)
    if count < len(groups):
        raise ValueError(f"executions-per-round must be >= {len(groups)} to cover all phases")
    chosen = [group[len(group) // 2] for group in groups.values()]
    chosen_ids = {j["id"] for j in chosen}
    others = [j for j in jobs if j["id"] not in chosen_ids]
    if others and count > len(chosen):
        chosen += [others[i] for i in np.linspace(0, len(others) - 1, count - len(chosen), dtype=int)]
    while len(chosen) < count:
        chosen.append(chosen[len(chosen) % len(groups)])
    frequency = defaultdict(int)
    for job in chosen:
        frequency[job_phase(job, window)] += 1
    weights = {name: len(group) / frequency[name] for name, group in groups.items()}
    return chosen, weights, {name: len(group) for name, group in groups.items()}


@torch.no_grad()
def bench(args, progress):
    model, rate = load_decoder(args)
    seq = legacy.sequences(args)
    rows = []
    meta = {}
    for mode in [x for x in args.variants if x != "current"]:
        for execution in args.executions:
            candidate = make(args, model, mode, execution)
            anchor = make(args, model, "current", "graph")
            label = f"{mode}/{execution}"
            try:
                for batch in args.batches:
                    # All calls are retained; weighted totals use their corpus frequency.
                    chosen, weights, corpus_phases = weighted_jobs(
                        seq, batch, args.executions_per_round, model.config.sliding_window
                    )
                    snapshots = {}
                    for side, wrapper in (("current", anchor), ("candidate", candidate)):
                        snapshots[side] = {}
                        for job in {j["id"]: j for j in chosen}.values():
                            codes, caches = prepare(model, wrapper, job, args.device)
                            snapshots[side][job["id"]] = (codes, copy.deepcopy(caches))
                            release(wrapper, caches)

                    def restore(side, job):
                        codes, history = snapshots[side][job["id"]]
                        return codes, copy.deepcopy(history)

                    for side, wrapper in (("current", anchor), ("candidate", candidate)):
                        for i in range(args.warmup):
                            codes, caches = restore(side, chosen[i % len(chosen)])
                            legacy.measured(model, wrapper, codes, caches)
                            release(wrapper, caches)
                    total = 4 * args.executions_per_round
                    for round_id, side in enumerate(("current", "candidate", "candidate", "current")):
                        wrapper = anchor if side == "current" else candidate
                        for i in range(args.executions_per_round):
                            job = chosen[i % len(chosen)]
                            codes, caches = restore(side, job)
                            phase_name = job_phase(job, model.config.sliding_window)
                            before = dict(getattr(wrapper, "routes", {}))
                            wall, event, output = legacy.measured(model, wrapper, codes, caches)
                            delta = {
                                k: v - before.get(k, 0)
                                for k, v in getattr(wrapper, "routes", {}).items()
                                if isinstance(k, str)
                            }
                            rows.append(
                                dict(
                                    pair=label,
                                    mode=side,
                                    batch=batch,
                                    actual_batch=len(codes),
                                    phase=phase_name,
                                    corpus_weight=weights[phase_name],
                                    corpus_phases=corpus_phases,
                                    round=round_id,
                                    job=job["id"],
                                    execution=i,
                                    wall_ms=wall,
                                    device_ms=event,
                                    codec_frames=sum(x.shape[-1] for x in codes),
                                    audio_seconds=sum(x.numel() for x in output) / rate,
                                    counters=delta,
                                )
                            )
                            release(wrapper, caches)
                            done = round_id * args.executions_per_round + i + 1
                            if done % 20 == 0:
                                progress.update(done, total, f"ABBA {label} batch={batch}")
                    (args.results / "timings.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
                    )
                meta[label] = dict(
                    candidate=candidate.statistics(),
                    current=anchor.statistics(),
                    peak_memory_bytes=torch.npu.max_memory_allocated(),
                    note="内存峰值包含同时驻留的对照与候选",
                )
                write_json(args.results / "graph.json", meta)
                if args.profile:
                    profile(args, model, candidate, seq, label, progress)
                    profile(args, model, anchor, seq, "current", progress)
            finally:
                release(candidate, list(getattr(candidate, "owners", {}).values()))
            del candidate, anchor
            torch.npu.empty_cache()
    report(args, progress)


def profile(args, model, wrapper, seq, label, progress):
    import torch_npu

    for i, job in enumerate(legacy.make_jobs(seq, 4, 6)):
        codes, caches = prepare(model, wrapper, job, args.device)
        torch.npu.synchronize()
        directory = args.results / "profile" / label.replace("/", "-") / f"batch{i}"
        directory.mkdir(parents=True, exist_ok=True)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        ) as prof:
            run(model, wrapper, codes, caches)
            torch.npu.synchronize()
        write_json(
            directory / "operators.json",
            [
                dict(
                    name=e.key,
                    count=e.count,
                    self_cpu_us=e.self_cpu_time_total,
                    self_device_us=getattr(e, "self_device_time_total", None),
                )
                for e in prof.key_averages()
            ],
        )
        release(wrapper, caches)
    progress.log(f"独立六批 profiling 完成：{label}")


def report(args, progress):
    rows = [json.loads(x) for x in (args.results / "timings.jsonl").read_text().splitlines() if x.strip()]
    summaries = []
    lines = [
        "# Code2Wav Transformer 增量 KV 性能",
        "",
        "性能独立运行，不代表正确性或音质通过。设备事件不是纯 kernel 时间。",
        "",
        "|配置|batch|阶段|current p50/p95 ms|候选 p50/p95 ms|加速比|配对95%区间|",
        "|---|---:|---|---|---|---:|---|",
    ]
    rng = np.random.default_rng(42)
    for pair, batch in sorted({(r["pair"], r["batch"]) for r in rows}):
        selected = [r for r in rows if r["pair"] == pair and r["batch"] == batch]
        for phase in ["all"] + sorted({r["phase"] for r in selected}):
            group = [r for r in selected if phase == "all" or r["phase"] == phase]
            sides = {}
            for side in ("current", "candidate"):
                values = [r for r in group if r["mode"] == side]
                wall = sum(r["wall_ms"] * r.get("corpus_weight", 1) for r in values)
                sides[side] = dict(
                    wall_p50_p95=np.percentile([r["wall_ms"] for r in values], [50, 95]).tolist(),
                    device_p50_p95=np.percentile([r["device_ms"] for r in values], [50, 95]).tolist(),
                    wall_total_ms=wall,
                    executions=len(values),
                    estimated_corpus_wall_ms=wall / 2,
                    codec_frames_per_s=sum(r["codec_frames"] * r.get("corpus_weight", 1) for r in values) * 1000 / wall,
                    audio_seconds_per_s=sum(r["audio_seconds"] * r.get("corpus_weight", 1) for r in values)
                    * 1000
                    / wall,
                )
            pairs = np.asarray(
                [
                    [
                        sum(r["wall_ms"] * r.get("corpus_weight", 1) for r in group if r["job"] == j and r["mode"] == s)
                        for s in ("current", "candidate")
                    ]
                    for j in sorted({r["job"] for r in group})
                ]
            )
            sampled = pairs[rng.integers(0, len(pairs), (1000, len(pairs)))].mean(1)
            ci = np.percentile(sampled[:, 0] / sampled[:, 1], [2.5, 97.5]).tolist()
            speed = sides["current"]["wall_total_ms"] / sides["candidate"]["wall_total_ms"]
            summaries.append(dict(pair=pair, batch=batch, phase=phase, sides=sides, speedup=speed, interval=ci))

            def fmt(side):
                return "/".join(f"{x:.3f}" for x in sides[side]["wall_p50_p95"])

            lines.append(f"|{pair}|{batch}|{phase}|{fmt('current')}|{fmt('candidate')}|{speed:.3f}|{ci}|")
    lines += [
        "",
        "all 按完整采集请求的阶段调用频率加权估算，包含结束短块和滚动重算；不是完整请求实测。"
        "p50/p95 为测量样本分位数。重复帧的 bootstrap 仅作探索。",
        "",
        "state/kv 搬运、实际 Transformer 位置数见 timings.jsonl；图路由和内存见 graph.json。"
        "stream_kv 与 current 语义不同，须另查波形和盲听结果。",
    ]
    write_json(args.results / "summary.json", summaries)
    (args.results / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def service(args, progress):
    import runtime

    original = runtime.deployment

    def deployment(args, mode, directory, capture=None):
        import yaml

        path = original(args, "current", directory, capture)
        config = yaml.safe_load(path.read_text())
        extra = config["connectors"]["connector_of_shared_memory"]["extra"]
        extra["code2wav_transformer_execution"] = mode
        extra["code2wav_incremental_execution"] = "graph"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    saved_manifest = read_json(args.results / "manifest.json")
    runtime.deployment = deployment
    try:
        load_script("incremental_service", "service_bench.py").run(args)
    finally:
        runtime.deployment = original
        write_json(args.results / "manifest.json", saved_manifest)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("check", "bench", "report", "service"))
    common_args(parser)
    parser.add_argument("--input", type=Path, default=Path(INPUT))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("current", "growth_kv", "stream_kv"),
        default=["current", "growth_kv", "stream_kv"],
    )
    parser.add_argument("--executions", nargs="+", choices=("eager", "graph"), default=["eager", "graph"])
    parser.add_argument("--batches", nargs="+", type=int, choices=(1, 2, 3, 4), default=[1, 2, 3, 4])
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--executions-per-round", type=int, default=60)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 16])
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.executions_per_round < 1
        or args.requests < 1
        or any(x < 1 or x > 16 for x in args.concurrencies)
    ):
        parser.error("计时次数、请求数和并发参数无效")
    if args.command not in ("report", "service") and not args.device.startswith("npu"):
        parser.error("真实实验要求 NPU；CPU 请运行 test_cpu.py")
    if args.command != "report" and (
        args.results.resolve() == args.input.resolve() or args.results.resolve() in args.input.resolve().parents
    ):
        parser.error("结果目录不能覆盖采集输入")
    args.results.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.results)
    if args.command != "report":
        data = manifest(args)
        data["baseline"] = "current Code2Wav graph @78a08520c; kv_frame unchanged"
        for p in list(Path(__file__).parent.glob("*.py")) + [
            ROOT / "vllm_omni/model_executor/models/qwen3_tts/code2wav_incremental_kv.py"
        ]:
            data["code"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        if args.command != "service":
            data["input_sha256"] = input_identity(args)
        write_json(args.results / "manifest.json", data)
    try:
        globals()[args.command](args, progress)
    except Exception as exc:
        write_json(args.results / "error.json", dict(command=args.command, error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
