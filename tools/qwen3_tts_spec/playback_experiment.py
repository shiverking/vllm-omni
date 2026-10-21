#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproducible zero-training RVQ/playback experiments; see PLAYBACK_ZH.md."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def policy_module():
    spec = importlib.util.spec_from_file_location(
        "rvq_policy_offline", ROOT / "vllm_omni/model_executor/models/common/qwen3_tts_deadline_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(config):
    paths = list((ROOT / "vllm_omni/model_executor/models/common").glob("qwen3*predictor*.py"))
    paths += [ROOT / "vllm_omni/model_executor/models/common/qwen3_tts_deadline_policy.py", Path(__file__)]
    paths += list((ROOT / "vllm_omni/core/sched").glob("playback_*.py"))
    paths += [
        ROOT / p
        for p in (
            "tools/qwen3_tts_spec/playback_service.py",
            "vllm_omni/engine/playback_stage.py",
            "vllm_omni/engine/stage_engine_core_proc.py",
            "vllm_omni/benchmarks/gstreamer_player.py",
            "vllm_omni/core/sched/audio_interaction.py",
            "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py",
            "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code_predictor_vllm.py",
            "vllm_omni/model_executor/models/common/qwen3_tts_playback_session.py",
            "vllm_omni/entrypoints/openai/api_server.py",
            "vllm_omni/entrypoints/openai/serving_speech.py",
        )
    ]
    capture = Path(config["input"])
    packages = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    repositories = {}
    for name in ("vllm_repo", "ascend_repo"):
        head = subprocess.run(["git", "-C", config[name], "rev-parse", "HEAD"], capture_output=True)
        diff = subprocess.run(["git", "-C", config[name], "diff", "--no-ext-diff", "HEAD"], capture_output=True)
        repositories[name] = dict(
            head=head.stdout.decode().strip() if head.returncode == 0 else None,
            tracked_diff=hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None,
        )
    return dict(
        code={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths},
        predictor=sha(capture / "predictor.pt"),
        manifest=sha(capture / "manifest.json"),
        frames={p.name: sha(p) for p in sorted((capture / "frames").glob("*.pt"))},
        sampling=config["sampling"],
        batch_buckets=config["batch_buckets"],
        device=config["device"],
        frames_per_request=config["frames_per_request"],
        chunk_frames=config["chunk_frames"],
        left_context_frames=config["left_context_frames"],
        playback_rate=config["playback_rate"],
        feedback_ms=config["feedback_ms"],
        packages=packages,
        repositories=repositories,
    )


def prepare(args):
    if args.config.exists():
        raise ValueError("configuration already exists; choose a new run directory")
    base = "/softwarePlatform/y00899301"
    config = dict(
        schema="playback_experiment_v1",
        input=args.input or base + "/qwen3_tts_zero_train_spec_run01",
        results=str(args.config.parent),
        device=args.device,
        model="/softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        vllm_repo="/vllm-workspace/vllm",
        ascend_repo="/vllm-workspace/vllm-ascend",
        sampling=["sampling", "greedy"],
        batch_buckets=getattr(args, "batch_buckets", [1]),
        frames_per_request=3,
        warmup=20,
        seeds=[17, 29, 43],
        chunk_frames=4,
        left_context_frames=2,
        playback_rate=1.0,
        feedback_ms=50,
        startup_buffer_ms=100,
        broker_url="http://127.0.0.1:8091",
        server_url="http://127.0.0.1:8000",
        independent_confirmation="missing_local_text_manifest",
    )
    capture = read(Path(config["input"]) / "manifest.json")
    config["dataset"] = capture["dataset"]
    counts = {
        split: sum(r["split"] == split for r in capture["dataset"]) for split in ("calibration", "validation", "test")
    }
    if counts == {"calibration": 80, "validation": 40, "test": 200}:
        if len({" ".join(r["text"].split()) for r in capture["dataset"]}) != 320:
            raise ValueError("formal capture dataset contains duplicate text")
        config["independent_confirmation"] = "formal_captures_present_evaluation_pending"
    if args.text_manifest:
        rows = read(args.text_manifest)
        if not isinstance(rows, list) or any(not isinstance(r.get("text"), str) or not r["text"].strip() for r in rows):
            raise ValueError("text manifest must be a list of nonempty text objects")
        texts = sorted({" ".join(r["text"].split()) for r in rows})
        if len(texts) < 320:
            raise ValueError("independent dataset requires at least 320 distinct texts")
        existing = {" ".join(r["text"].split()) for r in capture["dataset"]}
        if existing.intersection(texts):
            raise ValueError("independent texts overlap the smoke corpus")
        random.Random(17).shuffle(texts)
        # Store separately: existing frame captures still belong to smoke data.
        independent = [
            dict(
                id=hashlib.sha256(t.encode()).hexdigest()[:16],
                text=t,
                split="calibration" if i < 80 else "validation" if i < 120 else "test",
            )
            for i, t in enumerate(texts[:320])
        ]
        write(args.config.parent / "independent_dataset.json", independent)
        config["independent_confirmation"] = "texts_prepared_new_captures_required"
    write(args.config, config)
    print(f"配置已创建：{args.config}。原捕获数据保持不变。", flush=True)


def runtime(config):
    import experiment as exp
    import torch

    from vllm_omni.model_executor.models.common.qwen3_code_predictor_adaptive import AdaptiveExecutor

    ns = SimpleNamespace(
        output=Path(config["input"]),
        device=config["device"],
        vllm_repo=Path(config["vllm_repo"]),
        ascend_repo=Path(config["ascend_repo"]),
    )
    model = exp.load_predictor(ns)
    model._vllm_config.scheduler_config.max_num_seqs = max(config["batch_buckets"])
    model.configure_mtp_execution_buckets(config["batch_buckets"])
    model.requires_grad_(False)
    return exp, torch, AdaptiveExecutor, ns, model


def subset(frames, count=3):
    grouped = {}
    for frame in frames:
        grouped.setdefault(frame["request_id"], []).append(frame)
    selected = []
    for rid in sorted(grouped):
        rows = sorted(grouped[rid], key=lambda r: r["frame_id"])
        indices = sorted({round(i * (len(rows) - 1) / max(1, count - 1)) for i in range(count)})
        selected.extend(rows[i] for i in indices)
    return selected


def inputs_for(exp, torch, ns, model, frames, mode):
    rows = [exp.inputs_for(ns, model, f, greedy=mode == "greedy") for f in frames]
    return {k: torch.cat([r[k] for r in rows], dim=0) if torch.is_tensor(v) else v for k, v in rows[0].items()}


def groups(frames, batch):
    for offset in range(0, len(frames), batch):
        selected = frames[offset : offset + batch]
        # Repeated tail rows are explicit, never counted as new independent data.
        yield selected + [selected[-1]] * (batch - len(selected))


def timed(torch, device, function):
    api = getattr(torch, device.split(":")[0]) if device.startswith(("npu", "cuda")) else None
    if api:
        api.synchronize()
        start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
        start.record()
    before = time.perf_counter()
    value = function()
    if api:
        end.record()
        end.synchronize()
    wall = (time.perf_counter() - before) * 1000
    return value, wall, start.elapsed_time(end) if api else wall


def calibrate(args, config):
    exp, torch, executor, ns, model = runtime(config)
    output = Path(config["results"])
    progress = exp.ReplayProgress(output, interval=10)
    frames = subset(exp.frames(ns, "calibration"), config["frames_per_request"])
    observations = []
    progress.begin("校准候选位置", len(frames))
    with torch.inference_mode():
        for frame in frames:
            inp = exp.inputs_for(ns, model, frame)
            prefix = exp.spec_module().initial_prefix(model, inp["layer0_embed"], inp["last_talker_hidden"])
            codes = frame["codes"].to(ns.device)
            for p in range(1, model._num_groups - 1):
                hidden = exp.spec_module().reference_forward(model, prefix)[:, -1]
                ranks = []
                for target in range(p + 1, min(p + 4, model._num_groups)):
                    top = model.lm_head[target - 1](hidden).topk(4, dim=-1).indices
                    hits = (top == codes[:, target : target + 1]).flatten().cpu().tolist()
                    ranks.append(hits.index(True) if any(hits) else -1)
                observations.append(ranks)
                embedded = model.model.codec_embedding[p - 1](codes[:, p : p + 1])
                prefix = torch.cat((prefix, model.small_to_mtp_projection(embedded)), 1)
            progress.advance()
        frozen = dict(binding=binding(config), actions=policy_module().fit_actions(observations))
        frozen["sha256"] = policy_module().digest(frozen)
        write(output / "actions.json", frozen)
        costs, prefill_costs = [], []
        total = sum(math.ceil(len(frames) / b) for b in config["batch_buckets"]) * len(config["sampling"])
        progress.begin("校准位置／动作／实际批大小的联合成本", total)
        for mode in config["sampling"]:
            for batch in config["batch_buckets"]:
                point = output / "calibration_points" / f"{mode}-b{batch}.json"
                if point.exists():
                    previous = read(point)
                    if previous["actions"] != frozen["sha256"]:
                        raise ValueError("calibration checkpoint binding changed")
                    costs.extend(previous["costs"])
                    prefill_costs.extend(previous["prefill"])
                    for _ in groups(frames, batch):
                        progress.advance(f"restored {mode} B={batch}")
                    continue
                cost_offset, prefill_offset = len(costs), len(prefill_costs)
                engine = executor(
                    model,
                    frozen["actions"],
                    batch_size=batch,
                    do_sample=mode == "sampling",
                    execution="graph" if ns.device.startswith("npu") else "eager",
                )
                first_batch = next(groups(frames, batch))
                warm_inputs = inputs_for(exp, torch, ns, model, first_batch, mode)
                engine.prepare(warm_inputs)
                for _ in range(config["warmup"]):
                    engine.decode(**warm_inputs, choose=lambda p, b: "d3n16")
                for selected in groups(frames, batch):
                    inp = inputs_for(exp, torch, ns, model, selected, mode)
                    _, prefill_wall, prefill_device = timed(torch, ns.device, lambda engine=engine: engine.start(inp))
                    prefill_costs.append(
                        dict(
                            sampling=mode,
                            batch=batch,
                            wall_ms=prefill_wall,
                            device_ms=prefill_device,
                            split="calibration",
                        )
                    )
                    # Every action is probed from the SAME serial prefix. Restore
                    # is outside timing; graph capture is never charged as cost.
                    for p in range(1, model._num_groups - 1):
                        snapshot = [t.clone() for t in engine.bank.tensors()]
                        for action in frozen["actions"]:
                            for dst, source in zip(engine.bank.tensors(), snapshot):
                                dst.copy_(source)
                            (states, _, _), wall, device = timed(
                                torch, ns.device, lambda engine=engine: engine.advance(p, action, tuple(range(batch)))
                            )
                            for row, (delta, accepted) in enumerate(states):
                                costs.append(
                                    dict(
                                        split="calibration",
                                        sampling=mode,
                                        batch=batch,
                                        position=p,
                                        action=action,
                                        delta=delta,
                                        accepted=accepted,
                                        wall_ms=wall,
                                        device_ms=device,
                                        request_id=selected[row]["request_id"],
                                        frame_id=selected[row]["frame_id"],
                                        trial=len(costs) // batch,
                                        query_size=engine.last_context["query_size"],
                                    )
                                )
                        for dst, source in zip(engine.bank.tensors(), snapshot):
                            dst.copy_(source)
                        engine.advance(p, "serial", tuple(range(batch)))
                    progress.advance(f"{mode} B={batch}")
                write(
                    point,
                    dict(actions=frozen["sha256"], costs=costs[cost_offset:], prefill=prefill_costs[prefill_offset:]),
                )
                engine.close()
                del engine
                if ns.device.startswith("npu"):
                    torch.npu.synchronize()
                    torch.npu.empty_cache()
        write(output / "costs.json", costs)
        write(output / "prefill_costs.json", prefill_costs)
        fitted = policy_module().fit_policy(
            costs, terminal=model._num_groups - 1, binding=dict(source=frozen["binding"], actions=frozen["sha256"])
        )
        write(output / "policy.json", fitted)
        write(
            output / "manifest.json",
            dict(
                config=config,
                binding=frozen["binding"],
                environment=exp.environment(ns),
                scope="smoke_calibration",
                independent_confirmation=config["independent_confirmation"],
            ),
        )


def artifacts(config):
    output = Path(config["results"])
    actions, policy = read(output / "actions.json"), read(output / "policy.json")
    if policy_module().digest({k: v for k, v in actions.items() if k != "sha256"}) != actions["sha256"]:
        raise ValueError("action signature mismatch")
    if (
        actions["binding"] != binding(config)
        or policy["binding"]["actions"] != actions["sha256"]
        or policy["binding"]["source"] != actions["binding"]
    ):
        raise ValueError("code, input or action binding changed; recalibrate")
    return actions, policy_module().CompletionPolicy(policy)


def choose_function(name, policy, mode):
    if name == "fixed_full":
        return lambda p, b: "d3n16"
    if name == "mean_b1":
        return lambda p, b: policy.choose(p, 1, mode)
    if name == "mean":
        return lambda p, b: policy.choose(p, b, mode)
    if name == "deadline":
        start = time.perf_counter()
        return lambda p, b: policy.choose(p, b, mode, deadline_ms=50 - (time.perf_counter() - start) * 1000)
    if name.startswith("lambda:"):
        return lambda p, b: policy.choose(p, b, mode, price=float(name.split(":")[1]))
    if name == "greedy_rate":

        def choose(p, b):
            table = policy.payload["tables"][f"{mode}/b{b}"]["samples"][str(p)]
            return min(table, key=lambda a: (-sum(r[0] for r in table[a]) / sum(r[1] for r in table[a]), a))

        return choose
    return lambda p, b: name


def check(args, config):
    actions, policy = artifacts(config)
    if not args.probe_only:
        probe = read(Path(config["results"]) / "probe.json")
        if not probe.get("passed") or probe["binding"] != binding(config) or probe["actions"] != actions["sha256"]:
            raise ValueError("run the matching fixed three-frame probe before sampled checking")
    exp, torch, executor, ns, model = runtime(config)
    output = Path(config["results"])
    frames = subset(exp.frames(ns, "test"), config["frames_per_request"])
    if args.probe_only:
        all_test = exp.frames(ns, "test")
        frames = [f for f in all_test if f["request_id"] == "smoke011" and f["frame_id"] in (0, 2)]
        last = [f for f in all_test if f["request_id"] == "smoke030"]
        if len(frames) != 2 or not last:
            raise ValueError("fixed probe frames missing")
        frames.append(max(last, key=lambda f: f["frame_id"]))
    policies = [
        *actions["actions"],
        "fixed_full",
        "mean_b1",
        "mean",
        "deadline",
        "lambda:0.5",
        "lambda:1",
        "lambda:2",
        "greedy_rate",
    ]
    rows = []
    progress = exp.ReplayProgress(output, interval=10)
    total = sum(math.ceil(len(frames) / b) for b in config["batch_buckets"]) * len(config["sampling"]) * len(policies)
    progress.begin("抽样动态树、切换、节点及 batch 正确性", total)
    with torch.inference_mode():
        for mode in config["sampling"]:
            for batch in config["batch_buckets"]:
                engine = executor(
                    model,
                    actions["actions"],
                    batch_size=batch,
                    do_sample=mode == "sampling",
                    execution="graph" if ns.device.startswith("npu") else "eager",
                )
                full_engine = executor(
                    model,
                    {k: actions["actions"][k] for k in ("serial", "d3n16")},
                    batch_size=batch,
                    do_sample=mode == "sampling",
                    trim_tail=False,
                    execution="graph" if ns.device.startswith("npu") else "eager",
                )
                for selected in groups(frames, batch):
                    inp = inputs_for(exp, torch, ns, model, selected, mode)
                    expected = model(**inp)
                    for name in policies:
                        selected_engine = full_engine if name == "fixed_full" else engine
                        context = dict(
                            mode=mode,
                            batch=batch,
                            policy=name,
                            frames=[(f["request_id"], f["frame_id"]) for f in selected],
                        )
                        try:
                            actual, info = selected_engine.decode(
                                **inp,
                                choose=choose_function(name, policy, mode),
                                audit=args.probe_only or name == "d3n16" and any(f["frame_id"] == 0 for f in selected),
                                record_events=True,
                            )
                            if not torch.equal(actual, expected):
                                raise AssertionError("codec codes differ from stock")
                        except Exception as exc:
                            from vllm_omni.model_executor.models.common.qwen3_code_predictor_pipeline import (
                                PipelineCaptureError,
                            )

                            if not isinstance(exc, PipelineCaptureError):
                                torch.save(
                                    dict(
                                        inputs=inp,
                                        expected=expected,
                                        snapshot=selected_engine.snapshot(),
                                        context=context,
                                    ),
                                    output / "first_failure.pt",
                                )
                            write(output / "correctness.json", dict(passed=False, error=repr(exc), context=context))
                            raise
                        rows.append(
                            dict(
                                **context,
                                audit_max=info["audit_max_logit_error"],
                                graphs=info["graphs"],
                                capture_seconds=info["capture_seconds"],
                                subbatch_counts=info["subbatch_counts"],
                                events=info["events"],
                            )
                        )
                        progress.advance(f"{mode} B={batch} {name}")
                engine.close()
                full_engine.close()
                del engine
    write(
        output / ("probe.json" if args.probe_only else "correctness.json"),
        dict(
            passed=True,
            scope="probe" if args.probe_only else "sampled",
            binding=binding(config),
            actions=actions["sha256"],
            policy=policy.payload["sha256"],
            checks=rows,
        ),
    )


def certificate(config, actions, policy):
    cert = read(Path(config["results"]) / "correctness.json")
    if not cert.get("passed") or cert.get("scope") != "sampled" or cert.get("binding") != binding(config):
        raise ValueError("matching sampled correctness certificate required")
    if cert.get("actions") != actions["sha256"] or cert.get("policy") != policy.payload["sha256"]:
        raise ValueError("certificate actions/policy mismatch")


def measure(args, config):
    actions, policy = artifacts(config)
    certificate(config, actions, policy)
    exp, torch, executor, ns, model = runtime(config)
    frames = subset(exp.frames(ns, "test"), config["frames_per_request"])
    output = Path(config["results"])
    names = [
        "serial",
        "fixed_full",
        "d3n16",
        "greedy_rate",
        "mean_b1",
        "mean",
        "deadline",
        "lambda:0.5",
        "lambda:1",
        "lambda:2",
    ]
    records = []
    progress = exp.ReplayProgress(output, interval=10)
    progress.begin("帧级 ABBA 配对计时", len(config["sampling"]) * len(config["batch_buckets"]) * len(names) * 4)
    with torch.inference_mode():
        for mode in config["sampling"]:
            for batch in config["batch_buckets"]:
                engine = executor(
                    model,
                    actions["actions"],
                    batch_size=batch,
                    do_sample=mode == "sampling",
                    execution="graph" if ns.device.startswith("npu") else "eager",
                )
                full_engine = executor(
                    model,
                    {k: actions["actions"][k] for k in ("serial", "d3n16")},
                    batch_size=batch,
                    do_sample=mode == "sampling",
                    trim_tail=False,
                    execution="graph" if ns.device.startswith("npu") else "eager",
                )
                batches = [inputs_for(exp, torch, ns, model, fs, mode) for fs in groups(frames, batch)]
                for name in names:
                    selected_engine = full_engine if name == "fixed_full" else engine
                    for i in range(config["warmup"]):
                        inp = batches[i % len(batches)]
                        model(**inp)
                        selected_engine.decode(**inp, choose=choose_function(name, policy, mode))
                    for round_id, order in enumerate(
                        (("stock", name), (name, "stock"), (name, "stock"), ("stock", name))
                    ):
                        for trial, inp in enumerate(batches):
                            for label in order:
                                callback = (
                                    (lambda: model(**inp))
                                    if label == "stock"
                                    else (
                                        lambda engine=selected_engine: engine.decode(
                                            **inp, choose=choose_function(name, policy, mode)
                                        )
                                    )
                                )
                                result, wall, device = timed(torch, ns.device, callback)
                                info = dict(backbone_calls=model._num_groups - 1) if label == "stock" else result[1]
                                records.append(
                                    dict(
                                        sampling=mode,
                                        batch=batch,
                                        pair=name,
                                        method=label,
                                        round=round_id,
                                        trial=trial,
                                        wall_ms=wall,
                                        device_ms=device,
                                        backbone_calls=info["backbone_calls"],
                                        lookup_ms=info.get("lookup_ms", 0),
                                        regroup_host_ms=info.get("regroup_host_ms", 0),
                                        subbatch_counts=info.get("subbatch_counts", {}),
                                        accepted_counts=info.get("accepted_counts", {}),
                                        delta_counts=info.get("delta_counts", {}),
                                        graphs=info.get("graphs"),
                                        capture_seconds=info.get("capture_seconds"),
                                        frames=[
                                            (f["request_id"], f["frame_id"]) for f in list(groups(frames, batch))[trial]
                                        ],
                                    )
                                )
                        progress.advance(f"{mode} B={batch} {name} round={round_id}")
                engine.close()
                full_engine.close()
                del engine
    write(output / "measurements.json", dict(binding=binding(config), rows=records, independent_frames=len(frames)))
    report(args, config)


def simulate(args, config):
    actions, policy = artifacts(config)
    # Joint latency/advancement resampling, never future target uniforms.
    import numpy as np

    prefill = read(Path(config["results"]) / "prefill_costs.json")
    rows = []
    for key, table in policy.payload["tables"].items():
        mode, bucket = key.split("/")
        batch = int(bucket[1:])
        starts = [r["wall_ms"] for r in prefill if r["sampling"] == mode and r["batch"] == batch]
        for name in (
            "serial",
            "d3n16",
            "greedy_rate",
            "mean_b1",
            "mean",
            "deadline",
            "lambda:0.5",
            "lambda:1",
            "lambda:2",
        ):
            totals = []
            for trial in range(1024):
                rng = random.Random(17000 + trial)
                position, elapsed = 1, rng.choice(starts)
                while position < policy.payload["terminal"]:
                    action = (
                        policy.choose(position, batch, mode, deadline_ms=50 - elapsed)
                        if name == "deadline"
                        else choose_function(name, policy, mode)(position, batch)
                    )
                    delta, wall, _ = rng.choice(table["samples"][str(position)][action])
                    position, elapsed = position + delta, elapsed + wall
                totals.append(elapsed)
            rows.append(
                dict(
                    condition=key,
                    policy=name,
                    mean_ms=float(np.mean(totals)),
                    p95_ms=float(np.percentile(totals, 95)),
                    completion_samples_ms=totals,
                )
            )
    write(
        Path(config["results"]) / "simulation.json",
        dict(kind="empirical_model_only", rows=rows, binding=binding(config), policy=policy.payload["sha256"]),
    )
    print("已保存模型策略；这是离线预测，不是服务性能结论。")


def report(args, config):
    output = Path(config["results"])
    lines = ["# RVQ 完成时间与播放期限实验", "", "零训练；sampling 与 greedy 分开解释，不设置加速门槛。", ""]
    path = output / "measurements.json"
    if path.exists():
        import numpy as np

        data = read(path)
        if data["binding"] != binding(config):
            raise ValueError("measurements are from a different code/data/config binding")
        lines += [
            f"独立测试帧：{data['independent_frames']}。设备：{config['device']}。",
            "",
            "|模式|B|方法|墙钟 p50/p95 ms|相对 stock 总时间比（>1 更快）|主干调用/批|",
            "|---|---:|---|---|---:|---:|",
        ]
        aggregate = []
        prediction_file = read(output / "simulation.json") if (output / "simulation.json").exists() else None
        if prediction_file and prediction_file["binding"] != binding(config):
            raise ValueError("prediction artifact binding changed")
        predictions = prediction_file["rows"] if prediction_file else []
        for mode in config["sampling"]:
            for batch in config["batch_buckets"]:
                subset_rows = [r for r in data["rows"] if r["sampling"] == mode and r["batch"] == batch]
                for name in sorted({r["pair"] for r in subset_rows}):
                    pair = [r for r in subset_rows if r["pair"] == name]
                    a, b = [[r for r in pair if r["method"] == method] for method in ("stock", name)]
                    times = [r["wall_ms"] for r in b]
                    speedup = sum(r["wall_ms"] for r in a) / sum(times)
                    p50, p95 = np.percentile(times, [50, 95])
                    paired = []
                    for trial in sorted({r["trial"] for r in pair}):
                        paired.append([sum(r["wall_ms"] for r in side if r["trial"] == trial) for side in (a, b)])
                    paired = np.array(paired)
                    rng = np.random.default_rng(17)
                    bootstrap = [
                        paired[i, 0].sum() / paired[i, 1].sum()
                        for i in rng.integers(0, len(paired), size=(1000, len(paired)))
                    ]
                    predicted = next(
                        (
                            r["mean_ms"]
                            for r in predictions
                            if r["condition"] == f"{mode}/b{batch}" and r["policy"] == name
                        ),
                        None,
                    )
                    aggregate.append(
                        dict(
                            sampling=mode,
                            batch=batch,
                            policy=name,
                            wall_mean_ms=float(np.mean(times)),
                            speedup=speedup,
                            paired_input_batch_bootstrap95=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                            device_p50_p95_ms=np.percentile([r["device_ms"] for r in b], [50, 95]).tolist(),
                            predicted_frame_mean_ms=predicted,
                            prediction_error_ms=None if predicted is None else float(np.mean(times)) - predicted,
                            graph_and_host_lookup_ms=sum(r["lookup_ms"] for r in b) / len(b),
                            regroup_host_ms=sum(r["regroup_host_ms"] for r in b) / len(b),
                        )
                    )
                    lines.append(
                        f"|{mode}|{batch}|{name}|{p50:.3f}/{p95:.3f}|{speedup:.3f}|"
                        f"{np.mean([r['backbone_calls'] for r in b]):.2f}|"
                    )
        write(
            output / "summary.json",
            dict(
                rows=aggregate,
                binding=data["binding"],
                uncertainty="exploratory paired input-batch bootstrap, not independent request inference",
            ),
        )
        lines += [
            "",
            "fixed_full 与 d3n16 仅差帧尾 query 缩小；mean_b1 与 mean 隔离 batch 条件化。",
            "summary.json 保存设备事件分位数、探索性配对输入批次区间与模型预测误差。",
            "不足 batch 的末组显式重复最后一帧，仍只计原始独立帧数；实际执行数见原始行。",
        ]
    service_rows = []
    for path in sorted((output / "service").glob("*/measure-c*-seed*.json")):
        value = read(path)
        if value["binding"] != binding(config):
            continue
        completed = [r for r in value["requests"] if r["error"] is None]
        formal_valid = value["scope"] == "formal" and value["elapsed_s"] >= 600 and len(completed) >= 200
        service_rows.append(
            dict(
                path=str(path),
                method=value["method"],
                ablation=value.get("ablation", "none"),
                sampling=value["sampling"],
                concurrency=value["concurrency"],
                qos_fraction=value["qos_fraction"],
                throughput=value["completed_per_second"],
                formal_valid=formal_valid,
            )
        )
    if service_rows:
        lines += [
            "",
            "|服务方法／消融|模式|会话并发|满足首音频／断流 SLO 占比|完成请求/秒|正式样本有效|",
            "|---|---|---:|---:|---:|---|",
        ]
        for row in service_rows:
            lines.append(
                f"|{row['method']}/{row['ablation']}|{row['sampling']}|{row['concurrency']}|{row['qos_fraction']:.2%}|"
                f"{row['throughput']:.3f}|{row['formal_valid']}|"
            )
        write(output / "service_summary.json", service_rows)
    lines += [
        "",
        "设备事件时间不是纯 kernel 时间。CPU 运行仅验证工具。",
        "帧级结果不能证明播放连续性或服务容量；端到端结果必须单独测量。",
        f"独立确认状态：{config['independent_confirmation']}。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "calibrate", "check", "measure", "simulate", "serve", "bench", "report")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-buckets", nargs="+", type=int, default=[1])
    parser.add_argument("--text-manifest", type=Path)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--method",
        default="joint",
        choices=("native", "liveserve_ar", "liveserve_fixed", "position", "joint_ar", "joint"),
    )
    parser.add_argument("--sampling", choices=("sampling", "greedy"), default="sampling")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phase", choices=("warmup", "check", "calibration", "measure"), default="check")
    parser.add_argument("--trace", action="store_true", help="separate diagnostic service run, not formal timing")
    parser.add_argument("--stage-events", type=Path, help="fit stage costs from a controlled calibration service trace")
    parser.add_argument("--formal", action="store_true", help="independent data, >=600 s and >=200 requests")
    parser.add_argument("--ablation", choices=("none", "buffer_only", "no_code2wav", "no_batch"), default="none")
    parser.add_argument("--arrivals", type=Path, help="fixed open-loop offsets in seconds; keeps client queue delay")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency not in (1, 2, 4, 8, 16):
        parser.error("requests must be positive; concurrency must be 1/2/4/8/16")
    if args.batch_buckets not in ([1], [1, 2], [1, 2, 4], [1, 2, 4, 8]):
        parser.error("batch buckets must include every power of two up to the maximum")
    if args.command == "prepare":
        prepare(args)
        return
    config = read(args.config)
    if args.command == "calibrate" and args.stage_events:
        from playback_service import fit_stage_costs

        fit_stage_costs(args, config)
        return
    try:
        if args.command in ("serve", "bench"):
            from playback_service import bench, serve

            (serve if args.command == "serve" else bench)(args, config)
        else:
            globals()[args.command](args, config)
    except Exception as exc:
        # Host-only diagnostics also work after a damaged NPU capture context.
        write(
            Path(config["results"]) / f"failure-{args.command}.json",
            dict(command=args.command, error=repr(exc), traceback=traceback.format_exc()),
        )
        raise


if __name__ == "__main__":
    main()
