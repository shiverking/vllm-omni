# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Research CLI. Run from the repository with the installed NPU vLLM environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def revision(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def api_for(device):
    if device.type == "npu":
        import torch_npu  # noqa: F401
    return getattr(torch, device.type) if device.type in ("cuda", "npu") else None


def environment(args):
    device = torch.device(args.device)
    api = api_for(device)
    packages = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "vllm-omni"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cann = getattr(args, "cann_version", None)
    for path in (
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.info"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info"),
    ):
        if cann is None and path.exists():
            cann = path.read_text(encoding="utf-8")
            break
    return dict(
        device=str(device),
        device_name=api.get_device_name(device) if api else "CPU",
        packages=packages,
        cann=cann,
        cann_home=os.getenv("ASCEND_HOME_PATH"),
        repositories={
            "vllm-omni": revision(ROOT),
            "vllm": revision(args.vllm_repo),
            "vllm-ascend": revision(args.ascend_repo),
        },
        base_tag="v0.30.0",
        base_sha="a8576ccb725c4e21cd13c3eb5f9a546b21149d2b",
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "tools/qwen3_tts_spec/experiment.py",
                *sorted((ROOT / "vllm_omni/model_executor/models/common").glob("qwen3_code_predictor*.py")),
                ROOT / "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code_predictor_vllm.py",
                ROOT / "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py",
            ]
        },
    )


def spec_module():
    from vllm_omni.model_executor.models.common import qwen3_code_predictor_spec

    return qwen3_code_predictor_spec


def predictor_runtime_config(device):
    if device.type != "npu":
        return SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=1))

    from vllm.config import VllmConfig
    from vllm_ascend.ascend_config import init_ascend_config

    # Offline replay has no worker to initialize Ascend's global configuration.
    # Use a real, model-free config: Ascend accesses more than scheduler_config,
    # and a ModelConfig here would unnecessarily reload the full TTS checkpoint.
    runtime = VllmConfig()
    runtime.scheduler_config.max_num_seqs = 1
    init_ascend_config(runtime)
    return runtime


def load_predictor(args):
    device = torch.device(args.device)
    api = api_for(device)
    if api is not None:
        api.set_device(device)
    runtime = predictor_runtime_config(device)
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
        CodePredictorWrapper,
        CodePredictorWrapperConfig,
    )

    checkpoint = torch.load(args.output / "predictor.pt", map_location="cpu", weights_only=True)
    config = SimpleNamespace(**checkpoint["config"])
    wrapper = CodePredictorWrapper(
        vllm_config=runtime,
        cp_config=config,
        talker_hidden_size=checkpoint["talker_hidden_size"],
        wrapper_config=CodePredictorWrapperConfig(
            use_cuda_graphs=args.device.startswith("npu"), use_projection=True, sampling_mode="per_call"
        ),
    )
    dtype = getattr(torch, checkpoint["dtype"].split(".")[-1])
    wrapper.to(dtype=dtype)
    wrapper.load_state_dict(checkpoint["state"], strict=True)
    wrapper.to(args.device).eval().requires_grad_(False)
    if args.device.startswith("npu"):
        wrapper._prepare_npu_weights()
    return wrapper


def frames(args, split):
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    valid = {row["id"] for row in manifest["dataset"] if row["split"] == split}
    found = []
    identities = set()
    for path in sorted((args.output / "frames").glob("*.pt")):
        frame = torch.load(path, map_location="cpu", weights_only=True)
        if frame["split"] != split:
            continue
        identity = (frame["request_id"], frame["frame_id"])
        if frame["request_id"] not in valid or identity in identities:
            raise ValueError(f"unexpected or duplicate frame: {path}")
        identities.add(identity)
        found.append(frame)
    if {f["request_id"] for f in found} != valid:
        raise ValueError(f"missing captured requests in {split}; finish collect before continuing")
    return found


def inputs_for(args, model, frame, greedy=False):
    spec = spec_module()
    inputs = {k: v.to(args.device) for k, v in frame["inputs"].items()}
    inputs.update(frame["sampling"])
    inputs["do_sample"] = not greedy and frame["sampling"]["do_sample"]
    inputs["sample_uniforms"] = spec.frame_uniforms(
        frame["seed"], frame["request_id"], frame["frame_id"], model._num_groups, model.config.vocab_size, args.device
    )
    return inputs


def collect(args):
    """Prepare a server-side switch or sequentially request the fixed dataset."""
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        if (args.output / "manifest.json").exists():
            raise ValueError("use a fresh output directory; refusing to mix runs")
        if args.model_revision == "unknown":
            raise ValueError("provide --model-revision with the checkpoint commit or immutable local revision")
        lines = args.data.read_text(encoding="utf-8").splitlines()
        if len(lines) != 30:
            raise ValueError("expected exactly 30 seed_tts_smoke English texts")
        dataset = [
            dict(id=line.split("|")[0], text=line.split("|")[-1], split="calibration" if i < 10 else "test")
            for i, line in enumerate(lines)
        ]
        manifest = environment(args)
        manifest.update(
            model=args.model,
            model_revision=args.model_revision,
            seed=args.seed,
            dataset=dataset,
            dataset_sha256=digest(dataset),
            voice="Vivian",
            language="English",
            sampling=dict(temperature=0.9, top_k=50, top_p=1.0),
            graph=dict(stock="tag default predictor graphs", tree="eager explicit attention"),
            collection_is_performance=False,
        )
        write_json(args.output / "manifest.json", manifest)
        import yaml

        deploy = yaml.safe_load((ROOT / "vllm_omni/deploy/qwen3_tts.yaml").read_text(encoding="utf-8"))
        for stage in deploy["stages"]:
            stage["max_num_seqs"] = 1
        # Prevent synthetic larger MTP batches during graph warmup as well.
        deploy["stages"][0].setdefault("compilation_config", {})["cudagraph_capture_sizes"] = [1]
        (args.output / "deploy.yaml").write_text(yaml.safe_dump(deploy, sort_keys=False))
        write_json(
            args.output / "collect_config.json", dict(mode="collect", seed=args.seed, output=str(args.output.resolve()))
        )
        print(f"Set VLLM_OMNI_TTS_SPEC_CONFIG={args.output.resolve() / 'collect_config.json'} in server workers.")
        print("Start CustomVoice server with stage-0 max_num_seqs=1; then run collect without --prepare.")
        return
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    if not args.e2e_label and list((args.output / "frames").glob("*.pt")):
        raise ValueError("capture frames already exist; use a fresh experiment rather than mixing reruns")
    rows = []
    if args.e2e_label and args.e2e_label != "stock":
        correctness = json.loads((args.output / "correctness.json").read_text(encoding="utf-8"))
        if not correctness["passed"]:
            raise ValueError("full-request speculation requires passed replay correctness")
    dataset = manifest["dataset"] if not args.e2e_label else manifest["dataset"][10:]
    for row in dataset:
        if not args.e2e_label:
            write_json(args.output / "control.json", row)
        payload = dict(
            model=args.model,
            input=row["text"],
            voice="Vivian",
            language="English",
            task_type="CustomVoice",
            response_format="pcm",
            stream_format="audio",
            seed=args.seed,
        )
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/audio/speech",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start, first, size, error = time.perf_counter(), None, 0, None
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                while chunk := response.read1(65536):
                    first = first or time.perf_counter()
                    size += len(chunk)
        except Exception as exc:
            error = str(exc)
        elapsed = time.perf_counter() - start
        if size == 0 and error is None:
            error = "empty PCM response"
        duration = size / (24000 * 2)
        rows.append(
            dict(
                request_id=row["id"],
                label=args.e2e_label or "capture",
                ttfp_s=first - start if first else None,
                e2e_s=elapsed,
                audio_s=duration,
                rtf=elapsed / duration if duration else None,
                error=error,
            )
        )
        if error:
            break
    write_rows(args.output / f"requests-{args.e2e_label or 'capture'}.jsonl", rows)
    if not args.e2e_label:
        (args.output / "control.json").unlink(missing_ok=True)
        metadata = args.output / "capture_metadata.json"
        if metadata.exists():
            manifest.update(json.loads(metadata.read_text(encoding="utf-8")))
            write_json(args.output / "manifest.json", manifest)
        elif not any(row["error"] for row in rows):
            raise RuntimeError("no capture metadata: check that server workers received the research switch")
    if any(row["error"] for row in rows):
        raise RuntimeError("request failures recorded; do not interpret this run as successful")


@torch.inference_mode()
def analyze(args):
    if (args.output / "trees.json").exists():
        raise ValueError("trees already frozen; use a fresh experiment to recalibrate")
    model, spec = load_predictor(args), spec_module()
    observations = []
    for frame in frames(args, "calibration"):
        inputs = inputs_for(args, model, frame)
        prefix = spec.initial_prefix(model, inputs["layer0_embed"], inputs["last_talker_hidden"])
        codes = frame["codes"].to(args.device)
        states = []
        for position in range(1, model._num_groups):
            hidden = spec.reference_forward(model, prefix)[:, -1]
            states.append(hidden.cpu())
            ranks = []
            for future in range(position, min(position + 3, model._num_groups)):
                candidate = model.lm_head[future - 1](hidden).topk(4, dim=-1).indices
                match = (candidate == codes[:, future : future + 1]).cpu().reshape(-1).tolist()
                ranks.append(match.index(True) if True in match else -1)
            observations.append(ranks)
            embed = model.small_to_mtp_projection(
                model.model.codec_embedding[position - 1](codes[:, position : position + 1])
            )
            prefix = torch.cat((prefix, embed), dim=1)
        state_dir = args.output / "calibration_states"
        state_dir.mkdir(exist_ok=True)
        torch.save(torch.stack(states), state_dir / f"{frame['request_id']}-{frame['frame_id']:06d}.pt")
    configs = {"chain": [[0], [0, 0], [0, 0, 0]]}
    configs.update({f"tree{n}": spec.fit_tree(observations, n) for n in (8, 16, 32)})
    frozen = dict(
        configs=configs,
        calibration_observations=len(observations),
        dataset_sha256=json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))["dataset_sha256"],
    )
    frozen["sha256"] = digest(frozen)
    write_json(args.output / "trees.json", frozen)
    for name, paths in configs.items():
        write_json(args.output / f"{name}_config.json", dict(mode="tree", paths=paths, frozen_sha256=frozen["sha256"]))
    write_json(
        args.output / "coverage.json",
        {
            name: sum(
                sum(tuple(p[:d]) in {tuple(x) for x in paths} for d in range(1, len(p) + 1)) for p in observations
            )
            / len(observations)
            for name, paths in configs.items()
        },
    )


def measured(fn, device):
    api = api_for(device)
    if api:
        api.synchronize()
        start_event, end_event = api.Event(enable_timing=True), api.Event(enable_timing=True)
        start_event.record()
    start = time.perf_counter()
    result = fn()
    if api:
        end_event.record()
        api.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed, start_event.elapsed_time(end_event) if api else None


@torch.inference_mode()
def replay(args):
    model, spec = load_predictor(args), spec_module()
    test_frames = frames(args, "test")
    frozen = json.loads((args.output / "trees.json").read_text(encoding="utf-8"))
    signature = frozen.pop("sha256")
    if digest(frozen) != signature:
        raise ValueError("frozen tree digest changed")
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    if frozen["dataset_sha256"] != digest(manifest["dataset"]):
        raise ValueError("dataset changed after calibration")
    configs = frozen["configs"]
    events, diagnostics, mismatches = [], [], []
    for frame in test_frames:
        for greedy in (True, False):
            inputs = inputs_for(args, model, frame, greedy)
            baseline = model(**inputs)
            if not greedy and not torch.equal(baseline.cpu(), frame["codes"]):
                mismatches.append(
                    dict(
                        request_id=frame["request_id"],
                        frame_id=frame["frame_id"],
                        config="stock_export_replay",
                        greedy=False,
                        ar=frame["codes"].tolist(),
                        candidate=baseline.cpu().tolist(),
                    )
                )
            for name, paths in {"reference": [], **configs}.items():
                # Numerical audit is separate from diagnostic timing, otherwise
                # the serial node checks would inflate verification phase time.
                try:
                    _, audit = spec.decode(model, **inputs, paths=paths, audit=True)
                except AssertionError as exc:
                    write_json(
                        args.output / "correctness.json",
                        dict(
                            passed=False,
                            node_logit_failure=str(exc),
                            request_id=frame["request_id"],
                            frame_id=frame["frame_id"],
                            config=name,
                        ),
                    )
                    raise
                codes, info = spec.decode(model, **inputs, paths=paths, detailed=True, profile=True)
                different = (codes != baseline).cpu().nonzero().tolist()
                if different:
                    mismatches.append(
                        dict(
                            request_id=frame["request_id"],
                            frame_id=frame["frame_id"],
                            config=name,
                            greedy=greedy,
                            positions=different,
                            ar=baseline.cpu().tolist(),
                            candidate=codes.cpu().tolist(),
                        )
                    )
                identity = dict(request_id=frame["request_id"], frame_id=frame["frame_id"], config=name, greedy=greedy)
                events.extend(dict(**identity, **e) for e in info["events"])
                diagnostics.append(
                    dict(
                        **identity,
                        backbone_calls=info["backbone_calls"],
                        processed_positions=info["processed_positions"],
                        audit_max_logit_error=audit["audit_max_logit_error"],
                        phases=info["timings"],
                    )
                )
    write_rows(args.output / "events.jsonl", events)
    write_rows(args.output / "diagnostics.jsonl", diagnostics)
    write_json(
        args.output / "correctness.json",
        dict(passed=not mismatches, mismatches=mismatches, unique_frames=len(test_frames), tree_sha256=signature),
    )
    if mismatches:
        raise RuntimeError("code differences found; inspect correctness.json before performance testing")
    device = torch.device(args.device)
    api = api_for(device)
    if api:
        api.reset_peak_memory_stats()
    timing_rows = []
    total = max(200, len(test_frames))
    # Inputs/RNG construction and data transfer are outside frame timers.
    prepared = [inputs_for(args, model, frame, args.greedy) for frame in test_frames]
    for name, paths in {"reference": [], **configs}.items():
        for i in range(20):
            inp = prepared[i % len(prepared)]
            model(**inp)
            spec.decode(model, **inp, paths=paths)
        # Four rounds per comparator, ABBA with identical frame order.
        for comparator in ("stock",) if name == "reference" else ("stock", "reference"):
            for round_id, kind in enumerate((comparator, name, name, comparator)):
                for execution in range(total):
                    index = execution % len(prepared)
                    inp = prepared[index]
                    fn = (
                        (lambda: model(**inp))
                        if kind == "stock"
                        else (lambda: spec.decode(model, **inp, paths=[] if kind == "reference" else paths))
                    )
                    result, wall, event = measured(fn, device)
                    stock_flops = spec.matmul_flops(
                        model,
                        (model._num_groups - 1) * (model._num_groups + 1),
                        (model._num_groups - 1) * (model._num_groups + 1) ** 2,
                        model._num_groups - 1,
                        model._num_groups,
                    )
                    timing_rows.append(
                        dict(
                            config=name,
                            comparator=comparator,
                            kind=kind,
                            round=round_id,
                            execution=execution,
                            request_id=test_frames[index]["request_id"],
                            frame_id=test_frames[index]["frame_id"],
                            wall_ms=wall,
                            device_ms=event,
                            residual_tokens=model._num_groups - 1,
                            estimated_matmul_flops=stock_flops
                            if kind == "stock"
                            else result[1]["estimated_matmul_flops"],
                            backbone_calls=model._num_groups - 1 if kind == "stock" else result[1]["backbone_calls"],
                            processed_positions=(model._num_groups - 1) * (model._num_groups + 1)
                            if kind == "stock"
                            else result[1]["processed_positions"],
                            attention_pairs=(model._num_groups - 1) * (model._num_groups + 1) ** 2
                            if kind == "stock"
                            else result[1]["attention_pairs"],
                            attention_backend=(
                                "npu_fusion_attention_sparse_mode2" if device.type == "npu" else "torch_sdpa"
                            )
                            if kind == "stock"
                            else "explicit_fp32_matmul_softmax",
                        )
                    )
    write_rows(args.output / "timings.jsonl", timing_rows)
    write_json(
        args.output / "replay.json",
        dict(
            environment=environment(args),
            unique_frames=len(test_frames),
            executions_per_round=total,
            warmup_per_config=20,
            dtype=str(next(model.parameters()).dtype),
            peak_memory_bytes=api.max_memory_allocated() if api else None,
            graph=dict(stock_buckets=list(model._device_graphs), reference="eager", tree="eager"),
            tree_sha256=signature,
            sampling="greedy" if args.greedy else "shared_uniforms",
        ),
    )


def report(args):
    correctness = json.loads((args.output / "correctness.json").read_text(encoding="utf-8"))
    if not correctness["passed"] or not (args.output / "timings.jsonl").exists():
        write_json(args.output / "summary.json", dict(verdict="正确性未通过；性能不验收", correctness=correctness))
        (args.output / "report.md").write_text("正确性未通过，详见 correctness.json。\n", encoding="utf-8")
        return
    rows = read_rows(args.output / "timings.jsonl")
    replay_info = json.loads((args.output / "replay.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    summary = {}
    for name, comparator in sorted({(row["config"], row["comparator"]) for row in rows}):
        paired = [r for r in rows if r["config"] == name and r["comparator"] == comparator]
        ar = [r for r in paired if r["kind"] == comparator]
        candidate = [r for r in paired if r["kind"] == name]
        # Cluster bootstrap over requests, preserving frame pairing and repeated
        # executions. Repeated frames do not become independent observations.
        ids = sorted({r["request_id"] for r in ar})
        a = np.array([sum(r["wall_ms"] for r in ar if r["request_id"] == rid) for rid in ids])
        b = np.array([sum(r["wall_ms"] for r in candidate if r["request_id"] == rid) for rid in ids])
        draws = rng.integers(0, len(ids), size=(5000, len(ids)))
        ci = np.quantile(a[draws].sum(1) / b[draws].sum(1), [0.025, 0.975]).tolist()

        def stats(items):
            return dict(
                p50_ms=float(np.median([r["wall_ms"] for r in items])),
                p95_ms=float(np.quantile([r["wall_ms"] for r in items], 0.95)),
                calls_per_frame=float(np.mean([r["backbone_calls"] for r in items])),
                processed_positions_per_frame=float(np.mean([r["processed_positions"] for r in items])),
                attention_pairs_per_frame=float(np.mean([r["attention_pairs"] for r in items])),
                estimated_matmul_flops_per_frame=float(np.mean([r["estimated_matmul_flops"] for r in items])),
                effective_tokens_per_backbone_call=sum(r["residual_tokens"] for r in items)
                / sum(r["backbone_calls"] for r in items),
            )

        summary[f"{name}/{comparator}"] = dict(
            config=name,
            comparator=comparator,
            speedup=float(a.sum() / b.sum()),
            bootstrap95=ci,
            baseline=stats(ar),
            candidate=stats(candidate),
        )
    is_npu = replay_info["environment"]["device"].startswith("npu")
    verdict = "实现完成、性能待验证" if not is_npu else "零训练候选不足"
    if is_npu and any(v["candidate"]["calls_per_frame"] < v["baseline"]["calls_per_frame"] for v in summary.values()):
        verdict = "机制有潜力，执行尚未加速"
    e2e = {}
    for path in args.output.glob("requests-*.jsonl"):
        if path.stem != "requests-capture":
            e2e[path.stem.removeprefix("requests-")] = read_rows(path)
    passed = []
    for result in summary.values():
        name = result["config"]
        if name == "reference" or result["comparator"] != "stock" or not is_npu or not correctness["passed"]:
            continue
        baseline, candidate = e2e.get("stock", []), e2e.get(name, [])
        expected_ids = {
            row["id"] for row in json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))["dataset"][10:]
        }
        complete = (
            len(baseline) == len(candidate) == 20
            and {r["request_id"] for r in baseline} == expected_ids
            and {r["request_id"] for r in candidate} == expected_ids
            and all(
                not r["error"] and r["audio_s"] > 0 and r["ttfp_s"] is not None and r["rtf"] is not None
                for r in baseline + candidate
            )
        )
        # Predeclared initial engineering regression gate: <=10% for mean E2E,
        # RTF and p95 TTFP. Raw request measurements remain in the report.
        healthy = complete and all(
            np.mean([r[k] for r in candidate]) <= 1.1 * np.mean([r[k] for r in baseline]) for k in ("e2e_s", "rtf")
        )
        healthy = healthy and np.quantile([r["ttfp_s"] for r in candidate], 0.95) <= 1.1 * np.quantile(
            [r["ttfp_s"] for r in baseline], 0.95
        )
        if result["speedup"] >= 1.10 and result["bootstrap95"][0] > 1:
            if healthy:
                passed.append(name)
            else:
                verdict = "帧级加速达标；完整请求验收待完成或未通过"
    if passed:
        verdict = "可推进流式研究"
    phase_totals = {}
    for row in read_rows(args.output / "diagnostics.jsonl"):
        if row["greedy"] != (replay_info["sampling"] == "greedy"):
            continue
        totals = phase_totals.setdefault(row["config"], {})
        for phase in row["phases"]:
            totals[phase["phase"]] = totals.get(phase["phase"], 0) + phase["wall_ms"]
    phase_shares = {
        name: {k: v / sum(values.values()) for k, v in values.items()} for name, values in phase_totals.items()
    }
    output = dict(
        verdict=verdict,
        correctness=correctness,
        performance=summary,
        replay=replay_info,
        e2e=e2e,
        diagnostic_wall_phase_shares=phase_shares,
        note="30 texts are a feasibility sample, not generalization evidence; phase times are separate diagnostic runs",
    )
    write_json(args.output / "summary.json", output)
    text = f"# Qwen3-TTS zero-training experiment\n\n{verdict}\n\n"
    text += (
        "| Config/comparator | Wall speedup | Paired request-bootstrap 95% CI | "
        "Candidate p50 / p95 ms | Calls/frame |\n|---|---:|---|---|---:|\n"
    )
    for name, row in summary.items():
        c = row["candidate"]
        text += (
            f"| {name} | {row['speedup']:.3f} | {row['bootstrap95']} | "
            f"{c['p50_ms']:.3f} / {c['p95_ms']:.3f} | {c['calls_per_frame']:.2f} |\n"
        )
    text += "\nReference/tree attention: explicit FP32 matmul + softmax, eager. Stock uses tag backend/graphs.\n"
    text += "\nOne host count transfer per tree round is included. Processed positions are a work proxy, not FLOPs.\n"
    text += (
        "\nSee summary.json, diagnostics.jsonl and requests-*.jsonl "
        "for device metadata, phase timings and streaming metrics.\n"
    )
    (args.output / "report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("collect", "analyze", "replay", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--cann-version", help="explicit CANN version if local auto-discovery is unavailable")
    parser.add_argument("--data", type=Path, default=ROOT / "benchmarks/build_dataset/seed_tts_smoke/en/meta.lst")
    parser.add_argument("--vllm-repo", type=Path, default=ROOT.parent / "vllm")
    parser.add_argument("--ascend-repo", type=Path, default=ROOT.parent / "vllm-ascend")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--e2e-label", choices=("stock", "chain", "tree8", "tree16", "tree32"))
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    globals()[args.command](args)


if __name__ == "__main__":
    main()
