# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Calibrate, check and benchmark the isolated single-forward RVQ pipeline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import experiment as exp
import numpy as np
import torch
from diagnose_fusion import select_request_frames
from quick_experiment import summarize


def pipeline():
    from vllm_omni.model_executor.models.common import qwen3_code_predictor_pipeline

    return qwen3_code_predictor_pipeline


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint():
    paths = [
        *sorted((exp.ROOT / "vllm_omni/model_executor/models/common").glob("qwen3_code_predictor*.py")),
        *[
            exp.ROOT / "tools/qwen3_tts_spec" / name
            for name in ("experiment.py", "quick_experiment.py", "diagnose_fusion.py", "highperf_experiment.py")
        ],
        exp.ROOT / "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code_predictor_vllm.py",
    ]
    return {str(path.relative_to(exp.ROOT)): file_hash(path) for path in paths}


def source_fingerprint(args):
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_sha256"] != exp.digest(manifest["dataset"]):
        raise ValueError("source dataset digest changed")
    return dict(
        model_sha256=file_hash(args.input / "predictor.pt"),
        manifest_sha256=file_hash(args.input / "manifest.json"),
        dataset_sha256=manifest["dataset_sha256"],
    )


def frame_fingerprints(args):
    return {p.name: file_hash(p) for p in sorted((args.input / "frames").glob("*.pt"))}


def modes(args):
    return ("greedy", "sampling") if args.sampling == "both" else (args.sampling,)


def identity(frame):
    return dict(request_id=frame["request_id"], frame_id=frame["frame_id"])


def frozen_trees(args):
    frozen = json.loads(args.trees.read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in frozen.items() if k != "sha256"}
    if exp.digest(unsigned) != frozen["sha256"]:
        raise ValueError("tree signature changed")
    if frozen["candidate_layout"] != pipeline().CANDIDATE_LAYOUT:
        raise ValueError("old/current candidate layout cannot be used by the pipeline")
    if frozen["source"] != source_fingerprint(args):
        raise ValueError("tree model/dataset source changed")
    for paths in frozen["configs"].values():
        exp.spec_module().validate_tree(paths)
    return frozen


def selected_frames(args):
    selected = select_request_frames(exp.frames(args, "test"), args.frames_per_request)
    if not selected:
        raise ValueError("empty test selection")
    return selected


def fixed_probe_frames(frames):
    chosen = []
    for rid, number in (("smoke011", 0), ("smoke011", 2), ("smoke030", None)):
        found = [f for f in frames if f["request_id"] == rid and (number is None or f["frame_id"] == number)]
        if not found:
            raise ValueError(f"required small-check frame missing: {rid}/{number}")
        chosen.append(max(found, key=lambda f: f["frame_id"]))
    return chosen


def binding(args, frozen, selected):
    return dict(
        source=frozen["source"],
        tree_sha256=frozen["sha256"],
        code=code_fingerprint(),
        frames=frame_fingerprints(args),
        selected_frames=[identity(f) for f in selected],
        sampling=list(modes(args)),
        graph_scope=getattr(args, "graph_scope", "step"),
        pipeline_optimization=getattr(args, "pipeline_optimization", "optimized"),
        sampling_parameters=sorted(
            {
                json.dumps(
                    dict(
                        do_sample=mode == "sampling" and f["sampling"].get("do_sample", True),
                        temperature=f["sampling"].get("temperature", 0.9),
                        top_k=f["sampling"].get("top_k", 50),
                        top_p=f["sampling"].get("top_p", 1.0),
                    ),
                    sort_keys=True,
                )
                for f in selected
                for mode in modes(args)
            }
        ),
        device=args.device,
        environment=exp.environment(args),
    )


@torch.inference_mode()
def calibrate(args, progress):
    spec = exp.spec_module()
    model = exp.load_predictor(args)
    frames = exp.frames(args, "calibration")
    observations, reused, recomputed = [], 0, 0
    state_hashes = {}
    progress.begin("校准未来三个位置", len(frames))
    for frame in frames:
        inputs = exp.inputs_for(args, model, frame)
        state_path = args.input / "calibration_states" / f"{frame['request_id']}-{frame['frame_id']:06d}.pt"
        states = torch.load(state_path, map_location=args.device, weights_only=True) if state_path.exists() else None
        if states is not None:
            if tuple(states.shape) != (model._num_groups - 1, 1, model.config.hidden_size):
                raise ValueError(f"invalid saved calibration hidden shape: {state_path}")
            state_hashes[state_path.name] = file_hash(state_path)
        prefix = spec.initial_prefix(model, inputs["layer0_embed"], inputs["last_talker_hidden"])
        codes = frame["codes"].to(args.device)
        for position in range(1, model._num_groups - 1):
            if states is not None:
                hidden = states[position - 1]
                reused += 1
            else:
                hidden = spec.reference_forward(model, prefix)[:, -1]
                recomputed += 1
            ranks = []
            # hidden predicts known tail c_position. Only *later* positions
            # enter the new tree: the first candidate is c_(position+1).
            for future in range(position + 1, min(position + 4, model._num_groups)):
                candidates = model.lm_head[future - 1](hidden).topk(4, dim=-1).indices
                matches = (candidates == codes[:, future : future + 1]).flatten().cpu().tolist()
                ranks.append(matches.index(True) if True in matches else -1)
            observations.append(ranks)
            if states is None:
                embed = model.model.codec_embedding[position - 1](codes[:, position : position + 1])
                prefix = torch.cat((prefix, model.small_to_mtp_projection(embed)), dim=1)
        progress.advance(str(identity(frame)))
    frozen = dict(
        candidate_layout=pipeline().CANDIDATE_LAYOUT,
        source=source_fingerprint(args),
        calibration_frames=frame_fingerprints(args),
        calibration_state_hashes=state_hashes,
        reused_hidden_states=reused,
        recomputed_hidden_states=recomputed,
        observations=len(observations),
        configs={f"tree{n}": spec.fit_tree(observations, n) for n in (8, 16, 32)},
    )
    frozen["sha256"] = exp.digest(frozen)
    exp.write_json(args.results / "trees.json", frozen)
    exp.write_json(
        args.results / "manifest.json",
        dict(
            source=frozen["source"],
            code=code_fingerprint(),
            environment=exp.environment(args),
            candidate_layout=pipeline().CANDIDATE_LAYOUT,
        ),
    )
    for name, paths in frozen["configs"].items():
        for engine in ("pipeline_eager", "pipeline_graph"):
            exp.write_json(
                args.results / f"{name}_{engine}_config.json",
                dict(
                    mode="tree",
                    engine=engine,
                    paths=paths,
                    candidate_layout=pipeline().CANDIDATE_LAYOUT,
                    frozen_sha256=frozen["sha256"],
                    correctness=None,
                ),
            )


@contextmanager
def stock_eager(model):
    saved = model._device_graphs
    model._device_graphs = {}
    try:
        yield
    finally:
        model._device_graphs = saved


class Engines:
    def __init__(self, model, frozen, configs, progress=None, graph_scope="step", pipeline_optimization="optimized"):
        self.model, self.frozen, self.configs = model, frozen, configs
        self.runners = {}
        self.progress = progress
        self.graph_scope = graph_scope
        self.pipeline_optimization = pipeline_optimization
        self.released_graphs = {}

    def runner(self, name, inputs):
        mode = bool(inputs["do_sample"])
        key = (name, mode)
        if key not in self.runners:
            if name == "cached_ar":
                paths, cache, execution = (), True, "eager"
            else:
                config, engine = name.split("/", 1)
                paths = self.frozen["configs"][config]
                cache, execution = engine != "pipeline_nocache", "graph" if engine == "pipeline_graph" else "eager"
            self.runners[key] = pipeline().PipelineExecutor(
                self.model,
                paths,
                cache=cache,
                execution=execution,
                do_sample=mode,
                temperature=inputs.get("temperature", 0.9),
                top_k=inputs.get("top_k", 50),
                graph_scope=self.graph_scope,
                optimization=self.pipeline_optimization,
            )
            if self.progress:
                self.runners[key].capture_progress = lambda p, n, total: self.progress.message(
                    f"捕获 {name} sampling={mode} 位置={p}：{n}/{total}"
                )
        return self.runners[key]

    def names(self, graph):
        return [
            "stock_graph",
            "stock_eager",
            "cached_ar",
            *[
                f"{config}/{engine}"
                for config in self.configs
                for engine in (
                    ("current", "pipeline_nocache", "pipeline_eager", "pipeline_graph")
                    if graph
                    else ("current", "pipeline_nocache", "pipeline_eager")
                )
            ],
        ]

    def decode(self, name, inputs, **kwargs):
        if name.startswith("stock"):
            if name == "stock_eager":
                with stock_eager(self.model):
                    result = self.model(**inputs)
            else:
                result = self.model(**inputs)
            calls = self.model._num_groups - 1
            return result, dict(
                backbone_calls=calls,
                root_backbone_calls=calls,
                prefill_backbone_calls=0,
                verification_backbone_calls=0,
                supplemental_backbone_calls=0,
                graph_replays=calls if name == "stock_graph" and self.model._device_graphs else 0,
                events=[],
            )
        if name.endswith("/current"):
            config = name.split("/")[0]
            return exp.spec_module().decode(self.model, **inputs, paths=self.configs[config], **kwargs)
        return self.runner(name, inputs).decode(**inputs, **kwargs)

    def graph_info(self):
        return {
            **self.released_graphs,
            **{
                f"{name}/{'sampling' if sample else 'greedy'}": dict(
                    **runner.capture_metadata(),
                    graphs=len(runner.graphs),
                    cache=runner.cache,
                    execution=runner.execution,
                )
                for (name, sample), runner in self.runners.items()
            },
        }

    def release_graphs(self):
        """Never retain greedy, sampling and all tree budgets' graphs together."""
        removed = 0
        for key, runner in list(self.runners.items()):
            if runner.execution != "graph":
                continue
            name, sample = key
            record = dict(
                **runner.capture_metadata(),
                graphs=len(runner.graphs),
                cache=runner.cache,
                execution=runner.execution,
                released=True,
                resident_graphs=0,
            )
            runner.close()
            self.released_graphs[f"{name}/{'sampling' if sample else 'greedy'}"] = record
            del self.runners[key]
            removed += record["graphs"]
        if removed:
            gc.collect()
            if next(self.model.parameters()).device.type == "npu":
                torch.npu.empty_cache()
            if self.progress:
                self.progress.message(f"已在计时区间外释放 {removed} 张流水图；不跨采样模式驻留。")


def current_configs(args):
    old = json.loads((args.input / "trees.json").read_text(encoding="utf-8"))
    if exp.digest({k: v for k, v in old.items() if k != "sha256"}) != old["sha256"]:
        raise ValueError("current reference tree signature changed")
    if old["dataset_sha256"] != source_fingerprint(args)["dataset_sha256"]:
        raise ValueError("current reference tree dataset changed")
    return {k: v for k, v in old["configs"].items() if k in ("tree8", "tree16", "tree32")}, old["sha256"]


def save_failure(args, engines, engine, inputs, expected, error, context):
    # capture_end can leave the device/stream poisoned. Persist the original
    # exception *before* any device read and never snapshot a failed capture.
    chain, seen, cause = [], set(), error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        chain.append(repr(cause))
        cause = cause.__cause__ or cause.__context__
    metadata = dict(
        context=context,
        error=repr(error),
        exception_chain=chain,
        source_capture=str(getattr(args, "input", "")),
        graphs=engines.graph_info(),
        input_metadata={
            k: dict(shape=list(v.shape), dtype=str(v.dtype), device=str(v.device)) if torch.is_tensor(v) else repr(v)
            for k, v in inputs.items()
        },
    )
    exp.write_json(args.results / "failure.json", metadata)
    payload = dict(metadata)
    runner = engines.runners.get((engine, bool(inputs.get("do_sample", True))))
    unsafe = getattr(error, "device_snapshot_unsafe", False) or bool(runner and runner.capture_error)
    if unsafe:
        payload["device_snapshot_skipped"] = "capture failed; synchronous D2H may also fail"
    else:
        try:
            payload["inputs"] = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in inputs.items()}
            payload["expected"] = expected.detach().cpu() if expected is not None else None
            if engine.endswith("/current") and hasattr(error, "payload"):
                payload["node"] = error.payload
            elif runner:
                payload["pipeline"] = runner.snapshot()
        except Exception as snapshot_error:
            # The original exception remains the one re-raised by the caller.
            metadata["snapshot_error"] = repr(snapshot_error)
            payload["snapshot_error"] = repr(snapshot_error)
    if unsafe:
        metadata["device_snapshot_skipped"] = payload["device_snapshot_skipped"]
    exp.write_json(args.results / "failure.json", metadata)
    torch.save(payload, args.results / "failure.pt")


@torch.inference_mode()
def check(args, progress):
    frozen = frozen_trees(args)
    test_frames = exp.frames(args, "test")
    tiny = fixed_probe_frames(test_frames)
    probe_only = getattr(args, "probe_only", False)
    selected = tiny if probe_only else selected_frames(args)
    configs, old_hash = current_configs(args)
    requested = getattr(args, "configs", ["tree16"])
    configs = {name: configs[name] for name in requested}
    result = dict(
        passed=False,
        binding=binding(args, frozen, selected),
        current_tree_sha256=old_hash,
        checks=[],
        stage="initializing",
        graph_supported=args.device.startswith("npu"),
    )
    exp.write_json(args.results / "correctness.json", result)
    exp.write_json(args.results / "manifest.json", result["binding"])
    model = exp.load_predictor(args)
    optimization = getattr(args, "pipeline_optimization", "optimized")
    engines = Engines(model, frozen, configs, progress, getattr(args, "graph_scope", "step"), optimization)
    baseline = Engines(
        model, frozen, configs, graph_scope=getattr(args, "graph_scope", "step"), pipeline_optimization="baseline"
    )
    result["predictor"] = dict(
        groups=model._num_groups,
        hidden_size=model.config.hidden_size,
        vocab_size=model.config.vocab_size,
        dtype=str(next(model.parameters()).dtype),
    )
    first_per_request = {
        f["request_id"]: min(x["frame_id"] for x in selected if x["request_id"] == f["request_id"]) for f in selected
    }
    graph = args.device.startswith("npu")
    all_names = engines.names(graph)
    small_stages = [
        ("small_cached_ar", ["cached_ar"], tiny),
        ("small_eager", [n for n in all_names if n != "cached_ar" and "graph" not in n], tiny),
    ]
    if graph:
        small_stages.append(("small_graph", [n for n in all_names if "graph" in n], tiny))
    variants = pipeline().OPTIMIZATIONS if optimization == "optimized" else (optimization,)
    stages = [(*stage, variant) for variant in variants for stage in small_stages]
    if not probe_only:
        stages.extend(
            [
                ("sampled_cached_ar", ["cached_ar"], selected, optimization),
                (
                    "sampled_eager",
                    [n for n in all_names if n != "cached_ar" and "graph" not in n],
                    selected,
                    optimization,
                ),
            ]
        )
    if graph and not probe_only:
        stages.append(("sampled_graph", [n for n in all_names if "graph" in n], selected, optimization))
    events = []
    trace_changes, graph_records = [], {}
    for stage, names, frames, variant in stages:
        if engines.pipeline_optimization != variant:
            engines = Engines(model, frozen, configs, progress, getattr(args, "graph_scope", "step"), variant)
        result["stage"] = stage
        progress.begin(f"{stage} optimization={variant}", len(frames) * len(names) * len(modes(args)))
        for mode in modes(args):
            for frame in frames:
                baseline_info = {}
                inputs = exp.inputs_for(args, model, frame, greedy=mode == "greedy")
                expected = model(**inputs)
                # Establish that the stock path still reproduces the capture.
                if mode == "sampling" and not torch.equal(expected.cpu(), frame["codes"]):
                    torch.save(
                        dict(inputs=inputs, actual=expected.cpu(), captured=frame["codes"]), args.results / "failure.pt"
                    )
                    raise RuntimeError("stock sampling differs from captured codes")
                for name in names:
                    context = dict(
                        stage=stage, engine=name, sampling=mode, pipeline_optimization=variant, **identity(frame)
                    )
                    audit = (name == "cached_ar" and stage.startswith("small")) or (
                        "pipeline" in name
                        and (
                            stage.startswith("small")
                            or (
                                name.startswith("tree16/")
                                and frame["frame_id"] == first_per_request[frame["request_id"]]
                            )
                        )
                    )
                    try:
                        actual, info = engines.decode(name, inputs, audit=audit, record_events=True)
                        if not torch.equal(expected, actual):
                            raise AssertionError(
                                f"codec codes differ: expected={expected.cpu().tolist()}, "
                                f"actual={actual.cpu().tolist()}"
                            )
                        if "pipeline_" in name and variant != "baseline":
                            config = name.split("/")[0]
                            if config not in baseline_info:
                                base_codes, base_info = baseline.decode(
                                    f"{config}/pipeline_eager", inputs, record_events=True
                                )
                                if not torch.equal(expected, base_codes):
                                    raise AssertionError("baseline pipeline codes differ from stock")
                                baseline_info[config] = base_info
                            reference = baseline_info[config]["events"]
                            path_keys = ("position", "committed", "accepted", "accepted_path")
                            paths_equal = [{k: e[k] for k in path_keys} for e in reference] == [
                                {k: e[k] for k in path_keys} for e in info["events"]
                            ]
                            draft_equal = [e["draft_candidates"] for e in reference] == [
                                e["draft_candidates"] for e in info["events"]
                            ]
                            if not paths_equal or not draft_equal:
                                trace_changes.append(
                                    dict(
                                        **context,
                                        paths_equal=paths_equal,
                                        draft_equal=draft_equal,
                                        baseline=reference,
                                        candidate=info["events"],
                                    )
                                )
                    except Exception as exc:
                        save_failure(args, engines, name, inputs, expected, exc, context)
                        exp.write_rows(args.results / "events.jsonl", events)
                        exp.write_rows(args.results / "candidate-path-changes.jsonl", trace_changes)
                        result.update(failure=context, error=repr(exc), graphs=engines.graph_info())
                        exp.write_json(args.results / "correctness.json", result)
                        raise
                    result["checks"].append(
                        dict(
                            **context,
                            code_equal=True,
                            node_audit=audit,
                            audit_max_logit_error=info.get("audit_max_logit_error"),
                        )
                    )
                    events.extend(dict(**context, **event) for event in info["events"])
                    progress.advance(str(context))
                exp.write_json(args.results / "correctness.json", result)
            engines.release_graphs()
            graph_records.update({f"{variant}/{k}": v for k, v in engines.graph_info().items()})
    result.update(
        passed=True,
        stage="complete",
        engines=all_names,
        configs=list(configs),
        graphs=graph_records,
        candidate_or_path_changes=len(trace_changes),
        scope="fixed_probe_only" if probe_only else "sampled_frame_correctness",
    )
    exp.write_rows(args.results / "events.jsonl", events)
    exp.write_rows(args.results / "candidate-path-changes.jsonl", trace_changes)
    result["events_sha256"] = file_hash(args.results / "events.jsonl")
    exp.write_json(args.results / "correctness.json", result)
    (args.results / "report.md").write_text(
        f"# 树流水抽样正确性\n\n{len(selected)} 个独立帧，{len(result['checks'])} 项检查通过。\n\n"
        f"NPU graph 检查：{'已执行' if graph else '未执行；CPU 结果不能证明 NPU 正确性'}。\n\n"
        f"检查范围：{'仅固定 3 帧预检，不能用于 bench' if probe_only else '抽样帧检查'}。\n\n"
        f"优化档位：{optimization}；小检查档位：{', '.join(variants)}。\n\n"
        f"相对 baseline 的候选／接受路径变化：{len(trace_changes)} 项，详见 candidate-path-changes.jsonl。\n\n"
        "尚未运行完整请求；该结果不是性能结论。\n",
        encoding="utf-8",
    )


def require_certificate(certificate, expected_binding, old_hash, names, configs):
    if not certificate.get("passed") or certificate.get("stage") != "complete":
        raise ValueError("correctness certificate is incomplete or failed")
    if certificate.get("scope") != "sampled_frame_correctness":
        raise ValueError("probe-only certificate cannot authorize performance testing")
    if certificate.get("binding") != expected_binding or certificate.get("current_tree_sha256") != old_hash:
        raise ValueError("correctness artifact does not match code/model/trees/frames/device/environment/sampling")
    if not set(names).issubset(certificate.get("engines", [])) or not set(configs).issubset(
        certificate.get("configs", [])
    ):
        raise ValueError("requested engines/configs were not checked")


@torch.inference_mode()
def bench(args, progress):
    frozen, selected = frozen_trees(args), selected_frames(args)
    configs, old_hash = current_configs(args)
    for name in args.configs:
        if name not in frozen["configs"]:
            raise ValueError(f"unknown frozen configuration: {name}")
    if args.executions_per_round < len(selected) or args.warmup < 1:
        raise ValueError("each timing round must cover all selected frames; warmup must be positive")
    certificate = json.loads(args.correctness.read_text(encoding="utf-8"))
    fingerprint = binding(args, frozen, selected)
    model = exp.load_predictor(args)
    engines = Engines(
        model,
        frozen,
        {k: configs[k] for k in args.configs},
        progress,
        getattr(args, "graph_scope", "step"),
        getattr(args, "pipeline_optimization", "optimized"),
    )
    names = [n for n in engines.names(args.device.startswith("npu")) if n != "cached_ar"]
    require_certificate(certificate, fingerprint, old_hash, names, args.configs)
    events_path = args.correctness.parent / "events.jsonl"
    if file_hash(events_path) != certificate.get("events_sha256"):
        raise ValueError("correctness events changed since verification")
    acceptance = [
        r
        for r in exp.read_rows(events_path)
        if r["stage"].startswith("sampled_") and r["engine"].split("/")[0] in args.configs
    ]
    exp.write_rows(args.results / "correctness-events.jsonl", acceptance)
    exp.write_json(
        args.results / "manifest.json",
        dict(
            binding=fingerprint,
            current_tree_sha256=old_hash,
            correctness_sha256=file_hash(args.correctness),
            configs=args.configs,
            warmup=args.warmup,
            executions_per_round=args.executions_per_round,
            performance_pass_threshold=None,
            candidate_or_path_changes=certificate.get("candidate_or_path_changes", 0),
        ),
    )
    api = exp.api_for(torch.device(args.device))
    rows, profiles, memory = [], [], {}
    for mode in modes(args):
        prepared = [exp.inputs_for(args, model, f, greedy=mode == "greedy") for f in selected]
        # Initialize stock capture before any eager comparison.
        model(**prepared[0])
        if args.device.startswith("npu") and not model._device_graphs:
            raise RuntimeError("stock NPU graph was not captured; refusing an unlabeled eager baseline")
        for name in names:
            progress.begin(f"{mode} 准备及预热 {name}", args.warmup)
            i = 0
            try:
                if "/pipeline_" in name:
                    engines.runner(name, prepared[0]).prepare(prepared[0])
                for i in range(args.warmup):
                    engines.decode(name, prepared[i % len(prepared)])
                    progress.advance()
            except Exception as exc:
                save_failure(
                    args,
                    engines,
                    name,
                    prepared[i % len(prepared)],
                    None,
                    exc,
                    dict(stage=progress.stage, engine=name, sampling=mode),
                )
                exp.write_json(args.results / "graphs.json", engines.graph_info())
                raise
        groups = [(name, "stock_graph") for name in names if name != "stock_graph"]
        for config in args.configs:
            groups.append((f"{config}/pipeline_eager", f"{config}/pipeline_nocache"))
            if args.device.startswith("npu"):
                groups.append((f"{config}/pipeline_graph", f"{config}/pipeline_eager"))
        for name, comparator in groups:
            for round_id, kind in enumerate((comparator, name, name, comparator)):
                if api:
                    api.synchronize()
                    api.reset_peak_memory_stats()
                progress.begin(f"{mode} {name}/{comparator} round={round_id + 1}/4 {kind}", args.executions_per_round)
                round_rows = []
                for i in range(args.executions_per_round):
                    index = i % len(prepared)
                    try:
                        (_, info), wall, device_ms = exp.measured(
                            lambda: engines.decode(kind, prepared[index]), torch.device(args.device)
                        )
                    except Exception as exc:
                        save_failure(
                            args,
                            engines,
                            kind,
                            prepared[index],
                            None,
                            exc,
                            dict(stage=progress.stage, engine=kind, sampling=mode, **identity(selected[index])),
                        )
                        exp.write_json(args.results / "graphs.json", engines.graph_info())
                        raise
                    round_rows.append(
                        dict(
                            config=name,
                            comparator=comparator,
                            kind=kind,
                            sampling=mode,
                            round=round_id,
                            execution=i,
                            **identity(selected[index]),
                            wall_ms=wall,
                            device_ms=device_ms,
                            residual_tokens=model._num_groups - 1,
                            **{
                                key: info.get(key, 0)
                                for key in (
                                    "backbone_calls",
                                    "prefill_backbone_calls",
                                    "verification_backbone_calls",
                                    "supplemental_backbone_calls",
                                    "graph_replays",
                                )
                            },
                            root_backbone_calls=info.get("root_backbone_calls", info.get("prefill_backbone_calls", 0)),
                            draft_backbone_calls=info.get("root_backbone_calls", 0) if kind.endswith("/current") else 0,
                            ar_backbone_calls=model._num_groups - 1 if kind.startswith("stock") else 0,
                            attention_backend=(
                                "npu_fusion_attention_sparse_mode2" if args.device.startswith("npu") else "torch_sdpa"
                            )
                            if kind.startswith("stock")
                            else exp.spec_module().attention_backend(torch.device(args.device)),
                        )
                    )
                    progress.advance()
                with (args.results / "timings.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write("".join(json.dumps(row) + "\n" for row in round_rows))
                rows.extend(round_rows)
                if api:
                    memory[f"{mode}/{name}/{comparator}/{round_id}"] = dict(
                        allocated_peak=api.max_memory_allocated(), reserved_peak=api.max_memory_reserved()
                    )
        # Diagnostics use a separate eager executor, never the timed graph.
        probe = select_request_frames(
            [f for f in selected if f["request_id"] in sorted({f["request_id"] for f in selected})[:2]],
            3,
        )
        progress.begin(f"{mode} 独立 6 帧阶段分析", len(probe))
        for frame in probe:
            inputs = exp.inputs_for(args, model, frame, greedy=mode == "greedy")
            _, info = engines.decode(
                "tree16/pipeline_eager" if "tree16" in args.configs else f"{args.configs[0]}/pipeline_eager",
                inputs,
                record_events=True,
                profile=True,
            )
            profiles.append(dict(sampling=mode, **identity(frame), **info))
            progress.advance()
        engines.release_graphs()
    exp.write_rows(args.results / "profiles.jsonl", profiles)
    exp.write_json(
        args.results / "graphs.json",
        dict(
            engines.graph_info(),
            stock=dict(
                buckets=[str(key) for key in model._device_graphs],
                execution="graph" if model._device_graphs else "eager_cpu",
            ),
        ),
    )
    exp.write_json(args.results / "memory.json", memory)
    report(args, progress)


@torch.inference_mode()
def profile(args, progress):
    """Six-frame native graph traces; never used as formal latency results."""
    if not args.device.startswith("npu"):
        raise ValueError("native graph profiling requires NPU; CPU cannot validate its performance")
    if len(args.configs) != 1:
        raise ValueError("profile one tree budget per process")
    if args.warmup < 1:
        raise ValueError("profile warmup must be positive")
    frozen, selected = frozen_trees(args), selected_frames(args)
    configs, old_hash = current_configs(args)
    certificate = json.loads(args.correctness.read_text(encoding="utf-8"))
    fingerprint = binding(args, frozen, selected)
    config = args.configs[0]
    require_certificate(certificate, fingerprint, old_hash, ["stock_graph", f"{config}/pipeline_graph"], [config])
    import torch_npu

    probe = [f for f in selected if f["request_id"] in sorted({f["request_id"] for f in selected})[:2]]
    probe = select_request_frames(probe, 3)
    if len(probe) != 6:
        raise ValueError("profiling requires two requests with first/middle/last frames (6 distinct frames)")
    variants = pipeline().OPTIMIZATIONS if args.pipeline_optimization == "optimized" else (args.pipeline_optimization,)
    exp.write_json(
        args.results / "manifest.json",
        dict(
            binding=fingerprint,
            scope="six_frame_graph_diagnostic_not_benchmark",
            frames=[identity(f) for f in probe],
            optimizations=list(variants),
            warmup=args.warmup,
            correctness_sha256=file_hash(args.correctness),
        ),
    )
    model = exp.load_predictor(args)
    graphs = {}
    for mode in modes(args):
        prepared = [exp.inputs_for(args, model, f, greedy=mode == "greedy") for f in probe]
        expected = [model(**inputs).cpu() for inputs in prepared]
        for variant in ("stock", *variants):
            label = "stock_graph" if variant == "stock" else variant
            engines = Engines(
                model,
                frozen,
                {config: configs[config]},
                progress,
                getattr(args, "graph_scope", "step"),
                "baseline" if variant == "stock" else variant,
            )
            name = "stock_graph" if variant == "stock" else f"{config}/pipeline_graph"
            progress.begin(f"{mode}/{label} 预热（捕获不进入 profiler）", args.warmup)
            try:
                for i in range(args.warmup):
                    engines.decode(name, prepared[i % len(prepared)])
                    progress.advance()
                torch.npu.synchronize()
                outputs = []
                progress.begin(f"{mode}/{label} 六帧 profiling", len(probe))
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.results / mode / label)),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        export_type=torch_npu.profiler.ExportType.Text,
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    ),
                ) as profiler:
                    for i, inputs in enumerate(prepared):
                        with torch.profiler.record_function(f"{mode}/{label}/frame{i}"):
                            outputs.append(engines.decode(name, inputs)[0])
                        profiler.step()
                    torch.npu.synchronize()
                # Comparisons and progress writes stay outside the trace.
                for i, actual in enumerate(outputs):
                    if not torch.equal(actual.cpu(), expected[i]):
                        raise AssertionError(f"profile codes differ: {mode}/{label}/{identity(probe[i])}")
                    progress.advance()
            except Exception as exc:
                save_failure(
                    args,
                    engines,
                    name,
                    prepared[min(i, len(prepared) - 1)],
                    None,
                    exc,
                    dict(stage="native_profile", sampling=mode, pipeline_optimization=variant),
                )
                raise
            engines.release_graphs()
            graphs[f"{mode}/{label}"] = engines.graph_info()
            exp.write_json(args.results / "graphs.json", graphs)
    (args.results / "report.md").write_text(
        "# 六帧图执行诊断\n\nstock 与四档流水的 trace、kernel_details.csv、op_statistic.csv 分目录保存。\n\n"
        "捕获和预热在 profiler 外；未进行逐节点审计。该结果不替代无 profiler 的 ABBA 性能测试。\n",
        encoding="utf-8",
    )


def report(args, progress):
    rows = exp.read_rows(args.results / "timings.jsonl")
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    output = dict(
        performance_pass_threshold=None,
        scope="sampled_frames_not_complete_requests",
        device=manifest["binding"]["device"],
        graph_scope=manifest["binding"]["graph_scope"],
        pipeline_optimization=manifest["binding"].get("pipeline_optimization", "baseline"),
        candidate_or_path_changes=manifest.get("candidate_or_path_changes", 0),
        modes={},
    )
    lines = [
        "# 树流水性能实验",
        "",
        "不设置性能通过门槛；分别观察功能、调用次数和耗时。",
        "",
        f"设备：{output['device']}。独立帧：{len(manifest['binding']['selected_frames'])}。",
        f"图范围：{output['graph_scope']}。step 为整轮图；backbone 为 prefill＋共享主干图，其他步骤为 eager。",
        f"流水优化档位：{output['pipeline_optimization']}。",
        f"正确性轮相对 baseline 的候选／接受路径变化：{output['candidate_or_path_changes']} 项。"
        "存在变化时，调用数差异包含路径变化贡献，不能全部归为执行优化收益。",
        "CPU 运行只验证工具；NPU 结论必须来自真实 NPU 数据。",
        "",
        "| 模式／配置／对照 | 总调用/帧 | prefill/draft/AR/验证/补算 | 调用减少 | 墙钟 p50/p95 ms | 加速比 | 95%区间 |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for mode in sorted({r["sampling"] for r in rows}):
        mode_rows = [r for r in rows if r["sampling"] == mode]
        metrics = summarize(mode_rows)
        pairs = {f"{r['config']}/{r['comparator']}": (r["config"], r["comparator"]) for r in mode_rows}
        for comparison, item in metrics.items():
            name, comparator = pairs[comparison]
            candidates = [
                r for r in mode_rows if r["config"] == name and r["comparator"] == comparator and r["kind"] == name
            ]
            item["candidate"]["call_breakdown"] = {
                k: float(np.mean([r[k] for r in candidates]))
                for k in (
                    "prefill_backbone_calls",
                    "draft_backbone_calls",
                    "ar_backbone_calls",
                    "verification_backbone_calls",
                    "supplemental_backbone_calls",
                    "graph_replays",
                )
            }
            c, calls = item["candidate"], item["candidate"]["call_breakdown"]
            lines.append(
                f"| {mode}/{comparison} | {c['calls_per_frame']:.2f} | "
                f"{calls['prefill_backbone_calls']:.2f}/{calls['draft_backbone_calls']:.2f}/"
                f"{calls['ar_backbone_calls']:.2f}/{calls['verification_backbone_calls']:.2f}/"
                f"{calls['supplemental_backbone_calls']:.2f} | "
                f"{item['total_call_reduction_fraction']:.2%} | "
                f"{c['wall_p50_ms']:.3f}/{c['wall_p95_ms']:.3f} | {item['speedup']:.3f} | "
                f"{item['exploratory_request_bootstrap95']} |"
            )
        output["modes"][mode] = metrics
    profiles = exp.read_rows(args.results / "profiles.jsonl")
    output["functional_profiles"] = []
    for mode in sorted({r["sampling"] for r in profiles}):
        events = [e for row in profiles if row["sampling"] == mode for e in row["events"]]
        output["functional_profiles"].append(
            dict(
                sampling=mode,
                accepted_distribution=dict(Counter(e["accepted"] for e in events)),
                committed_distribution=dict(Counter(e["committed"] for e in events)),
                bootstrap_committed=sum(r["bootstrap_committed"] for r in profiles if r["sampling"] == mode),
                rounds=len(events),
            )
        )
    output["graphs"] = json.loads((args.results / "graphs.json").read_text(encoding="utf-8"))
    output["memory"] = json.loads((args.results / "memory.json").read_text(encoding="utf-8"))
    acceptance_rows = exp.read_rows(args.results / "correctness-events.jsonl")
    output["sampled_acceptance"] = {}
    for engine, mode in sorted({(r["engine"], r["sampling"]) for r in acceptance_rows}):
        events = [r for r in acceptance_rows if (r["engine"], r["sampling"]) == (engine, mode)]
        output["sampled_acceptance"][f"{engine}/{mode}"] = dict(
            rounds=len(events),
            accepted_distribution=dict(Counter(r["accepted"] for r in events)),
            committed_distribution=dict(Counter(r["committed"] for r in events)),
            independent_frames=len({(r["request_id"], r["frame_id"]) for r in events}),
        )
    lines.extend(
        [
            "",
            "current 使用旧候选位置；pipeline 三种模式使用相同的新树。",
            "比较 pipeline_nocache 与 pipeline_eager 隔离缓存收益；再比较 pipeline_graph 观察图执行收益。",
            "计时包含输入暂存、候选、主干、采样、状态同步与缓存提交。",
            "设备事件区间可能包含主机提交空隙，不是纯 kernel 时间。",
            "profiles.jsonl 是独立诊断轮；attention_gather_nested 包含在 backbone_inclusive 中，不能重复相加。",
            "summary.json 包含同批抽样帧的接受/提交分布；profiles.jsonl 仅用于 6 帧阶段分析。",
            "内存峰值包含同时驻留的执行器与图，不能视为单一配置独占内存。",
            "首次 prefill 产生 1 个 token；后续每轮提交数不重复计入已输出的待处理根 token。",
            "尚未测量完整请求或流式播放指标。",
        ]
    )
    exp.write_json(args.results / "summary.json", output)
    (args.results / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress.message(f"报告已保存：{args.results / 'report.md'}")


def parser():
    root = argparse.ArgumentParser(__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "check", "bench", "profile", "report"):
        command = commands.add_parser(name)
        command.add_argument("--results", type=Path, required=True)
        command.add_argument("--progress-interval", type=float, default=10)
        if name == "report":
            continue
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--device", default="npu:0")
        command.add_argument("--vllm-repo", type=Path, default=exp.ROOT.parent / "vllm")
        command.add_argument("--ascend-repo", type=Path, default=exp.ROOT.parent / "vllm-ascend")
        if name != "calibrate":
            command.add_argument("--trees", type=Path, required=True)
            command.add_argument("--frames-per-request", type=int, default=3)
            command.add_argument("--sampling", choices=("both", "greedy", "sampling"), default="both")
            command.add_argument(
                "--pipeline-optimization",
                choices=("baseline", "kv_commit", "batched", "optimized"),
                default="optimized",
            )
            command.add_argument(
                "--graph-scope",
                choices=("backbone", "step"),
                default="step",
                help="backbone captures two small graphs; step captures each complete position",
            )
            command.add_argument("--configs", nargs="+", choices=("tree8", "tree16", "tree32"), default=["tree16"])
        if name == "check":
            command.add_argument(
                "--probe-only", action="store_true", help="only fixed 3 frames; cannot authorize bench"
            )
        if name in ("bench", "profile"):
            command.add_argument("--correctness", type=Path, required=True)
            command.add_argument("--warmup", type=int, default=20)
            if name == "bench":
                command.add_argument("--executions-per-round", type=int, default=60)
    return root


@torch.inference_mode()
def run(args):
    if args.command != "report":
        args.output = args.input  # existing read-only capture reader contract
        args.results.mkdir(parents=True, exist_ok=False)
    progress = exp.ReplayProgress(args.results, args.progress_interval)
    started = time.perf_counter()
    exp.write_json(args.results / "status.json", dict(status="running", command=args.command))
    try:
        globals()[args.command](args, progress)
    except Exception as exc:
        exp.write_json(args.results / "status.json", dict(status="failed", error=repr(exc), command=args.command))
        progress.message(f"停止：{exc}")
        raise
    exp.write_json(
        args.results / "status.json",
        dict(status="complete", command=args.command, elapsed_seconds=time.perf_counter() - started),
    )


if __name__ == "__main__":
    run(parser().parse_args())
