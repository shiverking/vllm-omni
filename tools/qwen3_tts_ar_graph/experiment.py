# SPDX-License-Identifier: Apache-2.0
"""Independent KV/frame-graph experiment; no speculative-code imports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BUCKETS = [1, 2, 4, 8, 16]
CHECK_BATCHES = [1, 2, 3, 4, 5, 8, 9, 15, 16]
MODEL = "/softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def environment(args):
    versions = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    repos = {}
    for name, path in (("omni", ROOT), ("vllm", args.vllm_repo), ("ascend", args.ascend_repo)):
        try:
            repos[name] = git_revision(path)
        except (OSError, subprocess.CalledProcessError):
            repos[name] = None
    cann = None
    for name in ("version.cfg", "version.info", "aarch64-linux/ascend_toolkit_install.info"):
        path = Path("/usr/local/Ascend/ascend-toolkit/latest") / name
        if path.exists():
            cann = path.read_text()
            break
    api = device_api(args.device)
    return dict(
        device=args.device,
        device_name=api.get_device_name(torch.device(args.device)) if api else "CPU",
        packages=versions,
        repositories=repos,
        cann=cann,
    )


def device_api(device):
    if str(device).startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(torch.device(device))
        return torch.npu
    if str(device) != "cpu":
        raise ValueError("Only CPU diagnostics and Ascend NPU are supported")
    return None


class Progress:
    def __init__(self, root):
        self.root = root
        self.started = time.monotonic()
        self.last = 0

    def begin(self, label, total):
        self.label, self.total, self.done = label, total, 0
        self.stage = time.monotonic()
        self.message(f"{label}: 0/{total}")

    def message(self, message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.root / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def advance(self):
        self.done += 1
        now = time.monotonic()
        if now - self.last >= 10 or self.done == self.total:
            elapsed = now - self.stage
            eta = elapsed / self.done * max(0, self.total - self.done)
            self.message(f"{self.label}: {self.done}/{self.total}; elapsed={elapsed:.1f}s ETA~{eta:.1f}s")
            self.last = now


def capture_root(root):
    imported = root / "import.json"
    if imported.exists():
        ref = read_json(imported)
        source = Path(ref["source"])
        if file_hash(source / "predictor.pt") != ref["predictor_sha256"]:
            raise ValueError("Imported predictor changed")
        if file_hash(source / "manifest.json") != ref["manifest_sha256"]:
            raise ValueError("Imported manifest changed")
        return source
    return root


def select_frames(root, probe=False):
    manifest = read_json(root / "manifest.json")
    dataset = manifest["dataset"]
    if len(dataset) != 30 or len({row["id"] for row in dataset}) != 30:
        raise ValueError("Expected the fixed 30 unique English smoke requests")
    expected = {row["id"] for row in dataset[10:]}
    groups = defaultdict(list)
    for path in sorted((root / "frames").glob("*.pt")):
        frame = torch.load(path, map_location="cpu", weights_only=True)
        if frame["request_id"] not in expected:
            continue
        if frame["split"] != "test":
            raise ValueError("Test request has wrong split")
        frame["artifact"] = str(path.resolve())
        groups[frame["request_id"]].append(frame)
    if set(groups) != expected:
        raise ValueError("Missing test requests; finish collection")
    result = []
    for request in sorted(groups):
        rows = sorted(groups[request], key=lambda f: f["frame_id"])
        if len({f["frame_id"] for f in rows}) != len(rows) or len(rows) < 3:
            raise ValueError("Duplicate frames or fewer than three frames in a request")
        result.extend(rows[i] for i in (0, len(rows) // 2, len(rows) - 1))
    if probe:
        all_rows = [f for rows in groups.values() for f in rows]
        first = [next(f for f in all_rows if f["request_id"] == "smoke011" and f["frame_id"] == i) for i in (0, 2)]
        first.append(max(groups["smoke030"], key=lambda f: f["frame_id"]))
        return first
    return result


def normalize_state(source):
    state = dict(source)
    for name in list(state):
        if ".q_proj." in name:
            other = [name.replace(".q_proj.", f".{part}_proj.") for part in ("q", "k", "v")]
            state[name.replace(".q_proj.", ".qkv_proj.")] = torch.cat([state.pop(k) for k in other])
        elif ".gate_proj." in name:
            other = [name, name.replace(".gate_proj.", ".up_proj.")]
            state[name.replace(".gate_proj.", ".gate_up_proj.")] = torch.cat([state.pop(k) for k in other])
    return state


def verify_model_weights(checkpoint, model):
    from safetensors import safe_open

    source = {}
    files = sorted(Path(model).glob("*.safetensors"))
    if not files:
        raise ValueError("Model directory has no safetensors; cannot verify capture against serving weights")
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if "code_predictor." in key:
                    source[key.split("code_predictor.", 1)[1]] = stream.get_tensor(key)
    actual = normalize_state(source)
    expected = normalize_state(checkpoint["state"])
    for name, value in expected.items():
        if "rotary_emb." in name:
            continue
        if name not in actual or not torch.equal(value, actual[name].to(value.dtype)):
            raise ValueError(f"Captured predictor weights differ from model: {name}")


def validate_capture(root, model):
    manifest = read_json(root / "manifest.json")
    if "1.7B-CustomVoice" not in str(manifest.get("model", "")):
        raise ValueError("Capture must identify Qwen3-TTS-12Hz-1.7B-CustomVoice")
    checkpoint = torch.load(root / "predictor.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"]["num_code_groups"] != 16:
        raise ValueError("Expected 16 code groups")
    if Path(model, "config.json").exists():
        model_config = read_json(Path(model, "config.json"))
        expected = model_config["talker_config"]["code_predictor_config"]
        for field in ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads", "num_key_value_heads"):
            if checkpoint["config"][field] != expected[field]:
                raise ValueError(f"Captured model config differs: {field}")
    if Path(model).is_dir():
        verify_model_weights(checkpoint, model)
    select_frames(root)
    return checkpoint


def load_predictor(args, root):
    api = device_api(args.device)
    if api and not Path(args.model).is_dir():
        raise ValueError("NPU checks require the real local model directory")
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
        CodePredictorWrapper,
        CodePredictorWrapperConfig,
    )
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
    )

    if api:
        from vllm.config import VllmConfig
        from vllm_ascend.ascend_config import init_ascend_config

        runtime = VllmConfig()
        runtime.scheduler_config.max_num_seqs = 16
        init_ascend_config(runtime)
    else:
        runtime = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=16))
    checkpoint = validate_capture(root, args.model)
    config = Qwen3TTSTalkerCodePredictorConfig(**checkpoint["config"])
    w = CodePredictorWrapper(
        vllm_config=runtime,
        cp_config=config,
        talker_hidden_size=checkpoint["talker_hidden_size"],
        wrapper_config=CodePredictorWrapperConfig(
            use_cuda_graphs=api is not None, use_projection=True, sampling_mode="per_call"
        ),
    )
    dtype = getattr(torch, checkpoint["dtype"].split(".")[-1])
    w.to(dtype=dtype)
    state = normalize_state(checkpoint["state"])
    w.load_state_dict(state, strict=True)
    w.to(args.device).eval().requires_grad_(False)
    if api:
        w._prepare_npu_weights()
    w._setup_compile()
    if api and not all(bucket in w._device_graphs for bucket in BUCKETS):
        raise RuntimeError("Stock predictor graphs were not captured for every required batch bucket")
    return w


def binding(args, root, frames):
    paths = [
        *sorted((ROOT / "vllm_omni/model_executor/models/common").glob("qwen3_code_predictor*.py")),
        ROOT / "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py",
        ROOT / "vllm_omni/worker/gpu_model_runner.py",
        ROOT / "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code_predictor_vllm.py",
        *sorted(Path(__file__).parent.glob("*.py")),
    ]
    return dict(
        code={str(p.relative_to(ROOT)): file_hash(p) for p in paths},
        predictor=file_hash(root / "predictor.pt"),
        manifest=file_hash(root / "manifest.json"),
        model=args.model,
        model_files={p.name: file_hash(p) for p in sorted(Path(args.model).glob("*.safetensors"))}
        | (
            {"config.json": file_hash(Path(args.model) / "config.json")}
            if (Path(args.model) / "config.json").exists()
            else {}
        ),
        frames={f"{f['request_id']}/{f['frame_id']}": file_hash(Path(f["artifact"])) for f in frames},
        environment=environment(args),
        buckets=BUCKETS,
        check_batches=CHECK_BATCHES,
        modes=["stock", "eager", "kv_step", "kv_frame"],
        sampling=["greedy", "sampling"],
        temperature=0.9,
        top_k=50,
        top_p=1.0,
    )


def uniforms_for(frames, vocab, device, salt=0):
    rows = []
    for f in frames:
        identity = f"42/{f['request_id']}/{f['frame_id']}/{salt}"
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (2**63 - 1)
        rows.append(torch.rand((15, vocab), generator=torch.Generator().manual_seed(seed)))
    return torch.stack(rows).to(device)


def batch_inputs(frames, device):
    return {
        key: torch.cat([f["inputs"][key].reshape(1, -1) for f in frames], dim=0).to(device)
        for key in ("layer0_code", "layer0_embed", "last_talker_hidden")
    }


def batches(frames, size, count):
    return [[frames[(i * size + j) % len(frames)] for j in range(size)] for i in range(count)]


def binding_differences(saved, current, path="binding"):
    """Compare recorded and current provenance without hiding missing fields."""
    if isinstance(saved, dict) and isinstance(current, dict):
        differences = []
        for key in sorted(saved.keys() | current.keys()):
            child = f"{path}.{key}"
            if key not in saved or key not in current:
                differences.append(
                    dict(
                        field=child,
                        saved=saved.get(key),
                        current=current.get(key),
                        saved_present=key in saved,
                        current_present=key in current,
                    )
                )
            else:
                differences.extend(binding_differences(saved[key], current[key], child))
        return differences
    if saved != current:
        return [dict(field=path, saved=saved, current=current)]
    return []


def certificate(args, expected):
    reasons, differences = [], []
    cert = {}
    try:
        cert = read_json(args.correctness)
        if not isinstance(cert, dict):
            raise ValueError("证书根节点必须是 JSON 对象")
    except (OSError, ValueError) as exc:
        reasons.append(f"无法读取正确性证书: {exc}")
        cert = {}
    else:
        if not cert.get("passed"):
            reasons.append("正确性检查未通过，或缺少 passed 字段；先解决检查失败。")
        if cert.get("probe_only"):
            reasons.append("该证书仅覆盖三帧预检；请执行不带 --probe-only 的 60 帧抽样检查。")
        differences = binding_differences(cert.get("binding"), expected)
        if differences:
            reasons.append(f"证书与本次运行存在 {len(differences)} 项绑定差异；需在当前配置下重新检查。")
    if not reasons:
        return
    diagnostic = dict(
        certificate=str(args.correctness),
        reasons=reasons,
        passed=cert.get("passed"),
        probe_only=cert.get("probe_only"),
        previous_error=cert.get("error"),
        previous_context=cert.get("context"),
        differences=differences,
    )
    print(f"\n[证书校验失败] {args.correctness}", flush=True)
    for reason in reasons:
        print(f"  - {reason}", flush=True)
    for key in ("previous_error", "previous_context"):
        if diagnostic[key] is not None:
            print(f"  {key}: {json.dumps(diagnostic[key], ensure_ascii=False)}", flush=True)
    for difference in differences[:30]:
        print("  " + json.dumps(difference, ensure_ascii=False), flush=True)
    if len(differences) > 30:
        print(f"  另有 {len(differences) - 30} 项差异，见诊断文件。", flush=True)
    if getattr(args, "results", None) is not None:
        destination = Path(args.results) / "certificate_diagnostic.json"
        try:
            write_json(destination, diagnostic)
            print(f"完整诊断已保存: {destination}", flush=True)
        except OSError as exc:
            print(f"诊断文件保存失败: {exc}", flush=True)
    raise ValueError("正确性证书校验失败；具体原因见上方输出。性能测试未启动。")


def collect(args, progress):
    if args.import_from:
        source = args.import_from.resolve()
        validate_capture(source, args.model)
        write_json(
            args.results / "import.json",
            dict(
                source=str(source),
                predictor_sha256=file_hash(source / "predictor.pt"),
                manifest_sha256=file_hash(source / "manifest.json"),
            ),
        )
        write_json(args.results / "manifest.json", dict(imported_from=str(source), model=args.model))
        progress.message("已只读导入现有采集；原始文件未修改。")
        return
    if args.prepare:
        import yaml

        dataset_path = ROOT / "benchmarks/build_dataset/seed_tts_smoke/en/meta.lst"
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 30:
            raise ValueError("Expected 30 smoke texts")
        dataset = [
            dict(id=line.split("|")[0], text=line.split("|")[-1], split="calibration" if i < 10 else "test")
            for i, line in enumerate(lines)
        ]
        write_json(
            args.results / "manifest.json",
            dict(
                model=args.model,
                dataset=dataset,
                seed=42,
                model_revision=args.model_revision,
                environment=environment(args),
            ),
        )
        (args.results / "frames").mkdir()
        deploy = yaml.safe_load((ROOT / "vllm_omni/deploy/qwen3_tts.yaml").read_text(encoding="utf-8"))
        deploy["connectors"]["connector_of_shared_memory"]["extra"]["code_predictor_ar_capture"] = str(
            args.results.resolve()
        )
        for stage in deploy["stages"]:
            stage["max_num_seqs"] = 1
        (args.results / "deploy.yaml").write_text(yaml.safe_dump(deploy, sort_keys=False), encoding="utf-8")
        progress.message("使用生成的 deploy.yaml 启动服务，再运行 collect --url。")
        return
    manifest = read_json(args.results / "manifest.json")
    if list((args.results / "frames").glob("*.pt")):
        raise ValueError("Refusing to overwrite an existing capture")
    progress.begin("采集请求", len(manifest["dataset"]))
    try:
        for row in manifest["dataset"]:
            write_json(args.results / "control.json", row)
            payload = dict(
                model=args.model,
                input=row["text"],
                voice="Vivian",
                language="English",
                task_type="CustomVoice",
                response_format="pcm",
                stream_format="audio",
                seed=42,
            )
            request = urllib.request.Request(
                args.url.rstrip("/") + "/v1/audio/speech",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                if not response.read():
                    raise RuntimeError("Empty audio")
            progress.advance()
    finally:
        (args.results / "control.json").unlink(missing_ok=True)
    validate_capture(args.results, args.model)


@torch.inference_mode()
def check(args, progress):
    from vllm_omni.model_executor.models.common.qwen3_code_predictor_ar import CachedARExecutor

    root = capture_root(args.input)
    frames = select_frames(root, args.probe_only)
    w = load_predictor(args, root)
    bind = binding(args, root, frames)
    engines = {"eager": CachedARExecutor(w, "eager")}
    if args.device.startswith("npu"):
        engines.update({m: CachedARExecutor(w, m) for m in ("kv_step", "kv_frame")})
    rows = []
    matrix = [
        (size, sample, batch)
        for size in CHECK_BATCHES
        for sample in (False, True)
        for batch in batches(frames, size, (len(frames) + size - 1) // size)
    ]
    progress.begin("逐位置/输出/重放/批处理检查", len(matrix))
    context, inputs = {}, {}
    stock_logits, audit, expected = [], [], None
    try:
        for size, sample, batch in matrix:
            inputs = batch_inputs(batch, args.device)
            inputs.update(do_sample=sample, sample_uniforms=uniforms_for(batch, w.config.vocab_size, args.device))
            context = dict(batch=size, sampling=sample, frames=[(f["request_id"], f["frame_id"]) for f in batch])
            stock_logits = []
            hooks = [
                head.register_forward_hook(lambda mod, inp, out: stock_logits.append(out.detach().clone()))
                for head in w.lm_head
            ]
            try:
                expected = w(**inputs)
            finally:
                for hook in hooks:
                    hook.remove()
            for name, executor in engines.items():
                context["engine"] = name
                audit, calls = [], []
                if name == "eager":
                    hook = w.model.register_forward_hook(lambda *unused: calls.append(1))
                    try:
                        actual = executor(**inputs, audit=audit)
                    finally:
                        hook.remove()
                    if len(calls) != 15:
                        raise AssertionError(f"Expected 15 actual backbone calls, got {len(calls)}")
                    for position, (left, right) in enumerate(zip(audit, stock_logits)):
                        context["position"] = position + 1
                        torch.testing.assert_close(left[:size], right, atol=0.02, rtol=0.02)
                else:
                    # Capture hooks establish real forward count; replay count is separate.
                    bucket = next(n for n in BUCKETS if n >= size)
                    if (bucket, sample) not in executor.graphs:
                        count = []
                        hook = w.model.register_forward_hook(lambda *unused: count.append(1))
                        try:
                            executor.prepare(bucket, sample)
                        finally:
                            hook.remove()
                        if len(count) != 60:  # 3 warmups + capture, each 15 forwards
                            raise AssertionError(f"Unexpected warmup/capture backbone count: {len(count)}")
                    actual = executor(**inputs)
                    audit = [logits[:size].detach().clone() for logits in executor.last_logits(size, sample)]
                    for position, (left, right) in enumerate(zip(audit, stock_logits)):
                        context["position"] = position + 1
                        torch.testing.assert_close(left, right, atol=0.02, rtol=0.02)
                if not torch.equal(actual, expected):
                    raise AssertionError(f"codes differ: {context}")
                saved = actual.clone()
                order = torch.arange(size - 1, -1, -1, device=args.device)
                reordered = {k: v[order] if torch.is_tensor(v) else v for k, v in inputs.items()}
                if not torch.equal(executor(**reordered), expected[order]):
                    raise AssertionError("Request row reordering changed codes")
                if not torch.equal(executor(**inputs), saved) or not torch.equal(actual, saved):
                    raise AssertionError("Replay/cache/output lifetime violation")
                changed = dict(inputs, sample_uniforms=uniforms_for(batch, w.config.vocab_size, args.device, salt=1))
                if not torch.equal(executor(**changed), w(**changed)):
                    raise AssertionError("Updated uniforms mismatch")
                rows.append(dict(context, passed=True))
            # Independent single-row baseline also catches padding/layout effects.
            for i in range(size):
                single = {k: v[i : i + 1] if torch.is_tensor(v) else v for k, v in inputs.items()}
                if not torch.equal(w(**single), expected[i : i + 1]):
                    raise AssertionError("Stock batch vs single differs; diagnose numerical layout before benchmarking")
            progress.advance()
        # Explicit bucket transition after the full matrix (including replacement rows).
        for size in (16, 1, 9, 2, 15, 3, 8, 1):
            batch = batches(frames[::-1], size, 1)[0]
            x = batch_inputs(batch, args.device)
            x.update(sample_uniforms=uniforms_for(batch, w.config.vocab_size, args.device, 2))
            for executor in engines.values():
                if not torch.equal(executor(**x), w(**x)):
                    raise AssertionError("Bucket transition/request replacement mismatch")
    except Exception as exc:
        torch.save(
            dict(
                context=context,
                inputs=inputs,
                stock_logits=stock_logits,
                audit=audit,
                expected=expected,
                caches={name: engine.snapshot() for name, engine in engines.items()},
            ),
            args.results / "failure.pt",
        )
        write_json(
            args.results / "correctness.json", dict(passed=False, error=repr(exc), context=context, binding=bind)
        )
        print("[正确性检查失败] " + json.dumps(context, ensure_ascii=False), flush=True)
        print(f"原因: {exc}", flush=True)
        print(f"诊断文件: {args.results / 'correctness.json'}；{args.results / 'failure.pt'}", flush=True)
        raise
    write_json(
        args.results / "correctness.json",
        dict(
            passed=True,
            probe_only=args.probe_only,
            binding=bind,
            checks=len(rows),
            unique_frames=len(frames),
            graph_verified=args.device.startswith("npu"),
            results=rows,
        ),
    )
    progress.message(f"通过 {len(rows)} 项；CPU 结果不授权 NPU 性能测试。")


def measured(api, fn):
    if api:
        api.synchronize()
        start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
    begin = time.perf_counter()
    if api:
        start.record()
    result = fn()
    if api:
        end.record()
        end.synchronize()
    return result, (time.perf_counter() - begin) * 1000, start.elapsed_time(end) if api else None


@torch.inference_mode()
def bench(args, progress):
    from vllm_omni.model_executor.models.common.qwen3_code_predictor_ar import CachedARExecutor

    if args.warmup < 20 or args.executions_per_round < 60:
        raise ValueError("Benchmark requires at least 20 warmups and 60 executions per round")
    root = capture_root(args.input)
    frames = select_frames(root)
    bind = binding(args, root, frames)
    certificate(args, bind)
    if not args.device.startswith("npu"):
        raise ValueError("Performance and graph conclusions require NPU")
    w = load_predictor(args, root)
    api = device_api(args.device)
    rows, metadata = [], []
    for candidate in ("kv_step", "kv_frame"):
        executor = CachedARExecutor(w, candidate)
        for sample in (False, True):
            for size in BUCKETS:
                executor.prepare(size, sample)
                prepared = [
                    (batch_inputs(batch, args.device), batch)
                    for batch in batches(frames, size, args.executions_per_round)
                ]

                def run(name, index, selected_executor=executor):
                    x, batch_frames = prepared[index % len(prepared)]
                    # Include deterministic uniform preparation and transfer in both paths.
                    return (w if name == "stock" else selected_executor)(
                        **x,
                        do_sample=sample,
                        sample_uniforms=uniforms_for(batch_frames, w.config.vocab_size, args.device)
                        if sample
                        else None,
                    )

                for i in range(args.warmup):
                    run("stock", i)
                    run(candidate, i)
                api.synchronize()
                progress.begin(f"{candidate} batch={size} sampling={sample} ABBA", 4 * len(prepared))
                for round_id, name in enumerate(("stock", candidate, candidate, "stock")):
                    for i in range(len(prepared)):
                        _, wall, device = measured(api, lambda run_fn=run: run_fn(name, i))
                        rows.append(
                            dict(
                                candidate=candidate,
                                engine=name,
                                sampling=sample,
                                batch=size,
                                round=round_id,
                                index=i,
                                wall_ms=wall,
                                device_ms=device,
                                backbone_calls=15,
                                graph_replays=15 if name in ("stock", "kv_step") else 1,
                                effective_frames=size,
                                padding=0,
                            )
                        )
                        progress.advance()
        metadata.append(
            dict(
                engine=candidate,
                capture_seconds=executor.capture_seconds,
                graph_count=sum(1 if candidate == "kv_frame" else 15 for _ in executor.graphs),
                memory_peak_bytes=api.max_memory_allocated(),
                graph_owner="CodePredictor",
                attention="npu_fusion_attention: prefill causal / decode dense cached",
            )
        )
        del run, executor
        api.synchronize()
        api.empty_cache()
    with (args.results / "timings.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    write_json(
        args.results / "manifest.json",
        dict(
            binding=bind,
            graphs=metadata,
            unique_frames=len(frames),
            stock_graph_scope="standalone predictor; service outer MTP scope recorded separately",
            memory_note="peak includes stock resident graphs; not candidate-exclusive",
            timing_note=(
                "includes CPU identity-bound uniform generation/H2D, input staging and output clone; "
                "upstream inputs on device"
            ),
        ),
    )
    report(args, progress)


@torch.inference_mode()
def profile(args, progress):
    import torch_npu

    from vllm_omni.model_executor.models.common.qwen3_code_predictor_ar import CachedARExecutor

    root = capture_root(args.input)
    frames = select_frames(root)
    certificate(args, binding(args, root, frames))
    if not args.device.startswith("npu"):
        raise ValueError("Native profiling requires NPU")
    w = load_predictor(args, root)
    api = device_api(args.device)
    progress.begin("独立图 profiling", 18)
    for mode in ("stock", "kv_step", "kv_frame"):
        executor = w if mode == "stock" else CachedARExecutor(w, mode)
        for sample in (False, True):
            for size in (1, 4, 16):
                data = []
                for batch in batches(frames[:6], size, 6):
                    x = batch_inputs(batch, args.device)
                    x.update(do_sample=sample, sample_uniforms=uniforms_for(batch, w.config.vocab_size, args.device))
                    data.append(x)
                for i in range(args.warmup):
                    executor(**data[i % len(data)])
                api.synchronize()
                destination = args.results / mode / ("sampling" if sample else "greedy") / f"batch{size}"
                destination.mkdir(parents=True)
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(destination)),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                ) as prof:
                    for x in data:
                        executor(**x)
                        prof.step()
                    api.synchronize()
                progress.advance()
        del executor
        api.synchronize()
        api.empty_cache()
    write_json(
        args.results / "manifest.json",
        dict(
            binding=binding(args, root, frames),
            scope="6 diagnostic batches; profiler timings are not benchmark timings",
        ),
    )


def report(args, progress):
    rows = [json.loads(line) for line in (args.results / "timings.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sampling"], row["batch"], row["candidate"])].append(row)
    summary = []
    lines = [
        "# KV cache＋整帧 AR 图实验",
        "",
        "不设置性能通过门槛；仅代表本次抽样帧级结果。",
        "",
        "| 模式 | batch | 配置 | 墙钟 p50/p95 ms | 设备事件 p50/p95 ms | 帧/s | 相对 stock |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for (sample, size, candidate), group in groups.items():
        base = [r for r in group if r["engine"] == "stock"]
        cand = [r for r in group if r["engine"] == candidate]
        wall = np.array([r["wall_ms"] for r in cand])
        device = np.array([r["device_ms"] for r in cand])
        ratio = sum(r["wall_ms"] for r in base) / wall.sum()
        # Paired batch-index bootstrap; exploratory, not independent-request CI.
        paired = []
        for i in sorted({r["index"] for r in group}):
            paired.append(
                (sum(r["wall_ms"] for r in base if r["index"] == i), sum(r["wall_ms"] for r in cand if r["index"] == i))
            )
        pair = np.asarray(paired)
        rng = np.random.default_rng(42)
        draws = pair[rng.integers(len(pair), size=(2000, len(pair)))].sum(axis=1)
        ci = np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975]).tolist()
        item = dict(
            sampling=sample,
            batch=size,
            candidate=candidate,
            speedup=ratio,
            wall_p50_ms=float(np.median(wall)),
            wall_p95_ms=float(np.quantile(wall, 0.95)),
            device_p50_ms=float(np.median(device)),
            device_p95_ms=float(np.quantile(device, 0.95)),
            frames_per_second=float(size * 1000 / wall.mean()),
            exploratory_paired_batch_ci=ci,
            baseline_wall_mean_ms=float(np.mean([r["wall_ms"] for r in base])),
            candidate_wall_mean_ms=float(wall.mean()),
            backbone_calls=15,
        )
        summary.append(item)
        lines.append(
            f"| {'sampling' if sample else 'greedy'} | {size} | {candidate} | "
            f"{item['wall_p50_ms']:.3f}/{item['wall_p95_ms']:.3f} | "
            f"{item['device_p50_ms']:.3f}/{item['device_p95_ms']:.3f} | "
            f"{item['frames_per_second']:.2f} | {ratio:.3f}× |"
        )
    lines += [
        "",
        "每帧主干调用仍为 15 次；整帧图减少 replay 次数，不代表一次主干。",
        "设备事件可能包含主机提交空隙，不是纯 kernel 时间。",
        "区间是配对 batch 的探索性 bootstrap；帧被循环复用，不作为泛化证据。",
        "服务并发、实际 batch 分布和音频指标另见服务实验产物。",
    ]
    write_json(args.results / "summary.json", summary)
    (args.results / "report.md").write_text("\n".join(lines), encoding="utf-8")
    progress.message("报告已保存。")


def parser():
    root = argparse.ArgumentParser(__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("collect", "check", "bench", "profile", "report"):
        p = commands.add_parser(name)
        p.add_argument("--results", type=Path, required=True)
        if name == "report":
            continue
        p.add_argument("--device", default="npu:0")
        p.add_argument("--model", default=MODEL)
        p.add_argument("--vllm-repo", type=Path, default=Path("/vllm-workspace/vllm"))
        p.add_argument("--ascend-repo", type=Path, default=Path("/vllm-workspace/vllm-ascend"))
        if name == "collect":
            p.add_argument("--import-from", type=Path)
            p.add_argument("--prepare", action="store_true")
            p.add_argument("--model-revision", default="local-checkpoint; predictor SHA256 recorded")
            p.add_argument("--url", default="http://127.0.0.1:8000")
        else:
            p.add_argument("--input", type=Path, required=True)
        if name == "check":
            p.add_argument("--probe-only", action="store_true")
        if name in ("bench", "profile"):
            p.add_argument("--correctness", type=Path, required=True)
            p.add_argument("--warmup", type=int, default=20)
            p.add_argument("--executions-per-round", type=int, default=60)
    return root


def main():
    args = parser().parse_args()
    resume_collect = args.command == "collect" and not args.prepare and not args.import_from
    args.results.mkdir(parents=True, exist_ok=args.command == "report" or resume_collect)
    progress = Progress(args.results)
    write_json(args.results / "status.json", dict(status="running", command=args.command))
    try:
        globals()[args.command](args, progress)
    except Exception as exc:
        write_json(args.results / "status.json", dict(status="failed", error=repr(exc)))
        raise
    write_json(args.results / "status.json", dict(status="complete", command=args.command))


if __name__ == "__main__":
    main()
