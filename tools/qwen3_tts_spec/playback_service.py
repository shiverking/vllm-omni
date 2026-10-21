# SPDX-License-Identifier: Apache-2.0
"""Loopback service harness; GStreamer measurements are never simulated."""

from __future__ import annotations

import base64
import concurrent.futures
import dataclasses
import json
import queue
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from playback_experiment import ROOT, artifacts, binding, certificate, read, write

from vllm_omni.core.sched.playback_broker import PlaybackArbiter, rpc
from vllm_omni.core.sched.playback_deadline import ChunkPlanner, StageCosts


def serve(args, config):
    import yaml

    actions, policy = artifacts(config)
    certificate(config, actions, policy)
    if config["device"] != "npu:0":
        raise ValueError("live serving requires the Ascend NPU; CPU simulation is a separate command")
    if config["batch_buckets"] != [1, 2, 4, 8]:
        raise ValueError("complete 1/2/4/8 frame checks before starting the concurrent service")
    if args.sampling not in config["sampling"]:
        raise ValueError("uncertified sampling mode")
    planner = None
    if args.method in ("joint", "joint_ar"):
        cost_payload = read(Path(config["results"]) / f"stage_costs_{args.sampling}.json")
        if cost_payload["binding"] != binding(config):
            raise ValueError("stage cost binding changed")
        planner = ChunkPlanner(policy, StageCosts(cost_payload), sampling=args.sampling, ablation=args.ablation)
    run = Path(config["results"]) / "service" / f"{args.method}-{args.sampling}-{args.ablation}-{int(time.time())}"
    run.mkdir(parents=True, exist_ok=False)
    runtime = dict(
        config,
        service_method=args.method,
        service_sampling=args.sampling,
        service_seed=args.seed,
        service_trace=args.trace,
        ablation=args.ablation,
    )
    write(run / "runtime.json", runtime)
    write(
        Path(config["results"]) / "active_service.json",
        dict(run_directory=str(run), events=str(run / "events.jsonl"), runtime=str(run / "runtime.json")),
    )
    # Inherit current transport/communication fixes, not obsolete LiveServe
    # implementations of engine plumbing from another release.
    deploy = dict(
        base_config=str(ROOT / "vllm_omni/deploy/qwen3_tts.yaml"),
        model_runner="v1",
        async_chunk=True,
        connectors={
            "connector_of_shared_memory": {
                "extra": {
                    "codec_chunk_frames": 4,
                    "initial_codec_chunk_frames": 4,
                    "codec_left_context_frames": 2,
                    "decode_batch_max_size": 8,
                    "decode_cudagraph_batch_sizes": [1, 2, 4, 8],
                }
            }
        },
        stages=[],
    )
    for stage in (0, 1):
        row = dict(
            stage_id=stage,
            devices="0",
            max_num_seqs=64 if args.method == "native" else 8,
            gpu_memory_utilization=0.3,
            async_scheduling=args.method == "native",
        )
        if args.method != "native":
            row["engine_extras"] = dict(additional_config=dict(tts_playback_experiment=str(run / "runtime.json")))
        if stage == 0:
            row["subtalker_sampling_params"] = dict(
                do_sample=args.sampling == "sampling", temperature=0.9, top_k=50, top_p=1
            )
        deploy["stages"].append(row)
    (run / "deploy.yaml").write_text(yaml.safe_dump(deploy, sort_keys=False), encoding="utf-8")
    pending = queue.SimpleQueue()

    def writer():
        with (run / "events.jsonl").open("w", encoding="utf-8") as stream:
            while (row := pending.get()) is not None:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if row.get("event") == "phase":
                    stream.flush()

    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()
    arbiter = PlaybackArbiter(args.method, planner=planner, emit=pending.put, ablation=args.ablation)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size > 1024 * 1024:
                    raise ValueError("oversized research control message")
                value = arbiter.dispatch(self.path.strip("/"), json.loads(self.rfile.read(size)))
                value = dict(value, run_directory=str(run))
            except Exception as exc:
                value = dict(error=repr(exc))
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    address = urlparse(config["broker_url"])
    if address.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("research control plane must bind to loopback")
    server = ThreadingHTTPServer((address.hostname, address.port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = urlparse(config["server_url"])
    command = [
        "vllm",
        "serve",
        config["model"],
        "--omni",
        "--deploy-config",
        str(run / "deploy.yaml"),
        "--host",
        endpoint.hostname,
        "--port",
        str(endpoint.port),
    ]
    write(
        run / "manifest.json",
        dict(
            binding=binding(config),
            config=runtime,
            command=command,
            controlled=args.method != "native",
            sampling=args.sampling,
        ),
    )
    print("服务启动；客户端需先 warmup，再 check，最后 measure。", flush=True)
    print("日志目录：" + str(run), flush=True)
    process = None
    try:
        process = subprocess.Popen(command, cwd=ROOT)
        code = process.wait()
        if code:
            raise RuntimeError(f"vllm exited with {code}")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        server.shutdown()
        server.server_close()
        pending.put(None)
        writer_thread.join(timeout=10)


def run_request(config, text, index, seed, split, phase, concurrency, arrived=None):
    from vllm_omni.benchmarks.gstreamer_player import GStreamerSilentPlayer

    identity = f"speech-rvq-{phase}-c{concurrency}-{seed}-{index}"
    rpc(config["broker_url"], "register", dict(id=identity, split=split, seed=seed))
    player = GStreamerSilentPlayer(rate=config["playback_rate"])
    done = threading.Event()
    input_ended = threading.Event()
    feedback_errors = []
    chunks = []

    def feedback():
        sequence = 0
        while not done.is_set():
            try:
                snapshot = player.snapshot()
                rpc(
                    config["broker_url"],
                    "feedback",
                    dict(
                        id=identity,
                        sequence=sequence,
                        received_ms=snapshot.received_audio_ms,
                        played_ms=snapshot.played_audio_ms,
                        rate=config["playback_rate"],
                        started=snapshot.playback_started,
                        received_chunks=len(chunks),
                        input_complete=input_ended.is_set(),
                    ),
                    timeout=10,
                )
                sequence += 1
            except Exception as exc:
                feedback_errors.append(repr(exc))
                return
            done.wait(config["feedback_ms"] / 1000)

    task = threading.Thread(target=feedback, daemon=True)
    task.start()
    start = time.perf_counter() if arrived is None else arrived
    first = None
    error = None
    try:
        payload = dict(
            model=config["model"],
            input=text,
            task_type="CustomVoice",
            voice="Vivian",
            language="English",
            response_format="pcm",
            stream_format="sse",
            seed=seed,
            stream=True,
        )
        request = urllib.request.Request(
            config["server_url"] + "/v1/audio/speech",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json", "x-tts-playback-request-id": identity},
        )
        with urllib.request.urlopen(request, timeout=1800 if phase == "warmup" else 120) as response:
            saw_done = False
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                value = json.loads(line[5:].strip())
                if value["type"] == "speech.audio.delta":
                    pcm = base64.b64decode(value["audio"], validate=True)
                    if not pcm:
                        continue
                    elapsed = (time.perf_counter() - start) * 1000
                    snapshot = player.push_pcm(pcm)
                    if first is None and snapshot.playback_started:
                        first = (time.perf_counter() - start) * 1000
                    chunks.append(
                        dict(index=len(chunks), received_ms=elapsed, pcm_bytes=len(pcm), **dataclasses.asdict(snapshot))
                    )
                elif value["type"] == "speech.audio.error":
                    raise RuntimeError(value)
                elif value["type"] == "speech.audio.done":
                    saw_done = True
            if not saw_done:
                raise RuntimeError("SSE terminated without speech.audio.done")
        e2e = (time.perf_counter() - start) * 1000
        if not chunks:
            raise RuntimeError("no playable audio received")
        snapshot = player.end_input()
        input_ended.set()
        if first is None and snapshot.playback_started:
            first = (time.perf_counter() - start) * 1000
        while snapshot.playback_buffer_ms > 0:
            time.sleep(0.01)
            snapshot = player.snapshot()
        final = player.finish()
    except Exception as exc:
        error = repr(exc)
        e2e = (time.perf_counter() - start) * 1000
        final = player.abort()
    finally:
        done.set()
        task.join(timeout=15)
        player.close()
        rpc(
            config["broker_url"],
            "feedback",
            dict(
                id=identity,
                sequence=2**31 - 1,
                received_ms=final.received_audio_ms,
                played_ms=final.played_audio_ms,
                rate=config["playback_rate"],
                started=final.playback_started,
                received_chunks=len(chunks),
                finished=True,
            ),
        )
        rpc(config["broker_url"], "finish", dict(id=identity))
    error = error or (str(feedback_errors) if feedback_errors else None)
    return dict(
        id=identity,
        client_arrival_s=start,
        index=index,
        split=split,
        seed=seed,
        first_audio_ms=first,
        e2e_ms=e2e,
        error=error,
        qos_pass=error is None and first is not None and first <= 1000 and final.max_underrun_ms <= 100,
        playback=dataclasses.asdict(final),
        chunks=chunks,
    )


def bench(args, config):
    from vllm_omni.benchmarks.gstreamer_player import require_gstreamer

    require_gstreamer()
    if args.formal and (
        config["independent_confirmation"] != "formal_captures_present_evaluation_pending"
        or args.seed not in (17, 29, 43)
    ):
        raise ValueError("formal runs require independent 80/40/200 captures and seed 17/29/43")
    status = rpc(config["broker_url"], "health")
    if status["failed"]:
        raise RuntimeError(status["failed"])
    if status["method"] != args.method:
        raise ValueError("bench method differs from the running server")
    run = Path(status["run_directory"])
    manifest = read(run / "manifest.json")
    if manifest["binding"] != binding(config) or manifest["sampling"] != args.sampling:
        raise ValueError("live service code/data/sampling binding changed")
    if manifest["config"]["ablation"] != args.ablation:
        raise ValueError("ablation differs from the running server")
    if args.phase == "measure" and manifest["config"]["service_trace"]:
        raise ValueError("detailed diagnostic tracing cannot be mixed into formal timing")
    if args.phase == "measure":
        check = read(run / "service_check.json")
        if not check["passed"] or check["binding"] != manifest["binding"] or check["concurrency"] < args.concurrency:
            raise ValueError("matching full-request check at this concurrency required before timing")
    rpc(config["broker_url"], "phase", dict(phase=args.phase))
    if args.phase == "check":
        write(run / "service_check.json", dict(passed=False, binding=manifest["binding"], concurrency=args.concurrency))
    control_rtt = []
    for _ in range(32):
        before = time.perf_counter()
        rpc(config["broker_url"], "health")
        control_rtt.append((time.perf_counter() - before) * 1000)
    split = "calibration" if args.phase in ("warmup", "calibration") else "test"
    texts = [row["text"] for row in config["dataset"] if row["split"] == split]
    if not texts:
        raise ValueError(f"no {split} texts")
    point = run / f"{args.phase}-c{args.concurrency}-seed{args.seed}.json"
    if point.exists():
        previous = read(point)
        if previous.get("complete") and previous["binding"] == binding(config):
            print(f"已完成，恢复时跳过：{point}")
            return
        raise ValueError("partial point exists; preserve it and select another seed/run")
    if point.with_suffix(".partial.json").exists():
        raise ValueError(
            "incomplete service point preserved; use another seed/run rather than mixing interrupted clocks"
        )
    start = time.perf_counter()
    steady_start = start + (30 if args.formal else 0)
    results = []
    requested = max(200, args.requests) if args.formal else args.requests
    arrivals = read(args.arrivals) if args.arrivals else None
    if arrivals is not None and (len(arrivals) != requested or arrivals != sorted(arrivals) or arrivals[0] < 0):
        raise ValueError("arrival offsets must be sorted, nonnegative, and match the requested count")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending, submitted = set(), 0

        def needs_more():
            completed = sum(r["client_arrival_s"] >= steady_start and r["error"] is None for r in results)
            return (
                submitted < requested or args.formal and (time.perf_counter() - steady_start < 600 or completed < 200)
            )

        while pending or needs_more():
            while len(pending) < args.concurrency and (needs_more()):
                if arrivals is not None and submitted >= len(arrivals):
                    if args.formal and not pending and needs_more():
                        raise ValueError(
                            "formal arrival trace does not cover ramp plus 600 s and 200 completed requests"
                        )
                    break
                due = start + arrivals[submitted] if arrivals is not None else None
                if due is not None and time.perf_counter() < due:
                    break
                pending.add(
                    pool.submit(
                        run_request,
                        config,
                        texts[submitted % len(texts)],
                        submitted,
                        args.seed,
                        split,
                        args.phase,
                        args.concurrency,
                        due,
                    )
                )
                submitted += 1
            if not pending:
                time.sleep(0.01)
                continue
            completed, pending = concurrent.futures.wait(
                pending, timeout=0.05, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in completed:
                results.append(future.result())
                if results[-1]["error"] is not None:
                    write(point.with_suffix(".failure.json"), dict(binding=manifest["binding"], requests=results))
                    for task in pending:
                        task.cancel()
                    raise RuntimeError(f"request failed; timing stopped: {results[-1]['error']}")
                elapsed = time.perf_counter() - start
                eta = max(
                    600 - elapsed if args.formal else 0, elapsed / len(results) * max(0, requested - len(results))
                )
                print(f"[{args.phase}] {len(results)}/{requested}+; elapsed={elapsed:.1f}s; ETA~{eta:.1f}s", flush=True)
                write(
                    point.with_suffix(".partial.json"),
                    dict(complete=False, binding=manifest["binding"], requests=results),
                )
    elapsed = time.perf_counter() - steady_start
    ramp = [r for r in results if r["client_arrival_s"] < steady_start]
    results = [r for r in results if r["client_arrival_s"] >= steady_start]
    payload = dict(
        binding=binding(config),
        complete=True,
        phase=args.phase,
        concurrency=args.concurrency,
        method=args.method,
        ablation=args.ablation,
        sampling=args.sampling,
        elapsed_s=elapsed,
        requests=results,
        ramp_requests=ramp,
        qos_fraction=sum(r["qos_pass"] for r in results) / len(results),
        completed_per_second=sum(r["error"] is None for r in results) / elapsed,
        scope="formal" if args.formal else "smoke",
        independent_confirmation=config["independent_confirmation"],
        control_rtt_ms=control_rtt,
        arrival_trace=arrivals,
    )
    write(point, payload)
    if args.phase == "check":
        write(
            run / "service_check.json",
            dict(
                binding=binding(config),
                concurrency=args.concurrency,
                passed=all(r["error"] is None for r in results),
                artifact=str(point),
            ),
        )
    print(f"结果：{point}；QoS 会话占比 {payload['qos_fraction']:.2%}。独立正式确认尚未完成。")


def fit_stage_costs(args, config):
    """Fit only explicitly labelled controlled calibration intervals."""
    if args.stage_events.suffix == ".json":
        args.stage_events = Path(read(args.stage_events)["events"])
    manifest = read(args.stage_events.parent / "manifest.json")
    if manifest["binding"] != binding(config) or not manifest["controlled"]:
        raise ValueError("stage calibration requires a matching controlled resource window")
    mode = manifest["sampling"]
    samples, calibration_ids = {}, set()

    def add(key, value):
        if value > 0:
            samples.setdefault(key, []).append(value)

    for line in args.stage_events.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["event"] == "arrival" and row["split"] == "calibration":
            calibration_ids.add(row["id"])
        if row["event"] == "delivery_confirmation" and row["phase"] == "calibration":
            if row["id"] not in calibration_ids:
                raise ValueError("delivery sample is not from calibration")
            add("delivery/b1", row["wall_ms"])
        if row["event"] != "stage" or row["grant"]["phase"] != "calibration":
            continue
        grant = row["grant"]
        if row.get("error") or any(j["id"] not in calibration_ids for j in grant["jobs"]):
            raise ValueError("stage calibration contains failures or non-calibration requests")
        batch, stage = len(grant["jobs"]), grant["stage"]
        if stage == "generate":
            if grant["strategy"] != "stock":
                raise ValueError("collect stage costs using liveserve_ar, separately from dynamic-tree evaluation")
            if not row.get("completed_frames") or row["cp_ms"] <= 0:
                continue
            add(f"predictor_stock/b{batch}", row["cp_ms"])
            add(f"talker/b{batch}", row["wall_ms"] - row["cp_ms"])
        else:
            add(f"{stage}/b{batch}", row["wall_ms"])
    # Prefill cost comes from the inner calibration, not a guessed fraction of
    # stock AR. Delivery uses stage release to playable-client confirmation on
    # the broker clock, including transport and feedback polling delay.
    inner = read(Path(config["results"]) / "prefill_costs.json")
    for row in inner:
        if row["sampling"] == mode:
            add(f"predictor_prefill/b{row['batch']}", row["wall_ms"])
    required = [f"{s}/b{b}" for s in ("predictor_stock", "talker", "code2wav", "prefill") for b in range(1, 9)]
    required += [f"predictor_prefill/b{b}" for b in config["batch_buckets"]] + ["delivery/b1"]
    missing = [key for key in required if key not in samples]
    write(
        Path(config["results"]) / f"stage_costs_{mode}.json",
        dict(
            split="calibration",
            binding=binding(config),
            samples=samples,
            missing_buckets=missing,
            delivery_model=(
                "server release to playable-client feedback receipt; includes feedback period; one server clock"
            ),
        ),
    )
    print(f"阶段模型已保存；缺失桶：{missing}。缺失桶在运行时明确拒绝，不插值。")
