# SPDX-License-Identifier: Apache-2.0
"""Real-audio codec statistics. No correctness certificates or performance gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import signal
import socket
import subprocess
import time
import urllib.request
import uuid
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "/softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice"
FORMAT = "codebook_statistics_v1"
SEEDS = (42, 2026, 3407)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def log(root, message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (root / "progress.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(args):
    packages = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hardware = None
    if os.name != "nt":
        try:
            hardware = subprocess.check_output(["npu-smi", "info"], text=True, stderr=subprocess.STDOUT, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
    cann = {}
    for info in (
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"),
        Path("/usr/local/Ascend/driver/version.info"),
    ):
        if info.is_file():
            cann[str(info)] = info.read_text(errors="replace")
    config = Path(args.model) / "config.json"
    return dict(
        repositories=dict(omni=revision(ROOT), vllm=revision(args.vllm_repo), ascend=revision(args.ascend_repo)),
        packages=packages,
        platform=platform.platform(),
        hardware=hardware,
        cann_driver=cann,
        device=args.device,
        model=args.model,
        model_config_sha256=digest(config) if config.exists() else None,
    )


def dataset():
    rows = [
        json.loads(line) for line in Path(__file__).with_name("texts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != 100 or len({r["id"] for r in rows}) != 100 or len({r["text"] for r in rows}) != 100:
        raise ValueError("Expected 100 unique texts and ids")
    # Hash ordering is platform-independent and frozen before any generation.
    training = {
        r["id"] for r in sorted(rows, key=lambda r: hashlib.sha256(("split-v1:" + r["id"]).encode()).hexdigest())[:70]
    }
    return [dict(r, split="statistics" if r["id"] in training else "test") for r in rows]


def requests_for(manifest):
    # Seed-major order gives a short --limit run multiple distinct texts.
    return [
        dict(row, text_id=row["id"], id=f"{row['id']}-s{seed}", seed=seed)
        for seed in manifest["seeds"]
        for row in manifest["dataset"]
    ]


def prepare(args):
    import yaml

    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        log(root, "已有冻结清单，保留原始配置和采集数据。")
        return
    if any((root / name).exists() for name in ("codes", "audio", "completed")):
        raise ValueError("Result contains capture directories without manifest; choose a new results directory")
    config = yaml.safe_load((ROOT / "vllm_omni/deploy/qwen3_tts_ar_stock.yaml").read_text(encoding="utf-8"))
    config["connectors"]["connector_of_shared_memory"]["extra"]["code_predictor_ar_capture"] = str(root)
    if not args.device.startswith("npu:") or not args.device[4:].isdigit():
        raise ValueError("Real audio generation requires --device npu:<index>")
    for stage in config["stages"]:
        stage.update(max_num_seqs=1, devices=args.device[4:], dtype="bfloat16")
    # The capture flag disables outer MTP wrapping; keep normal Talker/Codec setup.
    config["platforms"]["npu"]["stages"][0]["compilation_config"]["cudagraph_capture_sizes"] = [1]
    for name in ("audio", "codes", "completed", "failures", "analysis"):
        (root / name).mkdir()
    deploy = root / "deploy.yaml"
    deploy.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(
        root / "manifest.json",
        dict(
            format=FORMAT,
            model=args.model,
            voice="Vivian",
            language="English",
            execution="stock",
            dtype="bfloat16",
            device=args.device,
            seeds=list(SEEDS),
            dataset=dataset(),
            codebooks=16,
            vocabulary=2048,
            environment=environment(args),
            talker_sampling=config["stages"][0]["default_sampling_params"],
            residual_sampling=config["stages"][0]["subtalker_sampling_params"],
            deployment_sha256=digest(deploy),
            text_source_sha256=digest(Path(__file__).with_name("texts.jsonl")),
            note="Original text prompts, real generated audio. Diagnostic collection is not a latency benchmark.",
        ),
    )
    log(root, "冻结 100 条文本：统计集 70、测试集 30；3 个种子，共 300 段。")


def local_open(request, timeout):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def ready(process, url, root, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("服务退出，请查看 server.log")
        begin = time.monotonic()
        try:
            with local_open(url + "/health", 5) as response:
                if response.status == 200:
                    log(root, "服务已就绪，将直连发送语音请求（绕过代理）。")
                    return
        except OSError as exc:
            log(root, f"等待服务，20 秒检查一次：{exc}")
        time.sleep(max(0, min(20 - (time.monotonic() - begin), deadline - time.monotonic())))
    raise TimeoutError("服务启动超时；见 server.log")


def wav_info(body):
    with wave.open(io.BytesIO(body), "rb") as audio:
        frames, rate = audio.getnframes(), audio.getframerate()
        if not frames or not rate:
            raise ValueError("Empty WAV")
        raw = audio.readframes(frames)
        if len(raw) != frames * audio.getnchannels() * audio.getsampwidth():
            raise ValueError("Truncated WAV")
        return dict(sample_rate=rate, channels=audio.getnchannels(), samples=frames, duration_seconds=frames / rate)


def read_codes(root, request, attempt=None):
    path = root / "codes" / f"{request['id']}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("音频已返回，但未采集到码本；检查 deploy.yaml 和采集开关。")
    for i, row in enumerate(rows):
        if any(row[k] != request[k] for k in ("seed", "text_id", "split")) or row["request_id"] != request["id"]:
            raise ValueError("Request/code association mismatch")
        if row["frame_id"] != i or (attempt is not None and row["attempt"] != attempt):
            raise ValueError("Incomplete frame sequence or mixed generation attempts")
        codes = row["codes"]
        if len(codes) != 16 or not all(isinstance(x, int) for x in codes):
            raise ValueError("Expected sixteen integer codes")
        valid = all(0 <= x < 2048 for x in codes) and any(codes)
        if row["valid"] != valid:
            raise ValueError("Invalid frame validity marker")
    if not any(r["valid"] for r in rows):
        raise ValueError("No valid audio code frames")
    return rows


def is_complete(root, request):
    marker = root / "completed" / f"{request['id']}.json"
    try:
        meta = read_json(marker)
        if any(meta["request"][k] != request[k] for k in ("id", "text_id", "text", "seed", "split")):
            return False
        return (
            digest(root / meta["audio"]) == meta["audio_sha256"]
            and digest(root / meta["codes"]) == meta["codes_sha256"]
        )
    except (OSError, ValueError, KeyError):
        return False


def capture_one(root, request, manifest, url, timeout):
    attempt = uuid.uuid4().hex
    codes_path = root / "codes" / f"{request['id']}.jsonl"
    audio_path = root / "audio" / f"{request['id']}.wav"
    marker = root / "completed" / f"{request['id']}.json"
    # Only this known incomplete request's files are replaced. Never clear root.
    marker.unlink(missing_ok=True)
    codes_path.write_text("", encoding="utf-8")
    write_json(root / "control.json", dict(request, attempt=attempt, format=FORMAT))
    started = time.monotonic()
    payload = dict(
        model=manifest["model"],
        input=request["text"],
        voice="Vivian",
        language="English",
        task_type="CustomVoice",
        response_format="wav",
        stream=False,
        seed=request["seed"],
    )
    req = urllib.request.Request(
        url + "/v1/audio/speech", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with local_open(req, timeout) as response:
            body = response.read()
        info = wav_info(body)
        rows = read_codes(root, request, attempt)
        observed = rows[0]["sampling"]
        if any(row["sampling"] != manifest["residual_sampling"] for row in rows):
            raise ValueError("Actual residual sampling differs from the frozen deployment")
        temporary = audio_path.with_suffix(".wav.tmp")
        temporary.write_bytes(body)
        temporary.replace(audio_path)
        write_json(
            marker,
            dict(
                request=request,
                attempt=attempt,
                **info,
                frames=len(rows),
                valid_frames=sum(r["valid"] for r in rows),
                sampling=observed,
                generation_seconds=time.monotonic() - started,
                audio=str(audio_path.relative_to(root)),
                codes=str(codes_path.relative_to(root)),
                audio_sha256=digest(audio_path),
                codes_sha256=digest(codes_path),
            ),
        )
        (root / "failures" / f"{request['id']}.json").unlink(missing_ok=True)
        return info
    except BaseException as exc:
        write_json(
            root / "failures" / f"{request['id']}.json",
            dict(request=request, attempt=attempt, error=repr(exc), elapsed_seconds=time.monotonic() - started),
        )
        raise
    finally:
        (root / "control.json").unlink(missing_ok=True)


def stop_server(process):
    # Kill our process group, including children if the parent already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def generate(args):
    if os.name == "nt":
        raise RuntimeError("请在 Linux Ascend 服务器生成真实音频。分析工具可在 CPU 运行。")
    import fcntl

    root = args.results.resolve()
    manifest = read_json(root / "manifest.json")
    with (root / "generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("该目录已有生成任务，避免请求串扰，请勿并行采集。") from exc
        requests = requests_for(manifest)
        pending = [r for r in requests if not is_complete(root, r)]
        done = len(requests) - len(pending)
        if args.limit is not None:
            pending = pending[: args.limit]
        if not pending:
            log(root, f"已完成 {done}/{len(requests)}，无需生成。")
            return
        # Refuse to attach to or stop an unrelated service.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        url = f"http://127.0.0.1:{args.port}"
        command = [
            "vllm-omni",
            "serve",
            manifest["model"],
            "--deploy-config",
            str(root / "deploy.yaml"),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--trust-remote-code",
            "--omni",
        ]
        frozen_args = argparse.Namespace(**{**vars(args), "model": manifest["model"], "device": manifest["device"]})
        launch = dict(
            command=command,
            environment=environment(frozen_args),
            started=time.time(),
            deployment_sha256=digest(root / "deploy.yaml"),
        )
        write_json(root / f"launch-{time.time_ns()}.json", launch)
        (root / "control.json").unlink(missing_ok=True)
        with (root / "server.log").open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== generation launch {time.ctime()} ===\n")
            stream.flush()
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                log(
                    root,
                    f"启动 stock 单请求服务；已完成 {done}/300，本次待生成 {len(pending)}；日志 {root / 'server.log'}",
                )
                ready(process, url, root, args.startup_timeout)
                started = time.monotonic()
                for index, request in enumerate(pending, 1):
                    log(root, f"发送 {request['id']}，split={request['split']}，本次 {index}/{len(pending)}")
                    info = capture_one(root, request, manifest, url, args.request_timeout)
                    elapsed = time.monotonic() - started
                    log(
                        root,
                        f"完成 {done + index}/300；音频 {info['duration_seconds']:.2f}s；"
                        f"本次耗时 {elapsed:.1f}s，剩余 ETA~{elapsed / index * (len(pending) - index):.1f}s",
                    )
            finally:
                (root / "control.json").unlink(missing_ok=True)
                stop_server(process)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "generate", "analyze", "report"))
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--vllm-repo", type=Path, default=Path("/vllm-workspace/vllm"))
    p.add_argument("--ascend-repo", type=Path, default=Path("/vllm-workspace/vllm-ascend"))
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--limit", type=int, help="本次最多生成多少个未完成请求；省略则继续完成全部请求")
    p.add_argument("--startup-timeout", type=float, default=1800)
    p.add_argument("--request-timeout", type=float, default=600)
    return p


def main():
    args = parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.command in ("analyze", "report"):
        from analysis import analyze, report

        {"analyze": analyze, "report": report}[args.command](args.results.resolve())
    else:
        {"prepare": prepare, "generate": generate}[args.command](args)


if __name__ == "__main__":
    main()
