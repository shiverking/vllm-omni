# SPDX-License-Identifier: Apache-2.0
"""Shared I/O and server lifecycle; no correctness certificates."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL = "/softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice"
RESULTS = "/softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01"
OPTIMIZED_RESULTS = "/softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_optimized_run01"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest(args):
    repos = {}
    for name, path in (("omni", ROOT), ("vllm", args.vllm_repo), ("ascend", args.ascend_repo)):
        try:
            repos[name] = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            repos[name] = None
    packages = {}
    for name in ("torch", "torch-npu", "transformers", "vllm", "vllm-ascend"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    files = list((ROOT / "tools/qwen3_tts_code2wav_graph").glob("*.py"))
    files += [
        ROOT / "vllm_omni/model_executor/models/qwen3_tts" / name
        for name in (
            "npu_segmented_graph_wrapper.py",
            "segmented_graph_wrapper.py",
            "qwen3_tts_code2wav.py",
            "code2wav_optimization.py",
            "code2wav_state_pool.py",
            "code2wav_capture.py",
            "tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py",
        )
    ]
    cann = Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg")
    return dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        repositories=repos,
        packages=packages,
        code={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        cann=cann.read_text() if cann.exists() else None,
        baseline="kv_frame@13350ff06; existing Code2Wav cache",
        model_config_sha256={
            str(p.relative_to(Path(args.model))): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(args.model) / "config.json", Path(args.model) / "speech_tokenizer/config.json")
            if p.exists()
        },
        sampling=dict(
            voice="Vivian",
            language="English",
            task_type="CustomVoice",
            seed=42,
            subtalker=dict(temperature=0.9, top_k=50, top_p=1.0),
        ),
        dtype="bfloat16" if args.device.startswith("npu") else "float32",
        correctness_certificate_required=False,
    )


class Progress:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()

    def log(self, text):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
        print(line, flush=True)
        with (self.root / "progress.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def update(self, done, total, label, started=None):
        elapsed = time.monotonic() - (self.start if started is None else started)
        eta = elapsed / max(done, 1) * max(total - done, 0)
        self.log(f"{label}: {done}/{total}; elapsed={elapsed:.1f}s ETA~{eta:.1f}s")


def direct_open(request, timeout=1800):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def client_env():
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
    env.update(NO_PROXY="127.0.0.1,localhost,::1", no_proxy="127.0.0.1,localhost,::1", PYTHONUNBUFFERED="1")
    return env


def texts():
    path = ROOT / "benchmarks/build_dataset/seed_tts_smoke/en/meta.lst"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [dict(id=f"smoke{i + 1:03d}", text=line.split("|")[-1], source_line=line) for i, line in enumerate(lines)]


def speech(url, model, text, cancel=False, wav=True):
    body = dict(
        model=model,
        input=text,
        voice="Vivian",
        language="English",
        task_type="CustomVoice",
        seed=42,
        response_format="wav" if wav else "pcm",
    )
    if not wav:
        body["stream_format"] = "audio"
    request = urllib.request.Request(
        url + "/v1/audio/speech", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with direct_open(request) as response:
        return response.read1(4096) if cancel else response.read()


def deployment(args, mode, directory, capture=None):
    name = "baseline" if mode == "baseline" else "npu_graph"
    config = yaml.safe_load((ROOT / f"vllm_omni/deploy/qwen3_tts_kv_frame_code2wav_{name}.yaml").read_text())
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]
    telemetry = directory / "telemetry"
    telemetry.mkdir(parents=True, exist_ok=True)
    # Only this tool's regular snapshots are replaced, never directories or input.
    for old in telemetry.glob("*.json"):
        if not old.is_symlink():
            old.unlink()
    extra["code2wav_graph_telemetry"] = str(telemetry)
    extra["code2wav_graph_optimization"] = mode if mode in ("constants", "pooled", "fused") else "current"
    if capture:
        extra["code2wav_capture_dir"] = str(capture)
        # Collection must execute Python hooks, so disable Stage 1 outer capture
        # only for collection. Formal baseline serving retains its configuration.
        config["stages"][1]["enforce_eager"] = True
        config["platforms"]["npu"]["stages"][1] = dict(
            stage_id=1, enforce_eager=True, compilation_config=dict(cudagraph_mode="NONE")
        )
    for stage in config["stages"]:
        stage["devices"] = args.device.split(":")[-1]
    path = directory / "deploy.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@contextlib.contextmanager
def server(args, mode, directory, progress, capture=None):
    if os.name == "nt" or not args.device.startswith("npu:"):
        raise RuntimeError("服务实验请在 Linux Ascend 主机运行")
    directory.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    try:
        direct_open(url + "/health", timeout=2).close()
    except OSError:
        pass
    else:
        raise RuntimeError(f"端口 {args.port} 已有服务；不会关闭或测试外部进程")
    config = deployment(args, mode, directory, capture)
    command = [
        "vllm-omni",
        "serve",
        args.model,
        "--deploy-config",
        str(config),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--omni",
    ]
    write_json(directory / "launch.json", dict(command=command))
    with (directory / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 1800
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"服务退出，查看 {directory / 'server.log'}")
                tick = time.monotonic()
                try:
                    with direct_open(url + "/health", timeout=5) as reply:
                        if reply.status == 200:
                            break
                except OSError as exc:
                    progress.log(f"{mode} 等待服务；本机直连，每20秒检查：{exc}")
                if time.monotonic() > deadline:
                    raise TimeoutError("等待服务超过1800秒")
                time.sleep(max(0, 20 - (time.monotonic() - tick)))
            progress.log(f"{mode} 服务就绪；日志：{directory / 'server.log'}")
            yield url
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def common_args(parser):
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--vllm-repo", type=Path, default=Path("/vllm-workspace/vllm"))
    parser.add_argument("--ascend-repo", type=Path, default=Path("/vllm-workspace/vllm-ascend"))
    parser.add_argument("--port", type=int, default=8091)
