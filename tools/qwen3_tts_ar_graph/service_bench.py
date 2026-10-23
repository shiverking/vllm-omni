# SPDX-License-Identifier: Apache-2.0
"""Sequentially launch three servers and reuse the tag TTS streaming benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import yaml
from experiment import (
    MODEL,
    ROOT,
    Progress,
    binding,
    capture_root,
    read_json,
    select_frames,
    write_json,
)

CONCURRENCIES = (1, 4, 16)
ROUNDS = 1
WARMUP_REQUESTS = 3
HEALTH_INTERVAL = 20


def local_urlopen(request, timeout):
    # Private opener: ignore proxy settings for this script's loopback requests
    # without changing urllib's global opener or the server environment.
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def benchmark_client_env():
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            del env[key]
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost,::1"
    return env


def ready(process, url, timeout=1800):
    deadline = time.monotonic() + timeout
    last_error = "尚未响应"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited; inspect server.log")
        attempt_started = time.monotonic()
        try:
            with local_urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    print(f"本机直连健康检查通过: {url}/health", flush=True)
                    return
                last_error = f"HTTP {response.status}"
        except OSError as exc:
            last_error = str(exc)
        print(f"等待本机服务就绪（每 20 秒检查）: {url}/health；{last_error}", flush=True)
        delay = min(HEALTH_INTERVAL - (time.monotonic() - attempt_started), deadline - time.monotonic())
        if delay > 0:
            time.sleep(delay)
    raise TimeoutError("Server did not become healthy within 1800 seconds")


def run_benchmark(command, output, progress, label):
    """Keep the full client log and report progress while a round is running."""
    log_path = output / "client.log"
    started = time.monotonic()
    env = benchmark_client_env()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as client_log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=client_log, stderr=subprocess.STDOUT, env=env)
        try:
            while True:
                try:
                    code = child.wait(timeout=20)
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                    break
                except subprocess.TimeoutExpired:
                    with log_path.open("rb") as reader:
                        reader.seek(0, 2)
                        reader.seek(max(0, reader.tell() - 4096))
                        lines = reader.read().decode("utf-8", errors="replace").replace("\r", "\n").splitlines()
                    latest = next((line.strip() for line in reversed(lines) if line.strip()), "等待客户端进度输出")
                    progress.message(f"{label}：运行中，耗时 {time.monotonic() - started:.0f}s；{latest[-300:]}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    progress.message(f"{label}：客户端完成，耗时 {time.monotonic() - started:.1f}s")


def audio_request(url, model, text, cancel=False):
    payload = dict(
        model=model,
        input=text,
        voice="Vivian",
        language="English",
        task_type="CustomVoice",
        response_format="pcm",
        stream_format="audio",
        seed=42,
    )
    request = urllib.request.Request(
        url + "/v1/audio/speech", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    print(f"发送本机语音请求（绕过代理）: cancel={cancel}, text_chars={len(text)}", flush=True)
    with local_urlopen(request, timeout=1800) as response:
        body = response.read1(4096) if cancel else response.read()
    if not body or len(body) % 2:
        raise RuntimeError("Empty or malformed PCM response")
    return hashlib.sha256(body).hexdigest()


def snapshot(directory):
    return {p.name: read_json(p) for p in sorted(directory.glob("batches-*.json"))}


def summarize(results):
    lines = [
        "# 原始 AR 多并发服务实验",
        "",
        "本服务测试不校验帧级正确性证书；性能结果不代表新增正确性验证。",
        "",
        "客户端并发不等于实际推理 batch。batch 遥测包含预热请求，原始快照按轮保存。",
        "",
        "| 配置 | 并发 | 轮次 | 成功请求 | 请求/s | 音频/s | TTFP p50/p95 | E2E p50/p95 | "
        "RTF p50/p95 | underrun p50/p95 s | 错误率 |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---:|",
    ]
    summaries = []

    def get(data, *keys):
        return next((data[k] for k in keys if k in data), None)

    for path in sorted(results.glob("*/c*/round*/result.json")):
        if not (path.parent / "batch-after.json").exists():
            continue
        data = read_json(path)
        mode, concurrency, round_id = path.parts[-4:-1]
        item = dict(mode=mode, concurrency=int(concurrency[1:]), round=round_id, raw=data)
        before = read_json(path.parent / "batch-before.json")
        after = read_json(path.parent / "batch-after.json")
        distribution = {}
        for worker, value in after.items():
            old = {
                (row["real_batch"], row["execution_rows"], row["outer_mtp_graph"]): row["calls"]
                for row in before.get(worker, {}).get("rows", [])
            }
            for row in value["rows"]:
                key = (row["real_batch"], row["execution_rows"], row["outer_mtp_graph"])
                label = str(key)
                distribution[label] = distribution.get(label, 0) + row["calls"] - old.get(key, 0)
        item["actual_batch_distribution_including_warmup"] = distribution
        summaries.append(item)
        completed = get(data, "completed")
        duration = get(data, "duration")
        audio_duration = get(data, "total_audio_duration_s", "total_audio_duration", "audio_duration_total")
        throughput = get(data, "request_throughput")
        audio_throughput = get(data, "audio_throughput")
        if audio_throughput is None and audio_duration is not None and duration:
            audio_throughput = audio_duration / duration

        def pair(metric):
            return f"{get(data, 'median_' + metric, 'p50_' + metric)}/{get(data, 'p95_' + metric)}"

        lines.append(
            f"| {mode} | {concurrency[1:]} | {round_id} | {completed} | {throughput} | "
            f"{audio_throughput} | {pair('audio_ttfp_ms')} | {pair('e2el_ms')} | "
            f"{pair('audio_rtf')} | {pair('audio_underrun_s')} | "
            f"{(120 - completed) / 120 if completed is not None else None} |"
        )
    lines += [
        "",
        "None 表示上游 benchmark 未提供该字段，不按 0 处理。",
        "错误率、underrun 及单位以每轮 result.json 原始字段为准；全部原始指标保存在 summary.json。",
        "batch-before.json / batch-after.json 保存实际 batch 与外层图范围；telemetry/ 保存累计计数。",
    ]
    write_json(results / "summary.json", summaries)
    (results / "report.md").write_text("\n".join(lines), encoding="utf-8")


def reset_results(args):
    """Replace only the explicitly selected output directory, never its inputs."""
    original = Path(args.results)
    target = original.resolve()
    protected = [ROOT.resolve(), Path.home().resolve(), Path.cwd().resolve()]
    inputs = []
    for name in ("input", "import_from", "correctness", "failure", "model", "vllm_repo", "ascend_repo"):
        value = getattr(args, name, None)
        if value:
            path = Path(value).resolve()
            inputs.append(path)
            if name in ("input", "import_from") and (path / "import.json").is_file():
                inputs.append(Path(read_json(path / "import.json")["source"]).resolve())
    if original.is_symlink() or target == Path(target.anchor):
        raise ValueError(f"拒绝清理根目录或链接目录: {target}")
    if any(target == path or target in path.parents for path in protected + inputs):
        raise ValueError(f"结果目录包含输入、模型或代码目录，拒绝清理: {target}")
    if any(path in target.parents for path in inputs):
        raise ValueError(f"结果目录位于输入目录内部，拒绝清理: {target}")
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"结果路径不是目录: {target}")
        print(f"清理服务多并发测试旧结果并重新运行: {target}", flush=True)
        shutil.rmtree(target)
    target.mkdir(parents=True)


def prepare_results(args):
    # Service benchmarks are independent of frame-correctness certificates.
    args._results_created = False
    reset_results(args)
    args._results_created = True
    root = capture_root(args.input)
    write_json(
        args.results / "service_run.json",
        dict(
            correctness_certificate_required=False,
            correctness_certificate_checked=False,
            legacy_correctness_argument=str(args.correctness) if getattr(args, "correctness", None) else None,
            note="服务性能测试不校验帧级证书；本次运行不代表新增帧级正确性验证。",
        ),
    )
    print("服务多并发测试：不校验正确性证书，开始准备服务。", flush=True)
    return root


def run(args):
    if os.name == "nt":
        raise RuntimeError("Run server experiments on the Linux Ascend host")
    root = prepare_results(args)
    progress = Progress(args.results)
    dataset = args.results / "dataset/en"
    dataset.mkdir(parents=True)
    original = (ROOT / "benchmarks/build_dataset/seed_tts_smoke/en/meta.lst").read_text(encoding="utf-8").splitlines()
    if len(original) != 30:
        raise ValueError("Expected 30 texts")
    # Expand explicitly instead of relying on random oversampling upstream.
    expanded = []
    for i in range(120):
        fields = original[i % 30].split("|")
        fields[0] = f"ar{i:04d}"
        expanded.append("|".join(fields))
    (dataset / "meta.lst").write_text("\n".join(expanded) + "\n", encoding="utf-8")
    url = f"http://127.0.0.1:{args.port}"
    # Do not accidentally benchmark or stop a pre-existing service.
    try:
        local_urlopen(url + "/health", timeout=2).close()
    except OSError:
        pass
    else:
        raise RuntimeError("Port already serves a healthy process; choose a free --port")
    progress.begin("服务多并发总进度", 3 * len(CONCURRENCIES) * ROUNDS)
    for mode in ("stock", "kv_step", "kv_frame"):
        progress.message(f"{mode}：准备配置与运行信息，随后启动服务；并发 {list(CONCURRENCIES)}，每档 {ROUNDS} 轮")
        directory = args.results / mode
        directory.mkdir()
        config = yaml.safe_load((ROOT / f"vllm_omni/deploy/qwen3_tts_ar_{mode}.yaml").read_text(encoding="utf-8"))
        config["connectors"]["connector_of_shared_memory"]["extra"]["code_predictor_ar_telemetry"] = str(
            directory / "telemetry"
        )
        deploy = directory / "deploy.yaml"
        deploy.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [
            "vllm-omni",
            "serve",
            args.model,
            "--deploy-config",
            str(deploy),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--trust-remote-code",
            "--omni",
        ]
        write_json(directory / "launch.json", dict(command=command, binding=binding(args, root, select_frames(root))))
        with (directory / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                ready(process, url)
                progress.message(f"{mode}: 服务就绪，执行预热和取消/替换检查")
                text = original[0].split("|")[-1]
                before = audio_request(url, args.model, text)
                audio_request(url, args.model, " ".join(row.split("|")[-1] for row in original), cancel=True)
                after = audio_request(url, args.model, text)
                write_json(
                    directory / "cancellation.json",
                    dict(
                        passed=before == after,
                        before=before,
                        after=after,
                        note="close streaming response early, then compare identical seeded request PCM",
                    ),
                )
                if before != after:
                    raise RuntimeError("Seeded output changed after cancellation; diagnose before performance run")
                for concurrency in CONCURRENCIES:
                    for repeat in range(ROUNDS):
                        label = f"{mode} / 并发 {concurrency} / 第 {repeat + 1}/{ROUNDS} 轮"
                        progress.message(f"开始 {label}：预热 {WARMUP_REQUESTS} 请求，正式 120 请求")
                        output = directory / f"c{concurrency}" / f"round{repeat}"
                        output.mkdir(parents=True)
                        time.sleep(2)  # background telemetry flush, outside measurement
                        write_json(output / "batch-before.json", snapshot(directory / "telemetry"))
                        bench = [
                            "vllm",
                            "bench",
                            "serve",
                            "--omni",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(args.port),
                            "--model",
                            args.model,
                            "--backend",
                            "openai-audio-speech",
                            "--endpoint",
                            "/v1/audio/speech",
                            "--dataset-name",
                            "seed-tts-text",
                            "--dataset-path",
                            str(dataset.parent),
                            "--seed-tts-locale",
                            "en",
                            "--disable-shuffle",
                            "--seed",
                            "42",
                            "--num-prompts",
                            "120",
                            "--num-warmups",
                            str(WARMUP_REQUESTS),
                            "--max-concurrency",
                            str(concurrency),
                            "--request-rate",
                            "inf",
                            "--extra-body",
                            json.dumps(dict(voice="Vivian", language="English", task_type="CustomVoice")),
                            "--percentile-metrics",
                            "ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun",
                            "--metric-percentiles",
                            "50,95",
                            "--save-result",
                            "--save-detailed",
                            "--result-dir",
                            str(output),
                            "--result-filename",
                            "result.json",
                        ]
                        write_json(output / "command.json", bench)
                        run_benchmark(bench, output, progress, label)
                        time.sleep(2)
                        write_json(output / "batch-after.json", snapshot(directory / "telemetry"))
                        result = read_json(output / "result.json")
                        if result.get("completed") != 120:
                            raise RuntimeError("Service requests failed; inspect result.json before continuing")
                        progress.advance()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                summarize(args.results)
        time.sleep(3)
    summarize(args.results)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--correctness", type=Path, help="兼容旧命令；服务测试不读取或校验此证书")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--vllm-repo", type=Path, default=Path("/vllm-workspace/vllm"))
    p.add_argument("--ascend-repo", type=Path, default=Path("/vllm-workspace/vllm-ascend"))
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--stock-batch-policy", choices=("strict", "record-stable"), default="strict")
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        if getattr(args, "_results_created", False):
            write_json(args.results / "status.json", dict(status="failed", error=repr(exc)))
        raise
    write_json(args.results / "status.json", dict(status="complete"))


if __name__ == "__main__":
    main()
