# SPDX-License-Identifier: Apache-2.0
"""Compare two services, both using the existing kv_frame Code Predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "qwen3_tts_code2wav_graph"))
import runtime
import yaml
from runtime import (
    ROOT,
    Progress,
    client_env,
    common_args,
    manifest,
    read_json,
    server,
    write_json,
)

_original_deployment = runtime.deployment


def deployment(args, mode, directory, capture=None):
    # Both modes use exactly the same Code2Wav/Talker deployment.
    path = _original_deployment(args, "baseline", directory, capture)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]
    extra["code_predictor_keep_codebooks"] = 16 if mode == "full16" else 10
    extra["code_predictor_ar_telemetry"] = str(directory / "telemetry")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


runtime.deployment = deployment


def texts():
    from generate_zh import TEXTS

    return [dict(id=f"zh{i + 1:03d}", text=text, source_line=f"zh{i + 1:03d}|||{text}") for i, text in enumerate(TEXTS)]


def speech(url, model, text, cancel=False, wav=True):
    body = dict(
        model=model,
        input=text,
        voice="Vivian",
        language="Chinese",
        task_type="CustomVoice",
        seed=42,
        response_format="wav" if wav else "pcm",
    )
    if not wav:
        body["stream_format"] = "audio"
    req = urllib.request.Request(
        url + "/v1/audio/speech", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with runtime.direct_open(req) as response:
        return response.read1(4096) if cancel else response.read()


def snapshots(directory):
    return {p.name: read_json(p) for p in directory.glob("*.json")}


def client(command, output, progress, label):
    tick = time.monotonic()
    path = output / "client.log"
    with path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=client_env())
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    with path.open("rb") as reader:
                        reader.seek(0, 2)
                        reader.seek(max(0, reader.tell() - 4096))
                        lines = reader.read().decode("utf-8", errors="replace").replace("\r", "\n").splitlines()
                    tail = next((x.strip() for x in reversed(lines) if x.strip()), "等待客户端")
                    progress.log(f"{label}，耗时 {time.monotonic() - tick:.0f}s；{tail[-250:]}")
            return process.returncode
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def summarize(root, entries):
    lines = [
        "# kv_frame：完整16码本 vs keep10 服务性能",
        "",
        "两组均为 kv_frame；full16 每帧15次主干，keep10 每帧9次；并发不等于实际 batch。",
        "性能独立运行。keep10省略尾部生成、Talker反馈及Codec贡献，可能改变后续生成与音频时长。",
        "",
        "| 配置 | 并发 | 成功/总数 | 请求/s | 音频秒/s | TTFP p50/p95 ms | "
        "E2E p50/p95 ms | RTF p50/p95 | underrun p50/p95 s | 音频总时长 s | 错误率 |",
        "|---|---:|---|---:|---:|---|---|---|---|---:|---:|",
    ]
    for item in entries:
        r = item.get("result", {})

        def pair(name):
            return f"{r.get('median_' + name, r.get('p50_' + name))}/{r.get('p95_' + name)}"

        n = r.get("completed")
        audio = next(
            (r[k] for k in ("total_audio_duration_s", "total_audio_duration", "audio_duration_total") if k in r), None
        )
        rate = r.get("audio_throughput", audio / r["duration"] if audio is not None and r.get("duration") else None)
        error = (item["requests"] - n) / item["requests"] if n is not None else None
        lines.append(
            f"| {item['mode']} | {item['concurrency']} | {n}/{item['requests']} | "
            f"{r.get('request_throughput')} | {rate} | {pair('audio_ttfp_ms')} | {pair('e2el_ms')} | "
            f"{pair('audio_rtf')} | {pair('audio_underrun_s')} | {audio} | {error} |"
        )
    comparisons = []
    lines += ["", "| 并发 | 请求吞吐比 keep10/full16 | E2E 中位耗时比 full16/keep10 |", "|---:|---:|---:|"]
    for concurrency in sorted({r["concurrency"] for r in entries if r["concurrency"] > 0}):
        sides = {r["mode"]: r.get("result", {}) for r in entries if r["concurrency"] == concurrency}
        baseline, candidate = sides.get("full16", {}), sides.get("keep10", {})

        def ratio(a, b):
            return a / b if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else None

        throughput = ratio(candidate.get("request_throughput"), baseline.get("request_throughput"))
        latency = ratio(
            baseline.get("median_e2el_ms", baseline.get("p50_e2el_ms")),
            candidate.get("median_e2el_ms", candidate.get("p50_e2el_ms")),
        )
        comparisons.append(
            dict(concurrency=concurrency, request_throughput_ratio=throughput, e2e_median_speedup=latency)
        )
        lines.append(f"| {concurrency} | {throughput} | {latency} |")
    lines += [
        "",
        "None 表示未提供。原始指标、执行错误与各档遥测快照保存在 summary.json。",
        "遥测快照是累计计数，包含预热；按 before/after 差值比较正式档位。",
        "重复运行覆盖本轮已管理结果，不删除采集输入或整个输出目录。",
    ]
    write_json(
        root / "summary.json", dict(runs=entries, comparisons=comparisons, correctness_certificate_required=False)
    )
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    root = args.results.resolve()
    # No recursive deletion: NFS busy directories can be reused. Only managed
    # regular output files are overwritten; input/model/repo must remain outside.
    for protected in (ROOT.resolve(), Path(args.model).resolve()):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError("服务结果目录不能包含代码/模型或位于其内部")
    root.mkdir(parents=True, exist_ok=True)
    progress = Progress(root)
    info = manifest(args)
    info.update(experiment="kv_frame_keep10_vs_full16", language="Chinese", keep_codebooks=[16, 10])
    info["baseline"] = "full16 kv_frame, same Code2Wav deployment as keep10"
    for path in (
        Path(__file__),
        ROOT / "vllm_omni/model_executor/models/common/qwen3_code_predictor.py",
        ROOT / "vllm_omni/model_executor/models/common/qwen3_code_predictor_ar.py",
        ROOT / "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py",
        ROOT / "vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py",
    ):
        info["code"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    info["sampling"]["language"] = "Chinese"
    write_json(root / "manifest.json", info)
    dataset = root / "dataset/zh"
    dataset.mkdir(parents=True, exist_ok=True)
    original = texts()
    lines = []
    for i in range(args.requests):
        fields = original[i % len(original)]["source_line"].split("|")
        fields[0] = f"c2w{i:04d}"
        lines.append("|".join(fields))
    (dataset / "meta.lst").write_text("\n".join(lines) + "\n", encoding="utf-8")
    entries = []
    summarize(root, entries)
    total = 2 * len(args.concurrencies)
    done = 0
    for mode in ("full16", "keep10"):
        directory = root / mode
        try:
            with server(args, mode, directory, progress) as url:
                audio_dir = directory / "audio"
                audio_dir.mkdir(exist_ok=True)
                for example in original:
                    progress.log(f"{mode} 保存真实生成试听音频 {example['id']}")
                    (audio_dir / (example["id"] + ".wav")).write_bytes(speech(url, args.model, example["text"]))
                # Diagnostic only: changed output does not block independent timing.
                try:
                    before = speech(url, args.model, original[0]["text"], wav=False)
                    speech(url, args.model, " ".join(r["text"] for r in original), cancel=True, wav=False)
                    after = speech(url, args.model, original[0]["text"], wav=False)
                    write_json(
                        directory / "cancellation.json",
                        dict(
                            equal=before == after,
                            before=hashlib.sha256(before).hexdigest(),
                            after=hashlib.sha256(after).hexdigest(),
                        ),
                    )
                except Exception as exc:
                    write_json(directory / "cancellation.json", dict(error=repr(exc)))
                for concurrency in args.concurrencies:
                    output = directory / f"c{concurrency}" / "round0"
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "result.json").unlink(missing_ok=True)
                    label = f"{mode} 并发{concurrency}：预热3，正式{args.requests}，1轮"
                    progress.log(label)
                    write_json(output / "batch-before.json", snapshots(directory / "telemetry"))
                    command = [
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
                        "zh",
                        "--disable-shuffle",
                        "--seed",
                        "42",
                        "--num-prompts",
                        str(args.requests),
                        "--num-warmups",
                        "3",
                        "--max-concurrency",
                        str(concurrency),
                        "--request-rate",
                        "inf",
                        "--extra-body",
                        json.dumps(dict(voice="Vivian", language="Chinese", task_type="CustomVoice", seed=42)),
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
                    write_json(output / "command.json", command)
                    code = client(command, output, progress, label)
                    write_json(output / "batch-after.json", snapshots(directory / "telemetry"))
                    item = dict(
                        mode=mode,
                        concurrency=concurrency,
                        requests=args.requests,
                        returncode=code,
                        result=read_json(output / "result.json") if (output / "result.json").exists() else {},
                        before=read_json(output / "batch-before.json"),
                        after=read_json(output / "batch-after.json"),
                    )
                    entries.append(item)
                    summarize(root, entries)
                    done += 1
                    progress.update(done, total, "两组服务对比")
        except Exception as exc:
            progress.log(f"{mode} 失败：{exc}；保留诊断并继续另一组")
            entries.append(dict(mode=mode, concurrency=0, requests=args.requests, error=repr(exc)))
            summarize(root, entries)
    failures = sum(
        bool(r.get("error")) or bool(r.get("returncode")) or r.get("result", {}).get("completed") != r["requests"]
        for r in entries
    )
    write_json(
        root / "status.json", dict(status="completed_with_errors" if failures else "complete", failures=failures)
    )
    if failures:
        progress.log(f"服务对比已结束，{failures} 项存在错误；详见 summary.json，不将其记为成功测试。")


def main():
    parser = argparse.ArgumentParser(__doc__)
    common_args(parser)
    parser.add_argument(
        "--results", type=Path, default=Path("/softwarePlatform/y00899301/qwen3_tts_keep10_perf_run01/service")
    )
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 16])
    args = parser.parse_args()
    if args.requests < 1 or any(n < 1 or n > 16 for n in args.concurrencies):
        parser.error("请求数必须为正，并发范围1..16")
    run(args)


if __name__ == "__main__":
    main()
