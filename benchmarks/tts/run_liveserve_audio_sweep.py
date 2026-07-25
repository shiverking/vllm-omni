#!/usr/bin/env python3
"""Run the reproducible single-NPU LiveServe scheduling experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CONCURRENCIES = (1, 2, 4, 8)
LOAD_FACTORS = (0.50, 0.75, 1.00, 1.25)
STRATEGIES = {
    "legacy": "qwen3_tts_910b4_single_legacy.yaml",
    "bounded_k2": "qwen3_tts_910b4_single_bounded_k2.yaml",
    "liveserve_audio": "qwen3_tts_910b4_single_liveserve_audio.yaml",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "vllm_omni" / "deploy"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_arrival_offsets(num_requests: int, request_rate: float, seed: int) -> list[float]:
    """Generate one deterministic Poisson trace as absolute offsets."""
    if num_requests <= 0 or request_rate <= 0 or not math.isfinite(request_rate):
        raise ValueError("num_requests and finite request_rate must be positive")
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(1, num_requests):
        offsets.append(offsets[-1] + rng.expovariate(request_rate))
    return offsets


def write_arrival_trace(path: Path, *, num_requests: int, request_rate: float, seed: int) -> None:
    payload = {
        "schema_version": 1,
        "distribution": "poisson",
        "request_rate": request_rate,
        "seed": seed,
        "arrival_offsets_s": generate_arrival_offsets(num_requests, request_rate, seed),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_benchmark_command(
    args: argparse.Namespace,
    *,
    result_path: Path,
    manifest_path: Path,
    concurrency: int | None = None,
    arrival_trace: Path | None = None,
    export_manifest: bool = False,
) -> list[str]:
    command = [
        args.bench_bin,
        "bench",
        "serve",
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--backend",
        "openai-audio-speech",
        "--endpoint",
        "/v1/audio/speech",
        "--dataset-name",
        "seed-tts",
        "--dataset-path",
        str(args.dataset_path),
        "--seed-tts-locale",
        "en",
        "--extra-body",
        '{"task_type":"Base"}',
        "--seed",
        "0",
        "--num-prompts",
        str(args.num_requests),
        "--num-warmups",
        str(args.num_warmups),
        "--percentile-metrics",
        "e2el,audio_ttfp,audio_rtf,audio_duration,audio_underrun",
        "--metric-percentiles",
        "50,90,99",
        "--enable-playback-feedback",
        "--goodput",
        f"audio_ttfp:{args.useful_audio_ttfp_ms}",
        "audio_continuity:1",
        "--useful-audio-ttfp-ms",
        str(args.useful_audio_ttfp_ms),
        "--save-result",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
    ]
    command += ["--workload-manifest-out" if export_manifest else "--workload-manifest-in", str(manifest_path)]
    if concurrency is not None:
        command += ["--max-concurrency", str(concurrency), "--request-rate", "inf"]
    if arrival_trace is not None:
        command += ["--arrival-trace-in", str(arrival_trace), "--request-rate", "1"]
    command += list(args.benchmark_extra_arg)
    return command


def build_server_command(args: argparse.Namespace, config_path: Path) -> list[str]:
    command = [
        args.serve_bin,
        "serve",
        args.model,
    ]
    if args.model_revision:
        command += ["--revision", args.model_revision]
    command += [
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--deploy-config",
        str(config_path),
        *args.server_extra_arg,
    ]
    return command


def fetch_scheduler_metrics(args: argparse.Namespace) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/v1/audio/speech/scheduling"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read())
    except Exception:
        return {"scheduling": {}, "feedback": {}}


def _connect_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host


def ensure_server_port_available(args: argparse.Namespace) -> None:
    """Reject a stale server instead of benchmarking the wrong process."""
    host = _connect_host(args.host)
    try:
        connection = socket.create_connection((host, args.port), timeout=1.0)
    except OSError:
        return
    connection.close()
    raise RuntimeError(
        f"{host}:{args.port} is already accepting connections. "
        "Stop the existing server or choose another --port before running the sweep."
    )


def _log_tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "<server log is not readable>"


def wait_for_server_health(
    args: argparse.Namespace,
    server: subprocess.Popen,
    server_log_path: Path,
) -> None:
    """Poll /health until the API server is ready or startup fails."""
    host = _connect_host(args.host)
    health_url = f"http://{host}:{args.port}/health"
    started = time.monotonic()
    deadline = started + args.server_startup_timeout_s
    last_error = "not checked"
    while time.monotonic() < deadline:
        return_code = server.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Server exited before becoming healthy (exit code {return_code}).\n"
                f"Last {server_log_path.name} lines:\n{_log_tail(server_log_path)}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    elapsed = time.monotonic() - started
                    print(f"Server is healthy after {elapsed:.1f}s: {health_url}", flush=True)
                    return
                last_error = f"HTTP {response.status}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        print(
            f"Waiting for server health ({elapsed:.1f}s, last={last_error}); "
            f"retrying in {args.health_poll_interval_s:g}s...",
            flush=True,
        )
        time.sleep(args.health_poll_interval_s)
    raise TimeoutError(
        f"Server did not become healthy within {args.server_startup_timeout_s:g}s "
        f"(last={last_error}).\nLast {server_log_path.name} lines:\n{_log_tail(server_log_path)}"
    )


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for section in ("scheduling", "feedback"):
        old = before.get(section) or {}
        new = after.get(section) or {}
        result[section] = {key: float(value) - float(old.get(key, 0)) for key, value in new.items()}
    return result


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def annotate_result(
    result_path: Path,
    *,
    strategy: str,
    mode: str,
    repeat: int,
    manifest_sha256: str,
    model: str,
    model_revision: str | None,
    config_sha256: str,
    scheduler_delta: dict[str, Any],
    concurrency: int | None = None,
    load_factor: float | None = None,
    request_rate: float | None = None,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["experiment"] = {
        "strategy": strategy,
        "mode": mode,
        "repeat": repeat,
        "concurrency": concurrency,
        "load_factor": load_factor,
        "request_rate": request_rate,
        "manifest_sha256": manifest_sha256,
        "model": model,
        "model_revision": model_revision,
        "common_config_sha256": config_sha256,
    }
    result["audio_scheduling_metrics"] = scheduler_delta
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return sum(values) / len(values) if values else float("nan")


def verify_results(records: list[dict[str, Any]], expected_manifest_sha256: str) -> list[str]:
    errors: list[str] = []
    identities = {
        (
            record["experiment"]["manifest_sha256"],
            record["experiment"]["model"],
            record["experiment"]["model_revision"],
            record["experiment"]["common_config_sha256"],
        )
        for record in records
    }
    if len(identities) != 1 or next(iter(identities))[0] != expected_manifest_sha256:
        errors.append("manifest、模型 revision 或公共部署参数不一致")
    for record in records:
        if int(record.get("failed", 0)) != 0 or int(record.get("completed", 0)) == 0:
            errors.append(f"{record['experiment']} 存在失败请求或没有完成请求")
        if int(record.get("feedback_failure_count", 0)) != 0:
            errors.append(f"{record['experiment']} 存在播放反馈发送失败")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        exp = record["experiment"]
        groups[(exp["mode"], exp["concurrency"], exp["load_factor"], exp["repeat"])].append(record)
    for cell, group in groups.items():
        if len(group) != len(STRATEGIES):
            continue
        durations = [float(item.get("total_audio_duration_s", 0)) for item in group]
        baseline = sum(durations) / len(durations)
        if baseline > 0 and (max(durations) - min(durations)) / baseline > 0.03:
            errors.append(f"cell={cell} 的策略间输出音频总时长差异超过 3%: {durations}")
    return errors


def write_report(output_dir: Path, records: list[dict[str, Any]], errors: list[str]) -> Path:
    fixed = [record for record in records if record["experiment"]["mode"] == "fixed"]
    legacy_c8 = [
        record
        for record in fixed
        if record["experiment"]["strategy"] == "legacy" and record["experiment"]["concurrency"] == 8
    ]
    no_bottleneck = bool(
        legacy_c8
        and mean(legacy_c8, "p90_audio_rtf") <= 1.0
        and mean(legacy_c8, "audio_continuity_ok_rate") >= 0.95
    )
    conclusion = "未进入调度瓶颈区" if no_bottleneck else "已进入或接近调度瓶颈区，请比较 useful throughput 与连续性"
    model = records[0]["experiment"]["model"] if records else DEFAULT_MODEL
    report = {
        "schema_version": 1,
        "model": model,
        "checks_passed": not errors,
        "check_errors": errors,
        "conclusion": conclusion,
        "records": records,
    }
    json_path = output_dir / "liveserve_audio_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 单卡 LiveServe 音频调度实验报告",
        "",
        f"- 模型：`{model}`",
        f"- 数据一致性检查：{'通过' if not errors else '失败'}",
        f"- 结论：**{conclusion}**",
        "",
        "| 策略 | 并发 | 请求吞吐 req/s | 音频吞吐 s/s | useful 吞吐 req/s | P90 TTFP ms | P90 RTF | 连续率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in STRATEGIES:
        for concurrency in CONCURRENCIES:
            cell = [
                record
                for record in fixed
                if record["experiment"]["strategy"] == strategy
                and record["experiment"]["concurrency"] == concurrency
            ]
            lines.append(
                f"| {strategy} | {concurrency} | {mean(cell, 'request_throughput'):.3f} | "
                f"{mean(cell, 'audio_throughput'):.3f} | {mean(cell, 'useful_request_throughput'):.3f} | "
                f"{mean(cell, 'p90_audio_ttfp_ms'):.1f} | {mean(cell, 'p90_audio_rtf'):.3f} | "
                f"{mean(cell, 'audio_continuity_ok_rate'):.1%} |"
            )
    if errors:
        lines += ["", "## 校验错误", "", *[f"- {error}" for error in errors]]
    markdown_path = output_dir / "liveserve_audio_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/liveserve_audio"))
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID or local model directory",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional Hugging Face commit, tag, or branch; do not use for a local model path",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--health-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--serve-bin", default="vllm")
    parser.add_argument("--bench-bin", default="vllm-omni")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--useful-audio-ttfp-ms", type=float, default=1000.0)
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument("--benchmark-extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.server_startup_timeout_s <= 0 or args.health_poll_interval_s <= 0:
        raise SystemExit("Server startup timeout and health poll interval must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "seed_tts_en_128_manifest.json"
    common_config_sha256 = sha256_file(DEPLOY_DIR / "qwen3_tts_910b4_single.yaml")
    records: list[dict[str, Any]] = []
    traces: dict[tuple[float, int], Path] = {}
    r_sat: float | None = None
    export_manifest = not manifest_path.exists()

    for strategy, config_name in STRATEGIES.items():
        config_path = DEPLOY_DIR / config_name
        server_log_path = args.output_dir / f"server_{strategy}.log"
        server_command = build_server_command(args, config_path)
        print("+", subprocess.list2cmdline(server_command), flush=True)
        if not args.dry_run:
            ensure_server_port_available(args)
        server_log = server_log_path.open("w", encoding="utf-8")
        server = None
        try:
            if not args.dry_run:
                server_env = os.environ.copy()
                server_env["VLLM_OMNI_ENABLE_PLAYBACK_FEEDBACK"] = "1"
                server = subprocess.Popen(server_command, env=server_env, stdout=server_log, stderr=subprocess.STDOUT)
            if server is not None:
                wait_for_server_health(args, server, server_log_path)
            for concurrency in CONCURRENCIES:
                for repeat in range(args.repeats):
                    result_path = args.output_dir / f"fixed_{strategy}_c{concurrency}_r{repeat}.json"
                    command = build_benchmark_command(
                        args,
                        result_path=result_path,
                        manifest_path=manifest_path,
                        concurrency=concurrency,
                        export_manifest=export_manifest,
                    )
                    before = fetch_scheduler_metrics(args) if not args.dry_run else {}
                    run_command(command, dry_run=args.dry_run)
                    if args.dry_run:
                        continue
                    export_manifest = False
                    manifest_sha256 = sha256_file(manifest_path)
                    after = fetch_scheduler_metrics(args)
                    records.append(
                        annotate_result(
                            result_path,
                            strategy=strategy,
                            mode="fixed",
                            repeat=repeat,
                            concurrency=concurrency,
                            manifest_sha256=manifest_sha256,
                            model=args.model,
                            model_revision=args.model_revision,
                            config_sha256=common_config_sha256,
                            scheduler_delta=counter_delta(before, after),
                        )
                    )
            if strategy == "legacy" and not args.dry_run:
                c8 = [
                    record
                    for record in records
                    if record["experiment"]["strategy"] == "legacy"
                    and record["experiment"]["concurrency"] == 8
                ]
                r_sat = mean(c8, "request_throughput")
                for factor in LOAD_FACTORS:
                    for repeat in range(args.repeats):
                        trace_path = args.output_dir / f"arrival_f{factor:.2f}_r{repeat}.json"
                        write_arrival_trace(
                            trace_path,
                            num_requests=args.num_requests,
                            request_rate=factor * r_sat,
                            seed=1000 + repeat,
                        )
                        traces[(factor, repeat)] = trace_path

            if not args.dry_run:
                assert r_sat is not None
                for factor in LOAD_FACTORS:
                    for repeat in range(args.repeats):
                        result_path = args.output_dir / f"poisson_{strategy}_f{factor:.2f}_r{repeat}.json"
                        before = fetch_scheduler_metrics(args)
                        run_command(
                            build_benchmark_command(
                                args,
                                result_path=result_path,
                                manifest_path=manifest_path,
                                arrival_trace=traces[(factor, repeat)],
                            ),
                            dry_run=False,
                        )
                        after = fetch_scheduler_metrics(args)
                        records.append(
                            annotate_result(
                                result_path,
                                strategy=strategy,
                                mode="poisson",
                                repeat=repeat,
                                load_factor=factor,
                                request_rate=factor * r_sat,
                                manifest_sha256=sha256_file(manifest_path),
                                model=args.model,
                                model_revision=args.model_revision,
                                config_sha256=common_config_sha256,
                                scheduler_delta=counter_delta(before, after),
                            )
                        )
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            server_log.close()

    if args.dry_run:
        print("Dry run complete; no server or benchmark process was started.")
        return
    errors = verify_results(records, sha256_file(manifest_path))
    report_path = write_report(args.output_dir, records, errors)
    if not args.skip_plots:
        from plot_results import plot_liveserve_report

        plot_liveserve_report(report_path, args.output_dir / "plots")
    if errors:
        raise SystemExit("Experiment completed but invariant checks failed; see report")


if __name__ == "__main__":
    main()
