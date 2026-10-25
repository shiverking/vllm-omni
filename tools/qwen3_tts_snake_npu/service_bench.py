# SPDX-License-Identifier: Apache-2.0
"""Service A/B: growth_kv graph + eager versus conservative SnakeBeta."""

import argparse
from pathlib import Path

from model_experiment import exp


def configure_deployment(config, mode):
    if mode not in ("baseline", "snake_conservative"):
        raise ValueError(f"Unknown SnakeBeta service variant: {mode}")
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]
    extra.update(
        code2wav_transformer_execution="growth_kv",
        code2wav_incremental_execution="graph",
        code2wav_graph_optimization="current",
        code2wav_snake_execution="eager" if mode == "baseline" else "triton_ascend",
        code2wav_snake_layout="conservative",
        code2wav_snake_block=2048,
        code2wav_snake_programs=32,
    )
    return config


def run(args):
    import runtime
    import yaml

    service = exp.load_script("snake_service", "service_bench.py")
    original_deployment, original_summary = runtime.deployment, service.summarize

    def deployment(args, mode, directory, capture=None):
        # Both sides start from the identical inner-graph deployment.
        path = original_deployment(args, "current", directory, capture)
        config = configure_deployment(yaml.safe_load(path.read_text(encoding="utf-8")), mode)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def summarize(root, entries):
        original_summary(root, entries)
        path = root / "report.md"
        text = path.read_text(encoding="utf-8")
        text = text.replace("# kv_frame 基线：Code2Wav 服务对比", "# SnakeBeta 算子接入：服务性能对照")
        text = text.replace(
            "baseline 保留原外层图，实验组由 Code2Wav 管理内层图。",
            "两组均为 growth_kv graph + kv_frame，仅切换 SnakeBeta：baseline 为原算子链，"
            "snake_conservative 为实测形状白名单。未测形状保留原算子。",
        )
        path.write_text(text, encoding="utf-8")

    runtime.deployment, service.summarize = deployment, summarize
    try:
        service.run(args)
    finally:
        runtime.deployment = original_deployment
        path = args.results / "manifest.json"
        if path.exists():
            data = exp.read_json(path)
            data["baseline"] = "growth_kv graph + kv_frame + eager SnakeBeta"
            data["candidate"] = "growth_kv graph + kv_frame + conservative SnakeBeta"
            for source in [Path(__file__), exp.ROOT / "vllm_omni/model_executor/models/common/snake_activation_npu.py"]:
                data["code"][str(source.relative_to(exp.ROOT))] = exp.hashlib.sha256(source.read_bytes()).hexdigest()
            exp.write_json(path, data)


def main():
    parser = argparse.ArgumentParser(__doc__)
    exp.common_args(parser)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 16])
    args = parser.parse_args()
    args.variants = ["baseline", "snake_conservative"]
    if args.requests < 1 or any(n < 1 or n > 16 for n in args.concurrencies):
        parser.error("请求数必须为正，并发范围1..16")
    run(args)


if __name__ == "__main__":
    main()
