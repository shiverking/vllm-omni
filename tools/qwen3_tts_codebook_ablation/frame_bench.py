# SPDX-License-Identifier: Apache-2.0
"""Independent full16/keep10 kv_frame paired timing, using existing AR capture."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ar_tools", ROOT / "tools/qwen3_tts_ar_graph/experiment.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", default=ar.MODEL)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--executions-per-round", type=int, default=60)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    if not args.device.startswith("npu:"):
        parser.error("正式计时要求 NPU")
    if args.warmup < 0 or args.executions_per_round < 1 or any(b not in ar.BUCKETS for b in args.batches):
        parser.error("warmup>=0，executions-per-round>=1，batch 必须为1/2/4/8/16")
    args.results.mkdir(parents=True, exist_ok=True)
    root = ar.capture_root(args.input)
    frames = ar.select_frames(root)
    wrapper = ar.load_predictor(args, root)
    from vllm_omni.model_executor.models.common.qwen3_code_predictor_ar import CachedARExecutor

    executors = {}
    for keep in (16, 10):
        wrapper.keep_codebooks = keep
        executors[keep] = CachedARExecutor(wrapper, "kv_frame")
        executors[keep].prepare_all()
    api = ar.device_api(args.device)
    rows, differences = [], []
    report = [
        "# kv_frame：full16 vs keep10 帧级计时",
        "",
        "同一输入隐藏状态；共同 uniforms。keep10 9次主干，full16 15次；各1次整帧图。",
        "独立性能实验，不读取正确性证书。前10码差异单独记录，不代表服务生成音质通过。",
        "",
        "| 模式 | batch | full16 p50/p95 ms | keep10 p50/p95 ms | 加速比（均值） |",
        "|---|---:|---|---|---:|",
    ]
    for sampling in (False, True):
        for batch in args.batches:
            jobs = ar.batches(frames, batch, max(args.warmup, args.executions_per_round))

            def run(keep, job):
                inputs = ar.batch_inputs(job, args.device)
                uniforms = ar.uniforms_for(job, wrapper.config.vocab_size, args.device)
                return executors[keep](**inputs, do_sample=sampling, sample_uniforms=uniforms[:, : keep - 1])

            for keep in (16, 10):
                for job in jobs[: args.warmup]:
                    run(keep, job)
            for index, job in enumerate(jobs[: args.executions_per_round]):
                full, short = run(16, job), run(10, job)
                if not torch.equal(full[:, :10], short[:, :10]):
                    differences.append(dict(sampling=sampling, batch=batch, job=index))
                    if len(differences) == 1:
                        torch.save(
                            dict(inputs=ar.batch_inputs(job, "cpu"), full=full.cpu(), short=short.cpu()),
                            args.results / "first_difference.pt",
                        )
            totals = {16: [], 10: []}
            for round_id, keep in enumerate((16, 10, 10, 16)):
                for index, job in enumerate(jobs[: args.executions_per_round]):
                    _, wall, device = ar.measured(api, lambda: run(keep, job))
                    totals[keep].append(wall)
                    rows.append(
                        dict(
                            sampling=sampling,
                            batch=batch,
                            keep=keep,
                            round=round_id,
                            job=index,
                            wall_ms=wall,
                            device_ms=device,
                            backbone_calls=keep - 1,
                            graph_replays=1,
                        )
                    )
                print(f"sampling={sampling} batch={batch} ABBA={round_id + 1}/4 keep={keep} 完成", flush=True)
            a, b = totals[16], totals[10]
            report.append(
                f"| {'sampling' if sampling else 'greedy'} | {batch} | "
                f"{np.percentile(a, 50):.3f}/{np.percentile(a, 95):.3f} | "
                f"{np.percentile(b, 50):.3f}/{np.percentile(b, 95):.3f} | {np.mean(a) / np.mean(b):.3f}× |"
            )
            (args.results / "timings.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            ar.write_json(args.results / "prefix_differences.json", differences)
            (args.results / "report.md").write_text("\n".join(report), encoding="utf-8")
    ar.write_json(
        args.results / "manifest.json",
        dict(
            model=args.model,
            input=str(root),
            device=args.device,
            keep=[16, 10],
            frames=len(frames),
            code_revision=ar.git_revision(ROOT),
            note="Wall timing includes common uniform preparation and input copies",
        ),
    )
    print(f"结果：{args.results / 'report.md'}；前10码差异记录 {len(differences)}", flush=True)


if __name__ == "__main__":
    main()
