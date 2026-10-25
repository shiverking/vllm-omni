# SPDX-License-Identifier: Apache-2.0
"""Independent same-codes keep16/keep10 eager versus graph check."""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "tools/qwen3_tts_code2wav_graph"
sys.path.insert(0, str(OLD))
from engine import compare, cpu_cache, eager_layout_class, load_decoder, run  # noqa: E402
from runtime import Progress, common_args, manifest, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("original_c2w_check", OLD / "experiment.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def activate(model, keep):
    if keep not in (10, 16):
        raise ValueError("Supported codebook counts: 10, 16")
    model.quantizer.active_codebooks = keep


@torch.inference_mode()
def check(args):
    progress = Progress(args.results)
    seq = legacy.sequences(args)
    model, _ = load_decoder(args)
    from vllm_omni.model_executor.models.qwen3_tts.npu_segmented_graph_wrapper import NPUGraphDecoderWrapper

    generator = torch.Generator().manual_seed(42)
    seq.append(
        dict(
            id="synthetic-boundaries",
            chunks=[
                torch.randint(model.config.codebook_size, (1, model.config.num_quantizers, n), generator=generator)
                for n in (1, 25, 25, 25, 25, 3, 25, 2)
            ],
        )
    )
    info = manifest(args)
    info.update(experiment="keep10_same_codes_graph", keep=[16, 10])
    write_json(args.results / "manifest.json", info)
    results = []
    for keep in (16, 10):
        (args.results / f"failure-keep{keep}.pt").unlink(missing_ok=True)
        wrapper = reference = None
        checks = []
        error = None
        context = {}
        try:
            # Set BEFORE either wrapper is created or warmed up.
            activate(model, keep)
            wrapper = NPUGraphDecoderWrapper(model)
            reference = eager_layout_class()(model)
            progress.log(f"keep{keep} 图捕获与同形状 eager 预热")
            wrapper.warmup(torch.device(args.device))
            reference.warmup(torch.device(args.device))
            total = sum(sum(1 for _ in legacy.batches(seq, b)) for b in (1, 2, 3, 4))
            done = 0
            for batch in (1, 2, 3, 4):
                caches, refs, native = {}, {}, {}
                audio = {}
                for start, position, active in legacy.batches(seq, batch):
                    keys = [(start, i) for i, _ in active]
                    codes = [r["chunks"][position].to(args.device).clone() for _, r in active]
                    if keep == 10:
                        for code in codes:
                            code[:, 10:] = 0
                    for mapping in (caches, refs, native):
                        for key in keys:
                            mapping.setdefault(key, dict(prefix_frames=0))
                    context = dict(keep=keep, batch=batch, position=position, requests=[r["id"] for _, r in active])
                    actual = run(model, wrapper, codes, [caches[k] for k in keys])
                    expected = run(model, reference, codes, [refs[k] for k in keys])
                    stock = run(model, None, codes, [native[k] for k in keys])
                    waves = [compare(a, b, 1e-5, 0.016) for a, b in zip(actual, expected)]
                    states = [legacy.compare_states(caches[k], refs[k], 1e-5, 0.016) for k in keys]
                    passed = all(x["passed"] for x in waves) and all(x["passed"] for s in states for x in s.values())
                    checks.append(
                        dict(
                            **context,
                            passed=passed,
                            waveform=waves,
                            state=states,
                            native=[compare(a, b, 1e-5, 0.016) for a, b in zip(actual, stock)],
                            boundary=[
                                dict(
                                    head=compare(
                                        a[..., : model.total_upsample], b[..., : model.total_upsample], 1e-5, 0.016
                                    ),
                                    tail=compare(
                                        a[..., -model.total_upsample :], b[..., -model.total_upsample :], 1e-5, 0.016
                                    ),
                                )
                                for a, b in zip(actual, expected)
                            ],
                        )
                    )
                    failure = args.results / f"failure-keep{keep}.pt"
                    if not passed and not failure.exists():
                        torch.save(
                            dict(
                                context=context,
                                codes=[c.cpu() for c in codes],
                                actual=[a.cpu() for a in actual],
                                expected=[b.cpu() for b in expected],
                                caches=[cpu_cache(caches[k]) for k in keys],
                            ),
                            failure,
                        )
                    for key, a, b in zip(keys, actual, expected):
                        audio.setdefault(key, ([], []))
                        audio[key][0].append(a.cpu().clone())
                        audio[key][1].append(b.cpu().clone())
                    done += 1
                    if done % 10 == 0 or done == total:
                        progress.update(done, total, f"keep{keep} 波形/状态")
                for key, (a, b) in audio.items():
                    checks.append(
                        dict(
                            keep=keep,
                            batch=batch,
                            key=key,
                            kind="concatenated",
                            **compare(torch.cat(a, -1), torch.cat(b, -1), 1e-5, 0.016),
                        )
                    )
        except Exception as exc:
            error = repr(exc)
            progress.log(f"keep{keep} 执行错误：{error}；记录后继续独立配置")
            write_json(args.results / f"error-keep{keep}.json", dict(error=error, context=context))
        results.append(
            dict(
                keep=keep,
                passed=error is None and all(r["passed"] for r in checks),
                error=error,
                checks=checks,
                graph=wrapper.statistics() if wrapper else None,
            )
        )
        write_json(args.results / "correctness.json", results)
        del wrapper, reference
    lines = [
        "# keep16/keep10：同 codes 波形检查",
        "",
        "只比较各自的 eager 与 graph；不要求 keep10 等同 keep16。性能命令独立。",
    ]
    for result in results:
        lines.append(
            f"keep{result['keep']}：通过={result['passed']}，检查项={len(result['checks'])}，错误={result['error']}"
        )
    (args.results / "report.md").write_text("\n\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    common_args(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if not args.device.startswith("npu:"):
        parser.error("NPU checks require npu device; use test_cpu.py locally")
    if args.results.resolve() == args.input.resolve() or args.results.resolve() in args.input.resolve().parents:
        parser.error("Results cannot overwrite input capture")
    args.results.mkdir(parents=True, exist_ok=True)
    check(args)


if __name__ == "__main__":
    main()
