# SPDX-License-Identifier: Apache-2.0
"""Collect bounded real SnakeBeta activations from recorded Code2Wav chunks."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


class Recorder:
    def __init__(self, directory, max_samples, max_bytes):
        self.directory = directory
        self.max_samples, self.max_bytes = max_samples, max_bytes
        self.rows, self.seen, self.counts = [], set(), {}
        self.bytes = 0
        self.context = {}

    def hook(self, name):
        def record(module, inputs):
            x = inputs[0]
            key = str(tuple(x.shape))
            self.counts[key] = self.counts.get(key, 0) + 1
            if key in self.seen:
                return
            self.seen.add(key)
            if not module._cached:
                module.precompute_exp_cache()
            tensors = dict(x=x, alpha=module._exp_alpha, inv_beta=module._inv_beta)
            size = sum(t.numel() * t.element_size() for t in tensors.values())
            if len(self.rows) >= self.max_samples or self.bytes + size > self.max_bytes:
                return
            file = f"sample{len(self.rows):03d}.pt"
            torch.save({k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}, self.directory / file)
            self.rows.append(
                dict(
                    file=file,
                    module=name,
                    shape=list(x.shape),
                    original_stride=list(x.stride()),
                    bytes=size,
                    context=dict(self.context),
                )
            )
            self.bytes += size
            self.flush()
            print(f"Captured {file}: {name} {list(x.shape)}, total={self.bytes / 2**20:.1f} MiB", flush=True)

        return record

    def flush(self):
        (self.directory / "index.json").write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        (self.directory / "coverage.json").write_text(
            json.dumps(
                dict(
                    calls_by_shape=self.counts,
                    observed_shapes=len(self.seen),
                    saved_shapes=len(self.rows),
                    saved_bytes=self.bytes,
                    note="One real sample per shape; bounded diagnostic capture, not timing.",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--request-limit", type=int, default=3)
    parser.add_argument("--batches", nargs="+", type=int, choices=(1, 2, 3, 4), default=[1, 4])
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--max-mib", type=int, default=512)
    args = parser.parse_args()
    if min(args.request_limit, args.max_samples, args.max_mib) < 1:
        parser.error("capture limits must be positive")
    if args.results.resolve() == args.input.resolve():
        parser.error("results must differ from the input capture directory")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools/qwen3_tts_code2wav_graph"))
    from engine import eager_layout_class, load_decoder, run

    from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta
    from vllm_omni.model_executor.models.qwen3_tts.code2wav_incremental_kv import IncrementalKVExecutor

    model, _ = load_decoder(args)
    current = eager_layout_class()(model)
    current.warmup(torch.device(args.device))
    runner = IncrementalKVExecutor(model, "growth_kv", "eager", current=current)
    runner.warmup(torch.device(args.device))
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "manifest.json").write_text(
        json.dumps(
            dict(
                args=vars(args),
                torch=torch.__version__,
                transformer="growth_kv",
                snake="eager",
                dtype=str(next(model.parameters()).dtype),
            ),
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )
    recorder = Recorder(args.results, args.max_samples, args.max_mib * 2**20)
    hooks = [
        m.register_forward_pre_hook(recorder.hook(name))
        for name, m in model.named_modules()
        if isinstance(m, SnakeBeta)
    ]
    requests = [r for r in json.loads((args.input / "requests.json").read_text()) if r.get("complete")]
    try:
        for batch in args.batches:
            for request in requests[: args.request_limit]:
                # Duplicate a real request across rows, explicitly for shape coverage.
                caches = [dict(prefix_frames=0) for _ in range(batch)]
                try:
                    for chunk, file in enumerate(request["chunks"]):
                        artifact = torch.load(args.input / file, map_location="cpu", weights_only=True)
                        for item in artifact["rows"]:
                            if item["request_id"] != request["internal_request_id"]:
                                raise ValueError("Capture request association mismatch")
                            codes = item["codes"].unsqueeze(0).to(args.device)
                            recorder.context = dict(
                                request=request["id"], chunk=chunk, batch=batch, duplicate_rows=True
                            )
                            run(model, runner, [codes] * batch, caches)
                    print(f"Completed request={request['id']} batch={batch}", flush=True)
                finally:
                    runner.release_requests(caches)
    finally:
        for hook in hooks:
            hook.remove()
        recorder.flush()


if __name__ == "__main__":
    main()
