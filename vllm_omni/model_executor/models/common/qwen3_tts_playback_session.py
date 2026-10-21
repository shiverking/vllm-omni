# SPDX-License-Identifier: Apache-2.0
"""Deployment-only bridge from stage grants to the adaptive RVQ executor."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .qwen3_code_predictor_adaptive import AdaptiveExecutor
from .qwen3_code_predictor_spec import frame_uniforms
from .qwen3_tts_deadline_policy import CompletionPolicy, digest


class PlaybackSession:
    def __init__(self, filename):
        self.config = json.loads(Path(filename).read_text(encoding="utf-8"))
        root = Path(self.config["results"])
        frozen = json.loads((root / "actions.json").read_text(encoding="utf-8"))
        signature = frozen.pop("sha256")
        if digest(frozen) != signature:
            raise ValueError("playback actions changed")
        self.actions = frozen["actions"]
        self.policy = CompletionPolicy(json.loads((root / "policy.json").read_text(encoding="utf-8")))
        certificate = json.loads((root / "correctness.json").read_text(encoding="utf-8"))
        if not certificate["passed"] or certificate["scope"] != "sampled" or certificate["actions"] != signature:
            raise ValueError("sampled adaptive correctness certificate required for live service")
        if (
            certificate["policy"] != self.policy.payload["sha256"]
            or self.policy.payload["binding"]["actions"] != signature
        ):
            raise ValueError("live policy differs from the certified policy")
        if self.config["service_sampling"] not in certificate["binding"]["sampling"]:
            raise ValueError("live sampling mode was not certified")
        self.buckets = certificate["binding"]["batch_buckets"]
        self.executors = {}
        self.grant = None
        self.request_ids = []
        self.frame_ids = {}
        self.completed = {}
        self.cp_ms = 0.0
        self.events = []
        self.weights_verified = False
        self.warm = False

    def begin(self, grant, request_map, retired=()):
        self.grant, self.request_map = grant, request_map
        self.grant_started = time.perf_counter()
        self.completed, self.cp_ms, self.events = {}, 0.0, []
        for rid in retired:
            self.frame_ids.pop(rid, None)

    def verify_weights(self, predictor):
        if self.weights_verified:
            return
        saved = torch.load(Path(self.config["input"]) / "predictor.pt", map_location="cpu", weights_only=False)
        weights = saved["state"]
        live = predictor.state_dict()
        for key, expected in weights.items():
            if key not in live or not torch.equal(live[key].detach().cpu(), expected.cpu()):
                raise ValueError(f"live predictor differs from certified weights: {key}")
        self.weights_verified = True

    @torch.inference_mode()
    def forward(self, predictor, original, inputs):
        # Engine warmup has no request/grant. Do not record it as real audio.
        if self.grant is None:
            return original(**inputs)
        self.verify_weights(predictor)
        batch = inputs["layer0_code"].shape[0]
        ids = self.request_ids[:batch]
        if len(ids) != batch or any(i not in self.request_map for i in ids):
            raise ValueError("live Code Predictor rows lack an exact request mapping")
        external = [self.request_map[i] for i in ids]
        mode = self.config["service_sampling"]
        values = dict(inputs, do_sample=mode == "sampling", temperature=0.9, top_k=50, top_p=1.0)
        # This seed is independent of dispatch order and candidate selection.
        if mode == "sampling":
            values["sample_uniforms"] = torch.cat(
                [
                    frame_uniforms(
                        self.grant["request_seeds"][rid],
                        rid,
                        self.frame_ids.get(rid, 0),
                        predictor._num_groups,
                        predictor.config.vocab_size,
                        inputs["layer0_code"].device,
                    )
                    for rid in external
                ]
            )
        values.pop("generators", None)
        values.pop("generator", None)
        if self.grant.get("phase") == "warmup" and not self.warm:
            # Capture every certified bucket before timing, even if warmup
            # traffic happens to select stock or produces only singleton rows.
            for b in self.buckets:
                key = (b, mode)
                part = {
                    k: v[:1].expand(b, *v.shape[1:]).contiguous() if torch.is_tensor(v) else v
                    for k, v in values.items()
                }
                engine = AdaptiveExecutor(
                    predictor, self.actions, batch_size=b, execution="graph", do_sample=mode == "sampling"
                )
                engine.prepare(part)
                self.executors[key] = engine
            self.warm = True
        strategy = self.grant["strategy"]
        before = time.perf_counter()
        if strategy == "stock":
            codes = original(**values)
            info = dict(backbone_calls=predictor._num_groups - 1, events=[])
        else:
            # Decompose a non-power-of-two live batch without dummy requests.
            # Stock retains its full native batch, including B=3/5/6/7.
            chunks, infos, offset = [], [], 0
            while offset < batch:
                b = max(size for size in self.buckets if size <= batch - offset)
                key = (b, mode)
                if key not in self.executors:
                    self.executors[key] = AdaptiveExecutor(
                        predictor, self.actions, batch_size=b, execution="graph", do_sample=mode == "sampling"
                    )
                engine = self.executors[key]
                part = {k: v[offset : offset + b] if torch.is_tensor(v) else v for k, v in values.items()}
                if not engine.prepared:
                    if self.grant.get("phase") != "warmup":
                        raise RuntimeError(f"uncaptured live bucket {key}; run the warmup phase first")
                    engine.prepare(part)

                def choose(p, actual_b):
                    if strategy == "fixed":
                        return "d3n16"
                    price = float(strategy.split(":")[1]) if strategy.startswith("lambda:") else 0.0
                    budget = self.grant["budget_ms"] - (time.perf_counter() - self.grant_started) * 1000
                    return self.policy.choose(
                        p,
                        1 if self.config.get("ablation") == "no_batch" else actual_b,
                        mode,
                        price=price,
                        deadline_ms=budget if strategy == "deadline" else None,
                    )

                result, detail = engine.decode(
                    **part, choose=choose, record_events=self.config.get("service_trace", False)
                )
                chunks.append(result)
                infos.append(detail)
                offset += b
            codes = torch.cat(chunks)
            info = dict(
                backbone_calls=sum(i["backbone_calls"] for i in infos), events=[e for i in infos for e in i["events"]]
            )
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - before) * 1000
        self.cp_ms += elapsed
        if self.grant.get("phase") == "check":
            expected = original(**values)
            if not torch.equal(expected, codes):
                torch.save(
                    dict(
                        inputs=values,
                        expected=expected,
                        actual=codes,
                        request_ids=external,
                        grant=self.grant,
                        states={str(k): e.snapshot() for k, e in self.executors.items()},
                    ),
                    Path(self.config["results"]) / "service_first_failure.pt",
                )
                raise AssertionError("live frame codes differ from stock; timing blocked")
        for rid in external:
            frame = self.frame_ids.get(rid, 0)
            self.frame_ids[rid] = frame + 1
            self.completed[rid] = self.completed.get(rid, 0) + 1
            if self.config.get("service_trace", False):
                self.events.append(
                    dict(
                        id=rid,
                        frame=frame,
                        chunk=frame // self.config["chunk_frames"],
                        strategy=strategy,
                        batch=batch,
                        wall_ms=elapsed,
                        **info,
                    )
                )
        return codes
