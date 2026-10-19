# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Internal research switch, intentionally absent from the Speech API."""

import hashlib
import json
import os
from pathlib import Path

import torch

from .qwen3_code_predictor_spec import decode, frame_uniforms, validate_tree


class ResearchSession:
    def __init__(self, filename):
        self.config = json.loads(Path(filename).read_text(encoding="utf-8"))
        self.mode = self.config["mode"]
        if self.mode not in ("collect", "tree", "reference"):
            raise ValueError("research mode must be collect, tree or reference")
        self.paths = validate_tree(self.config.get("paths", []))
        if self.mode == "tree" and not self.paths:
            raise ValueError("tree mode requires a frozen rank tree")
        if self.mode == "tree":
            frozen = json.loads((Path(filename).parent / "trees.json").read_text(encoding="utf-8"))
            signature = frozen.pop("sha256")
            actual = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
            if actual != signature or self.config.get("frozen_sha256") != signature:
                raise ValueError("frozen tree signature changed")
            if self.paths not in [validate_tree(p) for p in frozen["configs"].values()]:
                raise ValueError("live tree is not one of the calibrated frozen trees")
        self.frame = 0
        self.request = None
        self.exported = False

    @torch.inference_mode()
    def forward(self, predictor, original, layer0_code, layer0_embed, last_talker_hidden, kwargs):
        if layer0_code.shape[0] != 1:
            raise ValueError("Qwen3-TTS research requires max_num_seqs=1, including warmup")
        inputs = dict(layer0_code=layer0_code, layer0_embed=layer0_embed, last_talker_hidden=last_talker_hidden)
        if self.mode != "collect":
            # The runner supplies per-request, per-step uniforms. If it supplies
            # generators instead, draw the whole frame before any candidate work.
            if (
                kwargs.get("sample_uniforms") is None
                and kwargs.get("do_sample", True)
                and kwargs.get("temperature", 0.9) > 0
            ):
                generator = kwargs.get("generator")
                if kwargs.get("generators") is not None:
                    generator = kwargs["generators"][0]
                kwargs["sample_uniforms"] = torch.rand(
                    (1, predictor._num_groups - 1, predictor.config.vocab_size),
                    device=layer0_code.device,
                    generator=generator,
                )
            return decode(predictor, **inputs, **kwargs, paths=self.paths)[0]
        # Collection is explicitly not a performance run. Control is read once
        # per frame; tensors are saved once per complete frame, never per token.
        root = Path(self.config["output"])
        control = root / "control.json"
        if not control.exists():
            return original(**inputs, **kwargs)
        request = json.loads(control.read_text(encoding="utf-8"))
        if request["id"] != self.request:
            self.request, self.frame = request["id"], 0
        kwargs["sample_uniforms"] = frame_uniforms(
            self.config["seed"],
            self.request,
            self.frame,
            predictor._num_groups,
            predictor.config.vocab_size,
            layer0_code.device,
        )
        if not self.exported:
            torch.save(
                dict(
                    state={k: v.detach().cpu() for k, v in predictor.state_dict().items()},
                    config=predictor.config.to_dict(),
                    talker_hidden_size=layer0_embed.shape[-1],
                    dtype=str(next(predictor.model.parameters()).dtype),
                ),
                root / "predictor.pt",
            )
            self.exported = True
            metadata = dict(
                dtype=str(next(predictor.model.parameters()).dtype),
                sampling={
                    k: kwargs.get(k, default)
                    for k, default in (("do_sample", True), ("temperature", 0.9), ("top_k", 50), ("top_p", 1.0))
                },
                predictor_config=predictor.config.to_dict(),
                torch_version=str(torch.__version__),
            )
            (root / "capture_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        codes = original(**inputs, **kwargs)
        shard = dict(
            request_id=self.request,
            frame_id=self.frame,
            split=request["split"],
            inputs={k: v.detach().cpu() for k, v in inputs.items()},
            codes=codes.detach().cpu(),
            sampling={
                k: kwargs.get(k, default)
                for k, default in (("do_sample", True), ("temperature", 0.9), ("top_k", 50), ("top_p", 1.0))
            },
            seed=self.config["seed"],
        )
        path = root / "frames" / f"{self.request}-{self.frame:06d}-{os.getpid()}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(shard, path)
        self.frame += 1
        return codes
