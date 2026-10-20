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
        self.engine = self.config.get("engine", "current")
        if self.engine not in ("current", "pipeline_eager", "pipeline_graph"):
            raise ValueError("unknown research execution engine")
        if self.engine != "current" and self.mode != "tree":
            raise ValueError("pipeline research requires a frozen tree")
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
            if self.engine != "current":
                from .qwen3_code_predictor_pipeline import CANDIDATE_LAYOUT

                if frozen.get("candidate_layout") != CANDIDATE_LAYOUT:
                    raise ValueError("pipeline requires known-tail/next-three calibration")
                certificate_path = self.config.get("correctness")
                if not certificate_path:
                    raise ValueError("pipeline full requests require a sampled correctness certificate")
                certificate = json.loads(Path(certificate_path).read_text(encoding="utf-8"))
                binding = certificate.get("binding", {})
                names = [name for name, paths in frozen["configs"].items() if validate_tree(paths) == self.paths]
                if (
                    not certificate.get("passed")
                    or certificate.get("stage") != "complete"
                    or binding.get("tree_sha256") != signature
                    or binding.get("source") != frozen["source"]
                    or not any(f"{name}/{self.engine}" in certificate.get("engines", []) for name in names)
                ):
                    raise ValueError("pipeline correctness certificate does not match the frozen tree/engine")
                root = Path(__file__).resolve().parents[4]
                for relative, checksum in binding.get("code", {}).items():
                    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != checksum:
                        raise ValueError("pipeline code changed since correctness check")
                self.verified_sampling_modes = certificate["binding"]["sampling"]
                self.verified_sampling_parameters = certificate["binding"]["sampling_parameters"]
                self.verified_predictor = certificate["predictor"]
                self.pipeline_executors = {}
            elif frozen.get("candidate_layout"):
                raise ValueError("current engine cannot consume pipeline candidate layout")
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
            if self.engine != "current":
                from .qwen3_code_predictor_pipeline import PipelineExecutor

                sample = kwargs.get("do_sample", True)
                mode = "sampling" if sample else "greedy"
                if mode not in self.verified_sampling_modes:
                    raise ValueError("requested sampling mode was not checked")
                key = (sample, kwargs.get("temperature", 0.9), kwargs.get("top_k", 50))
                parameters = json.dumps(
                    dict(do_sample=sample, temperature=key[1], top_k=key[2], top_p=kwargs.get("top_p", 1.0)),
                    sort_keys=True,
                )
                shape = dict(
                    groups=predictor._num_groups,
                    hidden_size=predictor.config.hidden_size,
                    vocab_size=predictor.config.vocab_size,
                    dtype=str(next(predictor.parameters()).dtype),
                )
                if parameters not in self.verified_sampling_parameters or shape != self.verified_predictor:
                    raise ValueError("pipeline sampling parameters or predictor shape/dtype were not checked")
                if key not in self.pipeline_executors:
                    self.pipeline_executors[key] = PipelineExecutor(
                        predictor,
                        self.paths,
                        execution="graph" if self.engine == "pipeline_graph" else "eager",
                        do_sample=key[0],
                        temperature=key[1],
                        top_k=key[2],
                    )
                return self.pipeline_executors[key].decode(**inputs, **kwargs)[0]
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
