# SPDX-License-Identifier: Apache-2.0
"""Explicit sequential collection; never enabled for timing or production."""

import json
import re
from pathlib import Path

import torch


def capture_frame(wrapper, layer0_code, layer0_embed, last_talker_hidden, **kwargs):
    root = Path(wrapper.ar_capture_root)
    control = root / "control.json"
    wrapper._ar_capture_active = True
    try:
        inputs = dict(layer0_code=layer0_code, layer0_embed=layer0_embed, last_talker_hidden=last_talker_hidden)
        if not control.exists():
            return wrapper(**inputs, **kwargs)
        if layer0_code.shape[0] != 1:
            raise ValueError("Collection requires sequential single-request execution")
        request = json.loads(control.read_text())
        lightweight = request.get("format") == "codebook_statistics_v1"
        capture_key = request.get("attempt", request["id"]) if lightweight else request["id"]
        if getattr(wrapper, "_capture_request", None) != capture_key:
            wrapper._capture_request, wrapper._capture_frame = capture_key, 0
        frame = wrapper._capture_frame
        if not lightweight and not (root / "predictor.pt").exists():
            torch.save(
                dict(
                    state={k: v.detach().cpu() for k, v in wrapper.state_dict().items()},
                    config=wrapper.config.to_dict(),
                    talker_hidden_size=layer0_embed.shape[-1],
                    dtype=str(next(wrapper.parameters()).dtype),
                ),
                root / "predictor.pt",
            )
        codes = wrapper(**inputs, **kwargs)
        if lightweight:
            # Explicit diagnostic mode: synchronous small records, never timing.
            # This hook precedes Talker's invalid-c0 mapping and Codec filtering.
            ident = request["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
                raise ValueError("Invalid statistics request id")
            raw = codes.detach().cpu().reshape(-1).tolist()
            if len(raw) != 16:
                raise ValueError("Statistics capture requires 16 codebooks")
            valid = all(0 <= value < 2048 for value in raw) and any(raw)
            row = dict(
                request_id=ident,
                text_id=request["text_id"],
                seed=request["seed"],
                split=request["split"],
                attempt=request["attempt"],
                frame_id=frame,
                codes=raw,
                valid=valid,
                sampling={k: kwargs[k] for k in ("do_sample", "temperature", "top_k", "top_p")},
            )
            with (root / "codes" / f"{ident}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            wrapper._capture_frame += 1
            return codes
        torch.save(
            dict(
                request_id=request["id"],
                frame_id=frame,
                split=request["split"],
                inputs={k: v.detach().cpu() for k, v in inputs.items()},
                codes=codes.detach().cpu(),
                seed=42,
                sampling={k: kwargs[k] for k in ("do_sample", "temperature", "top_k", "top_p")},
            ),
            root / "frames" / f"{request['id']}-{frame:06d}.pt",
        )
        wrapper._capture_frame += 1
        return codes
    finally:
        wrapper._ar_capture_active = False
