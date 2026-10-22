# SPDX-License-Identifier: Apache-2.0
"""Explicit sequential collection; never enabled for timing or production."""

import json
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
        if getattr(wrapper, "_capture_request", None) != request["id"]:
            wrapper._capture_request, wrapper._capture_frame = request["id"], 0
        frame = wrapper._capture_frame
        if not (root / "predictor.pt").exists():
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
