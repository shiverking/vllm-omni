# SPDX-License-Identifier: Apache-2.0
"""Decoder loading and eager execution of exactly the graph's padded layout."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from runtime import ROOT


def load_decoder(args):
    sys.path.insert(0, str(ROOT))
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        from vllm_omni.platforms.npu.models.qwen3_tts_tokenizer_v2 import apply_qwen3_tts_tokenizer_v2_patch

        apply_qwen3_tts_tokenizer_v2_patch()
    from safetensors import safe_open

    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Config,
    )
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
    )

    config = Qwen3TTSTokenizerV2Config.from_pretrained(args.model, subfolder="speech_tokenizer")
    model = Qwen3TTSTokenizerV2Decoder._from_config(config.decoder_config).eval()
    weights = {}
    for path in sorted((Path(args.model) / "speech_tokenizer").glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key.startswith("decoder."):
                    weights[key.removeprefix("decoder.")] = stream.get_tensor(key)
    if not weights:
        raise ValueError("speech_tokenizer/*.safetensors 中未找到 decoder 权重")
    model.load_state_dict(weights, strict=True)
    model.to(device=args.device, dtype=torch.bfloat16 if args.device.startswith("npu") else torch.float32)
    model.precompute_snake_caches()
    model._initial_codec_chunk_frames = 1
    model._incremental_chunk_frames = 25
    model._incremental_chunk_ramp = []
    return model, config.output_sample_rate


def eager_layout_class():
    from vllm_omni.model_executor.models.qwen3_tts.npu_segmented_graph_wrapper import NPUGraphDecoderWrapper

    class EagerLayout(NPUGraphDecoderWrapper):
        """Diagnostic surrogate, never reported as graph performance."""

        def _is_capturing(self):
            return False

        def warmup(self, device, dtype=torch.long):
            self._device = device
            if self.pooled:
                for batch in self.capture_batch_sizes:
                    self._capture_pool(batch, None, device, dtype)
                    for target in self._xvec_previous_frames_by_target:
                        self._capture_pool(batch, target, device, dtype)
                self._warmed_up = True
                return
            model_dtype = next(self.decoder.parameters()).dtype
            for batch in self.capture_batch_sizes:
                state = {
                    "input": {
                        "codes": torch.zeros(
                            batch, self.num_quantizers, self.initial_chunk_frames, dtype=dtype, device=device
                        )
                    }
                }

                def first(state=state):
                    caches = {}
                    from vllm_omni.model_executor.models.qwen3_tts.code2wav_optimization import fixed_stage

                    with fixed_stage(self, state["input"]["codes"].shape[0], self.initial_chunk_frames):
                        state["output"] = self.decoder._decode_xvec_first_chunk(state["input"]["codes"], caches)
                    state["cache"] = caches

                state["graph"] = SimpleNamespace(replay=first)
                first()
                self.xvec_prefix_states[batch] = state
                for target, previous in self._xvec_previous_frames_by_target.items():
                    cache = self._make_dummy_xvec_cache(batch, device, model_dtype)
                    state = dict(
                        cache=cache,
                        input=dict(
                            codes=torch.zeros(
                                batch, self.num_quantizers, target - previous, dtype=dtype, device=device
                            ),
                            quantized=torch.zeros(
                                batch,
                                self.decoder.config.codebook_dim,
                                min(previous, self.prefix_length),
                                dtype=model_dtype,
                                device=device,
                            ),
                            conv=torch.zeros(
                                batch,
                                self._cached_conv_frames(previous),
                                self.decoder.config.latent_dim,
                                dtype=model_dtype,
                                device=device,
                            ),
                            mask=torch.ones(batch, target, dtype=torch.bool, device=device),
                        ),
                    )

                    def suffix(state=state, target=target):
                        x = state["input"]
                        from vllm_omni.model_executor.models.qwen3_tts.code2wav_optimization import fixed_stage

                        with fixed_stage(self, x["codes"].shape[0], target):
                            state["output"] = self._run_combined_suffix(
                                "xvec", x["codes"], x["quantized"], x["conv"], state["cache"], target, x["mask"]
                            )

                    state["graph"] = SimpleNamespace(replay=suffix)
                    suffix()
                    self.combined_states[("xvec", batch, target)] = state
            self._warmed_up = True

    return EagerLayout


def run(model, wrapper, codes, caches):
    if wrapper is not None and wrapper.optimization != "current":
        return [x[0] for x in wrapper._batched_request_decode(codes, caches)]
    lengths = [int(x.shape[-1]) for x in codes]
    batch = codes[0].new_zeros((len(codes), codes[0].shape[-2], max(lengths)))
    for i, x in enumerate(codes):
        batch[i, :, : lengths[i]].copy_(x[0] if x.ndim == 3 else x)
    if wrapper is None:
        return model.batched_chunked_decode(batch, lengths, caches=caches, max_batch_size=4)
    return wrapper.batched_chunked_decode_with_cudagraph(batch, lengths, caches=caches, max_batch_size=4)


def snapshot_cache(cache):
    """Only semantic state, never ping-pong aliases from another executor."""
    return {
        k: v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        for k, v in cache.items()
        if not k.startswith(("_suffix_", "_c2w_"))
    }


def cpu_cache(cache):
    result = {}
    for key in (
        "prefix_frames",
        "decoder_prefix_frames",
        "suffix_frames",
        "suffix_quantized",
        "suffix_conv",
        "ref_hidden",
        "ref_conv",
        "prefix_hidden",
    ):
        if key in cache:
            value = cache[key]
            result[key] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
    return result


def compare(left, right, atol, rtol):
    if left.shape != right.shape:
        return dict(passed=False, exact=False, shape_left=list(left.shape), shape_right=list(right.shape))
    a, b = left.float(), right.float()
    error = (a - b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return dict(
        passed=finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        finite=finite,
        exact=bool(torch.equal(left, right)),
        max_abs=float(error.max()) if error.numel() else 0.0,
        rmse=float(error.square().mean().sqrt()) if error.numel() else 0.0,
        outside=int((error > atol + rtol * b.abs()).sum()),
    )
