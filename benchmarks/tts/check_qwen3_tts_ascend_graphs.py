# SPDX-License-Identifier: Apache-2.0
"""Compare Qwen3-TTS eager and graph execution on Ascend, independent of serving."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--component", choices=("codec", "predictor"), default="codec")
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--buckets", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--codes", type=Path, help="Optional long tensor [batch, 16, frames], saved with torch.save.")
    parser.add_argument("--failure", type=Path, default=Path("qwen3_tts_graph_failure.pt"))
    args = parser.parse_args()
    if not 1 <= args.batch <= max(args.buckets):
        parser.error("batch must fit in the largest graph bucket")

    import torch
    import torch_npu  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm_ascend.ascend_config import init_ascend_config

    from vllm_omni.engine.arg_utils import OmniEngineArgs
    from vllm_omni.model_executor.models.qwen3_tts.npu_segmented_graph_wrapper import NPUGraphDecoderWrapper
    from vllm_omni.model_executor.models.qwen3_tts.npu_stream_decode import NPUStreamDecoder

    torch.npu.set_device(0)
    config = OmniEngineArgs(
        model=args.model,
        model_stage="qwen3_tts",
        model_arch="Qwen3TTSTalkerForConditionalGeneration",
        trust_remote_code=True,
        dtype="bfloat16",
        max_num_seqs=max(args.buckets),
        max_num_batched_tokens=512,
        max_model_len=4096,
        enforce_eager=True,
    ).create_engine_config()
    config.additional_config = {
        "initial_codec_chunk_frames": 4,
        "codec_chunk_frames": 8,
        "codec_left_context_frames": 72,
        "code2wav_execution": "baseline",
        "decode_batch_max_size": max(args.buckets),
    }
    init_ascend_config(config)
    device = torch.device(config.device_config.device)
    with torch.inference_mode(), set_current_vllm_config(config):
        if args.component == "predictor":
            run_predictor(args, torch, config, device)
            return
        adapter, _ = NPUStreamDecoder.load(args.model, config)
        graphs = NPUGraphDecoderWrapper(adapter.decoder, args.buckets)
        graphs.warmup(device)
        run_check(args, torch, adapter, graphs, device)


def run_check(args, torch, adapter, graphs, device):
    if args.codes:
        codes = torch.load(args.codes, weights_only=True, map_location="cpu")
    else:
        codes = torch.randint(1, 2048, (args.batch, 16, 99), generator=torch.Generator().manual_seed(42))
    if codes.ndim != 3 or codes.shape[:2] != (args.batch, 16) or not codes.shape[-1]:
        raise ValueError("expected nonempty [batch, 16, frames] codes")
    eager_caches = [adapter.new_cache() for _ in range(args.batch)]
    graph_caches = [adapter.new_cache() for _ in range(args.batch)]
    codes = codes.to(device=device, dtype=torch.long)
    maximum = 0.0
    for index, start in enumerate([0, *range(4, codes.shape[-1], 8)]):
        width = 4 if start == 0 else 8
        order = list(range(args.batch))[:: -1 if index % 2 else 1]
        inputs = [codes[row : row + 1, :, start : start + width] for row in order]
        adapter.graphs = None
        eager = adapter.decode_chunks(inputs, [eager_caches[row] for row in order])
        adapter.graphs = graphs
        captured = adapter.decode_chunks(inputs, [graph_caches[row] for row in order])
        for row, expected, actual in zip(order, eager, captured, strict=True):
            try:
                if not torch.isfinite(actual).all():
                    raise AssertionError("non-finite graph PCM")
                torch.testing.assert_close(actual, expected)
                left_cache, right_cache = eager_caches[row], graph_caches[row]
                for key in ("suffix_quantized", "suffix_conv"):
                    if key in left_cache or key in right_cache:
                        torch.testing.assert_close(right_cache[key], left_cache[key])
            except (AssertionError, KeyError):
                args.failure.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "codes": codes[row].cpu(),
                        "start": start,
                        "eager": expected.cpu(),
                        "graph": actual.cpu(),
                        "eager_cache": eager_caches[row],
                        "graph_cache": graph_caches[row],
                    },
                    args.failure,
                )
                raise
            maximum = max(maximum, (actual.float() - expected.float()).abs().max().item())
    print(f"batch={args.batch} frames={codes.shape[-1]} max_abs_pcm={maximum:.8g}")


def run_predictor(args, torch, config, device):
    from vllm.model_executor.model_loader import DefaultModelLoader
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.model_executor.models.qwen3_tts.npu_code_predictor import CachedARExecutor
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm import (
        Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM,
    )

    talker = config.model_config.hf_config.talker_config
    with set_default_torch_dtype(torch.bfloat16), device:
        predictor = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
            vllm_config=config, config=talker.code_predictor_config, talker_config=talker
        )
    loader = DefaultModelLoader(config.load_config)
    source = DefaultModelLoader.Source(model_or_path=args.model, revision=config.model_config.revision)
    prefix = "talker.code_predictor."
    loaded = predictor.load_weights(
        (name.removeprefix(prefix), weight)
        for name, weight in loader._get_weights_iterator(source)
        if name.startswith(prefix)
    )
    missing = set(dict(predictor.named_parameters())) - loaded
    if missing:
        raise RuntimeError(f"Predictor checkpoint missing weights: {sorted(missing)}")
    predictor.eval()
    eager = CachedARExecutor(predictor, "eager", sorted(set(args.buckets)))
    graph = CachedARExecutor(predictor, "kv_frame", sorted(set(args.buckets)))
    random = torch.Generator().manual_seed(42)
    shape = (args.batch, 1, talker.hidden_size)
    inputs = (
        torch.randint(1, graph.vocab, (args.batch, 1), generator=random).to(device),
        torch.randn(shape, generator=random).to(device=device, dtype=torch.bfloat16),
        torch.randn(shape, generator=random).to(device=device, dtype=torch.bfloat16),
    )
    uniforms = torch.rand(args.batch, graph.groups - 1, graph.vocab, generator=random).to(device).clamp_min_(1e-20)
    for sampling in (False, True):
        expected = eager(*inputs, do_sample=sampling, sample_uniforms=uniforms)
        actual = graph(*inputs, do_sample=sampling, sample_uniforms=uniforms)
        reordered = graph(*(tensor.flip(0) for tensor in inputs), do_sample=sampling, sample_uniforms=uniforms.flip(0))
        try:
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(reordered.flip(0), actual)
        except AssertionError:
            args.failure.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "inputs": [tensor.cpu() for tensor in inputs],
                    "uniforms": uniforms.cpu(),
                    "eager": expected.cpu(),
                    "graph": actual.cpu(),
                    "graph_reordered": reordered.cpu(),
                    "do_sample": sampling,
                },
                args.failure,
            )
            raise
    print(f"batch={args.batch} predictor codes matched in greedy and shared-uniform sampling")


if __name__ == "__main__":
    main()
