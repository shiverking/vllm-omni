# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU tests: real tiny decoder, eager graph-layout surrogate.

python tools/qwen3_tts_code2wav_graph/test_cpu.py
No vLLM installation or NPU is needed. This does not validate NPU capture.
"""

import importlib.util
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))


def bootstrap():
    # Avoid importing the serving stack on Windows CPU; only logger/platform
    # imports are substituted. Decoder, cache and routing code are real.
    if importlib.util.find_spec("vllm") is None:
        mod = ModuleType("vllm")
        mod.__path__ = []
        sys.modules["vllm"] = mod
        log = ModuleType("vllm.logger")
        log.init_logger = logging.getLogger
        sys.modules["vllm.logger"] = log
        platform = ModuleType("vllm.platforms")
        platform.current_platform = SimpleNamespace()
        sys.modules["vllm.platforms"] = platform
    for name in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        "vllm_omni.model_executor.models.common",
        "vllm_omni.model_executor.models.qwen3_tts",
        "vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz",
    ):
        module = ModuleType(name)
        module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        sys.modules.setdefault(name, module)


bootstrap()
from engine import compare, eager_layout_class, run, snapshot_cache  # noqa: E402

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2Decoder,
)


def decoder():
    torch.manual_seed(42)
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=16,
        upsample_rates=(2,),
        upsampling_ratios=(2,),
        sliding_window=72,
    )
    model = Qwen3TTSTokenizerV2Decoder._from_config(config).eval()
    model._initial_codec_chunk_frames = 1
    model._incremental_chunk_frames = 25
    return model


class StateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = decoder()
        cls.wrapper = eager_layout_class()(cls.model)
        with torch.no_grad():
            cls.wrapper.warmup(torch.device("cpu"))

    def assert_tensor(self, a, b):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)

    @torch.no_grad()
    def sequence(self, batch, sizes):
        caches = [dict(prefix_frames=0) for _ in range(batch)]
        reference = [dict(prefix_frames=0) for _ in range(batch)]
        saved = []
        for n in sizes:
            codes = [torch.randint(0, 32, (1, 2, n)) for _ in range(batch)]
            got = run(self.model, self.wrapper, codes, caches)
            # Serial eager reference avoids the existing padded-short eager
            # batching behavior; every actual short chunk updates its state.
            expected = [self.model.chunked_decode(x, c)[0] for x, c in zip(codes, reference)]
            for x, y, c, r in zip(got, expected, caches, reference):
                self.assert_tensor(x, y)
                self.assertEqual(c["suffix_frames"], r["suffix_frames"])
                self.assert_tensor(c["suffix_quantized"], r["suffix_quantized"])
                self.assert_tensor(c["suffix_conv"], r["suffix_conv"])
            if saved:
                self.assert_tensor(saved[0], saved[1])
            saved = [got[0], got[0].clone()]
        return caches

    @torch.no_grad()
    def test_capture_mask_without_host_probe(self):
        from unittest.mock import patch

        import transformers.masking_utils as masks

        from vllm_omni.model_executor.models.qwen3_tts.npu_segmented_graph_wrapper import capture_mask_context

        original = masks._ignore_causal_mask_sdpa
        for length in (26, 51, 76, 97):
            for padded in (False, True):
                padding = torch.ones(2, length, dtype=torch.bool)
                if padded:
                    padding[1, -3:] = False
                kwargs = dict(
                    batch_size=2,
                    q_length=length,
                    kv_length=length,
                    attention_mask=padding,
                    local_size=72,
                    mask_function=masks.sliding_window_causal_mask_function(72),
                )
                expected = masks.sdpa_mask(**kwargs, allow_is_causal_skip=False)
                with capture_mask_context(), patch.object(masks, "fast_all", side_effect=AssertionError("host probe")):
                    actual = masks.sdpa_mask(**kwargs)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertIs(masks._ignore_causal_mask_sdpa, original)

    @torch.no_grad()
    def test_constants_match_current_and_invalidate(self):
        from vllm_omni.model_executor.models.qwen3_tts.code2wav_optimization import FrozenCodebook

        model = decoder()
        wrapper = eager_layout_class()(model, optimization="constants")
        wrapper.warmup(torch.device("cpu"))
        for batch in (1, 3, 4):
            left = [dict(prefix_frames=0) for _ in range(batch)]
            right = [dict(prefix_frames=0) for _ in range(batch)]
            for length in (1, 25, 25, 25, 25, 3, 25):
                codes = [torch.randint(0, 32, (1, 2, length)) for _ in range(batch)]
                a = run(model, wrapper, codes, left)
                b = run(self.model, self.wrapper, codes, right)
                for x, y in zip(a, b):
                    self.assert_tensor(x, y)
        book = next(m for m in model.modules() if isinstance(m, FrozenCodebook))
        expected = book.embedding_sum / book.cluster_usage.clamp(min=book.epsilon)[:, None]
        torch.testing.assert_close(book._normalized_embedding, expected, atol=0, rtol=0)
        book.load_state_dict(book.state_dict())
        self.assertIsNone(book._normalized_embedding)
        book.decode(torch.zeros(1, 1, dtype=torch.long))
        book.to(dtype=torch.float64)
        self.assertIsNone(book._normalized_embedding)
        self.assertGreater(wrapper.statistics()["derived_buffer_bytes"], 0)

    @torch.no_grad()
    def test_pooled_state_lifecycle(self):
        model = decoder()
        wrapper = eager_layout_class()(model, optimization="pooled")
        wrapper.warmup(torch.device("cpu"))
        self.assertEqual(wrapper.statistics()["graph_count"], 15)
        for batch in (1, 2, 3, 4, 5):
            caches = [dict(prefix_frames=0) for _ in range(batch)]
            refs = [dict(prefix_frames=0) for _ in range(batch)]
            saved = None
            for length in (1, 25, 25, 25, 25, 3, 25, 2):
                codes = [torch.randint(0, 32, (1, 2, length)) for _ in caches]
                actual = run(model, wrapper, codes, caches)
                expected = run(self.model, self.wrapper, codes, refs)
                for a, b, c, r in zip(actual, expected, caches, refs):
                    self.assert_tensor(a, b)
                    self.assert_tensor(c["suffix_quantized"], r["suffix_quantized"])
                    self.assert_tensor(c["suffix_conv"], r["suffix_conv"])
                if saved is not None:
                    self.assert_tensor(*saved)
                saved = actual[0], actual[0].clone()
                caches.reverse()
                refs.reverse()
            wrapper.release_requests(caches)
            self.assertEqual(len(wrapper.state_pool.free), 16)
            self.assertTrue(all(list(c) == ["prefix_frames"] for c in caches))
        # Slot exhaustion is explicit and release permits a new request.
        caches = [dict(prefix_frames=0) for _ in range(16)]
        wrapper.import_histories(caches, torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            wrapper.import_histories([dict(prefix_frames=0)], torch.device("cpu"))
        wrapper.release_requests(caches[:1])
        wrapper.import_histories([dict(prefix_frames=0)], torch.device("cpu"))
        wrapper.reset_pool()
        self.assertEqual(len(wrapper.state_pool.free), 16)

    def test_graph_shapes(self):
        self.assertEqual(self.wrapper.statistics()["graph_count"], 15)
        self.assertEqual(self.wrapper._xvec_previous_frames_by_target, {26: 1, 51: 26, 76: 51, 97: 72})

    def test_growth_rolling_and_padding(self):
        for batch in (1, 2, 3, 4, 5):
            with self.subTest(batch=batch):
                self.sequence(batch, [1, 25, 25, 25, 25, 25])

    def test_short_then_continue_and_end(self):
        for batch in (1, 3):
            for sizes in ([1, 7, 25, 25, 25, 25, 2], [1, 25, 25, 25, 3, 25], [2, 25, 25, 25]):
                with self.subTest(batch=batch, sizes=sizes):
                    self.sequence(batch, sizes)

    @torch.no_grad()
    def test_reorder_replace_and_isolation(self):
        caches = [dict(prefix_frames=0) for _ in range(4)]
        codes = [torch.randint(0, 32, (1, 2, 1)) for _ in caches]
        before = run(self.model, self.wrapper, codes, caches)
        independent = run(self.model, self.wrapper, codes, [dict(prefix_frames=0) for _ in caches])
        for a, b in zip(before, independent):
            self.assert_tensor(a, b)
        order = [2, 0, 3]
        selected = [caches[i] for i in order]
        copied = [snapshot_cache(c) for c in selected]
        chunk = [torch.randint(0, 32, (1, 2, 25)) for _ in selected]
        expected = [self.model.chunked_decode(x, c)[0] for x, c in zip(chunk, copied)]
        actual = run(self.model, self.wrapper, chunk, selected)
        for a, b in zip(actual, expected):
            self.assert_tensor(a, b)
        # Replacement request and a rolling/growing request in the same batch.
        fresh = dict(prefix_frames=0)
        mixed = run(self.model, self.wrapper, [codes[0], chunk[0]], [fresh, selected[0]])
        self.assert_tensor(mixed[0], before[0])
        self.assertEqual(fresh["suffix_frames"], 1)

    def test_compare_nonfinite_and_shape(self):
        self.assertFalse(compare(torch.tensor([float("nan")]), torch.zeros(1), 1e-5, 1e-4)["passed"])
        self.assertFalse(compare(torch.zeros(2), torch.zeros(1), 1e-5, 1e-4)["passed"])

    @torch.no_grad()
    def test_capture_roundtrip(self):
        from vllm_omni.model_executor.models.qwen3_tts.code2wav_capture import record_decode

        with tempfile.TemporaryDirectory() as tmp:
            codes = torch.ones(2, 1, dtype=torch.long)
            record_decode(tmp, [("internal", codes)], [torch.ones(1, 4)], [True], ["external"], 24000)
            data = torch.load(next(Path(tmp).glob("*.pt")), weights_only=True)
            self.assertEqual(data["rows"][0]["external_id"], "external")
            self.assertTrue(data["rows"][0]["finished"])
            self.assert_tensor(data["rows"][0]["codes"], codes)

    @torch.no_grad()
    def test_tools_check_bench_report_and_resume(self):
        import contextlib
        import io
        import wave
        from unittest.mock import patch

        import experiment as exp
        from runtime import read_json

        from vllm_omni.model_executor.models.qwen3_tts.code2wav_capture import record_decode

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "capture"
            capture.mkdir()
            args = SimpleNamespace(results=capture, limit=3, model="unused", device="cpu", probe_only=False)
            progress = exp.Progress(capture)
            count = [0]

            @contextlib.contextmanager
            def fake_server(*args, **kwargs):
                yield "http://127.0.0.1"

            def fake_speech(*args, **kwargs):
                count[0] += 1
                for n in (1, 25, 25, 25, 2):
                    codes = torch.randint(32, (2, n))
                    record_decode(
                        capture / "chunks",
                        [(f"req{count[0]}", codes)],
                        [torch.zeros(1, n * 4)],
                        [n == 2],
                        [f"external{count[0]}"],
                        24000,
                    )
                stream = io.BytesIO()
                with wave.open(stream, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(24000)
                    w.writeframes(bytes(200))
                return stream.getvalue()

            with patch.object(exp, "server", fake_server), patch.object(exp, "speech", fake_speech):
                exp.collect(args, progress)
                exp.collect(args, progress)
            self.assertEqual(count[0], 3)
            self.assertEqual(len(exp.sequences(SimpleNamespace(input=capture, probe_only=False))), 3)
            args.input = capture
            args.results = root / "check"
            args.results.mkdir()
            ref = eager_layout_class()(self.model)
            ref.warmup(torch.device("cpu"))
            with patch.object(exp, "setup", return_value=(self.model, 24000, self.wrapper, ref)):
                exp.check(args, exp.Progress(args.results))
            self.assertTrue(read_json(args.results / "correctness.json")["passed"])
            args.results = root / "bench"
            args.results.mkdir()
            args.batches = [1, 3]
            args.executions_per_round = 4
            args.warmup = 1
            args.profile = False

            def fake_measure(model, wrapper, codes, caches):
                return (2.0 if wrapper is None else 1.0), 0.5, run(model, wrapper, codes, caches)

            fake_npu = SimpleNamespace(reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 123)
            with (
                patch.object(exp, "setup", return_value=(self.model, 24000, self.wrapper, None)),
                patch.object(exp, "measured", fake_measure),
                patch.object(torch, "npu", fake_npu, create=True),
            ):
                exp.bench(args, exp.Progress(args.results))
            result = read_json(args.results / "summary.json")
            self.assertTrue(result["comparisons"])
            self.assertTrue(all(x["speedup"] == 2.0 for x in result["comparisons"]))
            # These mock times live only in TemporaryDirectory and are never
            # emitted as real performance evidence.

    def test_capture_failure_is_explicit(self):
        from unittest.mock import patch

        from vllm_omni.model_executor.models.qwen3_tts.npu_segmented_graph_wrapper import NPUGraphDecoderWrapper

        wrapper = NPUGraphDecoderWrapper(self.model)
        npu = SimpleNamespace(graph_pool_handle=lambda: object())
        with (
            patch.object(torch, "npu", npu, create=True),
            patch.object(wrapper, "_capture_xvec_prefix", side_effect=RuntimeError("synthetic capture failure")),
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                wrapper.warmup(SimpleNamespace(type="npu"))
        self.assertFalse(wrapper._warmed_up)

    def test_deployment_keeps_predictor_and_batch_limit(self):
        import yaml
        from runtime import deployment

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(device="npu:0")
            configs = []
            for mode in ("baseline", "code2wav_graph"):
                directory = Path(tmp) / mode
                directory.mkdir()
                configs.append(yaml.safe_load(deployment(args, mode, directory).read_text(encoding="utf-8")))
            for c in configs:
                extra = c["connectors"]["connector_of_shared_memory"]["extra"]
                self.assertEqual(extra["code_predictor_ar_execution"], "kv_frame")
                self.assertEqual(extra["decode_batch_max_size"], 4)
                self.assertEqual(extra["code2wav_graph_batch_sizes"], [1, 2, 4])
            self.assertEqual(configs[0]["stages"][0], configs[1]["stages"][0])
            self.assertTrue(configs[1]["stages"][1]["enforce_eager"])

    @torch.no_grad()
    def test_short_row_does_not_disable_graph_for_full_rows(self):
        caches = [dict(prefix_frames=0) for _ in range(3)]
        first = [torch.randint(32, (1, 2, 1)) for _ in caches]
        run(self.model, self.wrapper, first, caches)
        reference = [snapshot_cache(c) for c in caches]
        codes = [torch.randint(32, (1, 2, n)) for n in (25, 3, 25)]
        self.wrapper.routes.clear()
        actual = run(self.model, self.wrapper, codes, caches)
        expected = [self.model.chunked_decode(x, c)[0] for x, c in zip(codes, reference)]
        for a, b in zip(actual, expected):
            self.assert_tensor(a, b)
        self.assertEqual(caches[1]["suffix_frames"], 4)
        self.assertEqual(self.wrapper.routes[("graph", "suffix_26", 2, 2)], 1)
        self.assertEqual(self.wrapper.routes[("eager", "short_chunk", 1, 1)], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
