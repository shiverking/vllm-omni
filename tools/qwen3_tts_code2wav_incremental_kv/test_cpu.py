# SPDX-License-Identifier: Apache-2.0
"""Standalone real-model CPU tests; no NPU capture claims."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("c2w_test_support", ROOT / "tools/qwen3_tts_code2wav_graph/test_cpu.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
sys.path.insert(0, str(ROOT))
from vllm_omni.model_executor.models.qwen3_tts.code2wav_incremental_kv import (  # noqa: E402
    IncrementalKVExecutor,
    causal_window_mask,
    continuous_reference,
)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_window_includes_self(self):
        mask = causal_window_mask(25, 71, 72, "cpu", torch.float32)[0, 0]
        self.assertEqual((mask[0] == 0).sum().item(), 72)
        self.assertEqual((mask[-1] == 0).sum().item(), 72)
        self.assertTrue((mask[-1, :24] < 0).all())

    @torch.no_grad()
    def test_reference_growth(self):
        model = support.decoder()
        codes = torch.randint(0, 32, (1, 2, 51))
        cache = {"prefix_frames": 0}
        offset = 0
        for n in (1, 25, 25):
            expected = model.chunked_decode(codes[..., offset : offset + n], cache)
            offset += n
            got, _ = continuous_reference(model, codes[..., :offset], n)
            torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)

    @torch.no_grad()
    def test_growth_handoff(self):
        model = support.decoder()
        runner = IncrementalKVExecutor(model)
        for batch in (1, 2, 3, 4):
            cache = [dict(prefix_frames=0) for _ in range(batch)]
            stock = [dict(prefix_frames=0) for _ in range(batch)]
            saved = None
            for n in (1, 25, 25, 25, 25, 3, 25, 25):
                codes = [torch.randint(0, 32, (1, 2, n)) for _ in cache]
                got = runner._batched_request_decode(codes, cache)
                expected = [model.chunked_decode(x, c) for x, c in zip(codes, stock)]
                for a, b, c, d in zip(got, expected, cache, stock):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
                    for name in ("suffix_quantized", "suffix_conv"):
                        torch.testing.assert_close(c[name], d[name], atol=1e-5, rtol=1e-4)
                if saved:
                    torch.testing.assert_close(*saved)
                saved = (got[0], got[0].clone())
                cache.reverse()
                stock.reverse()
            runner.release_requests(cache)
            self.assertEqual(len(runner.owners), 0)

    @torch.no_grad()
    def test_stream_multilayer_and_partition(self):
        model = support.decoder()
        # Two layers are essential: historical representations carry old context.
        import copy

        layer = copy.deepcopy(model.pre_transformer.layers[0])
        layer.self_attn.layer_idx = 1
        model.pre_transformer.layers.append(layer)
        model.config.num_hidden_layers = 2
        for sizes in ((1, 25, 25, 25, 25, 25, 25), (1, 3, 25, 7, 25, 25, 25, 25)):
            runner = IncrementalKVExecutor(model, mode="stream_kv")
            caches = [dict(prefix_frames=0) for _ in range(3)]
            histories = [torch.empty(1, 2, 0, dtype=torch.long) for _ in caches]
            count = 0
            for n in sizes:
                codes = [torch.randint(0, 32, (1, 2, n)) for _ in caches]
                got = runner._batched_request_decode(codes, caches)
                count += n
                for row in range(3):
                    histories[row] = torch.cat((histories[row], codes[row]), -1)
                    expected, hidden = continuous_reference(model, histories[row], n)
                    torch.testing.assert_close(got[row], expected, atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(caches[row]["_ikv"]["hidden"], hidden[:, -12:], atol=1e-5, rtol=1e-4)
                    self.assertLessEqual(caches[row]["_ikv"]["k"].shape[-2], 71)
                caches.reverse()
                histories.reverse()
            self.assertEqual(runner.routes["transformer_positions"], count * 3)
            runner.release_requests(caches)

    @torch.no_grad()
    def test_capacity_and_release(self):
        model = support.decoder()
        runner = IncrementalKVExecutor(model, "stream_kv")
        caches = [{} for _ in range(16)]
        code = torch.randint(0, 32, (1, 2, 1))
        runner._batched_request_decode([code] * 16, caches)
        with self.assertRaisesRegex(RuntimeError, "16 active"):
            runner._batched_request_decode([code], [{}])
        runner.release_requests(caches)
        got = runner._batched_request_decode([code], [caches[0]])[0]
        expected, _ = continuous_reference(model, code, 1)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)

    @torch.no_grad()
    def test_full_check_tool(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch

        module_spec = importlib.util.spec_from_file_location(
            "incremental_tool", Path(__file__).with_name("experiment.py")
        )
        tool = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(tool)
        model = support.decoder()
        sequences = [
            dict(id=f"test{i}", chunks=[torch.randint(0, 32, (1, 2, n)) for n in (1, 25, 25, 25, 7)]) for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                results=Path(temp),
                device="cpu",
                variants=["current", "growth_kv", "stream_kv"],
                executions=["eager"],
                batches=[1, 2, 3, 4],
            )
            with (
                patch.object(tool, "load_decoder", return_value=(model, 24000)),
                patch.object(tool.legacy, "sequences", return_value=sequences),
            ):
                tool.check(args, tool.Progress(Path(temp)))
            result = tool.read_json(Path(temp) / "correctness.json")
            self.assertTrue(result["passed"], result["errors"])
            self.assertTrue((Path(temp) / "listening.json").exists())
            # Deliberate numerical failures must not skip later chunks or audio.
            original_compare = tool.compare

            def mismatching_compare(*values):
                return {**original_compare(*values), "passed": False}

            args.variants = ["growth_kv", "stream_kv"]
            args.batches = [1]
            with (
                patch.object(tool, "load_decoder", return_value=(model, 24000)),
                patch.object(tool.legacy, "sequences", return_value=sequences),
                patch.object(tool, "compare", side_effect=mismatching_compare),
            ):
                tool.check(args, tool.Progress(Path(temp)))
            result = tool.read_json(Path(temp) / "correctness.json")
            self.assertTrue(result["complete"])
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "completed_with_differences")
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["mismatch_records"], 2 * (3 * 5 + 13))
            self.assertEqual(len(tool.read_json(Path(temp) / "listening.json")), 8)
            chosen, weights, counts = tool.weighted_jobs(sequences, 2, 12, 72)
            self.assertEqual(set(counts), {tool.job_phase(j) for j in chosen})
            self.assertAlmostEqual(sum(weights[tool.job_phase(j)] for j in chosen), sum(counts.values()))
            args.variants = ["growth_kv", "stream_kv"]
            args.batches = [1, 3]
            args.warmup = 1
            args.executions_per_round = 8
            args.profile = False
            actual_make = tool.make

            def eager_make(args, model, mode, execution):
                return actual_make(args, model, mode, "eager")

            def measured(model, wrapper, codes, caches):
                return 2.0, 1.0, tool.run(model, wrapper, codes, caches)

            with (
                patch.object(tool, "load_decoder", return_value=(model, 24000)),
                patch.object(tool.legacy, "sequences", return_value=sequences),
                patch.object(tool, "make", side_effect=eager_make),
                patch.object(tool.legacy, "measured", side_effect=measured),
                patch.object(
                    torch, "npu", SimpleNamespace(max_memory_allocated=lambda: 0, empty_cache=lambda: None), create=True
                ),
            ):
                tool.bench(args, tool.Progress(Path(temp)))
            self.assertTrue(all(r["speedup"] == 1.0 for r in tool.read_json(Path(temp) / "summary.json")))

    @torch.no_grad()
    def test_kv_layers_against_full_reference(self):
        model = support.decoder()
        runner = IncrementalKVExecutor(model, "stream_kv")
        cache = {}
        codes = torch.randint(0, 32, (1, 2, 151))
        offset = 0
        positions = []
        hook = model.pre_transformer.register_forward_pre_hook(
            lambda mod, args, kwargs: positions.append(kwargs["inputs_embeds"].shape[1]), with_kwargs=True
        )
        try:
            for n in (1, 25, 25, 25, 25, 25, 25):
                runner._batched_request_decode([codes[..., offset : offset + n]], [cache])
                offset += n
            self.assertEqual(positions, [1, 25, 25, 25, 25, 25, 25])
        finally:
            hook.remove()
        _, _, k, v = continuous_reference(model, codes, 25, return_kv=True)
        torch.testing.assert_close(cache["_ikv"]["k"], k[..., -71:, :], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(cache["_ikv"]["v"], v[..., -71:, :], atol=1e-5, rtol=1e-4)

    @torch.no_grad()
    def test_error_and_concurrent_cleanup(self):
        from unittest.mock import patch

        model = support.decoder()
        runner = IncrementalKVExecutor(model, "stream_kv")
        cache = {}
        codes = torch.randint(0, 32, (1, 2, 1))
        runner._batched_request_decode([codes], [cache])
        with patch.object(model.pre_transformer, "forward", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                runner._batched_request_decode([codes], [cache])
        self.assertEqual(cache, {"prefix_frames": 0})
        self.assertFalse(runner.owners)
        runner._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                runner._batched_request_decode([codes], [cache])
        finally:
            runner._lock.release()
        got = runner._batched_request_decode([codes], [cache])[0]
        expected, _ = continuous_reference(model, codes, 1)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)

    @torch.no_grad()
    def test_mature_batch_different_absolute_positions(self):
        model = support.decoder()
        runner = IncrementalKVExecutor(model, "stream_kv")
        caches = [{}, {}]
        histories = []
        for row, sizes in enumerate(((1, 25, 25, 25), (1, 25, 25, 25, 25, 25))):
            parts = []
            for n in sizes:
                x = torch.randint(0, 32, (1, 2, n))
                parts.append(x)
                runner._batched_request_decode([x], [caches[row]])
            histories.append(torch.cat(parts, -1))
        codes = [torch.randint(0, 32, (1, 2, 25)) for _ in caches]
        before = runner.routes["eager"]
        got = runner._batched_request_decode(codes, caches)
        self.assertEqual(runner.routes["eager"] - before, 1)
        for i in range(2):
            expected, _ = continuous_reference(model, torch.cat((histories[i], codes[i]), -1), 25)
            torch.testing.assert_close(got[i], expected, atol=1e-5, rtol=1e-4)
        self.assertEqual([c["_ikv_total"] for c in caches], [101, 151])


if __name__ == "__main__":
    unittest.main()
