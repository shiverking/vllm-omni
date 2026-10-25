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
    causal_window_mask,
    continuous_reference,
    IncrementalKVExecutor,
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
            expected = model.chunked_decode(codes[..., offset:offset+n], cache)
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


if __name__ == "__main__":
    unittest.main()
