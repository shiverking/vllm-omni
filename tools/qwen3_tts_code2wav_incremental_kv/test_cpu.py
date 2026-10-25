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


if __name__ == "__main__":
    unittest.main()
