# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    "snake_npu", Path(__file__).resolve().parents[2] / "vllm_omni/model_executor/models/common/snake_activation_npu.py"
)
snake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snake)


class Tests(unittest.TestCase):
    def test_jit_language_is_global(self):
        # Emulate a compiler inspecting globals without inheriting closure cells.
        language = types.ModuleType("triton.language")
        language.constexpr = object()
        triton = types.ModuleType("triton")
        triton.language = language
        triton.jit = lambda fn: fn
        snake.kernel.cache_clear()
        try:
            with patch.dict(sys.modules, {"triton": triton, "triton.language": language}):
                function = snake.kernel()
                self.assertIs(function.__globals__["tl"], language)
                self.assertNotIn("tl", function.__code__.co_freevars)
        finally:
            snake.kernel.cache_clear()
            snake.__dict__.pop("tl", None)

    def test_flat_indexing_and_cropped_strides(self):
        for b, c, t in ((1, 3, 1), (4, 7, 37), (2, 5, 129)):
            storage = torch.arange(b * c * (t + 9)).reshape(b, c, t + 9)
            x = storage[..., 4 : 4 + t]
            offset = torch.arange(b * c * t)
            row, time = offset // t, offset % t
            channel = row % c
            address = row // c * x.stride(0) + channel * x.stride(1) + time + x.storage_offset()
            torch.testing.assert_close(storage.flatten()[address], x.flatten())
            for block in (128, 1024):
                tasks = (x.numel() + block - 1) // block
                visited = [
                    i
                    for pid in range(40)
                    for task in range(pid, tasks, 40)
                    for i in range(task * block, min((task + 1) * block, x.numel()))
                ]
                self.assertEqual(sorted(visited), list(range(x.numel())))

    def test_auto_policy_and_override(self):
        self.assertEqual(snake.select_config((4, 1024, 37), "auto"), ("flat", 1024, 40))
        self.assertEqual(snake.select_config((1, 96, 1920), "auto"), ("channel", 2048, 32))
        self.assertEqual(snake.select_config((4, 96, 71040), "auto"), ("channel", 4096, 40))
        self.assertEqual(snake.select_config((4, 96, 71040), "flat", 512, 16), ("flat", 512, 16))
        with self.assertRaises(ValueError):
            snake.select_config((1, 1, 1), "invalid")

    def test_bounded_real_capture(self):
        capture_spec = importlib.util.spec_from_file_location("snake_capture", Path(__file__).with_name("capture.py"))
        capture = importlib.util.module_from_spec(capture_spec)
        capture_spec.loader.exec_module(capture)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = capture.Recorder(Path(tmp), 1, 1024)
            module = types.SimpleNamespace(_cached=True, _exp_alpha=torch.ones(3), _inv_beta=torch.ones(3))
            x = torch.arange(24.).reshape(1, 3, 8)
            hook = recorder.hook("snake")
            hook(module, (x,))
            x.zero_()
            hook(module, (x,))
            hook(module, (torch.zeros(2, 3, 8),))
            self.assertEqual(len(recorder.rows), 1)
            saved = torch.load(Path(tmp) / "sample000.pt", weights_only=True)
            self.assertEqual(saved["x"].sum().item(), 276)
            self.assertEqual(sum(recorder.counts.values()), 3)
            self.assertEqual(len(recorder.seen), 2)

    def test_rounding_order(self):
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            x = torch.randn(2, 3, 513).to(dtype)
            a, ib = torch.randn(3).exp().to(dtype), torch.randn(3).exp().to(dtype)
            z = (x.float() * a.float()[None, :, None]).to(dtype).float()
            s = torch.sin(z).to(dtype).float()
            sq = (s * s).to(dtype).float()
            term = (ib.float()[None, :, None] * sq).to(dtype).float()
            actual = (x.float() + term).to(dtype)
            torch.testing.assert_close(actual, snake.reference(x, a, ib), atol=0, rtol=0)

    def test_validation(self):
        x = torch.ones(1, 3, 17)
        a = torch.ones(3)
        snake.validate(x, a, a, torch.empty_like(x), 2048, 32)
        cropped = torch.ones(2, 3, 30)[..., 4:21]
        snake.validate(cropped, a, a, torch.empty(cropped.shape), 2048, 32)
        for block in (0, 127, 1500, 16384):
            with self.assertRaises(ValueError):
                snake.validate(x, a, a, x, block, 32)
        with self.assertRaises(ValueError):
            snake.validate(x.transpose(1, 2), a, a, x, 2048, 32)
        with self.assertRaises(ValueError):
            snake.fused_snake(x, a, a)

    def test_tile_coverage(self):
        for b, c, t in ((1, 3, 1), (2, 5, 2049), (4, 2, 8193)):
            block = 2048
            tiles = (t + block - 1) // block
            seen = set()
            for program in range(7):
                for task in range(program, b * c * tiles, 7):
                    channel = (task // tiles) % c
                    row = task // tiles
                    self.assertEqual(channel, row % c)
                    for offset in range((task % tiles) * block, min((task % tiles + 1) * block, t)):
                        position = row * t + offset
                        self.assertNotIn(position, seen)
                        seen.add(position)
            self.assertEqual(len(seen), b * c * t)


if __name__ == "__main__":
    unittest.main()
