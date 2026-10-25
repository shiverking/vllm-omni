# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "snake_npu", Path(__file__).resolve().parents[2] / "vllm_omni/model_executor/models/common/snake_activation_npu.py"
)
snake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snake)


class Tests(unittest.TestCase):
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
