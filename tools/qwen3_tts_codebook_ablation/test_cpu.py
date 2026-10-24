# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path

import torch
from experiment import decode_audio, read_request, retain_codebooks, save_wav
from generate_zh import captured_rows

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "decoder_test_support", ROOT / "tools/qwen3_tts_code2wav_graph/test_cpu.py"
)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = support.decoder()
        # Tiny untrained codebooks start at zero; populate nontrivial vectors.
        with torch.no_grad():
            for group in (cls.model.quantizer.rvq_first, cls.model.quantizer.rvq_rest):
                for layer in group.vq.layers:
                    layer._codebook.embedding_sum.normal_()

    @torch.inference_mode()
    def test_full_identity_and_restore(self):
        codes = torch.randint(0, 32, (1, 2, 3))
        reference = self.model.quantizer.decode(codes)
        with retain_codebooks(self.model.quantizer, 2):
            torch.testing.assert_close(self.model.quantizer.decode(codes), reference, rtol=0, atol=0)
        self.assertNotIn("decode", self.model.quantizer.__dict__)
        with self.assertRaises(RuntimeError):
            with retain_codebooks(self.model.quantizer, 1):
                raise RuntimeError("restore on error")
        torch.testing.assert_close(self.model.quantizer.decode(codes), reference, rtol=0, atol=0)

    @torch.inference_mode()
    def test_semantic_only_is_not_zero_token(self):
        codes = torch.randint(1, 32, (1, 2, 3))
        expected = self.model.quantizer.rvq_first.decode(codes[:, :1])
        with retain_codebooks(self.model.quantizer, 1):
            actual = self.model.quantizer.decode(codes)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        codes[:, 1] = 0
        self.assertFalse(torch.equal(actual, self.model.quantizer.decode(codes)))

    def test_state_isolation_and_length(self):
        codes = torch.randint(0, 32, (1, 2, 28))
        full = decode_audio(self.model, codes, 2)
        short = decode_audio(self.model, codes, 1)
        again = decode_audio(self.model, codes, 2)
        torch.testing.assert_close(full, again, rtol=0, atol=0)
        self.assertEqual(full.shape, short.shape)
        self.assertFalse(torch.equal(full, short))
        self.assertTrue(torch.isfinite(short).all())

    @torch.inference_mode()
    def test_deployed_quantizer_ignores_tail_placeholders(self):
        quantizer = self.model.quantizer
        codes = torch.randint(0, 32, (1, 2, 4))
        expected = quantizer.rvq_first.decode(codes[:, :1])
        quantizer.active_codebooks = 1
        try:
            torch.testing.assert_close(quantizer.decode(codes), expected, atol=0, rtol=0)
            codes[:, 1] = 0
            torch.testing.assert_close(quantizer.decode(codes), expected, atol=0, rtol=0)
        finally:
            del quantizer.active_codebooks

    def test_wav_and_invalid_keep(self):
        with self.assertRaises(ValueError):
            with retain_codebooks(self.model.quantizer, 0):
                pass
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "test.wav"
            save_wav(path, torch.tensor([0.0, 0.5, -0.5]), 24000, 1.0)
            with wave.open(str(path)) as stream:
                self.assertEqual(stream.getnframes(), 3)
                self.assertEqual(stream.getframerate(), 24000)
                self.assertEqual(stream.getsampwidth(), 2)

    def test_input_filter_and_integrity(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            audio = root / "original.wav"
            save_wav(audio, torch.zeros(10), 24000, 1)
            codes = root / "codes.jsonl"
            rows = [
                dict(frame_id=i, request_id="sample", attempt="a", codes=value, valid=valid)
                for i, (value, valid) in enumerate([([1] * 16, True), ([0] * 16, False)])
            ]
            codes.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            meta = dict(
                request=dict(id="sample"),
                attempt="a",
                codes="codes.jsonl",
                audio="original.wav",
                codes_sha256=hashlib.sha256(codes.read_bytes()).hexdigest(),
                audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
            )
            marker = root / "completed.json"
            marker.write_text(json.dumps(meta), encoding="utf-8")
            _, loaded, _ = read_request(root, marker)
            self.assertEqual(tuple(loaded.shape), (1, 16, 1))
            codes.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed source"):
                read_request(root, marker)

    def test_chinese_capture_order_and_request_isolation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            paths = []
            for seq in (1, 0):
                path = root / f"{seq}.pt"
                torch.save(
                    dict(sequence=seq, rows=[dict(request_id="internal", codes=torch.full((16, 2), seq + 1))]), path
                )
                paths.append(path)
            rows = captured_rows(paths, dict(id="zh001-s42"), "attempt")
            self.assertEqual([row["codes"][0] for row in rows], [1, 1, 2, 2])
            self.assertEqual([row["frame_id"] for row in rows], [0, 1, 2, 3])
            self.assertTrue(all(row["valid"] for row in rows))
            torch.save(
                dict(sequence=2, rows=[dict(request_id="another", codes=torch.ones(16, 1, dtype=torch.long))]),
                root / "extra.pt",
            )
            with self.assertRaisesRegex(ValueError, "唯一对应"):
                captured_rows([*paths, root / "extra.pt"], dict(id="zh001-s42"), "attempt")


if __name__ == "__main__":
    unittest.main()
