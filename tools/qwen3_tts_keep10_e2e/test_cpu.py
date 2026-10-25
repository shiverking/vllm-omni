# SPDX-License-Identifier: Apache-2.0
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("c2w_cpu_support", ROOT / "tools/qwen3_tts_code2wav_graph/test_cpu.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.path.insert(0, str(Path(__file__).parent))
import service_bench as service  # noqa: E402


class Tests(unittest.TestCase):
    def test_four_deployments(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            args = SimpleNamespace(device="npu:0")
            configs = {}
            for mode in service.VARIANTS:
                directory = Path(tmp) / mode
                directory.mkdir()
                config = yaml.safe_load(service.deployment(args, mode, directory).read_text(encoding="utf-8"))
                extra = config["connectors"]["connector_of_shared_memory"]["extra"]
                self.assertEqual(extra["code_predictor_ar_execution"], "kv_frame")
                self.assertEqual(extra["code_predictor_keep_codebooks"], 10 if "keep10" in mode else 16)
                self.assertEqual(extra["code2wav_graph_optimization"], "current")
                self.assertEqual(extra["decode_batch_max_size"], 4)
                self.assertEqual(
                    extra["code2wav_execution"], "npu_graph" if mode in ("graph16", "keep10_graph") else "baseline"
                )
                extra.pop("code_predictor_keep_codebooks")
                extra.pop("code_predictor_ar_telemetry")
                extra.pop("code2wav_graph_telemetry")
                configs[mode] = config
            self.assertEqual(configs["baseline"], configs["keep10"])
            self.assertEqual(configs["graph16"], configs["keep10_graph"])

    @torch.inference_mode()
    def test_keep10_graph_layout_boundaries_and_discarded_codes(self):
        torch.set_num_threads(1)
        tiny = base.decoder()
        tiny.config.num_quantizers = 16
        model = base.Qwen3TTSTokenizerV2Decoder._from_config(tiny.config).eval()
        model._initial_codec_chunk_frames = 1
        model._incremental_chunk_frames = 25
        for module in model.modules():
            if hasattr(module, "embedding_sum"):
                module.embedding_sum.normal_(std=0.2)
                module.cluster_usage.uniform_(0.5, 2)
        model.quantizer.active_codebooks = 10
        codes = torch.randint(32, (3, 16, 25))
        expected = model.quantizer.decode(codes[:, :10])
        codes[:, 10:] = 99999
        torch.testing.assert_close(model.quantizer.decode(codes), expected, atol=0, rtol=0)
        wrapper = base.eager_layout_class()(model)
        wrapper.warmup(torch.device("cpu"))
        caches, refs = [dict(prefix_frames=0) for _ in range(3)], [dict(prefix_frames=0) for _ in range(3)]
        for n in (1, 25, 25, 25, 25, 3, 25, 2):
            codes = [torch.randint(32, (1, 16, n)) for _ in range(3)]
            for value in codes:
                value[:, 10:] = 99999
            actual = base.run(model, wrapper, codes, caches)
            expected = base.run(model, None, codes, refs)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
        saved = actual[0].clone()
        base.run(model, wrapper, [torch.zeros(1, 16, 1, dtype=torch.long)], [dict(prefix_frames=0)])
        torch.testing.assert_close(actual[0], saved, atol=0, rtol=0)

    def test_service_flow_english_audio_and_five_comparisons(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 24)
        calls = []

        @contextlib.contextmanager
        def server(args, mode, directory, progress):
            calls.append(mode)
            directory.mkdir(parents=True, exist_ok=True)
            yield "http://127.0.0.1:8091"

        def client(command, directory, progress, label):
            self.assertEqual(command[command.index("--seed-tts-locale") + 1], "en")
            self.assertIn("English", command[command.index("--extra-body") + 1])
            service.write_json(directory / "result.json", dict(completed=2, request_throughput=1, median_e2el_ms=2))
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                results=Path(tmp) / "service",
                model="/nonexistent/model",
                device="npu:0",
                requests=2,
                concurrencies=[1],
                port=8091,
            )
            with (
                patch.object(service, "server", server),
                patch.object(service, "client", client),
                patch.object(service, "speech", return_value=output.getvalue()),
                patch.object(service, "manifest", return_value=dict(code={}, sampling={})),
            ):
                service.run(args)
            self.assertEqual(calls, list(service.VARIANTS))
            summary = service.read_json(args.results / "summary.json")
            self.assertEqual(len(summary["comparisons"]), 5)
            for mode in calls:
                self.assertEqual(len(service.read_json(args.results / mode / "audio/index.json")), 30)
            self.assertEqual(service.read_json(args.results / "status.json")["status"], "complete")


if __name__ == "__main__":
    unittest.main()
