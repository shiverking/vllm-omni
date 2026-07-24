"""Tests for SeedTTSTextDataset, SeedTTSTextSampleRequest, SeedTTSDesignDataset,
and SeedTTSDesignSampleRequest.

vllm stubs are installed by tests/benchmarks/conftest.py before collection.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Load the data module directly (bypasses vllm_omni.__init__ heavy imports).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "vllm_omni" / "benchmarks" / "data_modules" / "seed_tts_dataset.py"
_MODULE_NAME = "vllm_omni.benchmarks.data_modules.seed_tts_dataset"

if _MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import (  # noqa: E402
    SeedTTSDesignDataset,
    SeedTTSDesignSampleRequest,
    SeedTTSDataset,
    SeedTTSTextDataset,
    SeedTTSTextSampleRequest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_wav(path: Path, *, sample_rate: int = 24000, frames: int = 2400) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


@pytest.fixture()
def seed_tts_root(tmp_path: Path) -> Path:
    """Minimal seed-tts-style directory with 5 entries."""
    locale_dir = tmp_path / "en"
    locale_dir.mkdir()
    wav_dir = locale_dir / "prompt-wavs"
    wav_dir.mkdir()
    for i in range(5):
        _write_wav(wav_dir / f"utt{i:03d}.wav")
    meta = "\n".join(f"utt{i:03d}|ref text {i}|prompt-wavs/utt{i:03d}.wav|target text {i}" for i in range(5))
    (locale_dir / "meta.lst").write_text(meta, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def mock_tokenizer(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.encode = lambda text, **kw: [0] * len(text.split())
    tokenizer.get_vocab.return_value = {"<pad>": 0}
    tokenizer.all_special_ids = []
    tokenizer.all_special_tokens = []
    tokenizer.vocab_size = 1
    tokenizer.__len__.return_value = 1
    return tokenizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_seed_tts_manifest_is_deterministic(seed_tts_root, mock_tokenizer, tmp_path):
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for path in paths:
        ds = SeedTTSDataset(
            dataset_path=str(seed_tts_root),
            random_seed=0,
            locale="en",
            workload_manifest_out=str(path),
        )
        requests = ds.sample(mock_tokenizer, num_requests=5, request_id_prefix="seedtts-")
        assert len(requests) == 5

    first = paths[0].read_bytes()
    second = paths[1].read_bytes()
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    manifest = json.loads(first)
    assert manifest["random_seed"] == 0
    assert len(manifest["requests"]) == 5
    assert set(manifest["requests"][0]) == {
        "request_id",
        "utterance_id",
        "target_text",
        "reference_text",
        "reference_wav_path",
        "wav_sha256",
        "wav_duration_seconds",
        "wav_sample_rate",
        "target_token_count",
    }


def test_seed_tts_manifest_replay_preserves_order(seed_tts_root, mock_tokenizer, tmp_path):
    manifest_path = tmp_path / "workload.json"
    generated = SeedTTSDataset(
        dataset_path=str(seed_tts_root), random_seed=0, locale="en", workload_manifest_out=str(manifest_path)
    ).sample(mock_tokenizer, num_requests=5, request_id_prefix="stable-")

    replayed = SeedTTSDataset(
        dataset_path=str(seed_tts_root),
        random_seed=999,
        locale="en",
        workload_manifest_in=str(manifest_path),
    ).sample(mock_tokenizer, num_requests=5, no_oversample=False)

    assert [r.request_id for r in replayed] == [r.request_id for r in generated]
    assert [r.seed_tts_utterance_id for r in replayed] == [r.seed_tts_utterance_id for r in generated]


def test_seed_tts_manifest_replay_rejects_missing_wav(seed_tts_root, mock_tokenizer, tmp_path):
    manifest_path = tmp_path / "workload.json"
    SeedTTSDataset(
        dataset_path=str(seed_tts_root), random_seed=0, locale="en", workload_manifest_out=str(manifest_path)
    ).sample(mock_tokenizer, num_requests=5)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = seed_tts_root / "en" / manifest["requests"][0]["reference_wav_path"]
    missing.unlink()

    replay = SeedTTSDataset(
        dataset_path=str(seed_tts_root), locale="en", workload_manifest_in=str(manifest_path)
    )
    with pytest.raises(FileNotFoundError, match="Manifest reference WAV is missing"):
        replay.sample(mock_tokenizer, num_requests=5)


def test_seed_tts_manifest_replay_rejects_checksum_change(seed_tts_root, mock_tokenizer, tmp_path):
    manifest_path = tmp_path / "workload.json"
    SeedTTSDataset(
        dataset_path=str(seed_tts_root), random_seed=0, locale="en", workload_manifest_out=str(manifest_path)
    ).sample(mock_tokenizer, num_requests=5)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = seed_tts_root / "en" / manifest["requests"][0]["reference_wav_path"]
    with changed.open("ab") as f:
        f.write(b"changed")

    replay = SeedTTSDataset(
        dataset_path=str(seed_tts_root), locale="en", workload_manifest_in=str(manifest_path)
    )
    with pytest.raises(ValueError, match="WAV SHA256 mismatch"):
        replay.sample(mock_tokenizer, num_requests=5)


def test_seed_tts_manifest_replay_rejects_meta_change(seed_tts_root, tmp_path):
    manifest_path = tmp_path / "workload.json"
    meta = seed_tts_root / "en" / "meta.lst"
    manifest = {
        "schema_version": 1,
        "locale": "en",
        "meta_sha256": "0" * 64,
        "requests": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert meta.is_file()
    with pytest.raises(ValueError, match="meta.lst SHA256 mismatch"):
        SeedTTSDataset(dataset_path=str(seed_tts_root), locale="en", workload_manifest_in=str(manifest_path))


def test_seed_tts_text_dataset_omits_ref_audio(seed_tts_root, mock_tokenizer):
    ds = SeedTTSTextDataset(
        dataset_path=str(seed_tts_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    requests = ds.sample(mock_tokenizer, num_requests=3)
    assert len(requests) == 3
    for req in requests:
        assert isinstance(req, SeedTTSTextSampleRequest)
        assert req.seed_tts_speech_extra is None or "ref_audio" not in (req.seed_tts_speech_extra or {})
        assert req.seed_tts_ref_wav_path == ""
        assert "target text" in req.prompt


# ---------------------------------------------------------------------------
# SeedTTSDesignDataset tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def seed_tts_design_root(tmp_path: Path) -> Path:
    """seed-tts-design directory with 5-field meta.lst entries."""
    locale_dir = tmp_path / "en"
    locale_dir.mkdir()
    meta = "\n".join(
        f"des{i:03d}|||target text {i}|A warm {['female', 'male'][i % 2]} voice with neutral accent." for i in range(5)
    )
    (locale_dir / "meta.lst").write_text(meta, encoding="utf-8")
    return tmp_path


def test_seed_tts_design_dataset_has_instructions(seed_tts_design_root, mock_tokenizer):
    ds = SeedTTSDesignDataset(
        dataset_path=str(seed_tts_design_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    requests = ds.sample(mock_tokenizer, num_requests=3)
    assert len(requests) == 3
    for req in requests:
        assert isinstance(req, SeedTTSDesignSampleRequest)
        extra = req.seed_tts_speech_extra or {}
        assert "instructions" in extra
        assert extra["instructions"], "instructions must be non-empty"
        assert extra.get("task_type") == "VoiceDesign"
        assert "ref_audio" not in extra
        assert req.seed_tts_ref_wav_path == ""


def test_seed_tts_design_dataset_rejects_missing_description(seed_tts_design_root, mock_tokenizer):
    """Lines without a voice_description should be skipped."""
    locale_dir = seed_tts_design_root / "en"
    # The bad line has 4 fields, not 5, so will be filtered
    meta = "bad|||target text without description\n" + "\n".join(
        f"ok|||target text {i}|A clear female voice." for i in range(9)
    )
    (locale_dir / "meta.lst").write_text(meta, encoding="utf-8")
    ds = SeedTTSDesignDataset(
        dataset_path=str(seed_tts_design_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    requests = ds.sample(mock_tokenizer, num_requests=10, no_oversample=True)
    assert len(requests) == 9  # since we filter the bad row out and don't oversample
    for req in requests:
        assert isinstance(req, SeedTTSDesignSampleRequest)
        assert req.seed_tts_utterance_id == "ok"


def test_attach_sets_seed_tts_row_even_without_extra_body():
    """seed_tts_row=True must be set for SeedTTSTextSampleRequest (no extra body)."""
    from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSTextSampleRequest

    req = SeedTTSTextSampleRequest(
        prompt="hello world",
        prompt_len=2,
        expected_output_len=100,
        multi_modal_data=None,
        request_id="test-0",
        seed_tts_speech_extra=None,
        seed_tts_ref_wav_path="",
    )
    assert req.seed_tts_speech_extra is None
    assert req.seed_tts_ref_wav_path == ""
    # The fix ensures that even with speech_extra=None, the function
    # sets seed_tts_row=True. We verify the source code has the fix.
    import inspect

    import vllm_omni.benchmarks.patch.patch as patch_mod

    src = inspect.getsource(patch_mod._attach_seed_tts_to_request_func_input)
    # seed_tts_row must be set BEFORE the 'if not ex: return' check
    row_pos = src.index("seed_tts_row")
    not_ex_pos = src.index("if not ex:")
    assert row_pos < not_ex_pos, "seed_tts_row must be set before 'if not ex: return'"


def test_seed_tts_whisper_transcribe_passes_attention_mask(monkeypatch):
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    calls = {}

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name
            self.device = None

        def to(self, device):
            self.device = device
            return self

    class FakeProcessor:
        def __call__(self, wav, *, sampling_rate, return_tensors, return_attention_mask=False):
            calls["return_attention_mask"] = return_attention_mask
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            assert len(wav) > 0
            return types.SimpleNamespace(
                input_features=FakeTensor("features"),
                attention_mask=FakeTensor("mask") if return_attention_mask else None,
            )

        def get_decoder_prompt_ids(self, *, language, task):
            assert language == "english"
            assert task == "transcribe"
            return [(1, 2)]

        def batch_decode(self, predicted_ids, *, skip_special_tokens):
            assert skip_special_tokens
            assert predicted_ids == [[42]]
            return ["hello"]

    class FakeModel:
        def generate(self, input_features, **kwargs):
            calls["input_features"] = input_features
            calls["generate_kwargs"] = kwargs
            return [[42]]

    monkeypatch.setattr(seed_tts_eval, "_ensure_en_asr", lambda: None)
    monkeypatch.setattr(seed_tts_eval, "_en_processor", FakeProcessor())
    monkeypatch.setattr(seed_tts_eval, "_en_model", FakeModel())
    monkeypatch.setattr(seed_tts_eval, "_device", "cuda:1")

    text = seed_tts_eval._transcribe_en_f32_16k(np.ones(1600, dtype=np.float32))

    assert text == "hello"
    assert calls["return_attention_mask"] is True
    assert calls["input_features"].device == "cuda:1"
    assert calls["generate_kwargs"]["attention_mask"].device == "cuda:1"
    assert calls["generate_kwargs"]["forced_decoder_ids"] == [(1, 2)]
