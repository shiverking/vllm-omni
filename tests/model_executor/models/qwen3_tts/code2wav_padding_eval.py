# SPDX-License-Identifier: Apache-2.0
"""Offline helpers for evaluating Code2Wav right-padding accuracy.

This module deliberately lives under ``tests``.  It can replay an already
captured NPU graph with padded codec inputs, but it does not change production
routing in :mod:`npu_graph_decoder_wrapper`.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import random
import warnings
import wave
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

SAMPLE_RATE = 24_000
PAD_MODES = ("zero", "repeat_last")


@dataclass(frozen=True)
class PaddingThresholds:
    exact_max_abs: float = 1e-4
    exact_mean_abs: float = 1e-6
    exact_cosine: float = 0.99999
    exact_snr_db: float = 80.0
    padded_cosine: float = 0.9999
    padded_snr_db: float = 60.0
    padded_p99_mean_abs: float = 1e-5
    padded_p99_max_abs: float = 1e-3
    padded_worst_max_abs: float = 5e-3
    boundary_p99_max_abs: float = 1e-3
    boundary_click_delta_db: float = 3.0
    pesq_median_drop: float = 0.01
    pesq_p95_drop: float = 0.03
    stoi_median_drop: float = 1e-4
    stoi_p95_drop: float = 1e-3
    corpus_error_rate_delta: float = 0.001
    subset_error_rate_delta: float = 0.003
    speaker_median_drop: float = 0.001
    speaker_p95_drop: float = 0.005


@dataclass(frozen=True)
class ManifestCase:
    case_id: str
    codes_path: Path
    text: str
    language: str
    speaker_id: str
    mode: str
    ref_context_frames: int


@dataclass
class WaveMetrics:
    max_abs: float
    mean_abs: float
    cosine: float
    snr_db: float
    mrstft_error: float
    log_mel_error: float
    boundary_max_abs: float = 0.0
    boundary_click_delta_db: float = 0.0
    pesq_drop: float | None = None
    stoi_drop: float | None = None
    asr_error_rate_delta: float | None = None
    speaker_cosine_drop: float | None = None


def load_manifest(path: Path) -> list[ManifestCase]:
    cases: list[ManifestCase] = []
    base = path.parent
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            missing = {"id", "codes_path", "text", "language", "speaker_id", "mode"} - value.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            codes_path = Path(value["codes_path"])
            if not codes_path.is_absolute():
                codes_path = base / codes_path
            if not codes_path.exists():
                raise FileNotFoundError(f"{path}:{line_number} codec tensor not found: {codes_path}")
            ref_frames = int(value.get("ref_context_frames", 0))
            if ref_frames < 0:
                raise ValueError(f"{path}:{line_number} ref_context_frames must be non-negative")
            cases.append(
                ManifestCase(
                    case_id=str(value["id"]),
                    codes_path=codes_path,
                    text=str(value["text"]),
                    language=str(value["language"]).lower(),
                    speaker_id=str(value["speaker_id"]),
                    mode=str(value["mode"]).lower(),
                    ref_context_frames=ref_frames,
                )
            )
    if not cases:
        raise ValueError(f"Padding precision manifest is empty: {path}")
    duplicate_ids = {case.case_id for case in cases if sum(c.case_id == case.case_id for c in cases) > 1}
    if duplicate_ids:
        raise ValueError(f"Duplicate case ids in manifest: {sorted(duplicate_ids)}")
    return cases


def validate_manifest_coverage(cases: Sequence[ManifestCase]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[f"mode:{case.mode}"] += 1
        counts[f"language:{case.language}"] += 1
    problems: list[str] = []
    if len(cases) < 300:
        problems.append(f"need >=300 cases, got {len(cases)}")
    for mode in ("custom_voice", "base_voice_clone", "voice_design"):
        if counts[f"mode:{mode}"] < 80:
            problems.append(f"mode {mode} needs >=80, got {counts[f'mode:{mode}']}")
    for language in ("zh", "en"):
        if counts[f"language:{language}"] < 100:
            problems.append(f"language {language} needs >=100, got {counts[f'language:{language}']}")
    speakers = {case.speaker_id for case in cases}
    if len(speakers) < 20:
        problems.append(f"need >=20 speakers, got {len(speakers)}")
    return problems


def load_codec_tensor(path: Path, *, num_quantizers: int, codebook_size: int) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("codes", "codec_codes", "audio_codes"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} must contain a codec tensor")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or int(value.shape[1]) != num_quantizers:
        raise ValueError(
            f"Expected [B,{num_quantizers},F] codec tensor in {path}, got {tuple(value.shape)}"
        )
    if value.shape[-1] <= 0:
        raise ValueError(f"Codec tensor has no frames: {path}")
    return value.to(dtype=torch.long).remainder(codebook_size)


def next_bucket(actual_frames: int, buckets: Sequence[int]) -> int | None:
    return next((int(bucket) for bucket in buckets if int(bucket) >= actual_frames), None)


def padded_codes(codes: torch.Tensor, padded_frames: int, pad_mode: str) -> torch.Tensor:
    actual_frames = int(codes.shape[-1])
    if actual_frames <= 0 or padded_frames < actual_frames:
        raise ValueError(f"Invalid padding {actual_frames}->{padded_frames}")
    if pad_mode not in PAD_MODES:
        raise ValueError(f"Unsupported pad mode {pad_mode!r}; expected one of {PAD_MODES}")
    if padded_frames == actual_frames:
        return codes.clone()
    result = codes.new_zeros((*codes.shape[:-1], padded_frames))
    result[..., :actual_frames].copy_(codes)
    if pad_mode == "repeat_last":
        result[..., actual_frames:].copy_(codes[..., -1:].expand(*codes.shape[:-1], padded_frames - actual_frames))
    return result


def graph_padded_decode(
    wrapper: Any,
    codes: torch.Tensor,
    padded_frames: int,
    pad_mode: str,
) -> torch.Tensor:
    """Replay a captured graph with test-only right padding."""
    actual_frames = int(codes.shape[-1])
    key = (int(codes.shape[0]), int(padded_frames))
    if key not in wrapper.graphs:
        raise KeyError(f"NPU graph {key} was not captured")
    static_input = wrapper.static_inputs[key]
    static_input.copy_(padded_codes(codes, padded_frames, pad_mode))
    wrapper.graphs[key].replay()
    expected_len = actual_frames * int(wrapper.decoder.total_upsample)
    static_output = wrapper.static_outputs[key]
    if int(static_output.shape[-1]) < expected_len:
        raise AssertionError(
            f"Graph {key} output is too short: {static_output.shape[-1]} < {expected_len}"
        )
    return static_output[..., :expected_len].clone()


def graph_mixed_padded_decode(
    wrapper: Any,
    codes: Sequence[torch.Tensor],
    padded_frames: int,
    pad_mode: str,
) -> list[torch.Tensor]:
    """Replay one graph for a mixed-length batch and trim every row exactly."""
    if not codes:
        return []
    batch_size = len(codes)
    key = (batch_size, int(padded_frames))
    if key not in wrapper.graphs:
        raise KeyError(f"NPU graph {key} was not captured")
    static_input = wrapper.static_inputs[key]
    for row, value in enumerate(codes):
        if value.shape[0] != 1:
            raise ValueError(f"Each mixed-batch codec tensor must have B=1, got {tuple(value.shape)}")
        static_input[row : row + 1].copy_(padded_codes(value, padded_frames, pad_mode))
    wrapper.graphs[key].replay()
    static_output = wrapper.static_outputs[key]
    upsample = int(wrapper.decoder.total_upsample)
    return [
        static_output[row : row + 1, ..., : int(value.shape[-1]) * upsample].clone()
        for row, value in enumerate(codes)
    ]


def eager_padded_decode(
    decoder: torch.nn.Module,
    codes: torch.Tensor,
    padded_frames: int,
    pad_mode: str,
) -> torch.Tensor:
    actual_frames = int(codes.shape[-1])
    output = decoder(padded_codes(codes, padded_frames, pad_mode))
    return output[..., : actual_frames * int(decoder.total_upsample)].clone()


def assert_waveform_contract(candidate: torch.Tensor, eager: torch.Tensor, label: str) -> None:
    if candidate.shape != eager.shape:
        raise AssertionError(f"{label}: shape {tuple(candidate.shape)} != {tuple(eager.shape)}")
    if candidate.dtype != eager.dtype or candidate.dtype != torch.float32:
        raise AssertionError(f"{label}: expected matching FP32 outputs")
    if not bool(torch.isfinite(candidate).all()):
        raise AssertionError(f"{label}: output contains NaN/Inf")
    if candidate.numel() and (float(candidate.min()) < -1.0 or float(candidate.max()) > 1.0):
        raise AssertionError(f"{label}: output is outside [-1,1]")


def _safe_cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference) * torch.linalg.vector_norm(candidate))
    return float(torch.dot(reference, candidate)) / denominator if denominator > 0 else 1.0


def _mrstft_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    errors: list[float] = []
    for n_fft in (256, 512, 1024, 2048):
        if reference.numel() < n_fft:
            continue
        hop = n_fft // 4
        window = torch.hann_window(n_fft, dtype=torch.float64)
        ref_spec = torch.stft(reference, n_fft, hop, window=window, return_complex=True).abs()
        cand_spec = torch.stft(candidate, n_fft, hop, window=window, return_complex=True).abs()
        numerator = torch.linalg.vector_norm(cand_spec - ref_spec)
        denominator = torch.linalg.vector_norm(ref_spec).clamp_min(torch.finfo(torch.float64).tiny)
        errors.append(float(numerator / denominator))
    return float(sum(errors) / len(errors)) if errors else 0.0


def _log_mel_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    try:
        from torchaudio.transforms import MelSpectrogram

        transform = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            power=1.0,
        )
        reference_features = transform(reference.float())
        candidate_features = transform(candidate.float())
    except Exception:
        window = torch.hann_window(1024, dtype=torch.float64)
        reference_features = torch.stft(reference, 1024, 256, window=window, return_complex=True).abs()
        candidate_features = torch.stft(candidate, 1024, 256, window=window, return_complex=True).abs()
    return float((torch.log1p(reference_features) - torch.log1p(candidate_features)).abs().mean())


def waveform_metrics(
    candidate: torch.Tensor,
    eager: torch.Tensor,
    *,
    boundary_offsets: Sequence[int] = (),
    boundary_radius: int = 0,
) -> WaveMetrics:
    assert_waveform_contract(candidate, eager, "metrics")
    candidate_cpu = candidate.detach().cpu().to(torch.float64).reshape(-1)
    eager_cpu = eager.detach().cpu().to(torch.float64).reshape(-1)
    error = candidate_cpu - eager_cpu
    max_abs = float(error.abs().max()) if error.numel() else 0.0
    mean_abs = float(error.abs().mean()) if error.numel() else 0.0
    signal = float(torch.sum(eager_cpu.square()))
    noise = float(torch.sum(error.square()))
    if noise == 0:
        snr = math.inf
    elif signal == 0:
        snr = -math.inf
    else:
        snr = 10.0 * math.log10(signal / max(noise, torch.finfo(torch.float64).tiny))

    boundary_max = 0.0
    click_delta_db = 0.0
    for offset in boundary_offsets:
        start = max(0, int(offset) - boundary_radius)
        end = min(error.numel(), int(offset) + boundary_radius)
        if end <= start:
            continue
        boundary_max = max(boundary_max, float(error[start:end].abs().max()))
        ref_diff = eager_cpu[max(0, start - 1) : end].diff().abs().max()
        cand_diff = candidate_cpu[max(0, start - 1) : end].diff().abs().max()
        ratio = float(cand_diff.clamp_min(1e-12) / ref_diff.clamp_min(1e-12))
        click_delta_db = max(click_delta_db, 20.0 * math.log10(ratio))

    return WaveMetrics(
        max_abs=max_abs,
        mean_abs=mean_abs,
        cosine=_safe_cosine(eager_cpu, candidate_cpu),
        snr_db=snr,
        mrstft_error=_mrstft_error(eager_cpu, candidate_cpu),
        log_mel_error=_log_mel_error(eager_cpu, candidate_cpu),
        boundary_max_abs=boundary_max,
        boundary_click_delta_db=click_delta_db,
    )


def optional_perceptual_metrics(candidate: torch.Tensor, eager: torch.Tensor) -> tuple[float | None, float | None]:
    """Return PESQ/STOI drops, or ``None`` when their local packages are absent."""
    eager_np = eager.detach().cpu().float().reshape(-1).numpy()
    candidate_np = candidate.detach().cpu().float().reshape(-1).numpy()
    pesq_drop: float | None = None
    stoi_drop: float | None = None
    try:
        from pesq import pesq

        # PESQ-WB is defined at 16 kHz.  Use a deterministic torch resample so
        # evaluation never downloads or initializes another model.
        import torchaudio.functional as audio_functional

        eager_16k = audio_functional.resample(eager.cpu(), SAMPLE_RATE, 16_000).reshape(-1).numpy()
        candidate_16k = audio_functional.resample(candidate.cpu(), SAMPLE_RATE, 16_000).reshape(-1).numpy()
        baseline = float(pesq(16_000, eager_16k, eager_16k, "wb"))
        pesq_drop = baseline - float(pesq(16_000, eager_16k, candidate_16k, "wb"))
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception as error:
        warnings.warn(f"PESQ evaluation failed and blocks production gating: {error}", stacklevel=2)
    try:
        from pystoi import stoi

        baseline = float(stoi(eager_np, eager_np, SAMPLE_RATE, extended=False))
        stoi_drop = baseline - float(stoi(eager_np, candidate_np, SAMPLE_RATE, extended=False))
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception as error:
        warnings.warn(f"STOI evaluation failed and blocks production gating: {error}", stacklevel=2)
    return pesq_drop, stoi_drop


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_value in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_value in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + int(ref_value != hyp_value),
                )
            )
        previous = current
    return previous[-1]


def text_error_rate(reference: str, hypothesis: str, language: str) -> float:
    if language.startswith("zh"):
        reference_units = [char for char in reference if not char.isspace()]
        hypothesis_units = [char for char in hypothesis if not char.isspace()]
    else:
        reference_units = reference.lower().split()
        hypothesis_units = hypothesis.lower().split()
    return _edit_distance(reference_units, hypothesis_units) / max(1, len(reference_units))


def embedding_cosine_drop(reference: Any, candidate: Any) -> float:
    reference_tensor = torch.as_tensor(reference, dtype=torch.float64).reshape(-1)
    candidate_tensor = torch.as_tensor(candidate, dtype=torch.float64).reshape(-1)
    return 1.0 - _safe_cosine(reference_tensor, candidate_tensor)


def assert_exact_metrics(metrics: WaveMetrics, thresholds: PaddingThresholds) -> None:
    assert metrics.max_abs <= thresholds.exact_max_abs, metrics
    assert metrics.mean_abs <= thresholds.exact_mean_abs, metrics
    assert metrics.cosine >= thresholds.exact_cosine, metrics
    assert metrics.snr_db >= thresholds.exact_snr_db, metrics


def percentile(values: Iterable[float], q: float) -> float:
    data = sorted(float(value) for value in values)
    if not data:
        return math.nan
    position = (len(data) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[lower]
    weight = position - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row.get("scope", "case"),
                row["batch_size"],
                row["actual_frames"],
                row["padded_frames"],
                row["pad_mode"],
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        summary: dict[str, Any] = {
            "scope": key[0],
            "batch_size": key[1],
            "actual_frames": key[2],
            "padded_frames": key[3],
            "pad_mode": key[4],
            "count": len(values),
        }
        for metric in (
            "max_abs",
            "mean_abs",
            "cosine",
            "snr_db",
            "mrstft_error",
            "log_mel_error",
            "boundary_max_abs",
            "boundary_click_delta_db",
        ):
            samples = [float(value[metric]) for value in values]
            summary[f"{metric}_p50"] = percentile(samples, 0.50)
            summary[f"{metric}_p95"] = percentile(samples, 0.95)
            summary[f"{metric}_p99"] = percentile(samples, 0.99)
            summary[f"{metric}_worst"] = min(samples) if metric in ("cosine", "snr_db") else max(samples)
        summaries.append(summary)
    return summaries


def production_gate(
    rows: Sequence[dict[str, Any]],
    *,
    thresholds: PaddingThresholds,
    optional_metrics_complete: bool,
    blind_complete: bool,
) -> dict[str, Any]:
    summaries = aggregate_rows(rows)
    failures: list[str] = []
    mode_failures: dict[str, list[str]] = defaultdict(list)
    approved_shapes: dict[str, list[list[int]]] = defaultdict(list)
    for summary in summaries:
        pad_mode = str(summary["pad_mode"])
        label = (
            f"B={summary['batch_size']},F={summary['actual_frames']},"
            f"P={summary['padded_frames']},mode={pad_mode}"
        )
        shape_failures: list[str] = []
        if summary["mean_abs_p99"] > thresholds.padded_p99_mean_abs:
            shape_failures.append(f"{label}: mean_abs_p99")
        if summary["max_abs_p99"] > thresholds.padded_p99_max_abs:
            shape_failures.append(f"{label}: max_abs_p99")
        if summary["max_abs_worst"] > thresholds.padded_worst_max_abs:
            shape_failures.append(f"{label}: max_abs_worst")
        if summary["cosine_worst"] < thresholds.padded_cosine:
            shape_failures.append(f"{label}: cosine")
        if summary["snr_db_worst"] < thresholds.padded_snr_db:
            shape_failures.append(f"{label}: snr")
        if summary["boundary_max_abs_p99"] > thresholds.boundary_p99_max_abs:
            shape_failures.append(f"{label}: boundary")
        if summary["boundary_click_delta_db_worst"] > thresholds.boundary_click_delta_db:
            shape_failures.append(f"{label}: boundary click delta")
        failures.extend(shape_failures)
        mode_failures[pad_mode].extend(shape_failures)
        if not shape_failures and summary["scope"] != "full":
            approved_shapes[pad_mode].append(
                [int(summary["batch_size"]), int(summary["actual_frames"]), int(summary["padded_frames"])]
            )
    optional_summary: dict[str, Any] = {}
    for row in rows:
        if row.get("exact_graph_max_abs") is None:
            continue
        relative_limit = max(
            thresholds.padded_p99_max_abs,
            2.0 * float(row["exact_graph_max_abs"]),
        )
        if float(row["max_abs"]) > relative_limit:
            pad_mode = str(row["pad_mode"])
            failure = f"case={row['case_id']},mode={pad_mode}: exceeds 2x exact-graph control"
            failures.append(failure)
            mode_failures[pad_mode].append(failure)
            shape = [int(row["batch_size"]), int(row["actual_frames"]), int(row["padded_frames"])]
            if shape in approved_shapes[pad_mode]:
                approved_shapes[pad_mode].remove(shape)
    if optional_metrics_complete:
        optional_modes = {str(row["pad_mode"]) for row in rows if row.get("pesq_drop") is not None}
        for pad_mode in sorted(optional_modes):
            mode_rows = [row for row in rows if row.get("pad_mode") == pad_mode]
            pesq_drops = [float(row["pesq_drop"]) for row in mode_rows if row.get("pesq_drop") is not None]
            stoi_drops = [float(row["stoi_drop"]) for row in mode_rows if row.get("stoi_drop") is not None]
            asr_deltas = [
                float(row["asr_error_rate_delta"])
                for row in mode_rows
                if row.get("asr_error_rate_delta") is not None
            ]
            speaker_drops = [
                float(row["speaker_cosine_drop"])
                for row in mode_rows
                if row.get("speaker_cosine_drop") is not None
            ]
            mode_summary = {
                "pesq_median_drop": percentile(pesq_drops, 0.50),
                "pesq_p95_drop": percentile(pesq_drops, 0.95),
                "stoi_median_drop": percentile(stoi_drops, 0.50),
                "stoi_p95_drop": percentile(stoi_drops, 0.95),
                "asr_mean_error_rate_delta": sum(asr_deltas) / len(asr_deltas),
                "speaker_median_drop": percentile(speaker_drops, 0.50),
                "speaker_p95_drop": percentile(speaker_drops, 0.95),
            }
            checks = (
                ("PESQ median drop", mode_summary["pesq_median_drop"], thresholds.pesq_median_drop),
                ("PESQ P95 drop", mode_summary["pesq_p95_drop"], thresholds.pesq_p95_drop),
                ("STOI median drop", mode_summary["stoi_median_drop"], thresholds.stoi_median_drop),
                ("STOI P95 drop", mode_summary["stoi_p95_drop"], thresholds.stoi_p95_drop),
                (
                    "ASR corpus error-rate delta",
                    mode_summary["asr_mean_error_rate_delta"],
                    thresholds.corpus_error_rate_delta,
                ),
                (
                    "speaker median cosine drop",
                    mode_summary["speaker_median_drop"],
                    thresholds.speaker_median_drop,
                ),
                (
                    "speaker P95 cosine drop",
                    mode_summary["speaker_p95_drop"],
                    thresholds.speaker_p95_drop,
                ),
            )
            for name, value, limit in checks:
                if value > limit:
                    failure = f"mode={pad_mode}: {name}"
                    failures.append(failure)
                    mode_failures[pad_mode].append(failure)
                    approved_shapes[pad_mode].clear()
            languages = {str(row["language"]) for row in mode_rows if row.get("language")}
            language_deltas: dict[str, float] = {}
            for language in languages:
                samples = [
                    float(row["asr_error_rate_delta"])
                    for row in mode_rows
                    if row.get("language") == language and row.get("asr_error_rate_delta") is not None
                ]
                if samples:
                    language_deltas[language] = sum(samples) / len(samples)
                    if language_deltas[language] > thresholds.subset_error_rate_delta:
                        failure = f"mode={pad_mode}: ASR subset error-rate delta: {language}"
                        failures.append(failure)
                        mode_failures[pad_mode].append(failure)
                        approved_shapes[pad_mode].clear()
            mode_summary["asr_language_error_rate_delta"] = language_deltas
            optional_summary[pad_mode] = mode_summary
    blockers: list[str] = []
    if not optional_metrics_complete:
        blockers.append("PESQ/STOI/ASR/speaker metrics incomplete")
    if not blind_complete:
        blockers.append("blind listening incomplete")
    modes = {str(row["pad_mode"]) for row in rows}
    passing_modes = sorted(mode for mode in modes if not mode_failures[mode])
    if "zero" in passing_modes:
        quality_status = "direct_pass"
    elif any(mode == "repeat_last" or mode.startswith("overlap_safe") for mode in passing_modes):
        quality_status = "conditional_pass"
    elif any(approved_shapes.values()):
        quality_status = "local_pass"
    else:
        quality_status = "fail"
    status = "fail" if quality_status == "fail" else "blocked" if blockers else quality_status
    return {
        "status": status,
        "quality_status": quality_status,
        "passing_modes": passing_modes,
        "failures": failures,
        "mode_failures": dict(mode_failures),
        "approved_shapes": dict(approved_shapes),
        "blockers": blockers,
        "summaries": summaries,
        "optional_metrics": optional_summary,
    }


def write_reports(output_dir: Path, rows: Sequence[dict[str, Any]], decision: dict[str, Any]) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "padding_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "padding_report.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(decision), handle, ensure_ascii=False, indent=2, allow_nan=False)


def save_wave(path: Path, waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = waveform.detach().cpu().float().reshape(-1).clamp(-1, 1)
    pcm = (samples * 32767.0).round().to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def export_blind_pairs(
    output_dir: Path,
    candidates: Sequence[dict[str, Any]],
    *,
    limit: int = 60,
    seed: int = 20260731,
) -> Path:
    """Export a deterministic mixture of worst-boundary, worst-global and normal pairs."""
    by_global = sorted(candidates, key=lambda item: float(item["metrics"].max_abs), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(items: Sequence[dict[str, Any]], count: int) -> None:
        for item in items:
            key = str(item["sample_id"])
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
            if len(selected) >= count:
                break

    categories = sorted({str(item.get("category", "normal")) for item in candidates})
    quota = max(1, limit // max(1, len(categories)))
    for category in categories:
        category_items = [item for item in by_global if item.get("category", "normal") == category]
        add(category_items, min(limit, len(selected) + quota))
    remaining = [item for item in candidates if str(item["sample_id"]) not in seen]
    random.Random(seed).shuffle(remaining)
    add(remaining, limit)

    blind_dir = output_dir / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = blind_dir / "blind_manifest.jsonl"
    rng = random.Random(seed)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(selected):
            sample_id = f"pair-{index:03d}"
            padded_side = rng.choice(("A", "B"))
            eager_side = "B" if padded_side == "A" else "A"
            save_wave(blind_dir / f"{sample_id}-{padded_side}.wav", item["padded"])
            save_wave(blind_dir / f"{sample_id}-{eager_side}.wav", item["eager"])
            public = {
                "sample_id": sample_id,
                "audio_a": f"{sample_id}-A.wav",
                "audio_b": f"{sample_id}-B.wav",
                "category": item.get("category", "normal"),
            }
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")
    # Write the answer key separately after the public manifest is complete.
    rng = random.Random(seed)
    answer_key = {f"pair-{index:03d}": rng.choice(("A", "B")) for index in range(len(selected))}
    with (blind_dir / "blind_key.json").open("w", encoding="utf-8") as handle:
        json.dump(answer_key, handle, ensure_ascii=False, indent=2)
    with (blind_dir / "blind_ratings_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "listener_id",
                "preferred",
                "distinguishable",
                "artifact_a",
                "artifact_b",
            ),
        )
        writer.writeheader()
        for index in range(len(selected)):
            writer.writerow({"sample_id": f"pair-{index:03d}"})
    with (blind_dir / "blind_ratings_template.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(len(selected)):
            handle.write(
                json.dumps(
                    {
                        "sample_id": f"pair-{index:03d}",
                        "listener_id": "",
                        "preferred": "tie",
                        "distinguishable": False,
                        "artifact_a": False,
                        "artifact_b": False,
                    }
                )
                + "\n"
            )
    return manifest_path


def _wilson_upper(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    if total <= 0:
        return 1.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center + spread) / denominator


def analyze_blind_results(results_path: Path, answer_key_path: Path) -> dict[str, Any]:
    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    listeners = {str(row["listener_id"]) for row in rows}
    samples = {str(row["sample_id"]) for row in rows}
    ratings_by_listener: dict[str, set[str]] = defaultdict(set)
    listeners_by_sample: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        listener = str(row["listener_id"])
        sample = str(row["sample_id"])
        ratings_by_listener[listener].add(sample)
        listeners_by_sample[sample].add(listener)
    complete_panel = all(len(value) >= 60 for value in ratings_by_listener.values()) and all(
        len(value) >= 12 for value in listeners_by_sample.values()
    )
    if len(listeners) < 12 or len(samples) < 60 or not complete_panel:
        return {
            "pass": False,
            "reason": (
                "need >=12 listeners, >=60 pairs, and every listener/pair fully rated; "
                f"got {len(listeners)} listeners and {len(samples)} pairs"
            ),
        }
    padded_worse = 0
    padded_artifacts = 0
    eager_artifacts = 0
    distinguishable = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        padded_side = answer_key[sample_id]
        eager_side = "B" if padded_side == "A" else "A"
        if str(row.get("preferred", "tie")).upper() == eager_side:
            padded_worse += 1
        padded_artifacts += int(bool(row.get(f"artifact_{padded_side.lower()}", False)))
        eager_artifacts += int(bool(row.get(f"artifact_{eager_side.lower()}", False)))
        distinguishable += int(bool(row.get("distinguishable", False)))
    upper = _wilson_upper(padded_worse, len(rows))
    artifact_delta = (padded_artifacts - eager_artifacts) / len(rows)
    return {
        "pass": upper <= 0.55 and artifact_delta <= 0.01,
        "ratings": len(rows),
        "listeners": len(listeners),
        "pairs": len(samples),
        "padded_worse_rate": padded_worse / len(rows),
        "padded_worse_one_sided_95_upper": upper,
        "artifact_rate_delta": artifact_delta,
        "distinguishable_rate": distinguishable / len(rows),
    }


def load_scorer(spec: str | None) -> Callable[..., Any] | None:
    """Load a local scorer factory/callable from ``module:attribute``."""
    if not spec:
        return None
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError(f"Scorer must use module:attribute syntax, got {spec!r}")
    scorer = getattr(importlib.import_module(module_name), attribute)
    if not callable(scorer):
        raise TypeError(f"Scorer {spec!r} is not callable")
    return scorer


def metrics_row(
    *,
    case_id: str,
    batch_size: int,
    actual_frames: int,
    padded_frames: int,
    pad_mode: str,
    metrics: WaveMetrics,
    seed: int | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "batch_size": batch_size,
        "actual_frames": actual_frames,
        "padded_frames": padded_frames,
        "padding_frames": padded_frames - actual_frames,
        "pad_mode": pad_mode,
        "seed": seed,
        **asdict(metrics),
    }
