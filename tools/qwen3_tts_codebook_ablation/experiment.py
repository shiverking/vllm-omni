# SPDX-License-Identifier: Apache-2.0
"""Listen to prefix-codebook ablations of the SAME generated codec sequence."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import shutil
import sys
import time
import wave
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def retain_codebooks(quantizer, keep):
    """Slice before RVQ summation, never replace removed indices with token zero."""
    if not 1 <= keep <= quantizer.max_n_q:
        raise ValueError(f"keep must be in [1, {quantizer.max_n_q}]")
    original = quantizer.decode
    had_override = "decode" in quantizer.__dict__
    old_override = quantizer.__dict__.get("decode")
    quantizer.decode = lambda codes: original(codes[:, :keep])
    try:
        yield
    finally:
        if had_override:
            quantizer.decode = old_override
        else:
            del quantizer.decode


def read_request(root, marker):
    meta = json.loads(marker.read_text(encoding="utf-8"))
    code_path = (root / meta["codes"]).resolve()
    audio_path = (root / meta["audio"]).resolve()
    for path, key in ((code_path, "codes_sha256"), (audio_path, "audio_sha256")):
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Input escapes capture root: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta[key]:
            raise ValueError(f"Incomplete/changed source: {path}")
    rows = [json.loads(x) for x in code_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    valid = []
    for i, row in enumerate(rows):
        if row["frame_id"] != i or row["request_id"] != meta["request"]["id"]:
            raise ValueError("Frame order/request mismatch")
        if row.get("attempt") != meta["attempt"]:
            raise ValueError("Mixed capture attempts")
        codes = row["codes"]
        if len(codes) != 16 or any(type(c) is not int for c in codes):
            raise ValueError("Expected 16 integer codes")
        expected = all(0 <= c < 2048 for c in codes) and any(codes)
        if bool(row["valid"]) != expected:
            raise ValueError("Frame validity mismatch")
        if expected:
            valid.append(codes)
    if not valid:
        raise ValueError("No valid frames")
    return meta, torch.tensor(valid, dtype=torch.long).T.unsqueeze(0), audio_path


@torch.inference_mode()
def decode_audio(model, codes, keep):
    # Match the fixed CustomVoice chunk schedule, with independent state per variant.
    cache = {"prefix_frames": 0}
    pieces = []
    with retain_codebooks(model.quantizer, keep):
        start = 0
        while start < codes.shape[-1]:
            end = min(start + (1 if start == 0 else 25), codes.shape[-1])
            piece = model(codes[..., start:end], caches=cache)
            pieces.append(piece.detach().float().cpu().reshape(-1))
            start = end
    audio = torch.cat(pieces)
    if not torch.isfinite(audio).all():
        raise ValueError(f"Nonfinite waveform at keep={keep}")
    return audio


def save_wav(path, audio, rate, gain):
    pcm = (audio.mul(gain).clamp(-1, 1) * 32767).round().to(torch.int16).numpy().astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())


def write_page(root, records):
    sections = []
    for record in records:
        request = record["request"]
        players = []
        for variant in record["variants"]:
            players.append(
                f"<p>保留 {variant['keep']} 个码本（c0～c{variant['keep'] - 1}） "
                f'<audio controls preload="none" src="{html.escape(variant["file"], quote=True)}"></audio></p>'
            )
        sections.append(
            f"<section><h2>{html.escape(request['id'])}</h2><p>{html.escape(request['text'])}</p>"
            f'<p>原服务音频 <audio controls preload="none" src="{record["original"]}"></audio></p>'
            + "".join(players)
            + "</section>"
        )
    page = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>码本删减试听</title>'
        "<style>body{max-width:1000px;margin:30px auto;font-family:sans-serif}section{border-top:1px solid #aaa}"
        "audio{vertical-align:middle;width:400px}</style><h1>同一段 codes：尾部码本删减试听</h1>"
        "<p>16 是重解码对照。各删减版本使用相同 codes、解码设置及同一增益；原服务音频保持原样。"
        "本实验仅评估听感，不测提前停止 AR 的实际速度。</p>" + "".join(sections) + "</html>"
    )
    (root / "listen.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="复用真实生成 codes，导出尾部码本删减试听 WAV")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", default="/softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--keep", type=int, nargs="+", default=[16, 14, 12, 8, 4, 1])
    parser.add_argument("--limit", type=int, default=20, help="独立文本数量；0 表示全部")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-ids", nargs="+", help="指定请求 ID，覆盖 seed/limit 选择")
    args = parser.parse_args()
    if any(k < 1 or k > 16 for k in args.keep) or args.limit < 0:
        parser.error("keep 范围 1～16，limit 不得为负")
    args.keep = sorted(set([16, *args.keep]), reverse=True)
    if args.results.resolve().is_relative_to(args.input.resolve()):
        parser.error("results 必须位于原采集目录之外")
    selected = []
    seen = set()
    for marker in sorted((args.input / "completed").glob("*.json")):
        request = json.loads(marker.read_text(encoding="utf-8"))["request"]
        if args.request_ids:
            if request["id"] not in args.request_ids:
                continue
        elif request["seed"] != args.seed or request["text_id"] in seen:
            continue
        selected.append(marker)
        seen.add(request["text_id"])
        if not args.request_ids and args.limit and len(selected) >= args.limit:
            break
    if not selected:
        parser.error("没有找到匹配的 completed 请求；输入应为原码本统计采集目录")
    if args.request_ids:
        found = {json.loads(p.read_text(encoding="utf-8"))["request"]["id"] for p in selected}
        if set(args.request_ids) != found:
            parser.error(f"未找到请求: {sorted(set(args.request_ids) - found)}")
    args.results.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "tools/qwen3_tts_code2wav_graph"))
    from engine import load_decoder

    print(f"加载 Code2Wav；请求 {len(selected)}，保留码本 {args.keep}", flush=True)
    model, rate = load_decoder(args)
    records = []
    started = time.monotonic()
    for i, marker in enumerate(selected):
        meta, codes, original = read_request(args.input, marker)
        identifier = meta["request"]["id"]
        if Path(identifier).name != identifier or identifier in (".", ".."):
            raise ValueError("Unsafe request ID")
        folder = args.results / identifier
        folder.mkdir(exist_ok=True)
        codes = codes.to(args.device)
        variants = []
        audios = {}
        for keep in args.keep:
            print(f"[{i + 1}/{len(selected)}] {identifier} 保留 {keep} 个码本", flush=True)
            audios[keep] = decode_audio(model, codes, keep)
        if len({x.numel() for x in audios.values()}) != 1:
            raise ValueError("Variant audio lengths differ")
        peak = max(float(x.abs().max()) for x in audios.values())
        gain = min(1.0, 0.99 / peak) if peak else 1.0
        for keep, audio in audios.items():
            path = folder / f"keep{keep:02d}.wav"
            save_wav(path, audio, rate, gain)
            variants.append(
                dict(
                    keep=keep,
                    file=path.relative_to(args.results).as_posix(),
                    samples=audio.numel(),
                    raw_peak=float(audio.abs().max()),
                )
            )
        shutil.copyfile(original, folder / "original.wav")
        record = dict(
            request=meta["request"],
            valid_frames=codes.shape[-1],
            sample_rate=rate,
            shared_gain=gain,
            original=f"{identifier}/original.wav",
            variants=variants,
            source_codes_sha256=meta["codes_sha256"],
        )
        (folder / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
        manifest = dict(
            model=args.model,
            device=args.device,
            keep=args.keep,
            source=str(args.input.resolve()),
            decoder_schedule=[1, 25],
            records=records,
            note="Same full codes re-decoded with truncated RVQ contributions; no AR speed measurement",
        )
        (args.results / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_page(args.results, records)
        elapsed = time.monotonic() - started
        eta = elapsed / (i + 1) * (len(selected) - i - 1)
        print(
            f"完成 {i + 1}/{len(selected)}；耗时 {elapsed:.1f}s；ETA {eta:.1f}s",
            flush=True,
        )
    print(f"试听页面：{args.results / 'listen.html'}", flush=True)


if __name__ == "__main__":
    main()
