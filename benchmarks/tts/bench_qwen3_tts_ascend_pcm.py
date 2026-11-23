# SPDX-License-Identifier: Apache-2.0
"""Measure streaming PCM delivery on an already running Qwen3-TTS service."""

import argparse
import array
import http.client
import json
import math
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SAMPLE_RATE = 24_000
PCM_THRESHOLD = 328  # -40 dBFS peak threshold for 16-bit PCM.


def request(args, item):
    sample_id, prompt = item
    request_body = {
        "model": args.model,
        "input": prompt,
        "voice": args.voice,
        "language": args.language,
        "task_type": args.task_type,
        "seed": args.seed,
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    req = urllib.request.Request(
        args.url, data=json.dumps(request_body).encode(), headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.perf_counter()
    arrival = []
    onset = audible = audible_packet = None
    samples = 0
    playout_end = 0.0
    underrun = 0.0
    pending = b""
    try:
        with opener.open(req, timeout=args.timeout) as response:
            if response.headers.get_content_type() not in {"audio/pcm", "application/octet-stream"}:
                raise ValueError(f"unexpected PCM response: {response.headers.get_content_type()}")
            while data := response.read1(65_536):
                now = time.perf_counter() - started
                data = pending + data
                complete_bytes = len(data) & ~1
                pending = data[complete_bytes:]
                if complete_bytes == 0:
                    continue
                if arrival:
                    underrun += max(0.0, now - playout_end)
                arrival.append(now)
                playout_start = max(now, playout_end)
                if onset is None:
                    pcm = array.array("h")
                    pcm.frombytes(data[:complete_bytes])
                    if sys.byteorder != "little":
                        pcm.byteswap()
                    for index, value in enumerate(pcm):
                        if abs(value) >= PCM_THRESHOLD:
                            onset = (samples + index) / SAMPLE_RATE
                            audible_packet = now
                            audible = playout_start + index / SAMPLE_RATE
                            break
                new_samples = complete_bytes // 2
                samples += new_samples
                playout_end = playout_start + new_samples / SAMPLE_RATE
        if pending or not samples or onset is None:
            raise ValueError(f"invalid PCM: samples={samples} trailing_bytes={len(pending)} onset={onset}")
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return {"id": sample_id, "error": str(exc), "samples": samples}

    duration = samples / SAMPLE_RATE
    elapsed = time.perf_counter() - started
    return {
        "id": sample_id,
        "audio_s": duration,
        "e2e_s": elapsed,
        "rtf": elapsed / duration,
        "first_packet_s": arrival[0],
        "audible_packet_s": audible_packet,
        "estimated_playback_onset_s": audible,
        "onset_in_pcm_s": onset,
        "playback_underrun_s": underrun,
        "packet_gaps_s": [later - earlier for earlier, later in zip(arrival, arrival[1:])],
    }


def percentiles(values):
    if not values:
        return None
    return {"p50": statistics.median(values), "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1]}


def run(args, texts, concurrency):
    def send(item):
        return request(args, item)

    print(f"concurrency={concurrency} warmup={args.warmup} requests={args.requests}", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        warmup = list(pool.map(send, [texts[i % len(texts)] for i in range(args.warmup)]))
        start = time.perf_counter()
        futures = [pool.submit(send, texts[i % len(texts)]) for i in range(args.requests)]
        for index, _ in enumerate(as_completed(futures), start=1):
            if index == args.requests or index % 10 == 0:
                elapsed = time.perf_counter() - start
                eta = elapsed * (args.requests - index) / index
                print(
                    f"concurrency={concurrency} {index}/{args.requests} elapsed={elapsed:.1f}s ETA~{eta:.1f}s",
                    flush=True,
                )
        wall = time.perf_counter() - start
        rows = [dict(future.result(), request_index=index) for index, future in enumerate(futures, start=1)]

    passed = [row for row in rows if "error" not in row]
    total_audio_duration = sum(row["audio_s"] for row in passed)
    metrics = (
        "first_packet_s",
        "audible_packet_s",
        "estimated_playback_onset_s",
        "onset_in_pcm_s",
        "e2e_s",
        "rtf",
        "playback_underrun_s",
    )
    return {
        "concurrency": concurrency,
        "task_type": args.task_type,
        "voice": args.voice,
        "language": args.language,
        "seed": args.seed,
        "success": len(passed),
        "error_rate": 1 - len(passed) / len(rows),
        "requests_per_s": len(passed) / wall,
        "audio_s_per_s": total_audio_duration / wall,
        "total_audio_duration_s": total_audio_duration,
        "packet_gap_s": percentiles([gap for row in passed for gap in row["packet_gaps_s"]]),
        **{metric: percentiles([row[metric] for row in passed]) for metric in metrics},
        "warmup_errors": [row for row in warmup if "error" in row],
        "requests": rows,
    }


def read_texts(path):
    texts = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # The smoke format leaves the two Seed-TTS reference fields empty.
        fields = line.split("|", 3)
        if len(fields) != 4 or not fields[0].strip() or not fields[3].strip():
            raise ValueError(f"{path}:{line_number}: expected id|||prompt or id|ref_text|ref_wav|target_text")
        texts.append((fields[0].strip(), fields[3]))
    if not texts:
        raise ValueError(f"{path}: no text samples")
    return texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091/v1/audio/speech")
    parser.add_argument("--model", default="Qwen3-TTS")
    parser.add_argument("--task-type", choices=("CustomVoice", "Base"), default="CustomVoice")
    parser.add_argument("--voice", default="Vivian", help="Built-in speaker or registered Base voice name")
    parser.add_argument("--language", default="English")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--texts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build_dataset/seed_tts_smoke/en/meta.lst",
        help="Text list in id|||prompt or Seed-TTS meta.lst format",
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 16, 32])
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0 or min(args.concurrency) < 1 or args.timeout <= 0:
        parser.error("requests, concurrency and timeout must be positive; warmup must be nonnegative")
    try:
        texts = read_texts(args.texts)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for concurrency in args.concurrency:
        results.append(run(args, texts, concurrency))
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
