# SPDX-License-Identifier: Apache-2.0
"""Generate Chinese audio and capture the actual Code2Wav input for ablation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
import uuid
import wave
from pathlib import Path

import torch
from experiment import read_request

ROOT = Path(__file__).resolve().parents[2]
TEXTS = [
    "你好，欢迎收听今天的语音测试。希望你度过轻松愉快的一天。",
    "清晨，阳光穿过窗帘，落在桌上的一杯热茶旁。窗外传来几声清脆的鸟鸣。",
    "这是十四，还是四十？是三十三，还是四十三？请仔细听清每一个数字。",
    "小溪穿过竹林，细细的水声伴着沙沙的风声，四周显得格外安静。",
    "今天的订单总额是一千二百三十四元五角六分，预计十月二十五日下午三点半送达。",
    "你真的决定了吗？如果现在出发，我们还能赶上最后一班地铁吗？",
    "太好了，我们终于成功了！这一路虽然不容易，但大家的努力都没有白费。",
    "别着急，先深呼吸，再慢慢说。无论遇到什么困难，我们都可以一起想办法。",
    "吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。四是四，十是十，十四是十四。",
    "妈妈轻轻地说，早点休息吧。孩子点了点头，把床头的灯调暗了一些。",
    "请将设备连接到无线网络，然后打开设置页面，检查音量、电量和系统版本。",
    "周末，我们沿着山路走了很久。起初天空晴朗，后来云渐渐多了起来，远处传来了雷声。"
    "大家加快脚步，在下雨之前赶到了山脚的小屋。屋里很暖和，主人端来了热汤，大家这才松了一口气。",
]


def captured_rows(paths, request, attempt):
    artifacts = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    artifacts.sort(key=lambda d: d["sequence"])
    ids = {row["request_id"] for artifact in artifacts for row in artifact["rows"]}
    if not artifacts or len(ids) != 1 or any(len(a["rows"]) != 1 for a in artifacts):
        raise ValueError("单请求采集未得到唯一对应的 Code2Wav 序列")
    result = []
    for artifact in artifacts:
        codes = artifact["rows"][0]["codes"]
        if codes.ndim != 2 or codes.shape[0] != 16:
            raise ValueError(f"Expected [16, T] codec input, got {tuple(codes.shape)}")
        for values in codes.T.tolist():
            valid = all(type(c) is int and 0 <= c < 2048 for c in values) and any(values)
            result.append(
                dict(request_id=request["id"], frame_id=len(result), codes=values, valid=valid, attempt=attempt)
            )
    if not any(r["valid"] for r in result):
        raise ValueError("没有有效码本帧")
    return result


def main():
    sys.path.insert(0, str(ROOT / "tools/qwen3_tts_code2wav_graph"))
    from runtime import Progress, common_args, direct_open, server, write_json

    parser = argparse.ArgumentParser(description="中文真实音频生成＋码本采集，供尾部码本删减试听")
    common_args(parser)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > len(TEXTS):
        parser.error(f"limit 范围 1～{len(TEXTS)}")
    root = args.results.resolve()
    for folder in ("audio", "codes", "completed", "raw"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    manifest = dict(model=args.model, language="Chinese", voice="Vivian", seed=args.seed, texts=TEXTS)
    manifest_path = root / "generation.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("此目录已有不同生成配置，请为本次实验选择独立 results")
    write_json(manifest_path, manifest)
    progress = Progress(root)
    with server(args, "baseline", root / "server", progress, capture=root / "raw") as url:
        for i, text in enumerate(TEXTS[: args.limit]):
            request = dict(
                id=f"zh{i + 1:03d}-s{args.seed}",
                text_id=f"zh{i + 1:03d}",
                text=text,
                seed=args.seed,
                language="Chinese",
            )
            marker = root / "completed" / f"{request['id']}.json"
            if marker.exists():
                read_request(root, marker)
                progress.update(i + 1, args.limit, "中文采集（跳过已完成）")
                continue
            started = time.monotonic()
            before = set((root / "raw").glob("*.pt"))
            payload = dict(
                model=args.model,
                input=text,
                voice="Vivian",
                language="Chinese",
                task_type="CustomVoice",
                seed=args.seed,
                response_format="wav",
            )
            req = urllib.request.Request(
                url + "/v1/audio/speech",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with direct_open(req, timeout=600) as response:
                audio = response.read()
            with wave.open(io.BytesIO(audio)) as stream:
                rate, samples = stream.getframerate(), stream.getnframes()
            paths = set((root / "raw").glob("*.pt")) - before
            attempt = uuid.uuid4().hex
            rows = captured_rows(paths, request, attempt)
            audio_path = root / "audio" / f"{request['id']}.wav"
            codes_path = root / "codes" / f"{request['id']}.jsonl"
            audio_path.write_bytes(audio)
            codes_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            write_json(
                marker,
                dict(
                    request=request,
                    attempt=attempt,
                    sample_rate=rate,
                    duration_s=samples / rate,
                    audio=audio_path.relative_to(root).as_posix(),
                    codes=codes_path.relative_to(root).as_posix(),
                    audio_sha256=hashlib.sha256(audio).hexdigest(),
                    codes_sha256=hashlib.sha256(codes_path.read_bytes()).hexdigest(),
                    generation_s=time.monotonic() - started,
                    capture="actual Code2Wav input; predictor kv_frame",
                ),
            )
            progress.update(i + 1, args.limit, "中文真实音频与码本采集")
    progress.log("中文采集完成；用 experiment.py 对此目录导出各保留码本版本。")


if __name__ == "__main__":
    main()
