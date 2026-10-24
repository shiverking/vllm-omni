# SPDX-License-Identifier: Apache-2.0
"""Opt-in chunk recording, outside graph capture and performance measurements."""

import itertools
import json
import os
import time
from pathlib import Path

import torch

_SEQUENCE = itertools.count()
_SESSION = time.time_ns()


def record_decode(directory, requests, waves, finished, external_ids, sample_rate):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    seq = next(_SEQUENCE)
    rows = [
        dict(
            request_id=key,
            external_id=external,
            codes=codes.detach().cpu().clone(),
            wav=wav.detach().cpu().clone(),
            finished=bool(end),
        )
        for (key, codes), wav, end, external in zip(requests, waves, finished, external_ids, strict=True)
    ]
    path = root / f"chunk-{_SESSION}-{os.getpid()}-{seq:08d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(dict(sequence=seq, sample_rate=sample_rate, rows=rows), tmp)
    tmp.replace(path)


def record_batch(owner, directory, lengths=(), force=False):
    from collections import Counter

    if not getattr(owner, "_code2wav_exit_flush_registered", False):
        import atexit
        import weakref

        reference = weakref.ref(owner)

        def final_flush():
            live = reference()
            if live is not None:
                record_batch(live, directory, force=True)

        atexit.register(final_flush)
        owner._code2wav_exit_flush_registered = True
    counters = getattr(owner, "_code2wav_research_batches", None)
    if counters is None:
        counters = Counter()
        owner._code2wav_research_batches = counters
    if lengths:
        counters[(len(lengths), tuple(lengths))] += 1
    now = time.monotonic()
    if not force and now - getattr(owner, "_code2wav_stats_last_flush", 0) < 5:
        return
    owner._code2wav_stats_last_flush = now
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    wrapper = getattr(owner.decoder, "_npugraph_wrapper", None)
    graph_stats = wrapper.flush_stats() if wrapper else None
    data = dict(
        owner="inner_code2wav" if wrapper else "baseline",
        batches=[dict(batch=b, frames=list(ns), calls=n) for (b, ns), n in sorted(counters.items())],
        graph=graph_stats,
    )
    path = root / f"batches-{os.getpid()}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def record_finished(directory, request_ids):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"finished-{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"request_ids": sorted(request_ids)}) + "\n")
