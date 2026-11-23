# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned PCM accumulation for a stage that decodes audio in place."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY


@dataclass
class AudioRequest:
    samples_per_frame: int
    chunk_frames: int
    active: bool = True
    emitted: bool = False
    direct_first_pending: bool = False
    first_audio_required: bool = False
    pending: list[torch.Tensor | np.ndarray] = field(default_factory=list)
    pending_samples: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def accept_first_audio(self) -> None:
        with self.lock:
            if self.active:
                self.direct_first_pending = True

    def push(self, pcm: torch.Tensor | np.ndarray, ended: bool) -> torch.Tensor | None:
        with self.lock:
            if not self.active:
                return None
            size = pcm.numel() if isinstance(pcm, torch.Tensor) else pcm.size
            if self.direct_first_pending:
                self.direct_first_pending = False
                if size:
                    pcm = pcm[self.samples_per_frame :]
                    size = max(0, size - self.samples_per_frame)
                    self.emitted = True
                    self.first_audio_required = True
            if size:
                self.pending.append(pcm)
                self.pending_samples += size
            frames = self.pending_samples // self.samples_per_frame
            out = None
            if frames and (ended or not self.emitted or frames >= self.chunk_frames):
                if all(isinstance(part, np.ndarray) for part in self.pending):
                    array = np.concatenate(self.pending) if len(self.pending) > 1 else self.pending[0]
                    out = torch.from_numpy(array)
                else:
                    parts = [torch.from_numpy(part) if isinstance(part, np.ndarray) else part for part in self.pending]
                    out = torch.cat(parts) if len(parts) > 1 else parts[0]
                self.pending.clear()
                self.pending_samples = 0
                self.emitted = True
            if ended:
                self.active = False
                self.pending.clear()
                self.pending_samples = 0
            return out

    def finish(self) -> None:
        with self.lock:
            self.active = False
            self.pending.clear()
            self.pending_samples = 0


class StreamingAudioBuffer:
    def __init__(self, samples_per_frame: int, chunk_frames: int) -> None:
        self.samples_per_frame = samples_per_frame
        self.chunk_frames = chunk_frames
        self.requests: dict[str, AudioRequest] = {}

    def add(self, request_id: str) -> AudioRequest:
        if request_id not in self.requests:
            self.requests[request_id] = AudioRequest(self.samples_per_frame, self.chunk_frames)
        return self.requests[request_id]

    def finish(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            state = self.requests.pop(request_id, None)
            if state is not None:
                # Already queued outputs retain this object, never a fresh lookup
                # by ID. A late copy cannot resurrect a cancelled request.
                state.finish()


class ChunkDecoder(Protocol):
    first_chunk_frames: int
    chunk_frames: int
    sample_rate: int

    def new_cache(self) -> dict[str, Any]: ...
    def prepare_reference(self, cache: dict[str, Any], ref_codes: torch.Tensor, cache_key: str | None) -> None: ...
    def release_cache(self, cache: dict[str, Any]) -> None: ...
    def decode_chunks(self, codes: list[torch.Tensor], caches: list[dict[str, Any]]) -> list[torch.Tensor]: ...


@dataclass
class PendingCodecRequest:
    cache: dict[str, Any]
    release_cache: Callable[[dict[str, Any]], None] | None = None
    frames: list[torch.Tensor] = field(default_factory=list)
    started: bool = False


class ChunkedAudioBuffer:
    """Accumulate one frame per active request, then batch complete chunks."""

    def __init__(self) -> None:
        self.requests: dict[str, PendingCodecRequest] = {}

    def discard(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            state = self.requests.pop(req_id, None)
            if state is not None and state.release_cache is not None:
                state.release_cache(state.cache)

    def decode(
        self,
        decoder: ChunkDecoder,
        frames: dict[str, torch.Tensor],
        ended: set[str],
        references: dict[str, tuple[torch.Tensor, str | None]] | None = None,
    ) -> dict[str, torch.Tensor]:
        for req_id, frame in frames.items():
            state = self.requests.get(req_id)
            if state is None:
                state = PendingCodecRequest(decoder.new_cache(), getattr(decoder, "release_cache", None))
                if references and req_id in references:
                    decoder.prepare_reference(state.cache, *references[req_id])
                self.requests[req_id] = state
            state.frames.append(frame)

        ready = []
        for req_id in dict.fromkeys((*frames, *sorted(ended))):
            state = self.requests.get(req_id)
            if state is None or not state.frames:
                continue
            threshold = decoder.chunk_frames if state.started else decoder.first_chunk_frames
            if len(state.frames) == threshold or req_id in ended:
                ready.append((req_id, state))
        if not ready:
            self.discard(ended)
            return {}

        try:
            codes = [torch.stack(state.frames, dim=-1).unsqueeze(0) for _, state in ready]
            waves = decoder.decode_chunks(codes, [state.cache for _, state in ready])
            result = {}
            for (req_id, state), wav in zip(ready, waves, strict=True):
                state.frames.clear()
                state.started = True
                if wav.numel():
                    result[req_id] = wav.reshape(-1).float()
            return result
        except BaseException:
            # A partially executed decoder batch may have modified these caches.
            self.discard(req_id for req_id, _ in ready)
            raise
        finally:
            self.discard(ended)


@dataclass
class ChunkedAudioOutput:
    """Already partitioned PCM, aligned by request order for the native output."""

    chunks: list[torch.Tensor | None]
    sample_rate: torch.Tensor

    def to_cpu(
        self,
        copy_stream: Any,
        copy_tensor: Callable[[torch.Tensor], torch.Tensor],
    ) -> ChunkedAudioOutput:
        return ChunkedAudioOutput(
            [copy_tensor(chunk) if chunk is not None else None for chunk in self.chunks], self.sample_rate
        )

    def get_output(self) -> list[dict[str, torch.Tensor] | None]:
        return [
            {"model_outputs": chunk, "sr": self.sample_rate} if chunk is not None else None for chunk in self.chunks
        ]


@dataclass
class StreamingAudioOutput:
    wav: torch.Tensor
    valid: torch.Tensor
    sample_rate: torch.Tensor
    event: torch.cuda.Event | None
    query_start_loc: np.ndarray
    requests: list[AudioRequest | None]
    length_end: np.ndarray

    def to_cpu(
        self,
        copy_stream: torch.cuda.Stream,
        copy_tensor: Callable[[torch.Tensor], torch.Tensor],
    ) -> StreamingAudioOutput:
        if self.event is not None:
            copy_stream.wait_event(self.event)
        return StreamingAudioOutput(
            copy_tensor(self.wav),
            copy_tensor(self.valid),
            self.sample_rate,
            None,
            self.query_start_loc,
            self.requests,
            self.length_end,
        )

    def get_output(self) -> list[dict[str, torch.Tensor] | None]:
        valid = self.valid.numpy().astype(bool)
        # Each D2H batch owns its host allocation. NumPy slices retain that
        # backing tensor while a request accumulates PCM across steps; no
        # per-request Torch views are needed until an audio chunk is emitted.
        # Keep the tensor path for dtypes NumPy cannot represent (e.g. BF16).
        wav = self.wav.numpy() if self.wav.dtype in (torch.float16, torch.float32, torch.float64) else None
        output: list[dict[str, torch.Tensor] | None] = []
        required_marker = None
        for i, state in enumerate(self.requests):
            payload = None
            start, end = self.query_start_loc[i : i + 2]
            if state is not None and end > start:
                if wav is not None:
                    if end - start == 1:
                        pcm = wav[start:end].reshape(-1) if valid[start] else wav[0:0].reshape(-1)
                    else:
                        pcm = wav[start:end][valid[start:end]].reshape(-1)
                elif end - start == 1:
                    pcm = self.wav[start:end].reshape(-1) if valid[start] else self.wav.new_empty(0)
                else:
                    pcm = self.wav[start:end][torch.from_numpy(valid[start:end])].reshape(-1)
                ended = bool(self.length_end[i]) or not valid[end - 1]
                chunk = state.push(pcm, ended)
                if chunk is not None:
                    payload = {"model_outputs": chunk, "sr": self.sample_rate}
                if state.first_audio_required and (chunk is not None or ended):
                    if required_marker is None:
                        required_marker = torch.tensor(True)
                    if payload is None:
                        payload = {}
                    payload[FIRST_AUDIO_REQUIRED_KEY] = required_marker
            output.append(payload)
        return output
