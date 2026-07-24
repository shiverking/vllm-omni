"""Platform-independent interaction state and urgency policy for streaming audio."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence


DEFAULT_PLAYBACK_SAFE_BUFFER_MS = 100.0
DEFAULT_INTERACTION_STATE_TTL_MS = 500.0


class AudioUrgency(IntEnum):
    """Lower values are scheduled first."""

    U0 = 0  # playback started and is close to underrun
    U1 = 1  # waiting for the first playable audio
    U2 = 2  # playback has a safe buffer
    FALLBACK = 3  # state is missing, stale, or terminal


@dataclass
class AudioInteractionState:
    request_id: str
    first_audio_received: bool = False
    received_audio_ms: float = 0.0
    played_audio_ms: float = 0.0
    playback_buffer_ms: float = 0.0
    last_update_monotonic_s: float = 0.0
    ready_since_monotonic_s: float | None = None
    finished: bool = False
    aborted: bool = False

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.ready_since_monotonic_s is None:
            self.ready_since_monotonic_s = self.last_update_monotonic_s
        self._validate_audio_positions(self.received_audio_ms, self.played_audio_ms)
        self.playback_buffer_ms = self.received_audio_ms - self.played_audio_ms

    @staticmethod
    def _validate_audio_positions(received_audio_ms: float, played_audio_ms: float) -> None:
        if received_audio_ms < 0 or played_audio_ms < 0:
            raise ValueError("received_audio_ms and played_audio_ms must be non-negative")
        if played_audio_ms > received_audio_ms:
            raise ValueError("played_audio_ms must not exceed received_audio_ms")

    def apply_feedback(
        self,
        *,
        received_audio_ms: float,
        played_audio_ms: float,
        first_audio_received: bool,
        now_monotonic_s: float,
        finished: bool = False,
        aborted: bool = False,
    ) -> None:
        """Apply a monotonic client update or raise ``ValueError``."""
        self._validate_audio_positions(received_audio_ms, played_audio_ms)
        if now_monotonic_s < self.last_update_monotonic_s:
            raise ValueError("feedback monotonic timestamp moved backwards")
        if received_audio_ms < self.received_audio_ms or played_audio_ms < self.played_audio_ms:
            raise ValueError("audio positions must move monotonically")
        if self.first_audio_received and not first_audio_received:
            raise ValueError("first_audio_received cannot move backwards")
        if self.finished and not finished:
            raise ValueError("finished cannot move backwards")
        if self.aborted and not aborted:
            raise ValueError("aborted cannot move backwards")

        self.received_audio_ms = received_audio_ms
        self.played_audio_ms = played_audio_ms
        self.playback_buffer_ms = received_audio_ms - played_audio_ms
        self.first_audio_received = first_audio_received
        self.last_update_monotonic_s = now_monotonic_s
        self.finished = finished
        self.aborted = aborted

    def is_stale(self, *, now_monotonic_s: float, ttl_ms: float = DEFAULT_INTERACTION_STATE_TTL_MS) -> bool:
        if ttl_ms < 0:
            raise ValueError("ttl_ms must be non-negative")
        return (now_monotonic_s - self.last_update_monotonic_s) * 1000.0 > ttl_ms

    def urgency(
        self,
        *,
        now_monotonic_s: float,
        safe_buffer_ms: float = DEFAULT_PLAYBACK_SAFE_BUFFER_MS,
        ttl_ms: float = DEFAULT_INTERACTION_STATE_TTL_MS,
    ) -> AudioUrgency:
        if safe_buffer_ms < 0:
            raise ValueError("safe_buffer_ms must be non-negative")
        if self.finished or self.aborted or self.is_stale(now_monotonic_s=now_monotonic_s, ttl_ms=ttl_ms):
            return AudioUrgency.FALLBACK
        if not self.first_audio_received:
            return AudioUrgency.U1
        if self.playback_buffer_ms < safe_buffer_ms:
            return AudioUrgency.U0
        return AudioUrgency.U2


def rank_audio_requests(
    request_ids: Sequence[str],
    states: Mapping[str, AudioInteractionState],
    *,
    now_monotonic_s: float,
    safe_buffer_ms: float = DEFAULT_PLAYBACK_SAFE_BUFFER_MS,
    ttl_ms: float = DEFAULT_INTERACTION_STATE_TTL_MS,
) -> list[str]:
    """Rank ready requests as U0, U1, U2, then stable FCFS fallback."""

    def sort_key(item: tuple[int, str]) -> tuple[float, float, int]:
        fcfs_index, request_id = item
        state = states.get(request_id)
        if state is None:
            return float(AudioUrgency.FALLBACK), 0.0, fcfs_index
        urgency = state.urgency(
            now_monotonic_s=now_monotonic_s,
            safe_buffer_ms=safe_buffer_ms,
            ttl_ms=ttl_ms,
        )
        if urgency in (AudioUrgency.U0, AudioUrgency.U2):
            secondary = state.playback_buffer_ms
        elif urgency == AudioUrgency.U1:
            secondary = state.ready_since_monotonic_s or 0.0
        else:
            secondary = 0.0
        return float(urgency), secondary, fcfs_index

    return [request_id for _, request_id in sorted(enumerate(request_ids), key=sort_key)]
