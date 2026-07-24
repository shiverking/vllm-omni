from __future__ import annotations

import pytest

from vllm_omni.core.sched.audio_interaction import AudioInteractionState, AudioUrgency, rank_audio_requests


def _state(
    request_id: str,
    *,
    first: bool,
    received: float = 0,
    played: float = 0,
    updated: float = 10.0,
    ready: float | None = None,
) -> AudioInteractionState:
    return AudioInteractionState(
        request_id=request_id,
        first_audio_received=first,
        received_audio_ms=received,
        played_audio_ms=played,
        last_update_monotonic_s=updated,
        ready_since_monotonic_s=ready,
    )


def test_urgency_classification_and_safe_buffer_boundary():
    assert _state("u1", first=False).urgency(now_monotonic_s=10.1) == AudioUrgency.U1
    assert _state("u0", first=True, received=150, played=60).urgency(now_monotonic_s=10.1) == AudioUrgency.U0
    assert _state("u2", first=True, received=160, played=60).urgency(now_monotonic_s=10.1) == AudioUrgency.U2


def test_rank_orders_u0_u1_u2_then_missing_fallback():
    states = {
        "u2": _state("u2", first=True, received=300, played=100),
        "u1": _state("u1", first=False),
        "u0": _state("u0", first=True, received=150, played=100),
    }
    assert rank_audio_requests(["u2", "missing", "u1", "u0"], states, now_monotonic_s=10.1) == [
        "u0",
        "u1",
        "u2",
        "missing",
    ]


def test_same_urgency_ordering():
    states = {
        "u0-large": _state("u0-large", first=True, received=190, played=100),
        "u0-small": _state("u0-small", first=True, received=110, played=100),
        "u1-new": _state("u1-new", first=False, ready=9.0),
        "u1-old": _state("u1-old", first=False, ready=2.0),
        "u2-large": _state("u2-large", first=True, received=400, played=100),
        "u2-small": _state("u2-small", first=True, received=220, played=100),
    }
    ranked = rank_audio_requests(list(reversed(states)), states, now_monotonic_s=10.1)
    assert ranked == ["u0-small", "u0-large", "u1-old", "u1-new", "u2-small", "u2-large"]


def test_stale_and_terminal_states_use_fcfs_fallback():
    states = {
        "stale": _state("stale", first=True, received=101, played=100, updated=1.0),
        "finished": _state("finished", first=True, received=101, played=100),
        "live": _state("live", first=True, received=101, played=100),
    }
    states["finished"].finished = True
    assert rank_audio_requests(["stale", "finished", "missing", "live"], states, now_monotonic_s=10.1) == [
        "live",
        "stale",
        "finished",
        "missing",
    ]


def test_feedback_is_monotonic_and_updates_buffer():
    state = _state("r", first=False)
    state.apply_feedback(
        received_audio_ms=720,
        played_audio_ms=410,
        first_audio_received=True,
        now_monotonic_s=10.2,
    )
    assert state.playback_buffer_ms == 310
    with pytest.raises(ValueError, match="monotonically"):
        state.apply_feedback(
            received_audio_ms=700,
            played_audio_ms=410,
            first_audio_received=True,
            now_monotonic_s=10.3,
        )


def test_invalid_audio_positions_are_rejected():
    with pytest.raises(ValueError, match="must not exceed"):
        _state("bad", first=True, received=10, played=11)
