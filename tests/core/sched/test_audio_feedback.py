from __future__ import annotations

from types import SimpleNamespace

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin


class _FakeScheduler(OmniSchedulerMixin):
    def __init__(self):
        self.requests = {"internal-1": SimpleNamespace(request_id="internal-1")}
        self.chunk_transfer_adapter = SimpleNamespace(audio_scheduling_metrics={"u0_ready_count": 3})


def _payload(**overrides):
    payload = {
        "request_id": "internal-1",
        "received_audio_ms": 720.0,
        "played_audio_ms": 410.0,
        "first_audio_received": True,
        "finished": False,
        "aborted": False,
        "client_timestamp_ms": 1000.0,
    }
    payload.update(overrides)
    return payload


def test_feedback_accept_duplicate_stale_and_unknown():
    scheduler = _FakeScheduler()
    assert scheduler.update_audio_interaction_state(_payload())["status"] == "accepted"
    assert scheduler.audio_interaction_states["internal-1"].playback_buffer_ms == 310.0
    assert scheduler.update_audio_interaction_state(_payload())["status"] == "coalesced"
    assert scheduler.update_audio_interaction_state(_payload(client_timestamp_ms=999.0))["status"] == "stale"
    assert scheduler.update_audio_interaction_state(_payload(request_id="missing"))["status"] == "unknown_request"
    assert scheduler.audio_feedback_counters == {
        "accepted": 1,
        "coalesced": 1,
        "stale": 1,
        "unknown_request": 1,
        "state_cleaned": 0,
    }


def test_feedback_rejects_non_monotonic_audio_positions():
    scheduler = _FakeScheduler()
    scheduler.update_audio_interaction_state(_payload())
    result = scheduler.update_audio_interaction_state(
        _payload(received_audio_ms=700.0, client_timestamp_ms=1001.0)
    )
    assert result["status"] == "stale"
    assert scheduler.audio_interaction_states["internal-1"].received_audio_ms == 720.0


def test_terminal_feedback_and_finish_cleanup():
    scheduler = _FakeScheduler()
    result = scheduler.update_audio_interaction_state(_payload(finished=True))
    assert result["status"] == "accepted"
    assert "internal-1" not in scheduler.audio_interaction_states
    assert result["counters"]["state_cleaned"] == 1

    scheduler.update_audio_interaction_state(_payload(client_timestamp_ms=2000.0))
    scheduler._cleanup_audio_interaction_states(["internal-1"])
    assert "internal-1" not in scheduler.audio_interaction_states
    assert "internal-1" not in scheduler.audio_feedback_client_timestamps_ms


def test_late_terminal_feedback_is_idempotent():
    scheduler = _FakeScheduler()
    scheduler.requests.clear()

    result = scheduler.update_audio_interaction_state(_payload(finished=True))

    assert result["status"] == "already_finished"
    assert result["counters"]["unknown_request"] == 0


def test_reset_clears_metrics_without_removing_interaction_state():
    scheduler = _FakeScheduler()
    scheduler.update_audio_interaction_state(_payload())
    result = scheduler.reset_audio_scheduling_metrics()
    assert result["status"] == "reset"
    assert scheduler.chunk_transfer_adapter.audio_scheduling_metrics == {}
    assert scheduler.audio_feedback_counters == {
        "accepted": 0,
        "coalesced": 0,
        "stale": 0,
        "unknown_request": 0,
        "state_cleaned": 0,
    }
    assert "internal-1" in scheduler.audio_interaction_states
