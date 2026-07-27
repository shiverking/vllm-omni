# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.gstreamer_player import GStreamerSilentPlayer


class _Clock:
    now_ns = 0

    def get_time(self):
        return self.now_ns


class _Source:
    def emit(self, name, _buffer=None):
        return _Gst.FlowReturn.OK


class _Pipeline:
    def __init__(self):
        self.clock = _Clock()
        self.source = _Source()

    def get_by_name(self, name):
        return self.source if name == "src" else object()

    def set_state(self, _state):
        return _Gst.StateChangeReturn.SUCCESS

    def get_state(self, _timeout):
        return None

    def get_clock(self):
        return self.clock

    def get_base_time(self):
        return 0


class _Buffer:
    @staticmethod
    def new_allocate(_allocator, size, _params):
        return _Buffer()

    def fill(self, _offset, _data):
        return None


class _Gst:
    SECOND = 1_000_000_000
    State = SimpleNamespace(PLAYING=1, NULL=0)
    StateChangeReturn = SimpleNamespace(FAILURE=-1, SUCCESS=1)
    FlowReturn = SimpleNamespace(OK=0)
    Buffer = _Buffer
    pipeline = _Pipeline()

    @classmethod
    def parse_launch(cls, _description):
        cls.pipeline = _Pipeline()
        return cls.pipeline


def _advance_ms(value):
    _Gst.pipeline.clock.now_ns += int(value * 1_000_000)


def test_player_uses_startup_buffer_and_tracks_independent_underruns():
    player = GStreamerSilentPlayer(_gst=_Gst)
    first = player.push_pcm(b"\0" * 4800)  # 100 ms
    assert first.playback_started is True
    assert first.playback_buffer_ms == pytest.approx(100)

    _advance_ms(50)
    playing = player.snapshot()
    assert playing.played_audio_ms == pytest.approx(50)
    assert playing.playback_buffer_ms == pytest.approx(50)

    _advance_ms(100)
    starved = player.snapshot()
    assert starved.played_audio_ms == pytest.approx(100)
    assert starved.max_underrun_ms == pytest.approx(50)

    player.push_pcm(b"\0" * 4800)
    recovered = player.snapshot()
    assert recovered.playback_buffer_ms == pytest.approx(100)
    assert recovered.max_underrun_ms == pytest.approx(50)

    _advance_ms(150)
    player.snapshot()
    player.push_pcm(b"\0" * 4800)
    final = player.finish()
    assert final.max_underrun_ms == pytest.approx(50)
    assert final.total_underrun_ms == pytest.approx(100)
    assert final.underrun_event_count == 2
    player.close()


def test_sub_threshold_audio_starts_at_eos_without_startup_underrun():
    player = GStreamerSilentPlayer(_gst=_Gst)
    snapshot = player.push_pcm(b"\0" * 2400)
    assert snapshot.playback_started is False
    final = player.finish()
    assert final.playback_started is True
    assert final.max_underrun_ms == 0
    player.close()
