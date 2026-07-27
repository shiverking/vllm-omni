# SPDX-License-Identifier: Apache-2.0
"""Clocked, silent PCM playback for LiveServe benchmarks."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2
CHANNELS = 1
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
STARTUP_BUFFER_MS = 100.0
FEEDBACK_INTERVAL_S = 0.05
CONTINUITY_THRESHOLD_S = 0.1

_GST: Any | None = None


def require_gstreamer() -> Any:
    """Load and validate the GStreamer elements required by the benchmark."""
    global _GST
    if _GST is not None:
        return _GST
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
    except Exception as exc:  # pragma: no cover - depends on system packages
        raise RuntimeError(
            "GStreamer playback is required for audio benchmarks. Install "
            "python3-gi, gir1.2-gstreamer-1.0, gstreamer1.0-tools, and "
            "gstreamer1.0-plugins-base; then verify with: "
            "gst-inspect-1.0 appsrc queue clocksync fakesink"
        ) from exc
    missing = [name for name in ("appsrc", "queue", "clocksync", "fakesink") if Gst.ElementFactory.find(name) is None]
    if missing:
        raise RuntimeError(f"GStreamer is missing required elements: {', '.join(missing)}")
    _GST = Gst
    return Gst


@dataclass(frozen=True)
class PlaybackSnapshot:
    received_audio_ms: float
    played_audio_ms: float
    playback_buffer_ms: float
    playback_started: bool
    max_underrun_ms: float
    total_underrun_ms: float
    underrun_event_count: int


class GStreamerSilentPlayer:
    """Consume PCM against a GStreamer clock without opening an audio device."""

    def __init__(self, *, _gst: Any | None = None) -> None:
        self._gst = _gst or require_gstreamer()
        self._lock = threading.RLock()
        self._pipeline = self._gst.parse_launch(
            "appsrc name=src is-live=true format=time block=false "
            'caps="audio/x-raw,format=S16LE,rate=24000,channels=1,layout=interleaved" '
            "! queue name=playback_queue max-size-time=2000000000 "
            "! clocksync sync=true ! fakesink sync=false async=false"
        )
        self._source = self._pipeline.get_by_name("src")
        if self._source is None:
            raise RuntimeError("Failed to create GStreamer appsrc")
        self._pending = bytearray()
        self._received_ns = 0
        self._scheduled_end_ns: int | None = None
        self._started = False
        self._finished = False
        self._active_underrun_start_ns: int | None = None
        self._total_underrun_ns = 0
        self._max_underrun_ns = 0
        self._underrun_event_count = 0

    @staticmethod
    def _bytes_to_ns(size: int) -> int:
        return int(size * 1_000_000_000 / BYTES_PER_SECOND)

    def _running_time_ns(self) -> int:
        clock = self._pipeline.get_clock()
        if clock is None:
            return 0
        return max(0, int(clock.get_time()) - int(self._pipeline.get_base_time()))

    def _start(self) -> None:
        if self._started:
            return
        result = self._pipeline.set_state(self._gst.State.PLAYING)
        if result == self._gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer silent playback pipeline")
        self._pipeline.get_state(5 * self._gst.SECOND)
        self._started = True
        data = bytes(self._pending)
        self._pending.clear()
        if data:
            self._push_buffer(data, self._running_time_ns())

    def _refresh_underrun(self, now_ns: int) -> None:
        if not self._started or self._finished or self._scheduled_end_ns is None:
            return
        if now_ns > self._scheduled_end_ns and self._active_underrun_start_ns is None:
            self._active_underrun_start_ns = self._scheduled_end_ns
            self._underrun_event_count += 1

    def _finish_active_underrun(self, now_ns: int) -> None:
        if self._active_underrun_start_ns is None:
            return
        duration = max(0, now_ns - self._active_underrun_start_ns)
        self._total_underrun_ns += duration
        self._max_underrun_ns = max(self._max_underrun_ns, duration)
        self._active_underrun_start_ns = None

    def _push_buffer(self, data: bytes, now_ns: int) -> None:
        self._refresh_underrun(now_ns)
        self._finish_active_underrun(now_ns)
        pts = max(now_ns, self._scheduled_end_ns or now_ns)
        duration = self._bytes_to_ns(len(data))
        buffer = self._gst.Buffer.new_allocate(None, len(data), None)
        buffer.fill(0, data)
        buffer.pts = pts
        buffer.dts = pts
        buffer.duration = duration
        flow = self._source.emit("push-buffer", buffer)
        if flow != self._gst.FlowReturn.OK:
            raise RuntimeError(f"GStreamer push-buffer failed: {flow}")
        self._scheduled_end_ns = pts + duration

    def push_pcm(self, data: bytes) -> PlaybackSnapshot:
        if not data:
            return self.snapshot()
        with self._lock:
            if self._finished:
                raise RuntimeError("Cannot push PCM after playback finished")
            self._received_ns += self._bytes_to_ns(len(data))
            if not self._started:
                self._pending.extend(data)
                if self._received_ns >= int(STARTUP_BUFFER_MS * 1_000_000):
                    self._start()
            else:
                self._push_buffer(data, self._running_time_ns())
            return self._snapshot_locked()

    def _snapshot_locked(self) -> PlaybackSnapshot:
        now_ns = self._running_time_ns() if self._started else 0
        self._refresh_underrun(now_ns)
        buffer_ns = 0
        if self._started and self._scheduled_end_ns is not None:
            buffer_ns = max(0, self._scheduled_end_ns - now_ns)
        elif not self._started:
            buffer_ns = self._received_ns
        played_ns = max(0, min(self._received_ns, self._received_ns - buffer_ns))
        active_ns = (
            max(0, now_ns - self._active_underrun_start_ns)
            if self._active_underrun_start_ns is not None
            else 0
        )
        return PlaybackSnapshot(
            received_audio_ms=self._received_ns / 1_000_000,
            played_audio_ms=played_ns / 1_000_000,
            playback_buffer_ms=buffer_ns / 1_000_000,
            playback_started=self._started,
            max_underrun_ms=max(self._max_underrun_ns, active_ns) / 1_000_000,
            total_underrun_ms=(self._total_underrun_ns + active_ns) / 1_000_000,
            underrun_event_count=self._underrun_event_count,
        )

    def snapshot(self) -> PlaybackSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def finish(self) -> PlaybackSnapshot:
        with self._lock:
            if not self._started and self._pending:
                self._start()
            now_ns = self._running_time_ns() if self._started else 0
            self._refresh_underrun(now_ns)
            self._finish_active_underrun(now_ns)
            self._finished = True
            if self._started:
                self._source.emit("end-of-stream")
            return self._snapshot_locked()

    def abort(self) -> PlaybackSnapshot:
        return self.finish()

    def close(self) -> None:
        with self._lock:
            self._pipeline.set_state(self._gst.State.NULL)
