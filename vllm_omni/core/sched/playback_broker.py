# SPDX-License-Identifier: Apache-2.0
"""Loopback research arbiter. A grant covers a complete device stage step.

Leases are never expired into a second grant: an unresponsive worker fails the
experiment closed. Client feedback uses server receipt time, not clock offsets.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import uuid

from .audio_interaction import AudioInteractionState
from .playback_deadline import PlaybackState


def rpc(url, operation, payload=None, timeout=300):
    request = urllib.request.Request(
        url.rstrip("/") + "/" + operation,
        json.dumps(payload or {}, allow_nan=False).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if "error" in value:
        raise RuntimeError(value["error"])
    return value


class PlaybackArbiter:
    def __init__(self, method, *, planner=None, chunk_frames=4, emit=lambda row: None, ablation="none"):
        if method in ("joint", "joint_ar") and planner is None:
            raise ValueError("joint scheduling requires calibrated stage costs")
        self.method, self.planner, self.chunk_frames, self.emit = method, planner, chunk_frames, emit
        self.ablation = ablation
        self.condition = threading.Condition()
        self.states, self.frames, self.pending, self.grants = {}, {}, {}, {}
        self.delivery, self.seen_chunks = {}, {}
        self.seeds = {}
        self.active = None
        self.failure = None
        self.phase = "warmup"

    def dispatch(self, operation, payload):
        with self.condition:
            if self.failure and operation not in ("health", "feedback", "finish"):
                raise RuntimeError(self.failure)
            now = time.monotonic()
            if operation == "health":
                return dict(method=self.method, failed=self.failure, active=self.active, phase=self.phase)
            if operation == "phase":
                if self.active is not None or self.pending:
                    raise ValueError("cannot change experiment phase during device work")
                if payload["phase"] not in ("warmup", "check", "measure", "calibration"):
                    raise ValueError("unknown experiment phase")
                self.phase = payload["phase"]
                self.emit(dict(event="phase", phase=self.phase, time=now))
                return {}
            if operation == "register":
                rid = payload["id"]
                if rid in self.states:
                    raise ValueError("duplicate request identity")
                self.states[rid] = PlaybackState(rid, now, updated=now)
                self.frames[rid] = 0
                self.seeds[rid] = int(payload.get("seed", 17))
                self.delivery[rid], self.seen_chunks[rid] = [], 0
                self.emit(dict(event="arrival", id=rid, time=now, split=payload.get("split", "test")))
                return {}
            if operation == "feedback":
                state = self.states[payload["id"]]
                accepted = state.feedback(payload, now)
                if accepted:
                    rid = payload["id"]
                    received = int(payload.get("received_chunks", self.seen_chunks[rid]))
                    for index in range(self.seen_chunks[rid], received):
                        if self.delivery[rid]:
                            sent, phase = self.delivery[rid].pop(0)
                            self.emit(
                                dict(
                                    event="delivery_confirmation",
                                    id=rid,
                                    chunk=index,
                                    wall_ms=(now - sent) * 1000,
                                    phase=phase,
                                )
                            )
                        elif self.method != "native":
                            self.emit(dict(event="delivery_unmatched", id=rid, chunk=index))
                    self.seen_chunks[rid] = received
                self.emit(dict(payload, event="feedback", server_time=now, accepted=accepted))
                return dict(accepted=accepted)
            if operation == "finish":
                if payload["id"] in self.states:
                    self.states[payload["id"]].finished = True
                self.condition.notify_all()
                return {}
            if operation == "priorities":
                result = {}
                for rid in payload["ids"]:
                    if rid not in self.states:
                        raise ValueError(f"unregistered playback request: {rid}")
                    s = self.states[rid]
                    result[rid] = list(self._priority(s, now))
                return result
            if operation == "release":
                token = payload["token"]
                if token not in self.grants:
                    raise ValueError("unknown or already released device grant")
                grant = self.grants.pop(token)
                if payload.get("error"):
                    self.failure = "device stage failed: " + payload["error"]
                for rid, count in payload.get("completed_frames", {}).items():
                    self.frames[rid] += count
                if grant["stage"] == "code2wav" and not payload.get("error"):
                    for job in grant["jobs"]:
                        self.delivery[job["id"]].append((now, grant["phase"]))
                self.emit(dict(payload, event="stage", grant=grant, server_time=now))
                if self.active == token:
                    self.active = None
                self.condition.notify_all()
                return {}
            if operation != "claim":
                raise ValueError(f"unknown arbiter operation: {operation}")
            token = uuid.uuid4().hex
            jobs = []
            for rid in payload["ids"]:
                if rid not in self.states:
                    raise ValueError(f"unknown request cannot acquire device: {rid}")
                jobs.append(
                    dict(
                        id=rid,
                        remaining_frames=0
                        if payload["stage"] == "code2wav"
                        else self.chunk_frames - self.frames[rid] % self.chunk_frames,
                    )
                )
            if not jobs:
                raise ValueError("empty stage grant")
            self.pending[token] = dict(id=token, stage=payload["stage"], jobs=jobs)
            queued = now
            try:
                while True:
                    if self.failure:
                        raise RuntimeError(self.failure)
                    # Abort processing is owned by the engine. Allow its
                    # already-scheduled final step to finish, then release KV;
                    # a cancelled peer must not crash all live sessions.
                    if self.active is None or self.method == "native":
                        before = time.perf_counter()
                        chosen = self._choose(time.monotonic(), token)
                        overhead = (time.perf_counter() - before) * 1000
                        if chosen["id"] == token:
                            if self.method != "native":
                                self.active = token
                            self.pending.pop(token)
                            result = dict(
                                chosen,
                                token=token,
                                queue_ms=(time.monotonic() - queued) * 1000,
                                controller_ms=overhead,
                                phase=self.phase,
                                request_seeds={j["id"]: self.seeds[j["id"]] for j in jobs},
                            )
                            result["budget_ms"] -= overhead
                            self.grants[token] = result
                            return result
                        self.condition.notify_all()
                    if time.monotonic() - queued > (1800 if self.phase == "warmup" else 120):
                        self.failure = "device grant timeout; no replacement lease issued"
                        self.condition.notify_all()
                        raise TimeoutError(self.failure)
                    self.condition.wait(0.002)
            finally:
                self.pending.pop(token, None)

    def _choose(self, now, caller):
        if self.method == "native":
            return dict(self.pending[caller], strategy="stock", budget_ms=0)
        candidates = []
        strategies = ("stock", "mean", "deadline", "lambda:0.5", "lambda:1", "lambda:2")
        for value in self.pending.values():
            choices = strategies if value["stage"] == "generate" and self.method == "joint" else ("stock",)
            candidates.extend(dict(value, strategy=s) for s in choices)
        if self.planner is not None and self.method in ("joint", "joint_ar") and self.ablation != "buffer_only":
            ready = {j["id"] for candidate in candidates for j in candidate["jobs"]}
            waiting = [
                dict(
                    id=rid,
                    remaining_frames=self.chunk_frames - self.frames[rid] % self.chunk_frames,
                    needs_prefill=self.frames[rid] == 0,
                )
                for rid, state in self.states.items()
                if rid not in ready and not state.finished and not state.input_complete
            ]
            return self.planner.choose(candidates, self.states, now, waiting)

        # The ported LiveServe urgency classes prioritize interrupted playback,
        # then first audio, then buffered sessions, with FCFS tie breaking.
        def urgency(c):
            states = [self.states[j["id"]] for j in c["jobs"]]
            return min(self._priority(s, now) for s in states)

        chosen = min(candidates, key=urgency)
        strategy = {"liveserve_fixed": "fixed", "position": "mean"}.get(self.method, "stock")
        if self.ablation == "buffer_only" and self.method == "joint":
            strategy = "fixed" if min(self.states[j["id"]].slack_ms(now) for j in chosen["jobs"]) < 100 else "stock"
        return dict(
            chosen, strategy=strategy, budget_ms=min(self.states[j["id"]].slack_ms(now) for j in chosen["jobs"])
        )

    def _priority(self, state, now):
        if self.method in ("joint", "joint_ar") and self.ablation != "buffer_only":
            return state.slack_ms(now), state.arrived, state.request_id
        live = AudioInteractionState(
            request_id=state.request_id,
            first_audio_received=state.received_ms > 0,
            received_audio_ms=state.received_ms,
            played_audio_ms=state.played_ms,
            last_update_monotonic_s=state.updated,
            finished=state.finished,
        )
        return int(live.urgency(now_monotonic_s=now)), state.arrived, state.request_id
