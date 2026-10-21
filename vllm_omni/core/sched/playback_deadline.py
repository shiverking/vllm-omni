# SPDX-License-Identifier: Apache-2.0
"""Playback-aware, empirical next-chunk planning; no model or device imports."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class PlaybackState:
    request_id: str
    arrived: float
    received_ms: float = 0.0
    played_ms: float = 0.0
    rate: float = 1.0
    started: bool = False
    updated: float = 0.0
    sequence: int = -1
    finished: bool = False
    input_complete: bool = False

    def feedback(self, payload, now):
        seq = int(payload["sequence"])
        if seq <= self.sequence or self.finished:
            return False
        received, played, rate = (float(payload[k]) for k in ("received_ms", "played_ms", "rate"))
        if not all(math.isfinite(v) for v in (received, played, rate)) or not 0 <= played <= received or rate <= 0:
            raise ValueError("invalid playback feedback")
        if received < self.received_ms or played < self.played_ms:
            raise ValueError("playback positions moved backwards")
        self.received_ms, self.played_ms, self.rate = received, played, rate
        self.started = self.started or bool(payload["started"])
        self.updated, self.sequence = now, seq
        self.finished = bool(payload.get("finished", False))
        self.input_complete = self.input_complete or bool(payload.get("input_complete", False))
        return True

    def slack_ms(self, now, delivery_ms=0.0):
        if not self.started:
            return 1000 - (now - self.arrived) * 1000 - delivery_ms
        # Server receipt times only. Do not subtract a client's clock from ours.
        age = max(0.0, (now - self.updated) * 1000)
        buffer = max(0.0, (self.received_ms - self.played_ms) / self.rate - age)
        # Old feedback never provides a fresh positive safety margin.
        if age > 500:
            buffer = 0.0
        return buffer - delivery_ms


class StageCosts:
    def __init__(self, payload):
        if payload.get("split") != "calibration":
            raise ValueError("stage cost model must come from calibration")
        self.samples = payload["samples"]
        for key, rows in self.samples.items():
            if not rows or any(not math.isfinite(float(v)) or v <= 0 for v in rows):
                raise ValueError(f"invalid stage cost samples: {key}")

    def draw(self, stage, batch, rng):
        # No implicit batch-one extrapolation for missing buckets.
        return rng.choice(self.samples[f"{stage}/b{batch}"])


class ChunkPlanner:
    """Compare 32 common scenarios, then order misses/lateness/device cost.

    This is a short-horizon empirical planner, not a probability guarantee. Its
    randomness is independent of target sampling and recreated for each candidate.
    """

    def __init__(self, completion_policy, stage_costs, *, sampling="sampling", scenarios=32, ablation="none"):
        self.policy, self.costs = completion_policy, stage_costs
        self.sampling, self.scenarios = sampling, scenarios
        self.ablation = ablation
        if scenarios != 32:
            raise ValueError("the frozen experiment uses 32 common scenarios")

    def draw(self, stage, batch, rng):
        if stage == "code2wav" and self.ablation == "no_code2wav":
            return 0.0
        return self.costs.draw(stage, 1 if self.ablation == "no_batch" else batch, rng)

    def frame(self, batch, strategy, rng, budget_ms=None):
        if self.ablation == "no_batch":
            batch = 1
        if strategy == "stock":
            t = self.draw("predictor_stock", batch, rng)
            return t, t
        if batch not in (1, 2, 4, 8):
            elapsed, device, remaining = 0.0, 0.0, batch
            while remaining:
                size = max(b for b in (1, 2, 4, 8) if b <= remaining)
                wall, cost = self.frame(size, strategy, rng, None if budget_ms is None else budget_ms - elapsed)
                elapsed, device, remaining = elapsed + wall, device + cost, remaining - size
            return elapsed, device
        table = self.policy.payload["tables"][f"{self.sampling}/b{batch}"]
        elapsed = self.draw("predictor_prefill", batch, rng)
        device, position = elapsed, 1
        terminal = self.policy.payload["terminal"]
        while position < terminal:
            if strategy == "fixed":
                action = "d3n16"
            else:
                price = float(strategy.split(":")[1]) if strategy.startswith("lambda:") else 0.0
                deadline = budget_ms - elapsed if strategy == "deadline" and budget_ms is not None else None
                action = self.policy.choose(position, batch, self.sampling, price=price, deadline_ms=deadline)
            delta, wall, busy = rng.choice(table["samples"][str(position)][action])
            position += delta
            elapsed += wall
            device += busy
        return elapsed, device

    def choose(self, candidates, states, now, background_jobs=()):
        if not candidates:
            raise ValueError("no executable candidates")
        scored = []
        # Deduplicate requests present at multiple stages. Only ready codec
        # tasks can be selected; predicted generation is never counted as PCM.
        jobs = {j["id"]: dict(j) for j in background_jobs}
        for c in candidates:
            for job in c["jobs"]:
                previous = jobs.get(job["id"])
                # A ready chunk remains the next delivery even if Talker has
                # already started generating the following chunk.
                remaining = min(job.get("remaining_frames", 4), previous["remaining_frames"] if previous else 4)
                needs_prefill = c["stage"] == "prefill" or bool(previous and previous["needs_prefill"])
                jobs[job["id"]] = dict(job, remaining_frames=remaining, needs_prefill=needs_prefill)
        ordered = sorted(jobs.values(), key=lambda j: (states[j["id"]].slack_ms(now), states[j["id"]].arrived, j["id"]))
        for candidate in candidates:
            miss, late, busy_total = 0.0, 0.0, 0.0
            allocated = []
            batch = len(candidate["jobs"])
            budgets = [states[j["id"]].slack_ms(now) for j in candidate["jobs"]]
            for scenario in range(self.scenarios):
                rng = random.Random(17000 + scenario)
                delivery = self.draw("delivery", 1, rng)
                codec = self.draw("code2wav", batch, rng)
                remaining = max(jobs[j["id"]]["remaining_frames"] for j in candidate["jobs"])
                # Reserve the measured downstream work and future frames. The
                # frame policy only sees its allocated remainder, not futures.
                future = 0.0
                for _ in range(max(0, remaining - 1)):
                    f, _ = self.frame(batch, "mean", rng)
                    future += f + self.draw("talker", batch, rng)
                budget = min(budgets) - codec - delivery - future
                allocated.append(budget)
                if candidate["stage"] == "code2wav":
                    first, busy = codec, codec
                elif candidate["stage"] == "prefill":
                    first = self.draw("prefill", batch, rng)
                    busy = first
                else:
                    first, busy = self.frame(batch, candidate["strategy"], rng, budget)
                    talker = self.draw("talker", batch, rng)
                    first, busy = first + talker, busy + talker
                elapsed = first
                selected = {j["id"] for j in candidate["jobs"]}
                for job in ordered:
                    rid = job["id"]
                    frames = job.get("remaining_frames", 4)
                    if rid in selected and candidate["stage"] == "code2wav":
                        finish = first + delivery
                    else:
                        if job["needs_prefill"] and not (rid in selected and candidate["stage"] == "prefill"):
                            duration = self.draw("prefill", 1, rng)
                            elapsed += duration
                            busy += duration
                        frames -= int(rid in selected and candidate["stage"] == "generate")
                        for _ in range(max(0, frames)):
                            duration, cost = self.frame(1, "mean", rng)
                            talker = self.draw("talker", 1, rng)
                            elapsed += duration + talker
                            busy += cost + talker
                        duration = self.draw("code2wav", 1, rng)
                        elapsed += duration
                        busy += duration
                        finish = elapsed + delivery
                    overdue = max(0.0, finish - states[rid].slack_ms(now))
                    miss += int(overdue > 0)
                    late += overdue
                busy_total += busy
            score = (
                miss / self.scenarios,
                late / self.scenarios,
                busy_total / self.scenarios,
                min(states[j["id"]].arrived for j in candidate["jobs"]),
                candidate["id"],
            )
            scored.append((score, candidate, min(allocated)))
        score, chosen, budget = min(scored, key=lambda row: row[0])
        # Return only a measured-time budget, never a simulated target token.
        return dict(chosen, score=score[:3], budget_ms=budget)
