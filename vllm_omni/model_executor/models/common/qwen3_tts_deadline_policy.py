# SPDX-License-Identifier: Apache-2.0
"""Offline, empirical RVQ completion policies. No model training or target RNG.

All times are milliseconds. Rows are *joint* observations of advancement and
latency; their correlation must survive fitting. A position-only model is an
approximation, not an optimality guarantee for the underlying transformer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict

import numpy as np

ACTION_SIZES = ((1, 4), (2, 4), (2, 8), (2, 16), (3, 4), (3, 8), (3, 16), (3, 32))
LAMBDAS = (0.0, 0.5, 1.0, 2.0)
SCHEMA = "rvq_completion_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_actions(observations):
    """Force the requested depth, then grow an ancestor-closed rank tree."""
    counts = Counter(tuple(row[:d]) for row in observations for d in range(1, len(row) + 1))
    result = {"serial": []}
    for depth, budget in ACTION_SIZES:
        universe = [p for d in range(1, depth + 1) for p in itertools.product(range(4), repeat=d)]
        leaf = min((p for p in universe if len(p) == depth), key=lambda p: (-counts[p], p))
        chosen = [leaf[:d] for d in range(1, depth + 1)]
        while len(chosen) < budget:
            frontier = [p for p in universe if p not in chosen and (len(p) == 1 or p[:-1] in chosen)]
            chosen.append(min(frontier, key=lambda p: (-counts[p], len(p), p)))
        result[f"d{depth}n{budget}"] = [list(p) for p in sorted(chosen, key=lambda p: (len(p), p))]
    return result


def fit_policy(rows, *, terminal=15, horizon_ms=2000, binding=None):
    """Backward DP over positions, with a conservative 1-ms deadline grid.

    Missing mode/batch/position/action observations are rejected rather than filled
    using batch-one extrapolation. Every action must be measured at every position.
    """
    if terminal < 2 or horizon_ms < 1:
        raise ValueError("invalid terminal/horizon")
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row.get("split") != "calibration":
            raise ValueError("policy fitting only accepts calibration observations")
        p, batch, delta = row["position"], row["batch"], row["delta"]
        if not 1 <= p < terminal or batch not in (1, 2, 4, 8) or not 1 <= delta <= terminal - p:
            raise ValueError("invalid position, batch or newly committed count")
        wall, device = float(row["wall_ms"]), float(row["device_ms"])
        if not all(math.isfinite(v) and v >= 0 for v in (wall, device)) or wall == 0:
            raise ValueError("costs must be finite; wall time must be positive")
        if row["sampling"] not in ("greedy", "sampling"):
            raise ValueError("invalid sampling mode")
        grouped[f"{row['sampling']}/b{batch}"][p][row["action"]].append((delta, wall, device))
    if not grouped:
        raise ValueError("no calibration observations")
    tables = {}
    for key, positions in sorted(grouped.items()):
        if set(positions) != set(range(1, terminal)):
            raise ValueError(f"incomplete positions: {key}")
        actions = sorted(positions[1])
        if "serial" not in actions or any(sorted(v) != actions for v in positions.values()):
            raise ValueError(f"incomplete action coverage: {key}")
        values, policies = {}, {}
        for price in LAMBDAS:
            v, policy = np.zeros(terminal + 1), [None] * (terminal + 1)
            for p in range(terminal - 1, 0, -1):
                scores = {
                    a: sum(w + price * d + v[p + n] for n, w, d in positions[p][a]) / len(positions[p][a])
                    for a in actions
                }
                policy[p] = min(actions, key=lambda a: (scores[a], a))
                v[p] = scores[policy[p]]
            values[str(price)], policies[str(price)] = v.tolist(), policy
        f = np.zeros((terminal + 1, horizon_ms + 1))
        f[terminal, :] = 1.0
        choices = np.zeros((terminal + 1, horizon_ms + 1), dtype=np.int32)
        for p in range(terminal - 1, 0, -1):
            scores = []
            for action in actions:
                score = np.zeros(horizon_ms + 1)
                for n, w, _ in positions[p][action]:
                    cost = max(1, math.ceil(w))
                    if cost <= horizon_ms:
                        score[cost:] += f[p + n, : horizon_ms + 1 - cost]
                scores.append(score / len(positions[p][action]))
            scores = np.stack(scores)
            best = scores.max(axis=0)
            # Equal probabilities (including an impossible deadline) prefer the
            # action minimizing mean remaining wall time, not arbitrary order.
            order = sorted(
                range(len(actions)),
                key=lambda i: (
                    sum(w + values["0.0"][p + n] for n, w, _ in positions[p][actions[i]])
                    / len(positions[p][actions[i]]),
                    actions[i],
                ),
            )
            chosen = np.full(horizon_ms + 1, order[-1], dtype=np.int32)
            for i in reversed(order):
                chosen = np.where(scores[i] == best, i, chosen)
            f[p], choices[p] = best, chosen
        tables[key] = dict(
            actions=actions,
            values=values,
            policies=policies,
            deadline_actions=choices.tolist(),
            completion_probability=f.tolist(),
            samples={str(p): dict(a) for p, a in positions.items()},
        )
    payload = dict(schema=SCHEMA, terminal=terminal, horizon_ms=horizon_ms, binding=binding or {}, tables=tables)
    payload["sha256"] = digest(payload)
    return payload


class CompletionPolicy:
    def __init__(self, payload):
        check = {k: v for k, v in payload.items() if k != "sha256"}
        if payload.get("schema") != SCHEMA or digest(check) != payload.get("sha256"):
            raise ValueError("invalid policy signature/schema")
        self.payload = payload

    def choose(self, position, batch, sampling, *, price=0.0, deadline_ms=None):
        if not 1 <= position < self.payload["terminal"]:
            raise ValueError("no action at a completed or uninitialized position")
        table = self.payload["tables"][f"{sampling}/b{batch}"]
        if deadline_ms is not None:
            if not math.isfinite(deadline_ms):
                raise ValueError("deadline must be finite")
            if deadline_ms <= 0:
                return table["policies"]["0.0"][position]
            if deadline_ms <= self.payload["horizon_ms"]:
                return table["actions"][table["deadline_actions"][position][math.floor(deadline_ms)]]
        return table["policies"][str(float(price))][position]
