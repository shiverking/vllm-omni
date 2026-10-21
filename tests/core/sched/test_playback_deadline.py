# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
import time

import pytest

from vllm_omni.core.sched.playback_broker import PlaybackArbiter
from vllm_omni.core.sched.playback_deadline import ChunkPlanner, PlaybackState, StageCosts
from vllm_omni.model_executor.models.common.qwen3_tts_deadline_policy import CompletionPolicy, fit_policy


def test_feedback_sequence_clock_and_expiration():
    state = PlaybackState("a", 100, updated=100)
    assert state.slack_ms(100.25) == 750
    payload = dict(sequence=1, received_ms=1000, played_ms=200, rate=1.25, started=True)
    assert state.feedback(payload, 101)
    assert state.slack_ms(101.25) == pytest.approx(390)
    assert not state.feedback(payload, 101.3)
    assert state.slack_ms(101.6) == 0
    with pytest.raises(ValueError):
        state.feedback(dict(payload, sequence=2, received_ms=999), 102)


def test_arbiter_mutual_exclusion_failure_and_cancel_cleanup():
    arbiter = PlaybackArbiter("liveserve_ar")
    for rid in ("a", "b"):
        arbiter.dispatch("register", dict(id=rid))
    first = arbiter.dispatch("claim", dict(ids=["a"], stage="generate"))
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        second = pool.submit(arbiter.dispatch, "claim", dict(ids=["b"], stage="code2wav"))
        deadline = time.monotonic() + 1
        while not arbiter.pending and time.monotonic() < deadline:
            time.sleep(0.001)
        assert not second.done()
        arbiter.dispatch("finish", dict(id="b"))
        arbiter.dispatch("release", dict(token=first["token"], completed_frames={"a": 1}))
        grant = second.result(timeout=2)
        assert arbiter.frames["a"] == 1
        assert arbiter.active == grant["token"]
        arbiter.dispatch("release", dict(token=grant["token"], error="NPU failure"))
    with pytest.raises(RuntimeError, match="NPU failure"):
        arbiter.dispatch("claim", dict(ids=["a"], stage="generate"))
    assert not arbiter.grants


def test_common_scenario_planner_reserves_downstream_time():
    samples = []
    for batch in (1, 2):
        for p in range(1, 5):
            for action, delta, wall in (("serial", 1, 3), ("d3n16", min(2, 5 - p), 4)):
                samples.append(
                    dict(
                        split="calibration",
                        sampling="sampling",
                        batch=batch,
                        position=p,
                        action=action,
                        delta=delta,
                        wall_ms=wall,
                        device_ms=wall,
                    )
                )
    policy = CompletionPolicy(fit_policy(samples, terminal=5, horizon_ms=100))
    costs = StageCosts(
        dict(
            split="calibration",
            samples={
                f"{s}/b{b}": [v]
                for b in (1, 2)
                for s, v in (
                    ("predictor_stock", 16),
                    ("predictor_prefill", 2),
                    ("code2wav", 8),
                    ("talker", 1),
                    ("prefill", 20),
                    ("delivery", 2),
                )
            },
        )
    )
    planner = ChunkPlanner(policy, costs)
    states = {"a": PlaybackState("a", 0), "b": PlaybackState("b", 0)}
    candidates = [
        dict(id="g", stage="generate", strategy="mean", jobs=[dict(id="a", remaining_frames=2)]),
        dict(id="c", stage="code2wav", strategy="stock", jobs=[dict(id="b", remaining_frames=0)]),
    ]
    first = planner.choose(candidates, states, 0.96)
    second = planner.choose(list(reversed(candidates)), states, 0.96)
    assert first == second
    assert first["budget_ms"] < 40
    assert len(first["score"]) == 3
    with pytest.raises(KeyError):
        costs.draw("code2wav", 8, __import__("random").Random(1))


def test_test_split_cannot_fit_cost_model():
    with pytest.raises(ValueError, match="calibration"):
        StageCosts(dict(split="test", samples={"code2wav/b1": [1]}))


def test_stage_permit_covers_prefill_sampling_and_device_completion(tmp_path, monkeypatch):
    import contextlib
    import json
    from types import SimpleNamespace

    from vllm_omni.engine import playback_stage as module

    config = tmp_path / "runtime.json"
    config.write_text(json.dumps(dict(broker_url="unused", service_method="liveserve_ar")))
    request = SimpleNamespace(
        external_req_id="external", request_id="internal", arrival_time=0, num_computed_tokens=0, num_prompt_tokens=4
    )
    output = SimpleNamespace(num_scheduled_tokens={"internal": 4}, total_num_scheduled_tokens=4)
    operations = []
    scheduler = SimpleNamespace(
        requests={"internal": request},
        running=[request],
        waiting=[],
        has_requests=lambda: True,
        get_grammar_bitmask=lambda _: None,
        update_from_output=lambda a, b: {0: b},
    )

    def schedule():
        request.num_computed_tokens += 4
        return output

    scheduler.schedule = schedule

    def rpc(url, op, payload):
        operations.append(op)
        if op == "priorities":
            return {"external": [0, 0]}
        if op == "claim":
            assert payload["stage"] == "prefill"  # schedule already incremented computed count
            return dict(token="t", strategy="stock")
        if op == "release":
            assert operations[-2] == "worker_end"
        return {}

    monkeypatch.setattr(module, "rpc", rpc)

    def collective(method, args=()):
        operations.append(method.__name__)
        return [dict(completed_frames={}, cp_ms=0, events=[])]

    def execute(*args, **kwargs):
        operations.append("execute")
        return SimpleNamespace(result=lambda: None)

    def sample(*args):
        operations.append("sample")
        return "output"

    core = SimpleNamespace(
        scheduler=scheduler,
        model_executor=SimpleNamespace(collective_rpc=collective, execute_model=execute, sample_tokens=sample),
        log_error_detail=lambda _: contextlib.nullcontext(),
        log_iteration_details=lambda _: contextlib.nullcontext(),
        _process_aborts_queue=lambda: None,
    )
    assert module.PlaybackStage(core, config, 0).step() == ({0: "output"}, True)
    assert operations == ["priorities", "claim", "worker_begin", "execute", "sample", "worker_end", "release"]
