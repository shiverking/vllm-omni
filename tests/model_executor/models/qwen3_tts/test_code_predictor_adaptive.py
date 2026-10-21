# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch

from tests.model_executor.models.qwen3_tts.test_code_predictor_spec import research as research_fixture

research = research_fixture
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("sampling", [False, True])
@torch.inference_mode()
def test_switching_and_regrouping_matches_stock(research, batch, sampling):
    model, _, inputs = research
    policy = importlib.import_module("vllm_omni.model_executor.models.common.qwen3_tts_deadline_policy")
    module = importlib.import_module("vllm_omni.model_executor.models.common.qwen3_code_predictor_adaptive")
    inputs = {k: v.repeat(batch, *([1] * (v.ndim - 1))) if torch.is_tensor(v) else v for k, v in inputs.items()}
    inputs.update(do_sample=sampling, temperature=0.9, top_k=50)
    inputs["layer0_embed"] = inputs["layer0_embed"] + torch.randn_like(inputs["layer0_embed"])
    inputs["sample_uniforms"] = torch.rand_like(inputs["sample_uniforms"]).clamp(0.0001, 0.9999)
    expected = model(**inputs)
    actions = policy.fit_actions([[0, 0, 0], [1, 1, 1]])
    engine = module.AdaptiveExecutor(model, actions, batch_size=batch, do_sample=sampling)
    names = ["d3n16", "serial", "d1n4", "d2n8"]
    calls = []
    handle = model.model.register_forward_hook(lambda *_: calls.append(1))
    actual, info = engine.decode(**inputs, choose=lambda p, b: names[p % len(names)], audit=True, record_events=True)
    handle.remove()
    assert torch.equal(actual, expected)
    assert len(calls) == info["backbone_calls"] + info["audit_calls"]
    for row in range(batch):
        assert 1 + sum(e["delta"] for e in info["events"] if e["row"] == row) == model._num_groups - 1
    assert info["audit_max_logit_error"] < 1e-5
    other = dict(inputs, layer0_embed=inputs["layer0_embed"] + 7)
    engine.decode(**other)
    again, _ = engine.decode(**inputs, choose=lambda p, b: names[p % len(names)])
    assert torch.equal(actual, again)
    full = module.AdaptiveExecutor(
        model, {k: actions[k] for k in ("serial", "d3n16")}, batch_size=batch, do_sample=sampling, trim_tail=False
    )
    full_codes, _ = full.decode(**inputs, choose=lambda p, b: "d3n16", audit=True)
    assert torch.equal(full_codes, expected)
    assert full.steps[full.aliases[batch, model._num_groups - 2, "d3n16"]].query_size == 17
    order = torch.arange(batch - 1, -1, -1)
    shuffled = {k: v.index_select(0, order) if torch.is_tensor(v) else v for k, v in inputs.items()}
    reordered, _ = engine.decode(**shuffled, choose=lambda p, b: names[p % len(names)])
    assert torch.equal(reordered, expected.index_select(0, order))
    # All actions in a bucket alias committed state; tail query is physical Q=1.
    for step in engine.steps.values():
        assert step.codes.data_ptr() == engine.work[step.shared.batch].codes.data_ptr()
        if step.position == model._num_groups - 2:
            assert step.query_size == 1


def test_empirical_dp_and_deadline_boundary(research):
    from vllm_omni.model_executor.models.common.qwen3_tts_deadline_policy import CompletionPolicy, fit_policy

    rows = []
    for p in range(1, 5):
        for name, delta, wall in [("serial", 1, 2.0), ("tree", min(2, 5 - p), 3.0)]:
            rows.append(
                dict(
                    split="calibration",
                    sampling="sampling",
                    position=p,
                    batch=1,
                    action=name,
                    delta=delta,
                    wall_ms=wall,
                    device_ms=wall,
                )
            )
    payload = fit_policy(rows, terminal=5, horizon_ms=20)
    policy = CompletionPolicy(payload)
    assert policy.choose(1, 1, "sampling") == "tree"
    assert policy.choose(4, 1, "sampling") == "serial"
    assert policy.choose(4, 1, "sampling", deadline_ms=2) == "serial"
    assert payload["tables"]["sampling/b1"]["completion_probability"][4][1] == 0
    assert payload["tables"]["sampling/b1"]["completion_probability"][4][2] == 1
    assert policy.choose(4, 1, "sampling", deadline_ms=-1) == "serial"
    with pytest.raises(ValueError, match="calibration"):
        fit_policy([dict(r, split="test") for r in rows])
    with pytest.raises(ValueError, match="incomplete"):
        fit_policy(rows[:-1], terminal=5)


@torch.inference_mode()
def test_playback_artifact_roundtrip_and_gates(research, tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "tools/qwen3_tts_spec"))
    cli = importlib.import_module("playback_experiment")
    exp = importlib.import_module("experiment")
    module = importlib.import_module("vllm_omni.model_executor.models.common.qwen3_code_predictor_adaptive")
    model, _, inputs = research
    inputs = dict(inputs, temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
    capture, output = tmp_path / "capture", tmp_path / "run"
    (capture / "frames").mkdir(parents=True)
    output.mkdir()
    rows = [
        dict(id=rid, text=rid, split=split)
        for rid, split in (("smoke001", "calibration"), ("smoke011", "test"), ("smoke030", "test"))
    ]
    (capture / "manifest.json").write_text(json.dumps(dict(dataset=rows)), encoding="utf-8")
    torch.save(dict(state=model.state_dict()), capture / "predictor.pt")
    for row in rows:
        for frame_id in (0, 2):
            frame = dict(
                request_id=row["id"],
                frame_id=frame_id,
                split=row["split"],
                seed=17,
                inputs={k: v for k, v in inputs.items() if torch.is_tensor(v) and k != "sample_uniforms"},
                sampling=dict(do_sample=True, temperature=0.9, top_k=50, top_p=1.0),
                codes=model(**inputs),
            )
            torch.save(frame, capture / "frames" / f"{row['id']}-{frame_id}.pt")
    args = SimpleNamespace(
        config=output / "config.json", input=str(capture), device="cpu", text_manifest=None, probe_only=True
    )
    cli.prepare(args)
    config = cli.read(args.config)
    config.update(warmup=1, batch_buckets=[1, 2], sampling=["sampling"], frames_per_request=2)
    ns = SimpleNamespace(output=capture, device="cpu", vllm_repo=root, ascend_repo=root)
    monkeypatch.setattr(cli, "runtime", lambda _: (exp, torch, module.AdaptiveExecutor, ns, model))
    cli.calibrate(args, config)
    frozen, policy = cli.artifacts(config)
    with pytest.raises(FileNotFoundError):
        cli.certificate(config, frozen, policy)
    cli.check(args, config)
    assert cli.read(output / "probe.json")["passed"]
    args.probe_only = False
    cli.check(args, config)
    cli.certificate(config, frozen, policy)
    cli.measure(args, config)
    assert cli.read(output / "measurements.json")["rows"]
    # Resume is bound to code, data and frozen actions, not just a filename.
    old_costs = cli.read(output / "costs.json")
    cli.calibrate(args, config)
    assert cli.read(output / "costs.json") == old_costs
    config["playback_rate"] = 1.25
    with pytest.raises(ValueError, match="certificate"):
        cli.certificate(config, frozen, policy)
