# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import importlib
import itertools
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.qwen3_tts.test_code_predictor_dtype import (
    _load_target_classes,
    _make_tiny_config,
    _make_vllm_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_replay_progress_throttles_and_resets_stage_eta(tmp_path, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[4]
    module_spec = importlib.util.spec_from_file_location(
        "spec_experiment_progress", root / "tools/qwen3_tts_spec/experiment.py"
    )
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    clock = [0.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    progress = cli.ReplayProgress(tmp_path, interval=10)
    progress.begin("correctness", 4)
    capsys.readouterr()
    clock[0] = 5
    progress.advance()
    assert not capsys.readouterr().out
    clock[0] = 10
    progress.advance("frame=1")
    output = capsys.readouterr().out
    assert "2/4 (50.0%)" in output
    assert "ETA~00:00:10" in output
    progress.begin("performance", 1)
    clock[0] = 11
    progress.advance()
    output = capsys.readouterr().out
    assert "stage_elapsed=00:00:01; ETA~00:00:00; total_elapsed=00:00:11" in output
    assert "Replay started" in (tmp_path / "replay-progress.log").read_text(encoding="utf-8")
    for interval in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            cli.ReplayProgress(tmp_path, interval)


def test_offline_npu_initializes_ascend_before_weight_preparation(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    module_spec = importlib.util.spec_from_file_location(
        "spec_experiment_init", root / "tools/qwen3_tts_spec/experiment.py"
    )
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    events = []
    runtime = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=64), additional_config={})
    config_module = types.ModuleType("vllm.config")
    config_module.VllmConfig = lambda: runtime
    ascend_module = types.ModuleType("vllm_ascend.ascend_config")

    def initialize(config):
        assert config is runtime
        assert config.scheduler_config.max_num_seqs == 1
        events.append("init")

    ascend_module.init_ascend_config = initialize
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_config", ascend_module)
    device = SimpleNamespace(type="npu")
    monkeypatch.setattr(cli.torch, "device", lambda _: device)
    monkeypatch.setattr(cli, "api_for", lambda _: SimpleNamespace(set_device=lambda _: events.append("device")))
    monkeypatch.setattr(
        cli.torch, "load", lambda *a, **kw: dict(config={}, talker_hidden_size=32, dtype="torch.float32", state={})
    )

    class Wrapper:
        def __init__(self, *, vllm_config, **kwargs):
            assert vllm_config is runtime
            assert events == ["device", "init"]

        def to(self, *args, **kwargs):
            return self

        def load_state_dict(self, state, strict):
            assert strict

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

        def _prepare_npu_weights(self):
            assert "init" in events
            events.append("prepare")

    predictor_module = types.ModuleType("vllm_omni.model_executor.models.common.qwen3_code_predictor")
    predictor_module.CodePredictorWrapper = Wrapper
    predictor_module.CodePredictorWrapperConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, predictor_module.__name__, predictor_module)
    cli.load_predictor(SimpleNamespace(device="npu:0", output=Path("unused")))
    assert events == ["device", "init", "prepare"]
    cpu_runtime = cli.predictor_runtime_config(SimpleNamespace(type="cpu"))
    assert cpu_runtime.scheduler_config.max_num_seqs == 1
    assert events == ["device", "init", "prepare"]


@pytest.fixture
def research(mocker):
    config_mod, cp_mod = _load_target_classes(mocker)
    classes = (config_mod.Qwen3TTSTalkerCodePredictorConfig, config_mod.Qwen3TTSTalkerConfig, None, None, None)
    config, talker = _make_tiny_config(classes)
    config.num_code_groups = 7
    talker.num_code_groups = 7
    torch.manual_seed(731)
    predictor = cp_mod.Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
        vllm_config=_make_vllm_config(mocker, max_num_seqs=1), config=config, talker_config=talker
    ).eval()
    name = "vllm_omni.model_executor.models.common.qwen3_code_predictor_spec"
    # Remove cached modules left by earlier mocked-package fixture instances.
    mocker.patch.dict(sys.modules)
    sys.modules.pop(name, None)
    spec = importlib.import_module(name)
    inputs = dict(
        layer0_code=torch.tensor([[3]]),
        layer0_embed=torch.randn(1, 1, 32),
        last_talker_hidden=torch.randn(1, 1, 32),
        top_k=50,
        sample_uniforms=spec.frame_uniforms(7, "request", 3, 7, 64, "cpu"),
    )
    return predictor, spec, inputs


def test_layout_and_ancestor_closure(research):
    _, spec, _ = research
    paths = [(0,), (1,), (0, 2), (1, 3), (0, 2, 1)]
    positions, mask = spec.layout(3, paths, "cpu")
    assert positions.tolist() == [[0, 1, 2, 3, 3, 4, 4, 5]]
    assert mask[7].tolist() == [True, True, True, True, False, True, False, True]
    assert not mask[:3, 3:].any()
    for invalid in ([(1, 2)], [(4,)], [(0,), (0,)]):
        with pytest.raises(ValueError):
            spec.validate_tree(invalid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_canonical_attention_preserves_logical_ancestors(research, dtype):
    model, spec, inputs = research
    model.to(dtype=dtype)
    paths = spec.fit_tree([[2, 0, 1], [1, 0, 0]], 16)
    prefix = spec.initial_prefix(model, inputs["layer0_embed"], inputs["last_talker_hidden"])
    nodes = [model.model.codec_embedding[len(p) - 1](torch.tensor([[p[-1] + 1]])) for p in paths]
    packed = torch.cat([prefix, *nodes], dim=1)
    positions, mask = spec.attention_layout(2, tuple(paths), "cpu", model._num_groups + 1)
    path = (2, 0)
    row = 2 + paths.index(path)
    assert mask.ancestor_indices[row, :4].tolist() == [0, 1, 2 + paths.index((2,)), row]
    assert mask.ancestor_valid[row].sum() == 4
    hidden = model.model(packed, positions, mask)
    branch = [nodes[paths.index(path[:d])] for d in (1, 2)]
    serial = spec.reference_forward(model, torch.cat([prefix, *branch], dim=1))
    tolerance = 1e-6 if dtype == torch.float32 else 0.02
    torch.testing.assert_close(hidden[:, row], serial[:, -1], atol=tolerance, rtol=tolerance)
    changed = packed.clone()
    changed[:, 2 + paths.index((1,))] += 100
    isolated = model.model(changed, positions, mask)
    torch.testing.assert_close(hidden[:, row], isolated[:, row], atol=0, rtol=0)


@torch.inference_mode()
def test_tree_logits_match_serial_and_no_sibling_leak(research):
    model, spec, inputs = research
    prefix = spec.initial_prefix(model, inputs["layer0_embed"], inputs["last_talker_hidden"])
    paths = spec.fit_tree([[0, 0, 0], [1, 0, 2], [3, 1, 0]], 16)
    nodes = [model.model.codec_embedding[len(p) - 1](torch.tensor([[p[-1] + 1]])) for p in paths]
    packed = torch.cat([prefix, *nodes], dim=1)
    pos, mask = spec.layout(2, paths, "cpu")
    hidden = model.model(packed, pos, mask)
    for i, path in enumerate(paths):
        branch = [nodes[paths.index(path[:d])] for d in range(1, len(path) + 1)]
        serial = spec.reference_forward(model, torch.cat([prefix, *branch], dim=1))
        head = model.lm_head[len(path)]
        torch.testing.assert_close(head(hidden[:, 2 + i]), head(serial[:, -1]), atol=1e-6, rtol=1e-5)
    # Changing a root sibling cannot change any node on the other branch.
    changed = packed.clone()
    changed[:, 2 + paths.index((1,))] += 100
    perturbed = model.model(changed, pos, mask)
    for i, path in enumerate(paths):
        if path[0] != 1:
            torch.testing.assert_close(hidden[:, 2 + i], perturbed[:, 2 + i], atol=0, rtol=0)


@pytest.mark.parametrize("greedy", [True, False])
@torch.inference_mode()
def test_codes_match_stock_and_no_state_leak(research, greedy):
    model, spec, inputs = research
    inputs["do_sample"] = not greedy
    expected = model(**inputs)
    for budget in (8, 16, 32):
        paths = spec.fit_tree([[0, 0, 0], [1, 2, 0]], budget)
        actual, info = spec.decode(model, **inputs, paths=paths, detailed=True)
        assert torch.equal(expected, actual)
        assert sum(e["committed"] for e in info["events"]) == 6
        other = dict(inputs, layer0_embed=inputs["layer0_embed"] + 7)
        spec.decode(model, **other, paths=paths)
        again, _ = spec.decode(model, **inputs, paths=paths)
        assert torch.equal(actual, again)


@pytest.mark.parametrize(
    "prediction,count,accepted", [([2, 0, 0, 0], 1, 0), ([1, 2, 0, 0], 2, 1), ([1, 1, 1, 2], 4, 3)]
)
def test_zero_partial_and_full_coverage(research, prediction, count, accepted):
    _, spec, _ = research
    values, n, hits = spec.select_covered(torch.tensor(prediction), torch.tensor([0, 1, 2]), torch.tensor([1, 1, 1]), 4)
    assert int(n) == count and int(hits) == accepted
    assert values[:count].tolist() == prediction[:count]


def test_enumerated_target_distribution(research):
    _, spec, _ = research
    # Binary Markov target: P(0)=1/2; P(next=0|0)=3/4,
    # P(next=0|1)=1/4. Enumerate all 4x4 shared uniform cells.
    outcomes = {}
    for a, b in itertools.product((0.125, 0.375, 0.625, 0.875), repeat=2):
        root = int(a >= 0.5)
        after_zero = int(b >= 0.75)
        after_one = int(b >= 0.25)
        values, count, _ = spec.select_covered(
            torch.tensor([root, after_zero]), torch.tensor([0]), torch.tensor([0]), 2
        )
        result = values[: int(count)].tolist()
        if int(count) == 1:
            result.append(after_one)
        key = tuple(result)
        outcomes[key] = outcomes.get(key, 0) + 1 / 16
    assert outcomes == {(0, 0): 0.375, (0, 1): 0.125, (1, 0): 0.125, (1, 1): 0.375}


def test_gumbel_distribution(research):
    model, _, _ = research
    probs = torch.tensor([0.1, 0.3, 0.6])
    uniforms = torch.rand(100000, 3, generator=torch.Generator().manual_seed(71))
    draws = model._sample_codes_gumbel(probs.log().expand(100000, -1), uniforms=uniforms)
    counts = torch.bincount(draws.flatten(), minlength=3) / 100000
    torch.testing.assert_close(counts, probs, atol=0.005, rtol=0)


def test_tail_empty_batch_and_rng(research):
    model, spec, inputs = research
    u = spec.frame_uniforms(7, "request", 3, 7, 64, "cpu")
    assert torch.equal(u, inputs["sample_uniforms"])
    assert not torch.equal(u, spec.frame_uniforms(7, "another", 3, 7, 64, "cpu"))
    with pytest.raises(ValueError, match="exactly one"):
        spec.decode(model, **dict(inputs, layer0_code=torch.zeros(2, 1)))
    for groups in (1, 2, 3):
        model._num_groups = groups
        local = dict(inputs, do_sample=False)
        result, _ = spec.decode(model, **local, paths=[(0,), (0, 0), (0, 0, 0)])
        assert result.shape == (1, groups)


def test_fit_uses_complete_path_frequency(research):
    _, spec, _ = research
    observations = [[2, 3, 1]] * 50 + [[0, 0, 0]]
    fitted = spec.fit_tree(observations, 8)
    assert (2, 3, 1) in fitted
    assert len(fitted) == 8
    assert spec.validate_tree(fitted) == fitted


@torch.inference_mode()
def test_node_failure_preserves_inputs_and_diagnoses_topology(research, monkeypatch):
    model, spec, inputs = research

    def fail(*args, **kwargs):
        raise AssertionError("simulated device logit mismatch")

    monkeypatch.setattr(torch.testing, "assert_close", fail)
    with pytest.raises(spec.NodeAuditError, match="simulated") as error:
        spec.decode(model, **inputs, paths=[(0,), (1,), (0, 0)], audit=True)
    payload = error.value.payload
    assert payload["position"] == 1
    assert payload["rank_path"] == [0]
    assert payload["packed"].device.type == "cpu"
    result = spec.diagnose_node(model, payload)
    assert result["cpu_float64"]["max_abs"] < 1e-10
    assert result["same_shape_serial"]["max_abs"] < 1e-5
    # A deliberately invalid visibility mask must not pass the FP64 diagnosis.
    corrupted = dict(payload, mask=torch.ones_like(payload["mask"]))
    result = spec.diagnose_node(model, corrupted)
    assert result["cpu_float64"]["max_abs"] > 1e-5


@torch.inference_mode()
def test_projection_and_input_dtype(research):
    model, spec, inputs = research
    model.talker_config.hidden_size = 48
    projected = type(model)(
        vllm_config=model._vllm_config, config=model.config, talker_config=model.talker_config
    ).eval()
    inputs.update(
        layer0_embed=torch.randn(1, 1, 48).double(), last_talker_hidden=torch.randn(1, 1, 48).double(), do_sample=False
    )
    actual, _ = spec.decode(projected, **inputs, paths=[(0,), (0, 0), (0, 0, 0)])
    assert torch.equal(actual, projected(**inputs))


@torch.inference_mode()
def test_research_artifact_pipeline(research, tmp_path, monkeypatch):
    """Exercise real capture/export/reload, held-out replay and report on CPU."""
    model, spec, inputs = research
    root = Path(__file__).resolve().parents[4]
    module_spec = importlib.util.spec_from_file_location("spec_experiment", root / "tools/qwen3_tts_spec/experiment.py")
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    args = SimpleNamespace(
        output=tmp_path,
        device="cpu",
        model="synthetic-CustomVoice",
        model_revision="synthetic-test",
        seed=17,
        vllm_repo=root.parent / "vllm",
        ascend_repo=root.parent / "vllm-ascend",
        prepare=True,
        data=root / "benchmarks/build_dataset/seed_tts_smoke/en/meta.lst",
        e2e_label=None,
        greedy=False,
    )
    cli.collect(args)
    # Keep this integration test small, while the production CLI still requires
    # 20 warmups and 200 frame executions per ABBA round.
    model.config.num_code_groups = 3
    model.talker_config.num_code_groups = 3
    model = type(model)(vllm_config=model._vllm_config, config=model.config, talker_config=model.talker_config).eval()
    research_mod = importlib.import_module("vllm_omni.model_executor.models.common.qwen3_code_predictor_research")
    session = research_mod.ResearchSession(tmp_path / "collect_config.json")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest["dataset"]:
        cli.write_json(tmp_path / "control.json", row)
        session.forward(
            model,
            model.forward,
            inputs["layer0_code"],
            inputs["layer0_embed"],
            inputs["last_talker_hidden"],
            dict(do_sample=True, top_k=50),
        )
    assert len(list((tmp_path / "frames").glob("*.pt"))) == 30
    cli.analyze(args)
    # Exercise the actual environment-gated TTS wrapper and frozen config check.
    config_file = tmp_path / "tree16_config.json"
    monkeypatch.setenv("VLLM_OMNI_TTS_SPEC_CONFIG", str(config_file))
    live = type(model)(vllm_config=model._vllm_config, config=model.config, talker_config=model.talker_config).eval()
    live.load_state_dict(model.state_dict())
    shared = spec.frame_uniforms(17, "live", 0, 3, 64, "cpu")
    local = dict(inputs, sample_uniforms=shared)
    assert torch.equal(live(**local), model(**local))
    with pytest.raises(ValueError, match="max_num_seqs=1"):
        live(**dict(local, layer0_code=torch.ones(2, 1, dtype=torch.long)))
    monkeypatch.delenv("VLLM_OMNI_TTS_SPEC_CONFIG")
    with pytest.raises(ValueError, match="already frozen"):
        cli.analyze(args)
    cli.replay(args)
    cli.report(args)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["correctness"]["passed"]
    assert summary["verdict"] == "实现完成、性能待验证"
    assert summary["replay"]["unique_frames"] == 20
    assert summary["replay"]["executions_per_round"] == 200
    assert len(summary["performance"]) == 9
    assert (tmp_path / "report.md").exists()
    # A failed later correctness run must invalidate older timing artifacts.
    cli.write_json(tmp_path / "correctness.json", {"passed": False})
    cli.report(args)
    assert "未通过" in json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["verdict"]
