# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.qwen3_tts.test_code_predictor_spec import research as research_fixture

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

research = research_fixture


def pipeline_module():
    return importlib.import_module("vllm_omni.model_executor.models.common.qwen3_code_predictor_pipeline")


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("budget", [0, 8, 16, 32])
@torch.inference_mode()
def test_pipeline_matches_ar_nodes_and_actual_calls(research, cache, greedy, budget):
    model, spec, inputs = research
    paths = spec.fit_tree([[0, 0, 0], [1, 2, 1]], budget) if budget else ()
    inputs = dict(inputs, do_sample=not greedy)
    expected = model(**inputs)
    runner = pipeline_module().PipelineExecutor(model, paths, cache=cache, do_sample=not greedy)
    calls = []
    handle = model.model.register_forward_hook(lambda *_: calls.append(1))
    actual, info = runner.decode(**inputs, audit=True, record_events=True)
    handle.remove()
    assert torch.equal(expected, actual)
    assert len(calls) == info["backbone_calls"] + info["audit_backbone_calls"]
    assert info["prefill_backbone_calls"] == 1
    assert info["backbone_calls"] == 1 + len(info["events"])
    assert 1 + sum(e["committed"] for e in info["events"]) == model._num_groups - 1
    assert info["backbone_calls"] <= model._num_groups - 1
    assert info["audit_max_logit_error"] < 1e-5
    # A different frame and rejected candidates cannot contaminate the next run.
    runner.decode(**dict(inputs, layer0_embed=inputs["layer0_embed"] + 7))
    again, _ = runner.decode(**inputs)
    assert torch.equal(actual, again)
    assert not info["graph_replays"]


@pytest.mark.parametrize("groups", [2, 3, 4])
@torch.inference_mode()
def test_pipeline_tail_and_serial_cache(research, groups):
    model, spec, inputs = research
    model._num_groups = groups
    inputs = dict(inputs, do_sample=False, sample_uniforms=inputs["sample_uniforms"][:, : groups - 1])
    actual, info = pipeline_module().PipelineExecutor(model, do_sample=False).decode(**inputs, audit=True)
    assert torch.equal(actual, model(**inputs))
    assert info["backbone_calls"] == groups - 1
    assert info["verification_backbone_calls"] == groups - 2


@pytest.mark.parametrize(
    "predictions,count,hits,path,parent",
    [([2, 0, 0, 0], 1, 0, [0, 0], 0), ([1, 2, 0, 0], 2, 1, [0, 1, 0], 1), ([1, 1, 1, 2], 4, 3, [0, 1, 2, 3, 0], 3)],
)
def test_pending_tail_parent_is_not_fabricated(research, predictions, count, hits, path, parent):
    result = pipeline_module().select_path(
        torch.tensor(predictions), torch.tensor([0, 1, 2]), torch.tensor([1, 1, 1]), 4
    )
    assert int(result[1]) == count
    assert int(result[2]) == hits
    assert result[3][: len(path)].tolist() == path
    assert int(result[4]) == parent


def test_pipeline_rejects_unsupported_graph_and_batch(research):
    model, _, inputs = research
    cls = pipeline_module().PipelineExecutor
    with pytest.raises(ValueError, match="requires NPU"):
        cls(model, execution="graph")
    runner = cls(model)
    with pytest.raises(ValueError, match="exactly one"):
        runner.decode(**dict(inputs, layer0_code=torch.zeros(2, 1)))
    with pytest.raises(ValueError, match="uniforms"):
        runner.decode(**dict(inputs, sample_uniforms=None))


@torch.inference_mode()
def test_pipeline_multiple_layers_and_fixed_padding(research):
    model, spec, inputs = research
    model.config.num_hidden_layers = 3
    model = type(model)(vllm_config=model._vllm_config, config=model.config, talker_config=model.talker_config).eval()
    cls = pipeline_module().PipelineExecutor
    paths = spec.fit_tree([[0, 0, 0]], 32)
    runner = cls(model, paths)
    actual, _ = runner.decode(**inputs, audit=True)
    assert torch.equal(actual, model(**inputs))
    for position, meta in runner.layouts.items():
        assert meta["positions"].shape == (1, 33)
        assert meta["positions"].max() < runner.capacity
        assert meta["mask"].ancestor_valid.any(-1).all()
        # Logical positions and visibility of siblings are independent of packing.
        active = meta["active"]
        for i, path in enumerate(active):
            visible = meta["mask"].ancestor_indices[i + 1][meta["mask"].ancestor_valid[i + 1]].tolist()
            assert visible == list(range(position + 1)) + [runner.capacity] + [
                runner.capacity + 1 + active.index(path[:d]) for d in range(1, len(path) + 1)
            ]


@torch.inference_mode()
def test_graph_scheduling_replays_new_frame_inputs_and_tail_lengths(research):
    """A CPU replay stand-in checks host scheduling; it cannot validate NPUGraph."""
    model, spec, inputs = research
    cls = pipeline_module().PipelineExecutor
    runner = cls(model, spec.fit_tree([[0, 0, 0]], 16))
    runner.prepare(inputs)
    runner.execution = "graph"

    class Replay:
        def __init__(self, fn):
            self.fn = fn

        def replay(self):
            self.fn()

    runner.graphs = {0: Replay(runner._prefill), **{p: Replay(lambda p=p: runner._step(p)) for p in runner.layouts}}
    addresses = [runner.codes.data_ptr(), runner.last_hidden.data_ptr(), *[x.keys.data_ptr() for x in runner.layers]]
    for offset in (0, 7, -3, 0):
        local = dict(inputs, layer0_embed=inputs["layer0_embed"] + offset)
        actual, info = runner.decode(**local, audit=True)
        assert torch.equal(actual, model(**local))
        assert info["graph_replays"] == info["backbone_calls"]
        assert addresses == [
            runner.codes.data_ptr(),
            runner.last_hidden.data_ptr(),
            *[x.keys.data_ptr() for x in runner.layers],
        ]


@torch.inference_mode()
def test_highperf_artifact_pipeline_and_certificate_binding(research, tmp_path, monkeypatch):
    model, spec, inputs = research
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "tools/qwen3_tts_spec"))
    module_spec = importlib.util.spec_from_file_location(
        "highperf_test", root / "tools/qwen3_tts_spec/highperf_experiment.py"
    )
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    capture = tmp_path / "capture"
    common = dict(
        input=capture,
        output=capture,
        device="cpu",
        vllm_repo=root.parent / "vllm",
        ascend_repo=root.parent / "vllm-ascend",
        progress_interval=10,
    )
    collect_args = SimpleNamespace(
        **common,
        prepare=True,
        model="synthetic-CustomVoice",
        model_revision="synthetic",
        seed=17,
        data=root / "benchmarks/build_dataset/seed_tts_smoke/en/meta.lst",
    )
    cli.exp.collect(collect_args)
    model.config.num_code_groups = model.talker_config.num_code_groups = 3
    model = type(model)(vllm_config=model._vllm_config, config=model.config, talker_config=model.talker_config).eval()
    session_module = importlib.import_module("vllm_omni.model_executor.models.common.qwen3_code_predictor_research")
    session = session_module.ResearchSession(capture / "collect_config.json")
    manifest = json.loads((capture / "manifest.json").read_text())
    for row in manifest["dataset"]:
        cli.exp.write_json(capture / "control.json", row)
        for _ in range(3 if row["id"] == "smoke011" else 1):
            session.forward(
                model,
                model.forward,
                inputs["layer0_code"],
                inputs["layer0_embed"],
                inputs["last_talker_hidden"],
                dict(do_sample=True, top_k=50),
            )
    cli.exp.analyze(SimpleNamespace(**common))
    original = {p.name: p.read_bytes() for p in capture.glob("*.json")}
    calibration = tmp_path / "calibration"
    cli.run(SimpleNamespace(**common, command="calibrate", results=calibration))
    trees = calibration / "trees.json"
    frozen = json.loads(trees.read_text())
    assert frozen["candidate_layout"] == cli.pipeline().CANDIDATE_LAYOUT
    assert frozen["reused_hidden_states"] > 0
    checked = tmp_path / "checked"
    args = SimpleNamespace(
        **common, command="check", trees=trees, results=checked, frames_per_request=3, sampling="both"
    )
    cli.run(args)
    certificate = json.loads((checked / "correctness.json").read_text())
    assert certificate["passed"]
    assert not certificate["graph_supported"]
    assert any(r["node_audit"] and r["stage"] == "sampled_eager" for r in certificate["checks"])
    bench_args = SimpleNamespace(
        **common,
        command="bench",
        trees=trees,
        results=tmp_path / "bench",
        frames_per_request=3,
        sampling="both",
        correctness=checked / "correctness.json",
        configs=["tree16"],
        warmup=1,
        executions_per_round=22,
    )
    cli.run(bench_args)
    summary = json.loads((bench_args.results / "summary.json").read_text())
    assert summary["performance_pass_threshold"] is None
    assert set(summary["modes"]) == {"greedy", "sampling"}
    for mode in summary["modes"].values():
        assert "tree16/pipeline_eager/tree16/pipeline_nocache" in mode
        for name, row in mode.items():
            assert row["candidate"]["calls_per_frame"] <= 4
            if "pipeline_" in name:
                assert row["candidate"]["call_breakdown"]["prefill_backbone_calls"] == 1
    assert "尚未测量完整请求" in (bench_args.results / "report.md").read_text(encoding="utf-8")
    assert all((capture / name).read_bytes() == value for name, value in original.items())
    cli.require_certificate(
        certificate, certificate["binding"], certificate["current_tree_sha256"], ["tree16/pipeline_eager"], ["tree16"]
    )
    for field in ("code", "source", "frames", "sampling", "device", "graph_scope"):
        modified = dict(certificate["binding"], **{field: "changed"})
        with pytest.raises(ValueError, match="does not match"):
            cli.require_certificate(certificate, modified, certificate["current_tree_sha256"], [], [])
    with pytest.raises(ValueError, match="incomplete"):
        cli.require_certificate(
            dict(certificate, passed=False), certificate["binding"], certificate["current_tree_sha256"], [], []
        )
    # Full requests remain explicitly gated by the sampled artifact.
    live_config = calibration / "tree16_pipeline_eager_config.json"
    with pytest.raises(ValueError, match="require a sampled"):
        session_module.ResearchSession(live_config)
    config = json.loads(live_config.read_text())
    config["correctness"] = str(checked / "correctness.json")
    cli.exp.write_json(live_config, config)
    live = session_module.ResearchSession(live_config)
    local = dict(inputs, sample_uniforms=spec.frame_uniforms(17, "live", 0, 3, 64, "cpu"))
    expected = model(**local)
    actual = live.forward(
        model,
        model.forward,
        local["layer0_code"],
        local["layer0_embed"],
        local["last_talker_hidden"],
        dict(sample_uniforms=local["sample_uniforms"]),
    )
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="were not checked"):
        live.forward(
            model,
            model.forward,
            local["layer0_code"],
            local["layer0_embed"],
            local["last_talker_hidden"],
            dict(temperature=0.5, sample_uniforms=local["sample_uniforms"]),
        )
    # A short graph probe never grants the 60-frame performance certificate.
    args.results, args.probe_only = tmp_path / "probe", True
    cli.run(args)
    probe = json.loads((args.results / "correctness.json").read_text())
    assert probe["passed"] and probe["scope"] == "fixed_probe_only"
    assert len(probe["binding"]["selected_frames"]) == 3
    assert all(row["stage"].startswith("small_") for row in probe["checks"])
    with pytest.raises(ValueError, match="probe-only"):
        cli.require_certificate(probe, probe["binding"], probe["current_tree_sha256"], [], [])
    args.probe_only = False
    # A changed output must stop immediately and preserve the actual KV state.
    original_decode = cli.Engines.decode

    def corrupt(self, name, data, **kwargs):
        codes, info = original_decode(self, name, data, **kwargs)
        if name == "cached_ar":
            codes[:, -1] = (codes[:, -1] + 1) % self.model.config.vocab_size
        return codes, info

    monkeypatch.setattr(cli.Engines, "decode", corrupt)
    args.results = tmp_path / "failed"
    with pytest.raises(AssertionError, match="codec codes differ"):
        cli.run(args)
    assert not json.loads((args.results / "correctness.json").read_text())["passed"]
    failure = torch.load(args.results / "failure.pt", weights_only=True)
    assert failure["pipeline"]["keys"]
    assert not (args.results / "timings.jsonl").exists()


def test_graph_capture_failure_is_explicit_and_never_eager_fallback(research, monkeypatch):
    model, _, inputs = research
    runner = pipeline_module().PipelineExecutor(model)
    runner.execution = "graph"

    def fail_capture(*args, **kwargs):
        raise RuntimeError("operator not capturable")

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            graph_pool_handle=lambda: None, synchronize=lambda: None, NPUGraph=lambda: None, graph=fail_capture
        ),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="graph capture failed at position 0"):
        runner.prepare(inputs)
    assert "operator not capturable" in runner.capture_error
    assert not runner.prepared
    assert not runner.graphs


@torch.inference_mode()
def test_captured_steps_never_extract_device_scalars(research):
    from torch.utils._python_dispatch import TorchDispatchMode

    model, spec, inputs = research
    runner = pipeline_module().PipelineExecutor(model, spec.fit_tree([[0, 0, 0]], 32))
    runner.prepare(inputs)

    class NoScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("device scalar extraction inside captured step")
            return func(*args, **(kwargs or {}))

    # Only the outer decode round boundary may transfer count/accepted to CPU.
    with NoScalarRead():
        runner._prefill()
        for position in runner.layouts:
            runner._step(position)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_pipeline_low_precision_and_upstream_dtype_alignment(research, dtype):
    model, spec, inputs = research
    model.to(dtype=dtype)
    inputs = dict(
        inputs,
        do_sample=False,
        layer0_embed=inputs["layer0_embed"].double(),
        last_talker_hidden=inputs["last_talker_hidden"].double(),
    )
    runner = pipeline_module().PipelineExecutor(model, spec.fit_tree([[0, 0, 0]], 16), do_sample=False)
    actual, info = runner.decode(**inputs, audit=True)
    assert torch.equal(actual, model(**inputs))
    assert info["audit_max_logit_error"] <= 0.02


@torch.inference_mode()
def test_backbone_graph_reuses_two_graphs_for_all_positions(research, monkeypatch):
    """CPU graph stand-in verifies scheduling/metadata, not actual NPU kernels."""
    from contextlib import contextmanager

    model, spec, inputs = research
    runner = pipeline_module().PipelineExecutor(model, spec.fit_tree([[0, 0, 0]], 16), graph_scope="backbone")
    runner.execution = "graph"
    captured, reset = [], []

    class Graph:
        def replay(self):
            self.fn()

        def reset(self):
            reset.append(self)

    @contextmanager
    def capture(graph, **kwargs):
        captured.append(runner.capture_position)
        graph.fn = runner._prefill if runner.capture_position == 0 else runner._backbone
        yield

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            graph_pool_handle=lambda: None,
            synchronize=lambda: None,
            NPUGraph=Graph,
            graph=capture,
            memory_allocated=lambda: 100,
            memory_reserved=lambda: 200,
            mem_get_info=lambda: (300, 400),
        ),
        raising=False,
    )
    runner.prepare(inputs)
    assert captured == [0, "backbone"]
    assert len(runner.graphs) == 2
    for offset in (0, 7, -3, 0):
        local = dict(inputs, layer0_embed=inputs["layer0_embed"] + offset)
        calls = []
        handle = model.model.register_forward_hook(lambda *_: calls.append(1))
        actual, info = runner.decode(**local, audit=True)
        handle.remove()
        assert len(calls) == info["backbone_calls"] + info["audit_backbone_calls"]
        assert torch.equal(actual, model(**local))
        assert info["graph_scope"] == "backbone"
    assert runner.capture_metadata()["memory_before_capture"][0]["mem_get_info"] == [300, 400]
    runner.close()
    assert len(reset) == 2 and not runner.graphs and not runner.prepared


def load_highperf_cli(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "tools/qwen3_tts_spec"))
    module_spec = importlib.util.spec_from_file_location(
        "highperf_failure_test", root / "tools/qwen3_tts_spec/highperf_experiment.py"
    )
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    return cli


def test_capture_failure_diagnostics_never_read_the_damaged_device(research, monkeypatch, tmp_path):
    model, _, inputs = research
    cli = load_highperf_cli(monkeypatch)
    engines = cli.Engines(model, {}, {})
    runner = pipeline_module().PipelineExecutor(model)
    runner.capture_error = "allocation failure 207001"
    runner.capture_position = 3
    engines.runners[("tree32/pipeline_graph", True)] = runner
    error = pipeline_module().PipelineCaptureError("capture failed tree32 sampling position 3")
    error.__cause__ = RuntimeError("original allocation error 207001")
    reads = []

    def poisoned_cpu(tensor, *args, **kwargs):
        reads.append(1)
        raise RuntimeError("stream is still captured; secondary error 107030")

    with monkeypatch.context() as guarded:
        guarded.setattr(torch.Tensor, "cpu", poisoned_cpu)
        cli.save_failure(
            SimpleNamespace(results=tmp_path),
            engines,
            "tree32/pipeline_graph",
            inputs,
            inputs["layer0_code"],
            error,
            dict(stage="small_graph", frame_id=0),
        )
    assert not reads
    data = json.loads((tmp_path / "failure.json").read_text())
    assert "207001" in data["exception_chain"][-1]
    assert data["device_snapshot_skipped"]
    assert data["graphs"]["tree32/pipeline_graph/sampling"]["position"] == 3
    saved = torch.load(tmp_path / "failure.pt", weights_only=True)
    assert saved["device_snapshot_skipped"]
    assert "inputs" not in saved


def test_snapshot_failure_does_not_replace_original_error(research, monkeypatch, tmp_path):
    model, _, inputs = research
    cli = load_highperf_cli(monkeypatch)
    engines = cli.Engines(model, {}, {})

    def poisoned_cpu(tensor, *args, **kwargs):
        raise RuntimeError("secondary snapshot error")

    with monkeypatch.context() as guarded:
        guarded.setattr(torch.Tensor, "cpu", poisoned_cpu)
        cli.save_failure(
            SimpleNamespace(results=tmp_path),
            engines,
            "stock_graph",
            inputs,
            None,
            ValueError("original numerical mismatch"),
            {},
        )
    data = json.loads((tmp_path / "failure.json").read_text())
    assert "original numerical mismatch" in data["error"]
    assert "secondary snapshot error" in data["snapshot_error"]


def test_graph_sets_are_released_between_sampling_modes(research, monkeypatch):
    model, _, _ = research
    cli = load_highperf_cli(monkeypatch)
    engines = cli.Engines(model, {}, {})
    released = []

    class Graph:
        def reset(self):
            released.append(1)

    for mode in (False, True):
        for config in ("tree8", "tree16", "tree32"):
            runner = pipeline_module().PipelineExecutor(model)
            runner.execution = "graph"
            runner.graphs = {p: Graph() for p in range(15)}
            engines.runners[(f"{config}/pipeline_graph", mode)] = runner
    engines.release_graphs()
    assert len(released) == 90
    assert not engines.runners
    assert all(record["released"] and record["resident_graphs"] == 0 for record in engines.graph_info().values())
