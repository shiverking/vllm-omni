# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from .test_cached_ar import models  # noqa: F401
from .test_code_predictor_dtype import loaded_target_classes  # noqa: F401

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("ar_experiment_test", ROOT / "tools/qwen3_tts_ar_graph/experiment.py")
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "source"
    (root / "frames").mkdir(parents=True)
    dataset = [
        dict(id=f"smoke{i:03d}", text=f"text {i}", split="calibration" if i <= 10 else "test") for i in range(1, 31)
    ]
    exp.write_json(root / "manifest.json", dict(model=exp.MODEL, dataset=dataset))
    for row in dataset[10:]:
        for frame in range(3):
            torch.save(
                dict(
                    request_id=row["id"],
                    frame_id=frame,
                    split="test",
                    inputs=dict(
                        layer0_code=torch.tensor([1]),
                        layer0_embed=torch.randn(1, 32),
                        last_talker_hidden=torch.randn(1, 32),
                    ),
                ),
                root / "frames" / f"{row['id']}-{frame}.pt",
            )
    torch.save(dict(config=dict(num_code_groups=16)), root / "predictor.pt")
    return root


def test_select_fixed_samples(capture):
    assert len(exp.select_frames(capture)) == 60
    probe = exp.select_frames(capture, True)
    assert [(r["request_id"], r["frame_id"]) for r in probe] == [("smoke011", 0), ("smoke011", 2), ("smoke030", 2)]
    (capture / "frames/smoke011-0.pt").unlink()
    with pytest.raises(ValueError, match="fewer"):
        exp.select_frames(capture)


def test_uniforms_bound_to_identity_not_batch(capture):
    frames = exp.select_frames(capture)[:4]
    together = exp.uniforms_for(frames, 64, "cpu")
    independent = torch.cat([exp.uniforms_for([f], 64, "cpu") for f in frames])
    assert torch.equal(together, independent)
    assert not torch.equal(together, exp.uniforms_for(frames, 64, "cpu", 1))
    assert torch.equal(exp.uniforms_for(frames[::-1], 64, "cpu"), together.flip(0))


def test_certificate_rejects_changed_scope(tmp_path):
    path = tmp_path / "correctness.json"
    args = SimpleNamespace(correctness=path)
    exp.write_json(path, dict(passed=True, probe_only=False, binding={"code": "original"}))
    exp.certificate(args, {"code": "original"})
    with pytest.raises(ValueError):
        exp.certificate(args, {"code": "changed"})
    exp.write_json(path, dict(passed=True, probe_only=True, binding={"code": "original"}))
    with pytest.raises(ValueError):
        exp.certificate(args, {"code": "original"})


def test_import_is_read_only_and_detects_source_change(capture, tmp_path):
    output = tmp_path / "imported"
    output.mkdir()
    args = SimpleNamespace(import_from=capture, model=exp.MODEL, results=output)
    exp.collect(args, exp.Progress(output))
    assert exp.capture_root(output) == capture
    with (capture / "predictor.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="predictor changed"):
        exp.capture_root(output)


def test_cpu_probe_exercises_real_models(capture, tmp_path, models, monkeypatch):  # noqa: F811
    w, _ = models
    output = tmp_path / "check"
    output.mkdir()
    args = SimpleNamespace(input=capture, results=output, probe_only=True, device="cpu", model=exp.MODEL)
    monkeypatch.setattr(exp, "load_predictor", lambda *unused: w)
    monkeypatch.setattr(exp, "binding", lambda *unused: {"cpu": True})
    exp.check(args, exp.Progress(output))
    certificate = exp.read_json(output / "correctness.json")
    assert certificate["passed"] and certificate["probe_only"]
    assert not certificate["graph_verified"]


def test_failure_saved_before_performance(capture, tmp_path, models, monkeypatch):  # noqa: F811
    w, executor = models
    output = tmp_path / "failure"
    output.mkdir()
    args = SimpleNamespace(input=capture, results=output, probe_only=True, device="cpu", model=exp.MODEL)
    monkeypatch.setattr(exp, "load_predictor", lambda *unused: w)
    monkeypatch.setattr(exp, "binding", lambda *unused: {})
    original = type(executor).__call__

    def corrupt(self, *a, **kw):
        return original(self, *a, **kw) + 1

    monkeypatch.setattr(type(executor), "__call__", corrupt)
    with pytest.raises(AssertionError, match="codes differ"):
        exp.check(args, exp.Progress(output))
    assert (output / "failure.pt").exists()
    assert not exp.read_json(output / "correctness.json")["passed"]


@torch.inference_mode()
def test_bench_wiring_with_cpu_standins(capture, tmp_path, models, monkeypatch):  # noqa: F811
    import importlib

    w, eager = models
    module = importlib.import_module("vllm_omni.model_executor.models.common.qwen3_code_predictor_ar")
    real_executor = module.CachedARExecutor
    monkeypatch.setattr(module, "CachedARExecutor", lambda wrapper, mode: real_executor(wrapper, "eager"))
    output = tmp_path / "bench"
    output.mkdir()
    cert = tmp_path / "certificate.json"
    exp.write_json(cert, dict(passed=True, probe_only=False, binding={}))
    args = SimpleNamespace(
        input=capture,
        results=output,
        device="npu:0",
        model=exp.MODEL,
        correctness=cert,
        warmup=20,
        executions_per_round=60,
    )
    monkeypatch.setattr(exp, "load_predictor", lambda *unused: w)
    monkeypatch.setattr(exp, "binding", lambda *unused: {})
    monkeypatch.setattr(exp, "BUCKETS", [1])
    original_inputs, original_uniforms = exp.batch_inputs, exp.uniforms_for
    monkeypatch.setattr(exp, "batch_inputs", lambda frames, device: original_inputs(frames, "cpu"))
    monkeypatch.setattr(exp, "uniforms_for", lambda frames, vocab, device: original_uniforms(frames, vocab, "cpu"))
    api = SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda: 0, empty_cache=lambda: None)
    monkeypatch.setattr(exp, "device_api", lambda *unused: api)
    monkeypatch.setattr(exp, "measured", lambda api, fn: (fn(), 1.0, 0.9))
    exp.bench(args, exp.Progress(output))
    result = exp.read_json(output / "summary.json")
    assert len(result) == 4
    assert all(row["speedup"] == 1.0 for row in result)


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({"passed": False, "error": "logits differ", "context": {"position": 3}}, "正确性检查未通过"),
        ({"passed": True, "probe_only": True, "binding": {}}, "三帧预检"),
        ({"passed": True, "binding": {"code": {"executor.py": "old"}}}, "绑定差异"),
        ([], "无法读取"),
    ],
)
def test_certificate_explains_rejection(tmp_path, capsys, payload, reason):
    path = tmp_path / "correctness.json"
    exp.write_json(path, payload)
    args = SimpleNamespace(correctness=path, results=tmp_path / "results")
    with pytest.raises(ValueError, match="性能测试未启动"):
        exp.certificate(args, {})
    assert reason in capsys.readouterr().out
    report = exp.read_json(args.results / "certificate_diagnostic.json")
    assert report["reasons"]
    if isinstance(payload, dict) and payload.get("error"):
        assert report["previous_error"] == payload["error"]
        assert report["previous_context"] == payload["context"]


@pytest.mark.parametrize("contents", [None, "{invalid"])
def test_certificate_unreadable(tmp_path, capsys, contents):
    path = tmp_path / "correctness.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="证书校验失败"):
        exp.certificate(SimpleNamespace(correctness=path), {})
    assert "无法读取" in capsys.readouterr().out


def test_binding_differences_reports_exact_fields():
    differences = exp.binding_differences(
        {"code": {"executor.py": "old"}, "frames": {"a": None}},
        {"code": {"executor.py": "new"}, "frames": {"b": None}},
    )
    assert [row["field"] for row in differences] == [
        "binding.code.executor.py",
        "binding.frames.a",
        "binding.frames.b",
    ]
    assert differences[0]["saved"] == "old"
    assert differences[0]["current"] == "new"
    assert differences[1]["current_present"] is False


@torch.inference_mode()
def test_prefill_diagnostic_matches_cpu_and_removes_hooks(models, monkeypatch):  # noqa: F811
    import sys

    monkeypatch.setitem(sys.modules, "experiment", exp)
    spec = importlib.util.spec_from_file_location(
        "prefill_diagnostic", ROOT / "tools/qwen3_tts_ar_graph/diagnose_prefill.py"
    )
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    wrapper, executor = models
    buffers = executor._allocate(2)
    buffers["inputs"].normal_()
    short = buffers["inputs"][:, :2]
    full = torch.zeros(2, wrapper._num_groups + 1, short.shape[-1])
    full[:, :2].copy_(short)
    long_positions = torch.arange(full.shape[1]).unsqueeze(0).expand(2, -1)
    short_positions = long_positions[:, :2]
    long_trace = diagnostic.trace_prefill(wrapper, full, long_positions)
    short_trace = diagnostic.trace_prefill(wrapper, short, short_positions)
    cache_trace = diagnostic.trace_prefill(wrapper, short, short_positions, buffers["cache"])
    assert "layers.0.self_attn.o_proj.input" in cache_trace
    padded_trace = diagnostic.trace_prefill(wrapper, short, short_positions, buffers["cache"], pad_down_proj=True)
    assert all("forward" not in layer.mlp.down_proj.__dict__ for layer in wrapper.model.layers)
    for trace in (short_trace, cache_trace, padded_trace):
        assert diagnostic.compare(trace["head1.logits"], long_trace["head1.logits"])["outside_tolerance"] == 0
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in wrapper.model.modules())


def test_stock_row_difference_reports_first_code_and_its_logits():
    batch = torch.tensor([[0, 1, 2], [0, 2, 1]])
    single = torch.tensor([[0, 1, 2]])
    batch_logits = [torch.tensor([[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]])] * 2
    single_logits = [torch.tensor([[0.0, 2.0, 1.0]])] * 2
    assert exp.stock_row_difference(batch, single, batch_logits, single_logits, 0) is None
    result = exp.stock_row_difference(batch, single, batch_logits, single_logits, 1)
    assert result["position"] == 1
    assert result["different_positions"] == [1, 2]
    assert result["first_token_logits"]["stock_batch_top5"][0] == 2
    assert result["first_token_logits"]["stock_single_top5"][0] == 1


@pytest.mark.parametrize("policy", ["strict", "record-stable"])
def test_stock_variation_policy_is_explicit(capture, tmp_path, models, monkeypatch, policy):  # noqa: F811
    wrapper, _ = models
    output = tmp_path / "check_policy"
    output.mkdir()
    args = SimpleNamespace(
        input=capture, results=output, probe_only=True, device="cpu", model=exp.MODEL, stock_batch_policy=policy
    )
    monkeypatch.setattr(exp, "load_predictor", lambda *unused: wrapper)
    monkeypatch.setattr(exp, "binding", lambda *unused: {})
    monkeypatch.setattr(exp, "CHECK_BATCHES", [1])
    # Inject one comparison result to test policy plumbing independently of CPU rounding.
    seen = []

    def one_variation(*unused):
        if seen:
            return None
        seen.append(True)
        return dict(row=0, position=1, different_positions=[1])

    monkeypatch.setattr(exp, "stock_row_difference", one_variation)
    if policy == "strict":
        with pytest.raises(AssertionError, match="Stock batch vs single"):
            exp.check(args, exp.Progress(output))
        assert not exp.read_json(output / "correctness.json")["passed"]
    else:
        exp.check(args, exp.Progress(output))
        cert = exp.read_json(output / "correctness.json")
        assert cert["passed"] and not cert["stock_batch_invariant"]
        assert cert["correctness_scope"] == "cached_vs_stock_same_batch"
        assert len(cert["stock_batch_variations"]) == 1
        assert (output / "stock_batch_variation_first.pt").exists()


@torch.inference_mode()
def test_decode_diagnostic_uses_requested_position(models, monkeypatch):  # noqa: F811
    import sys

    monkeypatch.syspath_prepend(str(ROOT / "tools/qwen3_tts_ar_graph"))
    monkeypatch.setitem(sys.modules, "experiment", exp)
    spec = importlib.util.spec_from_file_location(
        "decode_diagnostic", ROOT / "tools/qwen3_tts_ar_graph/diagnose_decode.py"
    )
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    w, executor = models
    b = executor._allocate(2)
    b["inputs"].normal_()
    executor._backbone(b, 0)
    full = torch.zeros(2, 17, w.config.hidden_size)
    full[:, :3].copy_(b["inputs"][:, :3])
    positions = torch.arange(17).unsqueeze(0).expand(2, -1)
    reference = diagnostic.trace_position(w, full, positions, 2, 2)
    cached = diagnostic.trace_position(w, b["inputs"][:, 2:3], positions[:, 2:3], 2, 0, b["cache"], 2)
    torch.testing.assert_close(reference["head.logits"], cached["head.logits"], atol=2e-5, rtol=2e-5)
    assert "layers.0.self_attn.o_proj.input" in cached
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in w.model.modules())
