# SPDX-License-Identifier: Apache-2.0
import importlib.util
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    folder = ROOT / "tools/qwen3_tts_codebook_stats"
    exp = load("stats_experiment_test", folder / "experiment.py")
    monkeypatch.setitem(sys.modules, "experiment", exp)
    analysis = load("stats_analysis_test", folder / "analysis.py")
    capture = load(
        "stats_capture_test", ROOT / "vllm_omni/model_executor/models/common/qwen3_code_predictor_capture.py"
    )
    return exp, analysis, capture


@pytest.fixture
def prepared(modules, tmp_path):
    exp, _, _ = modules
    args = exp.parser().parse_args(["prepare", "--results", str(tmp_path / "run")])
    exp.prepare(args)
    return args.results, exp.read_json(args.results / "manifest.json")


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x01\x00" * 2400)
    return buffer.getvalue()


def record(ident="a", split="statistics", codes=None, ids=None):
    codes = codes if codes is not None else [[1, 1, 2, 3], [1, 1, 2, 3]]
    ids = ids if ids is not None else range(len(codes))
    return dict(
        request=dict(id=ident + "-s42", text_id=ident, split=split),
        audio=f"audio/{ident}.wav",
        rows=[dict(frame_id=i, codes=c, valid=True) for i, c in zip(ids, codes)],
    )


def test_dataset_split_freeze_and_prepare_resume(modules, prepared):
    exp, _, _ = modules
    root, manifest = prepared
    rows = exp.dataset()
    assert len(rows) == 100 and sum(r["split"] == "statistics" for r in rows) == 70
    original = (ROOT / "benchmarks/build_dataset/seed_tts_smoke/en/meta.lst").read_text(encoding="utf-8").splitlines()
    assert [r["text"] for r in rows[:30]] == [line.split("|")[-1] for line in original]
    requests = exp.requests_for(manifest)
    assert len(requests) == len({r["id"] for r in requests}) == 300
    for text in rows:
        group = [r for r in requests if r["text_id"] == text["id"]]
        assert {r["seed"] for r in group} == {42, 2026, 3407}
        assert {r["split"] for r in group} == {text["split"]}
    before = (root / "manifest.json").read_bytes()
    exp.prepare(exp.parser().parse_args(["prepare", "--results", str(root)]))
    assert (root / "manifest.json").read_bytes() == before


def fake_response(modules, root, manifest, monkeypatch, bad=False):
    exp, _, capture = modules
    seen = []

    def respond(request, timeout):
        payload = json.loads(request.data)
        control = exp.read_json(root / "control.json")
        assert payload["seed"] == control["seed"]
        assert payload["response_format"] == "wav" and payload["voice"] == "Vivian"

        class Wrapper:
            ar_capture_root = str(root)
            _ar_capture_active = False

            def __call__(self, **kw):
                seen.append(self._ar_capture_active)
                return torch.tensor([[control["seed"] % 10 + 1] + [2] * 15])

        wrapper = Wrapper()
        for _ in range(5):
            capture.capture_frame(
                wrapper, torch.ones(1), torch.zeros(1, 32), torch.zeros(1, 32), **manifest["residual_sampling"]
            )
        return io.BytesIO(b"not a wav" if bad else wav_bytes())

    monkeypatch.setattr(exp, "local_open", respond)
    return seen


def test_real_response_artifact_commit_resume_and_retry(modules, prepared, monkeypatch):
    exp, _, _ = modules
    root, manifest = prepared
    request = exp.requests_for(manifest)[0]
    fake_response(modules, root, manifest, monkeypatch, bad=True)
    with pytest.raises((wave.Error, EOFError)):
        exp.capture_one(root, request, manifest, "http://127.0.0.1:8091", 10)
    assert not exp.is_complete(root, request)
    assert (root / "failures" / f"{request['id']}.json").exists()
    fake_response(modules, root, manifest, monkeypatch)
    info = exp.capture_one(root, request, manifest, "http://127.0.0.1:8091", 10)
    assert info["duration_seconds"] == 0.1 and info["sample_rate"] == 24000
    assert exp.is_complete(root, request)
    assert len(exp.read_codes(root, request)) == 5  # previous attempt was replaced, not appended
    assert not (root / "control.json").exists() and not (root / "predictor.pt").exists()
    assert not list(root.glob("frames/*.pt"))
    assert not (root / "failures" / f"{request['id']}.json").exists()
    (root / "audio" / f"{request['id']}.wav").write_bytes(b"corrupt")
    assert not exp.is_complete(root, request)


def test_proxy_bypass(modules, monkeypatch):
    exp, _, _ = modules
    handlers = []

    class Opener:
        def open(self, request, timeout):
            return (request, timeout)

    monkeypatch.setattr(exp.urllib.request, "build_opener", lambda h: handlers.append(h) or Opener())
    assert exp.local_open("http://localhost/health", 5) == ("http://localhost/health", 5)
    assert handlers[0].proxies == {}


@pytest.mark.parametrize(
    "code,valid",
    [([0] * 16, False), ([1] * 16, True), ([2048] + [1] * 15, False), ([-1] + [1] * 15, False), ([1] + [0] * 15, True)],
)
def test_capture_filter_and_attempt_reset(modules, prepared, code, valid):
    exp, _, capture = modules
    root, manifest = prepared
    request = exp.requests_for(manifest)[0]
    exp.write_json(root / "control.json", dict(request, format=exp.FORMAT, attempt="first"))

    class Wrapper:
        ar_capture_root = str(root)
        _ar_capture_active = False

        def __call__(self, **kwargs):
            return torch.tensor([code])

    wrapper = Wrapper()
    for _ in range(2):
        result = capture.capture_frame(
            wrapper, torch.ones(1), torch.zeros(1), torch.zeros(1), **manifest["residual_sampling"]
        )
        assert result.tolist() == [code] and not wrapper._ar_capture_active
    path = root / "codes" / f"{request['id']}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["frame_id"] for r in rows] == [0, 1]
    assert all(r["valid"] == valid for r in rows)
    exp.write_json(root / "control.json", dict(request, format=exp.FORMAT, attempt="second"))
    capture.capture_frame(wrapper, torch.ones(1), torch.zeros(1), torch.zeros(1), **manifest["residual_sampling"])
    assert json.loads(path.read_text().splitlines()[-1])["frame_id"] == 0


def test_capture_disabled_no_artifacts(modules, tmp_path):
    _, _, capture = modules

    class Wrapper:
        ar_capture_root = str(tmp_path)
        _ar_capture_active = False

        def __call__(self, **kw):
            return torch.ones(1, 16, dtype=torch.long)

    wrapper = Wrapper()
    assert capture.capture_frame(wrapper, torch.ones(1), torch.ones(1), torch.ones(1)).shape == (1, 16)
    assert not list(tmp_path.iterdir()) and not wrapper._ar_capture_active


def test_entropy_independent_and_deterministic_mi(modules):
    _, a, _ = modules
    x = np.array([0, 0, 1, 1] * 20)
    y = np.array([0, 1, 0, 1] * 20)
    assert a.mutual_information(x, y) == pytest.approx(0)
    assert a.mutual_information(x, x) == pytest.approx(1)
    assert a.entropy([0, 10, 0]) == pytest.approx(0)


def test_fit_excludes_test_and_transitions_do_not_cross_gaps_or_requests(modules):
    _, a, _ = modules
    train = record(codes=[[1, 2, 3, 0], [2, 1, 0, 3], [3, 0, 1, 2]], ids=[0, 1, 3])
    test = record("b", "test", [[3, 3, 3, 3]] * 20)
    tables = a.Tables(4, 4).fit([train, test, record("c", codes=[[2, 2, 2, 2]])])
    assert tables.counts[0].sum() == 4
    assert sum(sum(v.values()) for (mode, _, _), v in tables.conditionals.items() if mode == "temporal") == 4
    before = tables.counts.copy(), dict(tables.conditionals)
    test["rows"] = record(codes=[[0, 0, 0, 1]])["rows"]
    other = a.Tables(4, 4).fit([train, test, record("c", codes=[[2, 2, 2, 2]])])
    np.testing.assert_array_equal(other.counts, before[0])
    assert dict(other.conditionals) == before[1]
    np.testing.assert_array_equal(tables.probability("main", 1, 99, 0), tables.priors[1])
    counts = tables.conditionals[("main", 1, 1)]
    expected = 32 * tables.priors[1].copy()
    expected[2] += 1
    np.testing.assert_allclose(tables.probability("main", 1, 1, 0), expected / (32 + sum(counts.values())))


def test_tree_ancestry_budget_branch_conditioning_tail_and_no_future(modules):
    _, a, _ = modules
    # Two different continuations conditioned on the hypothesized predecessor.
    train = record(codes=[[1, 1, 2, 3]] * 100 + [[2, 2, 3, 1]] * 100)
    tables = a.Tables(8, 4).fit([train])
    full = tables.tree("previous", 1, 1, 1)
    for budget in (8, 16, 32):
        nodes = full[:budget]
        assert len(nodes) <= budget
        assert all(len(path) == 1 or path[:-1] in nodes for path in nodes)
    assert (1, 2) in full and (2, 3) in full
    assert all(len(path) == 1 for path in tables.tree("mixed", 3, 1, 2))
    assert tables.tree("global", 4, 1, 2) == ()
    unchanged = tables.tree("previous", 1, 1, 1)
    assert a.path_coverage(full, (1, 2, 3)) == 3
    assert a.path_coverage(full, (7, 7, 7)) == 0
    assert tables.tree("previous", 1, 1, 1) == unchanged


def test_grouped_ci_keeps_repeated_seeds_together(modules):
    _, a, _ = modules
    rows = [dict(text_id="a", count=10, hit4=8), dict(text_id="b", count=10, hit4=2)]
    assert a.grouped_ci(rows, "hit4") == a.grouped_ci(rows * 3, "hit4")
    assert a.grouped_ci(rows[:1] * 3, "hit4") is None


def test_heldout_conditional_predictive_gain_and_temporal_pairing(modules):
    _, a, _ = modules
    codes = [[i, (i + 1) % 16, (i + 2) % 16, (i + 3) % 16] for i in range(16)]
    tables = a.Tables(16, 4).fit([record(codes=codes * 100)])
    rows, trees, examples = a.evaluate(tables, [record("test", "test", codes)])
    summary, tree_summary, _ = a.aggregate(rows, trees)
    overall = {r["mode"]: r for r in summary if r["position"] == "all"}
    assert overall["main"]["top1"] > overall["global"]["top1"]
    assert overall["main"]["nll_nats"] < overall["global"]["nll_nats"]
    assert overall["temporal"]["count"] == 15 * 3
    assert tree_summary and examples


def test_end_to_end_partial_analysis_and_saved_figures(modules, prepared, monkeypatch):
    exp, a, _ = modules
    root, manifest = prepared
    requests = exp.requests_for(manifest)
    selected = [r for r in requests if r["split"] == "statistics"][:2] + [r for r in requests if r["split"] == "test"][
        :2
    ]
    fake_response(modules, root, manifest, monkeypatch)
    for request in selected:
        exp.capture_one(root, request, manifest, "http://localhost", 10)
    a.analyze(root)
    summary = exp.read_json(root / "analysis/summary.json")
    assert summary["completed_audio"] == 4 and summary["statistics_audio"] == summary["test_audio"] == 2
    assert len(summary["incomplete_requests"]) == 296 and summary["coverage"]
    assert summary["raw_frames"] == 20 and summary["invalid_frames"] == 0
    assert "4/300" in (root / "report.md").read_text(encoding="utf-8")
    assert len(list((root / "analysis").glob("*.png"))) == 4
    assert (root / "analysis/conditional_counts.csv").exists()
    a.report(root)  # report is independent of generation and analysis execution
