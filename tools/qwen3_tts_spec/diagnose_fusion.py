# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolate shape/layout differences in one saved NPU tree audit failure."""

import argparse
import json
from pathlib import Path
from types import MethodType

import experiment as exp
import torch


def compact_ancestor_qkv(q, k, v, visible, width):
    """Diagnostic-only gather: one batch element per query, fixed KV width."""
    batch, heads, seq, dim = q.shape
    indices, valid = [], []
    for row in visible.cpu().tolist():
        ancestors = [i for i, allowed in enumerate(row) if allowed]
        if not 0 < len(ancestors) <= width:
            raise ValueError("ancestor count outside fixed KV width")
        indices.append(ancestors + [0] * (width - len(ancestors)))
        valid.append([True] * len(ancestors) + [False] * (width - len(ancestors)))
    index = torch.tensor(indices, device=q.device).reshape(-1)

    def gather(value):
        return (
            value.index_select(2, index)
            .reshape(batch, heads, seq, width, dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch * seq, heads, width, dim)
        )

    visibility = torch.tensor(valid, dtype=torch.bool, device=q.device)
    visibility = visibility[None].expand(batch, -1, -1).reshape(batch * seq, 1, 1, width)
    return q.transpose(1, 2).reshape(batch * seq, heads, 1, dim), gather(k), gather(v), visibility


def install_probe(model, compact_kv=False):
    """Process-local explicit-mask backend; never changes stock attention."""
    import torch_npu

    for module in model.model.modules():
        if not hasattr(module, "_forward_npu_attention"):
            continue
        original = module.forward

        def forward(self, hidden_states, position_embeddings, attention_mask=None, _original=original):
            if attention_mask is None:
                return _original(hidden_states, position_embeddings)
            batch, seq, _ = hidden_states.shape
            qraw, kraw, vraw = self._split_qkv(self.qkv_proj(hidden_states))
            q = self.q_norm(qraw.reshape(batch, seq, self.num_heads, self.head_dim)).transpose(1, 2)
            k = self.k_norm(kraw.reshape(batch, seq, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            v = vraw.reshape(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = position_embeddings
            q = torch_npu.npu_rotary_mul(q, cos.unsqueeze(1), sin.unsqueeze(1))
            k = torch_npu.npu_rotary_mul(k, cos.unsqueeze(1), sin.unsqueeze(1))
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            visible = getattr(attention_mask, "visibility", attention_mask)
            if compact_kv:
                q, k, v, visible = compact_ancestor_qkv(q, k, v, visible, model._num_groups + 1)
            attention = torch_npu.npu_fusion_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                self.num_heads,
                "BNSD",
                pse=None,
                padding_mask=None,
                atten_mask=(~visible).contiguous(),
                scale=float(self.scaling),
                keep_prob=1.0,
                pre_tockens=2147483647,
                next_tockens=2147483647,
                inner_precise=0,
                prefix=None,
                actual_seq_qlen=None,
                actual_seq_kvlen=None,
                sparse_mode=0,
                gen_mask_parallel=True,
                sync=True,
            )[0]
            if compact_kv:
                attention = attention.reshape(batch, seq, self.num_heads * self.head_dim)
            else:
                attention = attention.transpose(1, 2).reshape(batch, seq, -1)
            return self.o_proj(attention)

        module.forward = MethodType(forward, module)


def difference(left, right):
    delta = (left.double() - right.double()).abs()
    return dict(exact_equal=torch.equal(left, right), max_abs=float(delta.max()), mean_abs=float(delta.mean()))


def padded_position_ids(length, total, device):
    """Padding is physical storage, not additional logical RoPE positions."""
    if not 0 < length <= total:
        raise ValueError("expected 0 < logical length <= padded length")
    positions = torch.zeros((1, total), dtype=torch.long, device=device)
    positions[:, :length] = torch.arange(length, device=device)
    return positions


@torch.inference_mode()
def diagnose(args):
    import torch_npu  # noqa: F401

    payload = torch.load(args.output / "fusion_model_probe_node_failure.pt", map_location="cpu", weights_only=True)
    print("Loading predictor for four single-node forwards...", flush=True)
    model = exp.load_predictor(args)
    install_probe(model, args.compact_kv)
    packed = payload["packed"].to(args.device)
    serial = payload["serial"].to(args.device)
    positions = payload["positions"].to(args.device)
    mask = payload["mask"].to(args.device)
    indices = payload["serial_indices"]
    n, length = packed.shape[1], serial.shape[1]
    idx = torch.tensor(indices, device=args.device)
    assert torch.equal(packed.index_select(1, idx), serial), "Branch embeddings differ"
    assert torch.equal(positions.index_select(1, idx), torch.arange(length, device=args.device)[None])
    assert torch.equal(
        mask.index_select(0, idx).index_select(1, idx),
        torch.ones(length, length, dtype=torch.bool, device=args.device).tril(),
    )
    assert mask[idx].sum().item() == length * (length + 1) // 2, "Branch reads non-ancestors"

    padded = packed.new_zeros(packed.shape)
    padded[:, :length] = serial
    padded_mask = torch.eye(n, dtype=torch.bool, device=args.device)
    padded_mask[:length, :length] = torch.ones(length, length, dtype=torch.bool, device=args.device).tril()
    permutation = torch.tensor(indices + [i for i in range(n) if i not in indices], device=args.device)
    cases = {
        "serial": (
            serial,
            torch.arange(length, device=args.device)[None],
            padded_mask[:length, :length],
            list(range(length)),
        ),
        "serial_padded": (padded, padded_position_ids(length, n, args.device), padded_mask, list(range(length))),
        "tree": (packed, positions, mask, indices),
        "tree_compact": (
            packed.index_select(1, permutation),
            positions.index_select(1, permutation),
            mask.index_select(0, permutation).index_select(1, permutation),
            list(range(length)),
        ),
    }
    traces, logits = {}, {}
    # Fail on the host before any forward can launch an invalid RoPE lookup.
    rope_size = model.model.rotary_emb.cos_cached.shape[0]
    for label, (_, pos, _, _) in cases.items():
        cpu_positions = pos.cpu()
        if int(cpu_positions.min()) < 0 or int(cpu_positions.max()) >= rope_size:
            raise ValueError(f"{label}: logical positions exceed RoPE cache size {rope_size}")
    for label, (values, pos, visibility, branch) in cases.items():
        print(f"Forward: {label}, length={values.shape[1]}", flush=True)
        trace, handles = {}, []
        branch_idx = torch.tensor(branch, device=args.device)

        def hook(name):
            def record(module, inputs, result):
                if isinstance(result, torch.Tensor) and result.ndim >= 3 and result.shape[1] == values.shape[1]:
                    trace[name] = result.index_select(1, branch_idx).detach().cpu()

            return record

        def input_hook(name):
            def record(module, inputs):
                # o_proj's input is the reshaped fusion-attention result.
                # Observe it before GEMM so an output mismatch cannot be
                # incorrectly attributed to the projection itself.
                trace[name + ".input"] = inputs[0].index_select(1, branch_idx).detach().cpu()

            return record

        for name, module in model.model.named_modules():
            if name.endswith("self_attn.o_proj"):
                handles.append(module.register_forward_pre_hook(input_hook(name)))
            handles.append(module.register_forward_hook(hook(name)))
        try:
            hidden = model.model(values, pos, visibility)
            logits[label] = model.lm_head[payload["next_position"] - 1](hidden[:, branch[-1]]).cpu()
        finally:
            for handle in handles:
                handle.remove()
        traces[label] = trace

    report = dict(
        position=payload["position"],
        rank_path=payload["rank_path"],
        packed_length=n,
        serial_length=length,
        structural_checks_passed=True,
        comparisons={},
    )
    for left, right in (
        ("tree", "serial"),
        ("serial_padded", "serial"),
        ("tree", "serial_padded"),
        ("tree", "tree_compact"),
        ("tree_compact", "serial_padded"),
    ):
        rows = [
            dict(module=name, **difference(value, traces[right][name]))
            for name, value in traces[left].items()
            if name in traces[right]
        ]
        item = dict(
            logits=difference(logits[left], logits[right]),
            first_different_module=next((r for r in rows if not r["exact_equal"]), None),
            modules=rows,
        )
        report["comparisons"][f"{left} vs {right}"] = item
        print(
            json.dumps(
                dict(
                    comparison=f"{left} vs {right}",
                    logits=item["logits"],
                    first_different_module=item["first_different_module"],
                )
            ),
            flush=True,
        )
    report["saved_failure_reproduction"] = {
        "tree": difference(logits["tree"], payload["tree_logits"]),
        "serial": difference(logits["serial"], payload["serial_logits"]),
    }
    report["compact_kv"] = args.compact_kv
    stem = "fusion_compact_kv_diagnostic" if args.compact_kv else "fusion_layout_diagnostic"
    exp.write_json(args.output / f"{stem}.json", report)
    torch.save(dict(traces=traces, logits=logits), args.output / f"{stem}.pt")
    print(json.dumps(report["saved_failure_reproduction"]), flush=True)
    print(f"Saved {stem}.json and .pt. Diagnostic only; no acceptance gates changed.", flush=True)


@torch.inference_mode()
def probe_frames(args):
    """Small correctness probe, separate from full replay and performance."""
    import torch_npu  # noqa: F401

    frozen = json.loads((args.output / "trees.json").read_text(encoding="utf-8"))
    signature = frozen.pop("sha256")
    if exp.digest(frozen) != signature:
        raise ValueError("frozen tree digest changed")
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    if frozen["dataset_sha256"] != exp.digest(manifest["dataset"]):
        raise ValueError("dataset changed after calibration")
    targets = {("smoke011", 0), ("smoke011", 2), ("smoke030", 129)}
    selected = [f for f in exp.frames(args, "test") if (f["request_id"], f["frame_id"]) in targets]
    if len(selected) != len(targets):
        raise ValueError("missing one of the three diagnostic frames")
    configs = {"reference": [], **frozen["configs"]}
    total = len(selected) * 2 * len(configs)
    result = dict(
        scope="three_frame_probe_only", compact_kv=args.compact_kv, passed=False, tree_sha256=signature, checks=[]
    )
    stem = "fusion_compact_kv_probe" if args.compact_kv else "fusion_explicit_mask_probe"
    destination = args.output / f"{stem}.json"
    exp.write_json(destination, result)
    print(f"Loading predictor; {total} checks, stop at first failure.", flush=True)
    model, spec = exp.load_predictor(args), exp.spec_module()
    install_probe(model, args.compact_kv)
    for frame in selected:
        for greedy in (True, False):
            inputs = exp.inputs_for(args, model, frame, greedy)
            stock = model(**inputs).clone()
            for name, paths in configs.items():
                row = dict(request_id=frame["request_id"], frame_id=frame["frame_id"], greedy=greedy, config=name)
                print(f"Starting {len(result['checks']) + 1}/{total}: {row}", flush=True)
                try:
                    codes, info = spec.decode(model, **inputs, paths=paths, audit=True)
                    row["code_equal"] = torch.equal(stock, codes)
                    row["audit_max_logit_error"] = info["audit_max_logit_error"]
                    if not row["code_equal"]:
                        row.update(
                            positions=(codes != stock).nonzero().cpu().tolist(),
                            stock_codes=stock.cpu().tolist(),
                            candidate_codes=codes.cpu().tolist(),
                        )
                except Exception as exc:
                    row["error"] = repr(exc)
                    if isinstance(exc, spec.NodeAuditError):
                        artifact = args.output / f"{stem}_node_failure.pt"
                        torch.save(exc.payload, artifact)
                        row["node_artifact"] = artifact.name
                result["checks"].append(row)
                exp.write_json(destination, result)
                print(json.dumps(row), flush=True)
                if "error" in row or not row.get("code_equal", False):
                    print(f"FAILED; saved {destination}. Full replay gates unchanged.", flush=True)
                    return
    result["passed"] = True
    exp.write_json(destination, result)
    print(f"PASSED {total}/{total}; saved {destination}. Not a full correctness or performance pass.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--compact-kv", action="store_true", help="probe batched queries with contiguous ancestor KV")
    parser.add_argument("--probe-frames", action="store_true", help="check three real frames in both sampling modes")
    args = parser.parse_args()
    (probe_frames if args.probe_frames else diagnose)(args)
