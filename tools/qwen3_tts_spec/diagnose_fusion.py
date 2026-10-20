# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolate shape/layout differences in one saved NPU tree audit failure."""

import argparse
import json
from pathlib import Path
from types import MethodType

import experiment as exp
import torch


def install_probe(model):
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
            return self.o_proj(attention.transpose(1, 2).reshape(batch, seq, -1))

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
    install_probe(model)
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

        for name, module in model.model.named_modules():
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
    exp.write_json(args.output / "fusion_layout_diagnostic.json", report)
    torch.save(dict(traces=traces, logits=logits), args.output / "fusion_layout_diagnostic.pt")
    print(json.dumps(report["saved_failure_reproduction"]), flush=True)
    print("Saved fusion_layout_diagnostic.json and .pt. Diagnostic only; no acceptance gates changed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    diagnose(parser.parse_args())
