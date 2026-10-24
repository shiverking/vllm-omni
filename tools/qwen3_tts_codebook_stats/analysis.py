# SPDX-License-Identifier: Apache-2.0
"""Empirical statistics and held-out coverage, not target logits or acceptance."""

from __future__ import annotations

import csv
import heapq
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from experiment import is_complete, log, read_codes, read_json, requests_for, write_json

MODES = ("global", "main", "previous", "mixed", "temporal")
BUDGETS = (8, 16, 32)


def entropy(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if not total:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def mutual_information(x, y):
    if not len(x):
        return 0.0
    _, nx = np.unique(x, return_counts=True)
    _, ny = np.unique(y, return_counts=True)
    _, joint = np.unique(np.stack((x, y), axis=1), axis=0, return_counts=True)
    return max(0.0, entropy(nx) + entropy(ny) - entropy(joint))


def valid_sequence(rows):
    return [(r["frame_id"], np.asarray(r["codes"], dtype=np.int64)) for r in rows if r["valid"]]


class Tables:
    """Immutable after fitting. All caches depend only on known/proposed codes."""

    def __init__(self, vocabulary=2048, groups=16):
        self.vocabulary, self.groups = vocabulary, groups
        self.counts = np.zeros((groups, vocabulary), dtype=np.int64)
        self.conditionals = defaultdict(Counter)
        self.priors = None

    def fit(self, records):
        if self.priors is not None:
            raise ValueError("Tables are frozen after fitting")
        for record in records:
            if record["request"]["split"] != "statistics":
                continue
            previous = None
            for frame, codes in valid_sequence(record["rows"]):
                adjacent = previous is not None and frame == previous[0] + 1
                for pos, token in enumerate(codes):
                    self.counts[pos, token] += 1
                    if pos:
                        self.conditionals[("main", pos, int(codes[0]))][int(token)] += 1
                        self.conditionals[("previous", pos, int(codes[pos - 1]))][int(token)] += 1
                    if adjacent:
                        self.conditionals[("temporal", pos, int(previous[1][pos]))][int(token)] += 1
                previous = frame, codes
        self.priors = (self.counts + 1) / (self.counts.sum(axis=1, keepdims=True) + self.vocabulary)
        return self

    def support(self, mode, pos, main, prev, temporal=None):
        if mode == "global":
            return int(self.counts[pos].sum())
        if mode == "mixed":
            return min(self.support("main", pos, main, prev), self.support("previous", pos, main, prev))
        condition = {"main": main, "previous": prev, "temporal": temporal}[mode]
        return sum(self.conditionals.get((mode, pos, condition), {}).values())

    @lru_cache(maxsize=4096)
    def conditional(self, mode, pos, condition):
        counts = self.conditionals.get((mode, pos, condition), {})
        total = sum(counts.values())
        if not total:
            return self.priors[pos]
        result = 32.0 * self.priors[pos].copy()
        for token, count in counts.items():
            result[token] += count
        return result / (total + 32)

    def probability(self, mode, pos, main, prev, temporal=None):
        if mode == "global":
            return self.priors[pos]
        if mode == "mixed":
            return (self.conditional("main", pos, main) + self.conditional("previous", pos, prev)) * 0.5
        condition = {"main": main, "previous": prev, "temporal": temporal}[mode]
        return self.conditional(mode, pos, condition)

    def key(self, mode, main, prev, temporal=None):
        if mode == "global":
            return None, None, None
        return (
            main if mode in ("main", "mixed") else None,
            prev if mode in ("previous", "mixed") else None,
            temporal if mode == "temporal" else None,
        )

    def top(self, mode, pos, main, prev, temporal=None):
        return self._top(mode, pos, *self.key(mode, main, prev, temporal))

    @lru_cache(maxsize=16384)
    def _top(self, mode, pos, main, prev, temporal):
        prob = self.probability(mode, pos, main, prev, temporal)
        # Stable token-id tie break, consistent across all candidate budgets.
        ids = np.argsort(-prob, kind="stable")[:8]
        return tuple((int(t), float(prob[t])) for t in ids)

    def tree(self, mode, start, main, prev):
        if mode not in MODES[:4]:
            raise ValueError("Temporal prior is only a single-position control")
        main, prev, _ = self.key(mode, main, prev)
        return self._tree(mode, start, main, prev)

    @lru_cache(maxsize=8192)
    def _tree(self, mode, start, main, prev):
        depth = min(3, self.groups - start)
        if depth <= 0:
            return ()
        queue = [(-p, (token,)) for token, p in self.top(mode, start, main, prev)[:4]]
        heapq.heapify(queue)
        ordered = []
        while queue and len(ordered) < 32:
            neg_prob, path = heapq.heappop(queue)
            ordered.append(path)
            if len(path) < depth and len(ordered) < 32:
                for token, prob in self.top(mode, start + len(path), main, path[-1])[:4]:
                    heapq.heappush(queue, (neg_prob * prob, path + (token,)))
        return tuple(ordered)


def path_coverage(nodes, truth):
    nodes = set(nodes)
    length = 0
    for depth in range(1, len(truth) + 1):
        if tuple(truth[:depth]) not in nodes:
            break
        length = depth
    return length


def distribution(records, vocabulary, groups):
    all_codes = [codes for r in records for _, codes in valid_sequence(r["rows"])]
    if not all_codes:
        return dict(frames=0, positions=[], mutual_information_bits=[]), []
    array = np.stack(all_codes)
    positions, frequencies = [], []
    for pos in range(groups):
        count = np.bincount(array[:, pos], minlength=vocabulary)
        ranked = np.sort(count)[::-1]
        request_stats = []
        for record in records:
            seq = valid_sequence(record["rows"])
            if seq:
                c = np.bincount([int(x[pos]) for _, x in seq], minlength=vocabulary)
                request_stats.append(
                    [entropy(c), np.count_nonzero(c), *[np.sort(c)[::-1][:k].sum() / c.sum() for k in (1, 4, 8, 16)]]
                )
        request_mean = np.mean(request_stats, axis=0)
        positions.append(
            dict(
                position=pos,
                entropy_bits=entropy(count),
                used_tokens=int(np.count_nonzero(count)),
                concentration={str(k): float(ranked[:k].sum() / len(array)) for k in (1, 4, 8, 16)},
                request_mean_entropy_bits=float(request_mean[0]),
                request_mean_used_tokens=float(request_mean[1]),
                request_mean_concentration={str(k): float(request_mean[j + 2]) for j, k in enumerate((1, 4, 8, 16))},
            )
        )
        frequencies.extend(
            dict(position=pos, token=t, count=int(n), probability=float(n / len(array))) for t, n in enumerate(count)
        )
    mi = np.zeros((groups, groups))
    for i in range(groups):
        for j in range(i, groups):
            mi[i, j] = mi[j, i] = mutual_information(array[:, i], array[:, j])
    request_mi = np.zeros((groups, groups))
    request_arrays = [np.stack([c for _, c in valid_sequence(r["rows"])]) for r in records if valid_sequence(r["rows"])]
    for codes in request_arrays:
        for i in range(groups):
            for j in range(i, groups):
                value = mutual_information(codes[:, i], codes[:, j])
                request_mi[i, j] += value
                if i != j:
                    request_mi[j, i] += value
    request_mi /= len(request_arrays)
    temporal = []
    for pos in range(groups):
        equal, pairs, rates = 0, 0, []
        for record in records:
            seq = valid_sequence(record["rows"])
            values = [int(a[1][pos] == b[1][pos]) for a, b in zip(seq, seq[1:]) if b[0] == a[0] + 1]
            equal += sum(values)
            pairs += len(values)
            if values:
                rates.append(float(np.mean(values)))
        temporal.append(
            dict(
                position=pos,
                pairs=pairs,
                repeat_rate=equal / pairs if pairs else None,
                request_mean_repeat_rate=float(np.mean(rates)) if rates else None,
            )
        )
    return dict(
        frames=len(array),
        positions=positions,
        mutual_information_bits=mi.tolist(),
        request_mean_mutual_information_bits=request_mi.tolist(),
        temporal=temporal,
    ), frequencies


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_ci(rows, numerator, denominator="count", paired=None):
    by_text = defaultdict(lambda: np.zeros(2))
    for row in rows:
        by_text[row["text_id"]] += [row[numerator] - (row[paired] if paired else 0), row[denominator]]
    if len(by_text) < 2:
        return None
    values = np.asarray(list(by_text.values()))
    rng = np.random.default_rng(20261023)
    totals = values[rng.integers(len(values), size=(1000, len(values)))].sum(axis=1)
    return np.quantile(totals[:, 0] / totals[:, 1], [0.025, 0.975]).tolist()


def evaluate(tables, records, progress=None):
    rows, tree_rows, examples = [], [], []
    example_count = Counter()
    for index, record in enumerate(records, 1):
        request = record["request"]
        cells = defaultdict(lambda: np.zeros(7))
        trees = defaultdict(lambda: np.zeros(10))
        previous = None
        for frame, codes in valid_sequence(record["rows"]):
            adjacent = previous is not None and frame == previous[0] + 1
            for pos in range(1, tables.groups):
                main, prev, token = int(codes[0]), int(codes[pos - 1]), int(codes[pos])
                global_prob = tables.priors[pos]
                global_top = [t for t, _ in tables.top("global", pos, main, prev)]
                for mode in MODES:
                    if mode == "temporal" and not adjacent:
                        continue
                    temporal = int(previous[1][pos]) if adjacent else None
                    prob = tables.probability(mode, pos, main, prev, temporal)
                    top = [t for t, _ in tables.top(mode, pos, main, prev, temporal)]
                    support = tables.support(mode, pos, main, prev, temporal)
                    stratum = "unseen" if support == 0 else "sparse_1_31" if support < 32 else "frequent_32_plus"
                    cell = cells[(mode, pos, stratum)]
                    cell += [
                        1,
                        token in top[:1],
                        token in top[:4],
                        token in top[:8],
                        -math.log(prob[token]),
                        token in global_top[:4],
                        -math.log(global_prob[token]),
                    ]
                    if mode == "temporal":
                        continue
                    ordered = tables.tree(mode, pos, main, prev)
                    truth = tuple(int(x) for x in codes[pos : pos + 3])
                    for budget in BUDGETS:
                        nodes = ordered[:budget]
                        hit = path_coverage(nodes, truth)
                        target = trees[(mode, budget, pos)]
                        target[:6] += [1, hit, hit >= 2, hit == len(truth), len(nodes), len(truth)]
                        target[6 + hit] += 1
                        category = "multi_hit" if hit >= 2 else "miss" if hit == 0 else "partial"
                        key = mode, budget, category
                        if example_count[key] < 2:
                            examples.append(
                                dict(
                                    request_id=request["id"],
                                    text_id=request["text_id"],
                                    audio=record["audio"],
                                    frame_id=frame,
                                    position=pos,
                                    mode=mode,
                                    budget=budget,
                                    category=category,
                                    truth=list(truth),
                                    full_frame=codes.tolist(),
                                    nodes=[list(n) for n in nodes],
                                    continuous_hit=hit,
                                )
                            )
                            example_count[key] += 1
            previous = frame, codes
        for (mode, pos, stratum), value in cells.items():
            rows.append(
                dict(
                    request_id=request["id"],
                    text_id=request["text_id"],
                    mode=mode,
                    position=pos,
                    stratum=stratum,
                    **dict(
                        zip(
                            ("count", "hit1", "hit4", "hit8", "nll_sum", "global_hit4", "global_nll_sum"),
                            value.tolist(),
                        )
                    ),
                )
            )
        for (mode, budget, pos), value in trees.items():
            tree_rows.append(
                dict(
                    request_id=request["id"],
                    text_id=request["text_id"],
                    mode=mode,
                    budget=budget,
                    position=pos,
                    **dict(
                        zip(
                            (
                                "count",
                                "hit_sum",
                                "hit2",
                                "full",
                                "nodes",
                                "depth",
                                "length0",
                                "length1",
                                "length2",
                                "length3",
                            ),
                            value.tolist(),
                        )
                    ),
                )
            )
        if progress:
            progress(index, len(records), request["id"])
    return rows, tree_rows, examples


def aggregate(rows, tree_rows):
    coverage, trees = [], []
    for mode in MODES:
        for pos in ["all", *sorted({r["position"] for r in rows})]:
            selected = [r for r in rows if r["mode"] == mode and (pos == "all" or r["position"] == pos)]
            count = sum(r["count"] for r in selected)
            if not count:
                continue

            def total(name):
                return sum(r[name] for r in selected) / count

            by_request = defaultdict(lambda: np.zeros(5))
            for row in selected:
                by_request[row["request_id"]] += [row["hit4"], row["nll_sum"], row["count"], row["hit1"], row["hit8"]]
            rates = np.asarray(list(by_request.values()))
            coverage.append(
                dict(
                    mode=mode,
                    position=pos,
                    count=int(count),
                    top1=total("hit1"),
                    top4=total("hit4"),
                    top8=total("hit8"),
                    nll_nats=total("nll_sum"),
                    top4_gain=total("hit4") - total("global_hit4"),
                    nll_improvement=total("global_nll_sum") - total("nll_sum"),
                    top4_ci=grouped_ci(selected, "hit4"),
                    top4_gain_ci=grouped_ci(selected, "hit4", paired="global_hit4"),
                    nll_improvement_ci=grouped_ci(selected, "global_nll_sum", paired="nll_sum"),
                    top1_ci=grouped_ci(selected, "hit1"),
                    top8_ci=grouped_ci(selected, "hit8"),
                    nll_ci=grouped_ci(selected, "nll_sum"),
                    request_mean_top1=float(np.mean(rates[:, 3] / rates[:, 2])),
                    request_mean_top8=float(np.mean(rates[:, 4] / rates[:, 2])),
                    request_mean_top4=float(np.mean(rates[:, 0] / rates[:, 2])),
                    request_mean_nll=float(np.mean(rates[:, 1] / rates[:, 2])),
                    unseen_fraction=sum(r["count"] for r in selected if r["stratum"] == "unseen") / count,
                )
            )
    for mode in MODES[:4]:
        for budget in BUDGETS:
            for pos in ["all", *sorted({r["position"] for r in tree_rows})]:
                selected = [
                    r
                    for r in tree_rows
                    if r["mode"] == mode and r["budget"] == budget and (pos == "all" or r["position"] == pos)
                ]
                count = sum(r["count"] for r in selected)
                if not count:
                    continue
                per_request = defaultdict(lambda: np.zeros(4))
                for row in selected:
                    per_request[row["request_id"]] += [row["hit_sum"], row["hit2"], row["full"], row["count"]]
                rates = np.asarray(list(per_request.values()))
                trees.append(
                    dict(
                        mode=mode,
                        budget=budget,
                        position=pos,
                        rounds=int(count),
                        request_mean_hit=float(np.mean(rates[:, 0] / rates[:, 3])),
                        request_mean_hit2_fraction=float(np.mean(rates[:, 1] / rates[:, 3])),
                        request_mean_full_path_fraction=float(np.mean(rates[:, 2] / rates[:, 3])),
                        mean_hit_ci=grouped_ci(selected, "hit_sum"),
                        mean_hit=sum(r["hit_sum"] for r in selected) / count,
                        hit2_fraction=sum(r["hit2"] for r in selected) / count,
                        hit2_ci=grouped_ci(selected, "hit2"),
                        full_path_fraction=sum(r["full"] for r in selected) / count,
                        full_path_ci=grouped_ci(selected, "full"),
                        mean_nodes=sum(r["nodes"] for r in selected) / count,
                        length_histogram={str(i): int(sum(r[f"length{i}"] for r in selected)) for i in range(4)},
                    )
                )
    strata = []
    for mode in MODES:
        for name in ("unseen", "sparse_1_31", "frequent_32_plus"):
            selected = [r for r in rows if r["mode"] == mode and r["stratum"] == name]
            count = sum(r["count"] for r in selected)
            if count:
                strata.append(
                    dict(
                        mode=mode,
                        stratum=name,
                        count=int(count),
                        top4=sum(r["hit4"] for r in selected) / count,
                        nll_nats=sum(r["nll_sum"] for r in selected) / count,
                    )
                )
    return coverage, trees, strata


def analyze(root):
    import time

    manifest = read_json(root / "manifest.json")
    records, excluded = [], []
    for request in requests_for(manifest):
        if not is_complete(root, request):
            excluded.append(request["id"])
            continue
        marker = read_json(root / "completed" / f"{request['id']}.json")
        rows = read_codes(root, request, marker["attempt"])
        records.append(dict(request=request, rows=rows, audio=marker["audio"]))
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    stats = [r for r in records if r["request"]["split"] == "statistics"]
    tests = [r for r in records if r["request"]["split"] == "test"]
    log(root, f"分析完整音频 {len(records)}/300；统计集 {len(stats)}，测试集 {len(tests)}；未完成 {len(excluded)}")
    tables = Tables(manifest["vocabulary"], manifest["codebooks"]).fit(stats)
    distributions = {}
    for name, selected in (("statistics", stats), ("test", tests)):
        log(root, f"计算 {name} 的经验频率、熵和互信息。")
        distributions[name], frequencies = distribution(selected, tables.vocabulary, tables.groups)
        csv_write(output / f"frequencies_{name}.csv", frequencies)
    support_rows, conditional_rows = [], []
    for (mode, pos, condition), counts in sorted(tables.conditionals.items()):
        support_rows.append(
            dict(mode=mode, position=pos, condition=condition, samples=sum(counts.values()), targets=len(counts))
        )
        conditional_rows.extend(
            dict(
                mode=mode,
                position=pos,
                condition=condition,
                token=t,
                count=n,
                empirical_probability=n / sum(counts.values()),
                smoothed_probability=float(tables.conditional(mode, pos, condition)[t]),
            )
            for t, n in sorted(counts.items())
        )
    csv_write(output / "conditional_counts.csv", conditional_rows)
    csv_write(output / "condition_support.csv", support_rows)
    started = time.monotonic()

    def progress(i, total, ident):
        elapsed = time.monotonic() - started
        log(root, f"测试集候选覆盖 {i}/{total}，{ident}；耗时 {elapsed:.1f}s，ETA~{elapsed / i * (total - i):.1f}s")

    if stats and tests:
        rows, tree_rows, examples = evaluate(tables, tests, progress)
    else:
        rows, tree_rows, examples = [], [], []
        log(root, "尚无完整统计集或测试集音频，只输出已有分布，不伪造覆盖结果。")
    coverage, trees, strata = aggregate(rows, tree_rows)
    csv_write(output / "coverage_per_request.csv", rows)
    csv_write(output / "tree_per_request.csv", tree_rows)
    csv_write(output / "coverage.csv", coverage)
    csv_write(output / "tree_coverage.csv", trees)
    csv_write(output / "coverage_by_support.csv", strata)
    write_json(output / "examples.json", examples)
    write_json(
        output / "summary.json",
        dict(
            schema="codebook_statistics_analysis_v1",
            completed_audio=len(records),
            completed_texts=len({r["request"]["text_id"] for r in records}),
            statistics_audio=len(stats),
            test_audio=len(tests),
            statistics_texts=len({r["request"]["text_id"] for r in stats}),
            test_texts=len({r["request"]["text_id"] for r in tests}),
            incomplete_requests=excluded,
            failures=len(list((root / "failures").glob("*.json"))),
            raw_frames=sum(len(r["rows"]) for r in records),
            invalid_frames=sum(not f["valid"] for r in records for f in r["rows"]),
            distributions=distributions,
            coverage=coverage,
            trees=trees,
            support_strata=strata,
            smoothing=dict(global_additive=1, conditional_pseudocount=32, mixed_weights=[0.5, 0.5]),
            support_definition="mixed support=min(main,previous); frequent>=32, sparse=1..31, unseen=0",
            bootstrap=dict(unit="text: keep all seeds and frames together", resamples=1000, seed=20261023),
            note=(
                "Empirical token distribution, not model logits. "
                "Tree coverage is not speculative acceptance or speedup."
            ),
        ),
    )
    report(root)


def plots(output, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Re-reporting a partial/corrupted collection must not show stale figures.
    for name in ("distributions.png", "mutual_information.png", "candidate_coverage.png", "tree_coverage.png"):
        (output / name).unlink(missing_ok=True)
    selected = data["distributions"]["statistics"]
    if selected["positions"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for field, label in (("entropy_bits", "Frame pooled"), ("request_mean_entropy_bits", "Request mean")):
            axes[0].plot(
                [r["position"] for r in selected["positions"]],
                [r[field] for r in selected["positions"]],
                marker="o",
                label=label,
            )
        axes[0].set(xlabel="Codebook", ylabel="Empirical entropy (bits)")
        axes[0].legend()
        for k in (1, 4, 8, 16):
            axes[1].plot(
                [r["position"] for r in selected["positions"]],
                [r["concentration"][str(k)] for r in selected["positions"]],
                label=f"Top-{k}",
            )
        axes[1].set(xlabel="Codebook", ylabel="Marginal frequency mass")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output / "distributions.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(selected["mutual_information_bits"], origin="lower")
        fig.colorbar(im, ax=ax, label="Empirical MI (bits); small-sample biased")
        ax.set(xlabel="Codebook", ylabel="Codebook")
        fig.tight_layout()
        fig.savefig(output / "mutual_information.png", dpi=150)
        plt.close(fig)
    rows = [r for r in data["coverage"] if r["position"] == "all"]
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(len(rows))
        for j, k in enumerate((1, 4, 8)):
            axes[0].bar(x + (j - 1) * 0.25, [r[f"top{k}"] for r in rows], width=0.25, label=f"Top-{k}")
        axes[0].set_xticks(x, [r["mode"] for r in rows])
        axes[0].legend()
        axes[0].set_ylabel("Held-out token coverage")
        axes[1].bar(x, [r["top4_gain"] for r in rows], label="Top-4 gain")
        axes[1].plot(x, [r["unseen_fraction"] for r in rows], "o-", label="Unseen conditions")
        axes[1].set_xticks(x, [r["mode"] for r in rows])
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output / "candidate_coverage.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        for mode in MODES[:4]:
            values = [r for r in data["trees"] if r["position"] == "all" and r["mode"] == mode]
            ax.plot([r["budget"] for r in values], [r["hit2_fraction"] for r in values], marker="o", label=mode)
        ax.set(xlabel="Tree node budget", ylabel="Continuous coverage >= 2 (tail included)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "tree_coverage.png", dpi=150)
        plt.close(fig)


def report(root):
    output = root / "analysis"
    data = read_json(output / "summary.json")
    plots(output, data)
    rows = [r for r in data["coverage"] if r["position"] == "all"]
    lines = [
        "# 真实生成音频：码本分布与候选覆盖",
        "",
        f"完整音频 {data['completed_audio']}/300；独立文本 {data['completed_texts']}/100；"
        f"失败请求 {data['failures']}。",
        f"统计集 {data['statistics_texts']} 条文本/{data['statistics_audio']} 段；"
        f"测试集 {data['test_texts']} 条/{data['test_audio']} 段。",
        f"原始帧 {data['raw_frames']}；过滤无效帧 {data['invalid_frames']}。未完成请求详见 summary.json。",
        "",
        "固定 Vivian、英语、stock 单请求。经验码频率不是模型 softmax；候选覆盖不是实际投机接受率或加速。",
        "互信息存在高维小样本偏差，不代表因果关系。不同码本编号没有大小或同编号语义关系。",
        "",
        "## 独立测试集预测价值",
        "",
        "| 候选器 | Top-1 | Top-4 | Top-8 | Top-4 增益 | 增益 95% 区间 | NLL(nats) | 未见条件 |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['top1']:.2%} | {r['top4']:.2%} | {r['top8']:.2%} | "
            f"{r['top4_gain']:+.2%} | {r['top4_gain_ci']} | {r['nll_nats']:.3f} | {r['unseen_fraction']:.2%} |"
        )
    lines += [
        "",
        "global=全局频率；main=条件 c0；previous=前一码本；mixed=二者等权；temporal=上一有效帧同位置。",
        "时间先验仅评价相邻有效帧，增益对照也使用同一批帧。NLL 越低越好。",
        "区间按文本分组重采样，保留同文本的三个种子；不足两个测试文本时区间为 None。",
        "按请求平均、逐位置、频繁/稀疏/未见条件结果见 summary.json 和 CSV。",
        "",
        "## 连续路径覆盖",
        "",
        "| 候选器 | 节点预算 | 平均命中 | 连续≥2 | 完整路径 | 平均有效节点 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in data["trees"]:
        if r["position"] == "all":
            lines.append(
                f"| {r['mode']} | {r['budget']} | {r['mean_hit']:.3f} | "
                f"{r['hit2_fraction']:.2%} | {r['full_path_fraction']:.2%} | {r['mean_nodes']:.2f} |"
            )
    lines += [
        "",
        "树深度最多 3，帧尾裁剪；连续≥2的分母包含全部起点，完整路径按裁剪后的深度计算。",
        "子节点根据分支假设的前驱码生成，不读取真实未来码。",
        "",
        "## 探索结论",
        "",
    ]
    positions = data["distributions"]["statistics"]["positions"]
    if positions:
        low = min(positions, key=lambda r: r["entropy_bits"])
        high = max(positions, key=lambda r: r["entropy_bits"])
        lines.append(
            f"统计集经验熵最低为 c{low['position']}（{low['entropy_bits']:.3f} bits），"
            f"最高为 c{high['position']}（{high['entropy_bits']:.3f} bits）。"
            "这是边际分布描述，不等于给定上下文后的模型不确定性。"
        )
    if rows:
        gains = [r for r in rows if r["mode"] != "global"]
        best = max(gains, key=lambda r: r["top4_gain"])
        lines.append(f"本次整体 Top-4 增益最高：{best['mode']}（{best['top4_gain']:+.2%}）；这不是部署选择或加速证明。")
        positive = sorted(
            [
                r
                for r in data["coverage"]
                if r["position"] != "all"
                and r["mode"] not in ("global", "temporal")
                and r["top4_gain_ci"]
                and r["top4_gain_ci"][0] > 0
            ],
            key=lambda r: -r["top4_gain"],
        )[:5]
        lines.append(
            "可进一步检查的位置："
            + (
                "；".join(f"c{r['position']}/{r['mode']} 增益 {r['top4_gain']:+.2%}" for r in positive)
                or "尚未观察到增益区间下界大于零的帧内候选配置。"
            )
        )
        lines.append("逐位置筛选涉及多重比较，以上只用于后续假设；需新数据确认，不能将测试集重新用于拟合。")
    else:
        lines.append("统计集或测试集尚不完整到可进行覆盖评估；继续生成后可重跑 analyze。")
    for filename in ("distributions.png", "mutual_information.png", "candidate_coverage.png", "tree_coverage.png"):
        if (output / filename).exists():
            lines += ["", f"![{filename}](analysis/{filename})"]
    lines += ["", "## 典型序列与音频", ""]
    for r in read_json(output / "examples.json")[:24]:
        lines.append(
            f"- [{r['request_id']}]({r['audio']})，frame={r['frame_id']}、c{r['position']}，"
            f"{r['mode']}/tree{r['budget']}：连续覆盖 {r['continuous_hit']}；"
            f"真实后续码 {r['truth']}。"
        )
    lines += [
        "",
        "更多命中与失败样例及完整帧码本见 analysis/examples.json。",
        "音频与码本只按请求和生成顺序对应，不构成逐音素或精确采样点对齐。",
        "本轮没有训练、目标模型验证或性能测试；仅覆盖当前音色、语言和采样配置，稀疏条件仍可能缺少足够证据。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(root, f"报告已写入 {root / 'report.md'}")
