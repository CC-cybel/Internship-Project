#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


COLORS = {
    "blue": "#3B82F6",
    "green": "#10B981",
    "orange": "#F59E0B",
    "red": "#EF4444",
    "purple": "#8B5CF6",
    "gray": "#64748B",
    "slate": "#334155",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("status") == "ok" and isinstance(r.get("judge"), dict)]


def plot_overview(rows: list[dict[str, Any]], out: Path) -> None:
    ok = score_rows(rows)
    total = len(rows)
    ok_count = len(ok)
    gen_err = sum(1 for r in rows if r.get("status") == "generation_error")
    judge_err = sum(1 for r in rows if r.get("status") == "judge_error")
    missing = sum(1 for r in rows if r.get("status") == "missing_response")
    judged_mean = statistics.mean([r["judge"]["sample_score"] for r in ok]) if ok else 0
    full_mean = sum(r["judge"]["sample_score"] for r in ok) / total if total else 0
    strict = sum(bool(r["judge"].get("all_2_pass")) for r in ok) / ok_count if ok_count else 0
    soft = sum(bool(r["judge"].get("all_ge1_pass")) for r in ok) / ok_count if ok_count else 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    labels = ["Judged Mean", "Full Mean\n(gen errors=0)", "Strict Pass", "Soft Pass"]
    vals = [judged_mean, full_mean, strict, soft]
    bars = ax.bar(labels, vals, color=[COLORS["blue"], COLORS["purple"], COLORS["green"], COLORS["orange"]])
    ax.set_ylim(0, 1.05)
    ax.set_title("Instruction Following Scores")
    ax.set_ylabel("Score / Rate")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax = axes[1]
    status_labels = ["ok", "generation_error", "judge_error", "missing"]
    status_vals = [ok_count, gen_err, judge_err, missing]
    status_colors = [COLORS["green"], COLORS["red"], COLORS["orange"], COLORS["gray"]]
    ax.bar(status_labels, status_vals, color=status_colors)
    ax.set_title(f"Run Status (n={total})")
    ax.set_ylabel("Rows")
    for i, val in enumerate(status_vals):
        ax.text(i, val + max(1, total * 0.01), str(val), ha="center", va="bottom", fontsize=10)
    savefig(out / "01_overview.png")


def plot_score_distribution(rows: list[dict[str, Any]], out: Path) -> None:
    ok = score_rows(rows)
    scores = [r["judge"]["sample_score"] for r in ok]
    plt.figure(figsize=(8, 4.8))
    plt.hist(scores, bins=[i / 10 for i in range(11)], color=COLORS["blue"], edgecolor="white")
    plt.title("Sample Score Distribution")
    plt.xlabel("Sample score")
    plt.ylabel("Rows")
    plt.xlim(0, 1)
    savefig(out / "02_score_distribution.png")


def plot_bucket_scores(rows: list[dict[str, Any]], out: Path) -> None:
    ok = score_rows(rows)
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        buckets[int(r.get("instruction_count") or 0)].append(r)
    labels = [str(k) for k in sorted(buckets)]
    means = [statistics.mean([r["judge"]["sample_score"] for r in buckets[int(k)]]) for k in labels]
    strict = [sum(r["judge"].get("all_2_pass") for r in buckets[int(k)]) / len(buckets[int(k)]) for k in labels]
    soft = [sum(r["judge"].get("all_ge1_pass") for r in buckets[int(k)]) / len(buckets[int(k)]) for k in labels]

    x = range(len(labels))
    width = 0.24
    plt.figure(figsize=(8.5, 4.8))
    plt.bar([i - width for i in x], means, width=width, label="Mean", color=COLORS["blue"])
    plt.bar(list(x), strict, width=width, label="Strict pass", color=COLORS["green"])
    plt.bar([i + width for i in x], soft, width=width, label="Soft pass", color=COLORS["orange"])
    plt.xticks(list(x), labels)
    plt.ylim(0, 1.05)
    plt.xlabel("Instruction count")
    plt.ylabel("Score / Rate")
    plt.title("Scores by Instruction Count")
    plt.legend()
    savefig(out / "03_by_instruction_count.png")


def collect_instruction_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for r in score_rows(rows):
        for item in r["judge"].get("instruction_results") or []:
            item = dict(item)
            item["sample_id"] = r.get("candidate_id")
            items.append(item)
    return items


def plot_group_bar(items: list[dict[str, Any]], field: str, title: str, filename: str, out: Path, top_n: int | None = None) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for item in items:
        inst = item.get("instruction") or {}
        key = str(inst.get(field) or "unknown")
        groups[key].append(float(item.get("score", 0)) / 2)
    rows = sorted(((statistics.mean(vals), len(vals), key) for key, vals in groups.items()), key=lambda x: x[0])
    if top_n:
        rows = rows[:top_n]
    labels = [r[2] for r in rows]
    vals = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    height = max(4.5, 0.38 * len(labels) + 1.2)
    plt.figure(figsize=(10, height))
    bars = plt.barh(labels, vals, color=COLORS["blue"])
    plt.xlim(0, 1.05)
    plt.xlabel("Mean instruction score")
    plt.title(title)
    for bar, val, n in zip(bars, vals, counts):
        plt.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.2f} (n={n})", va="center", fontsize=9)
    savefig(out / filename)


def plot_failure_types(items: list[dict[str, Any]], out: Path) -> None:
    c = Counter(str(item.get("failure_type") or "unclear") for item in items if int(item.get("score", 0)) < 2)
    rows = c.most_common()
    labels = [x[0] for x in rows]
    vals = [x[1] for x in rows]
    plt.figure(figsize=(9, max(4.5, 0.4 * len(labels) + 1)))
    bars = plt.barh(labels[::-1], vals[::-1], color=COLORS["red"])
    plt.xlabel("Instruction failures")
    plt.title("Failure Types")
    for bar, val in zip(bars, vals[::-1]):
        plt.text(val + 0.5, bar.get_y() + bar.get_height() / 2, str(val), va="center")
    savefig(out / "06_failure_types.png")


def plot_lowest_atoms(items: list[dict[str, Any]], out: Path, top_n: int = 25) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for item in items:
        groups[str(item.get("atom_id"))].append(float(item.get("score", 0)) / 2)
    rows = sorted(((statistics.mean(vals), len(vals), key) for key, vals in groups.items()), key=lambda x: (x[0], -x[1]))[:top_n]
    labels = [r[2] for r in rows]
    vals = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    plt.figure(figsize=(11, max(7, 0.38 * len(labels) + 1)))
    bars = plt.barh(labels[::-1], vals[::-1], color=COLORS["purple"])
    plt.xlim(0, 1.05)
    plt.xlabel("Mean instruction score")
    plt.title(f"Lowest {top_n} Atom Scores")
    for bar, val, n in zip(bars, vals[::-1], counts[::-1]):
        plt.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.2f} (n={n})", va="center", fontsize=8)
    savefig(out / "07_lowest_atoms.png")


def write_summary(rows: list[dict[str, Any]], items: list[dict[str, Any]], out: Path) -> None:
    ok = score_rows(rows)
    total = len(rows)
    status = Counter(r.get("status") for r in rows)
    judged_mean = statistics.mean([r["judge"]["sample_score"] for r in ok]) if ok else 0
    full_mean = sum(r["judge"]["sample_score"] for r in ok) / total if total else 0
    strict = sum(bool(r["judge"].get("all_2_pass")) for r in ok) / len(ok) if ok else 0
    soft = sum(bool(r["judge"].get("all_ge1_pass")) for r in ok) / len(ok) if ok else 0
    lines = [
        "# Instruction Following Visualization Summary",
        "",
        f"- total_rows: {total}",
        f"- judged_ok: {len(ok)}",
        f"- status_counts: {dict(status)}",
        f"- judged_only_mean_score: {judged_mean:.4f}",
        f"- full_mean_score_generation_errors_as_zero: {full_mean:.4f}",
        f"- strict_all_2_pass_rate: {strict:.4f}",
        f"- soft_all_ge1_pass_rate: {soft:.4f}",
        "",
        "## Figures",
        "",
        "- [Overview](01_overview.png)",
        "- [Score Distribution](02_score_distribution.png)",
        "- [By Instruction Count](03_by_instruction_count.png)",
        "- [By Source](04_by_source.png)",
        "- [By Axis](05_by_axis.png)",
        "- [Failure Types](06_failure_types.png)",
        "- [Lowest Atom Scores](07_lowest_atoms.png)",
    ]
    (out / "visualization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = args.output_dir.resolve() if args.output_dir else run_dir / "visualization"
    rows = read_jsonl(run_dir / "evaluation_results.jsonl")
    items = collect_instruction_items(rows)
    plot_overview(rows, out)
    plot_score_distribution(rows, out)
    plot_bucket_scores(rows, out)
    plot_group_bar(items, "source", "Scores by Instruction Source", "04_by_source.png", out)
    plot_group_bar(items, "axis", "Scores by Instruction Axis", "05_by_axis.png", out)
    plot_failure_types(items, out)
    plot_lowest_atoms(items, out)
    write_summary(rows, items, out)
    print(out)


if __name__ == "__main__":
    main()
