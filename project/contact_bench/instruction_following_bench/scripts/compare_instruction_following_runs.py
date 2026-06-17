#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

COLORS = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6", "#EF4444", "#64748B"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("status") == "ok" and isinstance(r.get("judge"), dict)]


def items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in ok_rows(rows):
        for item in r["judge"].get("instruction_results") or []:
            item = dict(item)
            item["candidate_id"] = r.get("candidate_id")
            item["instruction_count"] = r.get("instruction_count")
            out.append(item)
    return out


def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | dict[str, int]]:
    ok = ok_rows(rows)
    total = len(rows)
    scores = [r["judge"]["sample_score"] for r in ok]
    return {
        "total": total,
        "judged_ok": len(ok),
        "generation_error": sum(1 for r in rows if r.get("status") == "generation_error"),
        "judge_error": sum(1 for r in rows if r.get("status") == "judge_error"),
        "mean_judged": statistics.mean(scores) if scores else 0.0,
        "mean_full": sum(scores) / total if total else 0.0,
        "median": statistics.median(scores) if scores else 0.0,
        "strict": sum(bool(r["judge"].get("all_2_pass")) for r in ok) / len(ok) if ok else 0.0,
        "soft": sum(bool(r["judge"].get("all_ge1_pass")) for r in ok) / len(ok) if ok else 0.0,
        "format_ok": sum(bool(r["judge"].get("format_ok")) for r in ok) / len(ok) if ok else 0.0,
        "status_counts": dict(Counter(str(r.get("status")) for r in rows)),
    }


def group_mean_by_sample(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    ok = ok_rows(rows)
    groups: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        key = str(r.get(field) or "unknown")
        groups[key].append(float(r["judge"]["sample_score"]))
    return {k: statistics.mean(v) for k, v in groups.items()}


def group_item_mean(run_items: list[dict[str, Any]], field: str) -> dict[str, tuple[float, int]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for item in run_items:
        inst = item.get("instruction") or {}
        if field == "atom_id":
            key = str(item.get("atom_id") or "unknown")
        elif field == "failure_type":
            if int(item.get("score", 0)) >= 2:
                continue
            key = str(item.get("failure_type") or "unclear")
        else:
            key = str(inst.get(field) or "unknown")
        groups[key].append(float(item.get("score", 0)) / 2)
    return {k: (statistics.mean(v), len(v)) for k, v in groups.items()}


def failure_counts(run_items: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter()
    for item in run_items:
        if int(item.get("score", 0)) < 2:
            c[str(item.get("failure_type") or "unclear")] += 1
    return dict(c)


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def bar_compare(names: list[str], values: dict[str, list[float]], title: str, ylabel: str, path: Path, ylim: tuple[float, float] | None = (0, 1.05)) -> None:
    labels = list(values.keys())
    x = range(len(labels))
    width = min(0.8 / max(1, len(names)), 0.35)
    plt.figure(figsize=(max(8, len(labels) * 1.1), 5))
    for i, name in enumerate(names):
        offsets = [j + (i - (len(names) - 1) / 2) * width for j in x]
        vals = values[name]
        bars = plt.bar(offsets, vals, width=width, label=name, color=COLORS[i % len(COLORS)])
        for bar, val in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=8, rotation=0)
    plt.xticks(list(x), labels, rotation=25, ha="right")
    if ylim:
        plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    save(path)


def horizontal_compare(names: list[str], maps: dict[str, dict[str, tuple[float, int]]], keys: list[str], title: str, path: Path) -> None:
    height = max(5, 0.42 * len(keys) + 1.5)
    plt.figure(figsize=(11, height))
    y = range(len(keys))
    h = min(0.8 / max(1, len(names)), 0.35)
    for i, name in enumerate(names):
        vals = [maps[name].get(k, (0, 0))[0] for k in keys]
        offsets = [j + (i - (len(names) - 1) / 2) * h for j in y]
        plt.barh(offsets, vals, height=h, label=name, color=COLORS[i % len(COLORS)])
    plt.yticks(list(y), keys)
    plt.xlim(0, 1.05)
    plt.xlabel("Mean instruction score")
    plt.title(title)
    plt.legend()
    save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", nargs=2, metavar=("NAME", "RUN_DIR"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    names = [x[0] for x in args.run]
    rows_by_name = {name: read_jsonl(Path(run_dir) / "evaluation_results.jsonl") for name, run_dir in args.run}
    items_by_name = {name: items(rows) for name, rows in rows_by_name.items()}
    m_by_name = {name: metrics(rows) for name, rows in rows_by_name.items()}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    overview_keys = ["mean_judged", "mean_full", "strict", "soft", "format_ok"]
    overview_labels = {
        "mean_judged": "Judged Mean",
        "mean_full": "Full Mean\n(gen=0)",
        "strict": "Strict Pass",
        "soft": "Soft Pass",
        "format_ok": "Format OK",
    }
    overview_values = {name: [float(m_by_name[name][k]) for k in overview_keys] for name in names}
    overview_values = {name: vals for name, vals in overview_values.items()}
    label_values = {overview_labels[k]: [float(m_by_name[name][k]) for name in names] for k in overview_keys}
    # transpose for grouped plotting helper
    values_for_helper = {name: [float(m_by_name[name][k]) for k in overview_keys] for name in names}
    old_xticks = overview_keys
    plt.figure(figsize=(10, 5))
    x = range(len(old_xticks))
    width = 0.34
    for i, name in enumerate(names):
        vals = values_for_helper[name]
        offs = [j + (i - (len(names)-1)/2) * width for j in x]
        plt.bar(offs, vals, width=width, label=name, color=COLORS[i])
        for xx, val in zip(offs, vals):
            plt.text(xx, val + 0.02, f"{val:.3f}", ha="center", fontsize=8)
    plt.xticks(list(x), [overview_labels[k] for k in old_xticks], rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score / Rate")
    plt.title("Overall Comparison")
    plt.legend()
    save(out / "01_overall_comparison.png")

    # status counts
    status_keys = sorted(set().union(*(set(m_by_name[n]["status_counts"].keys()) for n in names)))
    plt.figure(figsize=(max(8, len(status_keys) * 1.4), 5))
    x = range(len(status_keys))
    width = min(0.8 / max(1, len(names)), 0.28)
    max_val = 0
    for i, name in enumerate(names):
        vals = [int(m_by_name[name]["status_counts"].get(k, 0)) for k in status_keys]  # type: ignore[index]
        max_val = max(max_val, max(vals) if vals else 0)
        offs = [j + (i - (len(names) - 1) / 2) * width for j in x]
        bars = plt.bar(offs, vals, width=width, label=name, color=COLORS[i % len(COLORS)])
        for bar, val in zip(bars, vals):
            if val:
                plt.text(bar.get_x() + bar.get_width() / 2, val + max(1, max_val * 0.01), str(val), ha="center", va="bottom", fontsize=8)
    plt.xticks(list(x), status_keys, rotation=20, ha="right")
    plt.ylabel("Rows")
    plt.title("Status Counts")
    plt.legend()
    save(out / "02_status_counts.png")

    # instruction count sample scores
    bucket_maps = {}
    for name, rows in rows_by_name.items():
        ok = ok_rows(rows)
        d: dict[str, list[float]] = defaultdict(list)
        for r in ok:
            d[str(r.get("instruction_count") or 0)].append(float(r["judge"]["sample_score"]))
        bucket_maps[name] = {k: (statistics.mean(v), len(v)) for k, v in d.items()}
    bucket_keys = sorted(set().union(*(set(v.keys()) for v in bucket_maps.values())), key=lambda x: int(x))
    horizontal_compare(names, bucket_maps, bucket_keys, "Sample Score by Instruction Count", out / "03_by_instruction_count.png")

    # source and axis
    for field, filename, title in [("source", "04_by_source.png", "Instruction Score by Source"), ("axis", "05_by_axis.png", "Instruction Score by Axis")]:
        maps = {name: group_item_mean(items_by_name[name], field) for name in names}
        keys = sorted(set().union(*(set(v.keys()) for v in maps.values())), key=lambda k: min(maps[n].get(k, (999, 0))[0] for n in names))
        horizontal_compare(names, maps, keys, title, out / filename)

    # failures counts
    fmaps = {name: failure_counts(items_by_name[name]) for name in names}
    fkeys = sorted(set().union(*(set(v.keys()) for v in fmaps.values())), key=lambda k: -sum(fmaps[n].get(k, 0) for n in names))
    plt.figure(figsize=(max(9, len(fkeys) * 1.1), 5))
    x = range(len(fkeys))
    width = min(0.8 / max(1, len(names)), 0.35)
    max_val = 0
    for i, name in enumerate(names):
        vals = [fmaps[name].get(k, 0) for k in fkeys]
        max_val = max(max_val, max(vals) if vals else 0)
        offs = [j + (i - (len(names) - 1) / 2) * width for j in x]
        bars = plt.bar(offs, vals, width=width, label=name, color=COLORS[i % len(COLORS)])
        for bar, val in zip(bars, vals):
            if val:
                plt.text(bar.get_x() + bar.get_width() / 2, val + max(1, max_val * 0.01), str(val), ha="center", va="bottom", fontsize=8)
    plt.xticks(list(x), fkeys, rotation=25, ha="right")
    plt.ylabel("Failed instructions")
    plt.title("Failure Type Counts")
    plt.legend()
    save(out / "06_failure_types.png")

    # atom-level comparison
    atom_maps = {name: group_item_mean(items_by_name[name], "atom_id") for name in names}
    if len(names) == 2:
        a, b = names
        common = set(atom_maps[a]) & set(atom_maps[b])
        deltas = sorted(((atom_maps[b][k][0] - atom_maps[a][k][0], k, atom_maps[a][k][1], atom_maps[b][k][1]) for k in common), key=lambda x: x[0])
        selected = deltas[:15] + deltas[-15:]
        labels = [x[1] for x in selected]
        vals = [x[0] for x in selected]
        colors = ["#EF4444" if v < 0 else "#10B981" for v in vals]
        plt.figure(figsize=(12, max(7, 0.35 * len(labels) + 1)))
        bars = plt.barh(labels, vals, color=colors)
        plt.axvline(0, color="#334155", linewidth=1)
        plt.xlabel(f"Delta mean score ({b} - {a})")
        plt.title("Largest Atom Score Deltas")
        for bar, val in zip(bars, vals):
            plt.text(val + (0.015 if val >= 0 else -0.015), bar.get_y() + bar.get_height()/2, f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
        save(out / "07_atom_score_deltas.png")
    else:
        union_atoms = set().union(*(set(atom_maps[name].keys()) for name in names))
        atom_min_rows = []
        for atom in union_atoms:
            vals = [atom_maps[name].get(atom, (0.0, 0))[0] for name in names if atom in atom_maps[name]]
            counts = sum(atom_maps[name].get(atom, (0.0, 0))[1] for name in names)
            if vals:
                atom_min_rows.append((statistics.mean(vals), counts, atom))
        keys = [x[2] for x in sorted(atom_min_rows, key=lambda x: (x[0], -x[1]))[:30]]
        horizontal_compare(names, atom_maps, keys, "Lowest Atom Scores Across Runs", out / "07_lowest_atoms_comparison.png")

    summary = {
        "runs": {name: m_by_name[name] for name in names},
    }
    (out / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Instruction Following Run Comparison", ""]
    for name in names:
        m = m_by_name[name]
        lines.extend([
            f"## {name}",
            f"- total_rows: {m['total']}",
            f"- judged_ok: {m['judged_ok']}",
            f"- status_counts: {m['status_counts']}",
            f"- judged_mean_score: {float(m['mean_judged']):.4f}",
            f"- full_mean_score_generation_errors_as_zero: {float(m['mean_full']):.4f}",
            f"- strict_all_2_pass_rate: {float(m['strict']):.4f}",
            f"- soft_all_ge1_pass_rate: {float(m['soft']):.4f}",
            "",
        ])
    lines.extend([
        "## Figures",
        "- [Overall Comparison](01_overall_comparison.png)",
        "- [Status Counts](02_status_counts.png)",
        "- [By Instruction Count](03_by_instruction_count.png)",
        "- [By Source](04_by_source.png)",
        "- [By Axis](05_by_axis.png)",
        "- [Failure Types](06_failure_types.png)",
        "- [Atom Score Deltas / Lowest Atoms](07_atom_score_deltas.png or 07_lowest_atoms_comparison.png)",
    ])
    (out / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
