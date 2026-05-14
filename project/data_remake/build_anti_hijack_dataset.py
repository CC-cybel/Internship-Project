import argparse
import copy
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


FINAL_BLOCK_RE = re.compile(r"BEGIN_FINAL\s*(.*?)\s*END_FINAL", re.DOTALL)


@dataclass(frozen=True)
class TurnCandidate:
    dialog_id: int
    turn_idx: int
    history_assistant_turns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build anti-hijack A/B/C dataset from ShareGPT dialogues.")
    parser.add_argument("--input", required=True, help="Input JSON path (list of ShareGPT samples).")
    parser.add_argument("--output", required=True, help="Output dataset path (.json or .jsonl).")
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output file format.",
    )
    parser.add_argument("--num-samples", type=int, default=20000, help="Target number of generated anti samples.")
    parser.add_argument("--ratio-a", type=float, default=0.30, help="Class A ratio.")
    parser.add_argument("--ratio-b", type=float, default=0.10, help="Class B ratio.")
    parser.add_argument("--ratio-c", type=float, default=0.60, help="Class C ratio.")
    parser.add_argument(
        "--max-slices-per-dialog",
        type=int,
        default=6,
        help="Maximum generated samples per source dialogue. Set <=0 to disable.",
    )
    parser.add_argument(
        "--a-history-final-only-ratio",
        type=float,
        default=0.40,
        help="In class A, proportion of history assistant turns transformed to human-agent FINAL-only.",
    )
    parser.add_argument(
        "--b-min-history-assistant-turns",
        type=int,
        default=4,
        help="In class B, minimum number of history assistant turns before target turn.",
    )
    parser.add_argument(
        "--c-human-tail-turns",
        type=int,
        default=1,
        help="In class C, number of latest history assistant turns replaced as human-agent messages.",
    )
    parser.add_argument(
        "--human-agent-marker",
        default="【人工客服】",
        help="Marker prefix inserted for simulated human-agent replies.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-dialogs",
        type=int,
        default=0,
        help="Only use first N dialogues for quick debug. Set <=0 to disable.",
    )
    parser.add_argument(
        "--stats-output",
        default="",
        help="Optional path to save generation stats JSON. Default: <output>.stats.json",
    )
    return parser.parse_args()


def load_data(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of ShareGPT samples.")
    return data


def write_data(path: str, items: list[dict[str, Any]], fmt: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


def is_human_role(role: str) -> bool:
    return role in {"human", "user"}


def is_assistant_role(role: str) -> bool:
    return role in {"gpt", "assistant", "bot", "model"}


def normalize_role(role: Any) -> str:
    return str(role or "").strip().lower()


def is_strict_dual_format(text: str) -> bool:
    if not isinstance(text, str):
        return False
    p1 = text.find("BEGIN_META")
    p2 = text.find("END_META")
    p3 = text.find("BEGIN_FINAL")
    p4 = text.find("END_FINAL")
    return -1 not in (p1, p2, p3, p4) and p1 < p2 < p3 < p4


def extract_final_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = FINAL_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def collect_candidates(
    data: list[dict[str, Any]],
    b_min_history_assistant_turns: int,
    c_human_tail_turns: int,
) -> tuple[list[TurnCandidate], list[TurnCandidate], list[TurnCandidate], dict[str, int]]:
    pool_a: list[TurnCandidate] = []
    pool_b: list[TurnCandidate] = []
    pool_c: list[TurnCandidate] = []

    stats = defaultdict(int)
    for dialog_id, item in enumerate(data):
        conversations = item.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            stats["invalid_dialog"] += 1
            continue

        history_assistant = 0
        valid_turns = 0
        for idx, turn in enumerate(conversations):
            if not isinstance(turn, dict):
                continue
            role = normalize_role(turn.get("from", turn.get("role")))
            value = turn.get("value", turn.get("content", ""))
            if not isinstance(value, str):
                value = str(value)

            if is_assistant_role(role):
                if is_strict_dual_format(value):
                    cand = TurnCandidate(dialog_id=dialog_id, turn_idx=idx, history_assistant_turns=history_assistant)
                    if history_assistant >= 1:
                        pool_a.append(cand)
                    if history_assistant >= b_min_history_assistant_turns:
                        pool_b.append(cand)
                    if history_assistant >= c_human_tail_turns:
                        pool_c.append(cand)
                    valid_turns += 1
                history_assistant += 1

        if valid_turns == 0:
            stats["no_strict_assistant_turn_dialog"] += 1
        stats["valid_dialog"] += 1

    return pool_a, pool_b, pool_c, dict(stats)


def calc_target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError("Sum of ratios must be positive.")
    normalized = {k: v / ratio_sum for k, v in ratios.items()}

    exact = {k: total * normalized[k] for k in normalized}
    counts = {k: int(math.floor(v)) for k, v in exact.items()}
    remainder = total - sum(counts.values())
    if remainder > 0:
        frac_order = sorted(normalized.keys(), key=lambda k: (exact[k] - counts[k]), reverse=True)
        for i in range(remainder):
            counts[frac_order[i % len(frac_order)]] += 1
    return counts


def sample_with_budget(
    class_name: str,
    candidates: list[TurnCandidate],
    need: int,
    dialog_budget: dict[int, int] | None,
    rng: random.Random,
) -> list[TurnCandidate]:
    if need <= 0 or not candidates:
        return []

    shuffled = candidates[:]
    rng.shuffle(shuffled)
    picked: list[TurnCandidate] = []
    used = set()

    for cand in shuffled:
        if len(picked) >= need:
            break
        key = (cand.dialog_id, cand.turn_idx)
        if key in used:
            continue
        if dialog_budget is not None and dialog_budget.get(cand.dialog_id, 0) <= 0:
            continue
        picked.append(cand)
        used.add(key)
        if dialog_budget is not None:
            dialog_budget[cand.dialog_id] -= 1

    return picked


def replace_with_human_agent_final_only(text: str, marker: str) -> str:
    final_text = extract_final_text(text)
    if final_text:
        return f"{marker}\n{final_text}"
    return marker


def replace_with_final_only(text: str) -> str:
    return extract_final_text(text)


def build_sample_from_candidate(
    source_item: dict[str, Any],
    candidate: TurnCandidate,
    class_name: str,
    rng: random.Random,
    a_history_final_only_ratio: float,
    c_human_tail_turns: int,
    human_agent_marker: str,
) -> dict[str, Any]:
    conversations = source_item["conversations"]
    sliced = copy.deepcopy(conversations[: candidate.turn_idx + 1])

    assistant_hist_indices = [
        i
        for i, msg in enumerate(sliced[:-1])
        if is_assistant_role(normalize_role(msg.get("from", msg.get("role"))))
    ]

    if class_name == "A":
        to_replace = max(1, int(round(len(assistant_hist_indices) * a_history_final_only_ratio)))
        to_replace = min(to_replace, len(assistant_hist_indices))
        chosen = set(rng.sample(assistant_hist_indices, to_replace)) if to_replace > 0 else set()
        for i in chosen:
            old_value = sliced[i].get("value", "")
            sliced[i]["value"] = replace_with_human_agent_final_only(str(old_value), human_agent_marker)
    elif class_name == "B":
        for i in assistant_hist_indices:
            old_value = sliced[i].get("value", "")
            sliced[i]["value"] = replace_with_final_only(str(old_value))
    elif class_name == "C":
        k = min(max(c_human_tail_turns, 1), len(assistant_hist_indices))
        for i in assistant_hist_indices[-k:]:
            old_value = sliced[i].get("value", "")
            sliced[i]["value"] = replace_with_human_agent_final_only(str(old_value), human_agent_marker)
    else:
        raise ValueError(f"Unknown class: {class_name}")

    new_item: dict[str, Any] = {"conversations": sliced}
    if isinstance(source_item.get("system"), str):
        new_item["system"] = source_item["system"]
    if "tools" in source_item:
        new_item["tools"] = source_item["tools"]

    new_item["anti_class"] = class_name
    new_item["source_dialog_id"] = candidate.dialog_id
    new_item["target_turn_idx"] = candidate.turn_idx
    new_item["history_assistant_turns"] = candidate.history_assistant_turns
    return new_item


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    data = load_data(args.input)
    if args.max_dialogs > 0:
        data = data[: args.max_dialogs]

    pool_a, pool_b, pool_c, scan_stats = collect_candidates(
        data=data,
        b_min_history_assistant_turns=max(args.b_min_history_assistant_turns, 0),
        c_human_tail_turns=max(args.c_human_tail_turns, 1),
    )

    ratios = {"A": args.ratio_a, "B": args.ratio_b, "C": args.ratio_c}
    target_counts = calc_target_counts(max(args.num_samples, 0), ratios)

    max_slices = args.max_slices_per_dialog if args.max_slices_per_dialog > 0 else None
    dialog_budget = {i: max_slices for i in range(len(data))} if max_slices is not None else None

    class_to_pool = {"A": pool_a, "B": pool_b, "C": pool_c}

    # Prioritize scarce class first.
    order = sorted(
        ["A", "B", "C"],
        key=lambda k: (len(class_to_pool[k]) / max(target_counts[k], 1)),
    )

    selected: dict[str, list[TurnCandidate]] = {"A": [], "B": [], "C": []}
    for cls in order:
        selected[cls] = sample_with_budget(
            class_name=cls,
            candidates=class_to_pool[cls],
            need=target_counts[cls],
            dialog_budget=dialog_budget,
            rng=rng,
        )

    generated: list[dict[str, Any]] = []
    class_counts = {"A": 0, "B": 0, "C": 0}
    for cls in ["A", "B", "C"]:
        for cand in selected[cls]:
            source_item = data[cand.dialog_id]
            item = build_sample_from_candidate(
                source_item=source_item,
                candidate=cand,
                class_name=cls,
                rng=rng,
                a_history_final_only_ratio=args.a_history_final_only_ratio,
                c_human_tail_turns=max(args.c_human_tail_turns, 1),
                human_agent_marker=args.human_agent_marker,
            )
            generated.append(item)
            class_counts[cls] += 1

    rng.shuffle(generated)
    write_data(args.output, generated, args.output_format)

    stats = {
        "input": args.input,
        "output": args.output,
        "output_format": args.output_format,
        "seed": args.seed,
        "scan_stats": scan_stats,
        "candidate_pool_size": {
            "A": len(pool_a),
            "B": len(pool_b),
            "C": len(pool_c),
        },
        "target_counts": target_counts,
        "actual_counts": class_counts,
        "actual_total": len(generated),
        "config": {
            "num_samples": args.num_samples,
            "ratio_a": args.ratio_a,
            "ratio_b": args.ratio_b,
            "ratio_c": args.ratio_c,
            "max_slices_per_dialog": args.max_slices_per_dialog,
            "a_history_final_only_ratio": args.a_history_final_only_ratio,
            "b_min_history_assistant_turns": args.b_min_history_assistant_turns,
            "c_human_tail_turns": args.c_human_tail_turns,
            "human_agent_marker": args.human_agent_marker,
            "max_dialogs": args.max_dialogs,
        },
    }
    stats_path = args.stats_output or f"{args.output}.stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("✅ Anti dataset generation finished")
    print(f"Input dialogs: {len(data)}")
    print(
        "Candidate pools: "
        f"A={len(pool_a)}, B={len(pool_b)}, C={len(pool_c)}"
    )
    print(
        "Generated counts: "
        f"A={class_counts['A']}, B={class_counts['B']}, C={class_counts['C']}, total={len(generated)}"
    )
    if len(generated) < args.num_samples:
        print(
            f"⚠ Requested {args.num_samples}, but generated {len(generated)}. "
            "You can increase --max-slices-per-dialog or reduce --num-samples."
        )
    print(f"Dataset written to: {args.output}")
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()
