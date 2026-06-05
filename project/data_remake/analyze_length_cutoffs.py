import argparse
import json
import math
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计多轮对话样本 token 长度并给出 cutoff 建议")
    parser.add_argument(
        "--input",
        default="data_remake/outputs/normal/normal_s5_dual_resp.json",
        help="输入数据文件，支持 list 或 {'items': [...]}",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B-Base",
        help="Tokenizer 路径或 HF 名称",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="是否启用 trust_remote_code")
    parser.add_argument(
        "--round-to",
        type=int,
        default=128,
        help="推荐 cutoff 向上取整粒度（常用 64/128）",
    )
    parser.add_argument(
        "--model-max-len",
        type=int,
        default=None,
        help="可选：模型最大上下文长度，推荐值会被限制到该值以内",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="可选：将统计结果写入 json 文件",
    )
    return parser.parse_args()


def load_items(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        out: list[dict[str, Any]] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
        return out

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("输入根结构必须是 list 或 {'items': list}")

    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    return out


def convert_to_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    system_text = item.get("system")
    if isinstance(system_text, str) and system_text.strip():
        messages.append({"role": "system", "content": system_text.strip()})

    conversations = item.get("conversations", [])
    if not isinstance(conversations, list):
        return messages

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role_raw = str(turn.get("from", "")).strip().lower()
        value = turn.get("value", "")
        content = value if isinstance(value, str) else str(value)

        if role_raw in {"human", "user"}:
            messages.append({"role": "user", "content": content})
        elif role_raw in {"gpt", "assistant", "bot", "model"}:
            messages.append({"role": "assistant", "content": content})

    return messages


def get_token_length(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if isinstance(token_ids, list):
                return len(token_ids)
            if hasattr(token_ids, "shape"):
                return int(token_ids.shape[-1])
        except Exception:
            pass

    fallback_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return len(tokenizer.encode(fallback_text, add_special_tokens=True))


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def round_up(x: int, base: int) -> int:
    if base <= 1:
        return x
    return ((x + base - 1) // base) * base


def ratio_over(values: list[int], cutoff: int) -> float:
    if not values:
        return 0.0
    over = sum(1 for v in values if v > cutoff)
    return over / len(values)


def main() -> None:
    args = parse_args()
    items = load_items(args.input)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    lengths: list[int] = []
    for idx, item in enumerate(items, start=1):
        messages = convert_to_messages(item)
        if not messages:
            continue
        lengths.append(get_token_length(tokenizer, messages))

        if idx % 2000 == 0:
            print(f"processed: {idx}/{len(items)}")

    if not lengths:
        raise SystemExit("没有可统计的有效样本")

    sorted_lengths = sorted(lengths)
    p95 = percentile(sorted_lengths, 0.95)
    p99 = percentile(sorted_lengths, 0.99)

    p95_ceil = int(math.ceil(p95))
    p99_ceil = int(math.ceil(p99))

    rec_p95 = round_up(p95_ceil, args.round_to)
    rec_p99 = round_up(p99_ceil, args.round_to)

    if args.model_max_len is not None:
        rec_p95 = min(rec_p95, args.model_max_len)
        rec_p99 = min(rec_p99, args.model_max_len)

    final_recommend = rec_p99

    report = {
        "input": args.input,
        "tokenizer": args.tokenizer,
        "num_samples": len(lengths),
        "min": sorted_lengths[0],
        "mean": sum(lengths) / len(lengths),
        "median": percentile(sorted_lengths, 0.5),
        "p95": p95,
        "p99": p99,
        "max": sorted_lengths[-1],
        "round_to": args.round_to,
        "recommend_p95": rec_p95,
        "recommend_p99": rec_p99,
        "final_recommend": final_recommend,
        "truncation_ratio_if_use_recommend_p95": ratio_over(lengths, rec_p95),
        "truncation_ratio_if_use_recommend_p99": ratio_over(lengths, rec_p99),
    }

    print("\n===== Token Length Stats =====")
    print(f"samples: {report['num_samples']}")
    print(f"min/mean/median/max: {report['min']} / {report['mean']:.2f} / {report['median']:.2f} / {report['max']}")
    print(f"p95: {report['p95']:.2f}")
    print(f"p99: {report['p99']:.2f}")
    print("\n===== Cutoff Recommendation =====")
    print(f"recommend (coverage ~95%): {report['recommend_p95']}")
    print(f"recommend (coverage ~99%): {report['recommend_p99']}")
    print(f"final recommend: {report['final_recommend']}")
    print(
        "truncation ratio @95/@99 recommend: "
        f"{report['truncation_ratio_if_use_recommend_p95']:.4%} / {report['truncation_ratio_if_use_recommend_p99']:.4%}"
    )

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\njson report saved to: {args.json_output}")


if __name__ == "__main__":
    main()
