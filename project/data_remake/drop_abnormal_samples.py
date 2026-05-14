import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="丢弃异常 ShareGPT 样本（支持 JSON / JSONL）")
    parser.add_argument(
        "--input",
        required=True,
        help="输入文件路径，支持 json 或 jsonl",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出文件路径，默认自动在输入文件名后追加 _drop_abnormal",
    )
    parser.add_argument(
        "--keep-odd-turns",
        action="store_true",
        help="默认会丢弃奇数条对话消息（轮次异常）；加此参数可保留",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_drop_abnormal{input_path.suffix}")


def load_data(path: Path) -> tuple[str, Any, list[Any], int]:
    with path.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return "json", [], [], 0

    try:
        root = json.loads(content)
        if isinstance(root, list):
            return "json", root, root, len(root)
        if isinstance(root, dict) and isinstance(root.get("items"), list):
            return "json", root, root["items"], len(root["items"])
        raise ValueError("JSON 根结构必须是 list 或 {'items': [...]}。")
    except json.JSONDecodeError:
        items: list[Any] = []
        invalid_json_lines = 0
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                invalid_json_lines += 1
        return "jsonl", None, items, len(items) + invalid_json_lines


def sanitize_sharegpt_item(item: Any, drop_odd_turns: bool) -> tuple[Any, str]:
    if not isinstance(item, dict):
        return None, "item_not_dict"

    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None, "conversations_not_list"

    if drop_odd_turns and len(conversations) % 2 == 1:
        return None, "odd_turn_count"

    cleaned_conversations = []
    for turn in conversations:
        if not isinstance(turn, dict):
            return None, "turn_not_dict"

        if set(turn.keys()) != {"from", "value"}:
            return None, "turn_keys_invalid"

        role = turn.get("from", turn.get("role"))
        value = turn.get("value")

        if not isinstance(role, str):
            return None, "role_not_str"

        if not isinstance(value, str):
            return None, "value_not_str"

        cleaned_conversations.append({"from": role, "value": value})

    cleaned_item = {"conversations": cleaned_conversations}
    if isinstance(item.get("system"), str):
        cleaned_item["system"] = item["system"]
    if "tools" in item:
        cleaned_item["tools"] = item["tools"]

    return cleaned_item, "kept"


def build_output(root: Any, output_format: str, kept_items: list[dict[str, Any]]) -> Any:
    if output_format == "jsonl":
        return kept_items

    if isinstance(root, list):
        return kept_items

    if isinstance(root, dict) and isinstance(root.get("items"), list):
        copied = dict(root)
        copied["items"] = kept_items
        return copied

    raise ValueError("JSON 根结构必须是 list 或 {'items': [...]}。")


def write_output(path: Path, output_format: str, output_data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if output_format == "jsonl":
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            return

        json.dump(output_data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    output_format, root, items, total_input_count = load_data(input_path)
    drop_odd_turns = not args.keep_odd_turns

    kept_items: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {
        "kept": 0,
        "item_not_dict": 0,
        "conversations_not_list": 0,
        "odd_turn_count": 0,
        "turn_not_dict": 0,
        "turn_keys_invalid": 0,
        "role_not_str": 0,
        "value_not_str": 0,
    }

    for item in items:
        cleaned, reason = sanitize_sharegpt_item(item, drop_odd_turns=drop_odd_turns)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if cleaned is not None:
            kept_items.append(cleaned)

    dropped_invalid_json = max(total_input_count - len(items), 0)

    output_data = build_output(root, output_format, kept_items)
    write_output(output_path, output_format, output_data)

    total_dropped = total_input_count - len(kept_items)

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"format: {output_format}")
    print(f"drop odd turns: {drop_odd_turns}")
    print(f"total input samples: {total_input_count}")
    print(f"kept samples: {len(kept_items)}")
    print(f"dropped samples: {total_dropped}")
    print(f"drop ratio: {total_dropped / total_input_count:.4%}" if total_input_count else "drop ratio: N/A")
    if output_format == "jsonl":
        print(f"invalid jsonl lines: {dropped_invalid_json}")

    print("drop reasons:")
    for key in [
        "item_not_dict",
        "conversations_not_list",
        "odd_turn_count",
        "turn_not_dict",
        "turn_keys_invalid",
        "role_not_str",
        "value_not_str",
    ]:
        print(f"  - {key}: {reason_counts.get(key, 0)}")


if __name__ == "__main__":
    main()

