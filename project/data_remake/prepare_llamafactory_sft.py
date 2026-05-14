import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ShareGPT-format data for LLaMA-Factory SFT.")
    parser.add_argument("--input", required=True, help="Input JSON file path.")
    parser.add_argument("--output", required=True, help="Output JSON/JSONL file path.")
    parser.add_argument(
        "--assistant-mode",
        choices=["full", "response", "response_slot"],
        default="full",
        help="How to serialize assistant content when value is a dict.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="json",
        help="Output file format.",
    )
    parser.add_argument(
        "--strict-order",
        action="store_true",
        help="Drop samples that do not follow human/gpt alternating order.",
    )
    return parser.parse_args()


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported input format: expected list or dict with 'items'.")


def normalize_role(role):
    if not role:
        return None
    r = str(role).lower()
    if r in {"human", "user"}:
        return "human"
    if r in {"gpt", "assistant", "bot", "model"}:
        return "gpt"
    return role


def serialize_assistant_value(value, mode):
    if mode == "full":
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)
    if mode == "response":
        if isinstance(value, dict):
            return str(value.get("response", "")).strip()
        if isinstance(value, str):
            return value
        return str(value)
    if mode == "response_slot":
        if isinstance(value, dict):
            payload = {
                "slot_values": value.get("slot_values", {}),
                "response": value.get("response", ""),
            }
            return json.dumps(payload, ensure_ascii=False)
        if isinstance(value, str):
            return value
        return str(value)
    return str(value)


def serialize_value(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def is_valid_order(convs):
    for idx, msg in enumerate(convs):
        role = msg.get("from")
        if idx % 2 == 0:
            if role not in {"human", "observation"}:
                return False
        else:
            if role not in {"gpt", "function", "function_call"}:
                return False
    return True


def convert_item(item, assistant_mode):
    if not isinstance(item, dict):
        return None
    convs = item.get("conversations", [])
    if not isinstance(convs, list):
        return None

    new_convs = []
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = normalize_role(turn.get("from") or turn.get("role"))
        value = turn.get("value", turn.get("content"))
        if role is None:
            continue
        if role == "gpt":
            content = serialize_assistant_value(value, assistant_mode)
        else:
            content = serialize_value(value)
        new_convs.append({"from": role, "value": content})

    output = {"conversations": new_convs}
    if "system" in item and isinstance(item["system"], str):
        output["system"] = item["system"]
    if "tools" in item:
        output["tools"] = item["tools"]
    return output


def write_output(path, items, fmt):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    items = load_items(args.input)

    converted = []
    skipped = 0
    invalid_order = 0

    for item in items:
        new_item = convert_item(item, args.assistant_mode)
        if not new_item:
            skipped += 1
            continue
        if not is_valid_order(new_item["conversations"]):
            invalid_order += 1
            if args.strict_order:
                skipped += 1
                continue
        converted.append(new_item)

    write_output(args.output, converted, args.output_format)

    print("✅ 处理完成")
    print(f"总数: {len(items)}")
    print(f"输出: {len(converted)}")
    print(f"跳过: {skipped}")
    print(f"顺序异常: {invalid_order}")


if __name__ == "__main__":
    main()
