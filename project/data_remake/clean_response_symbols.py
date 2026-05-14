import argparse
import json
import os
import re


QUOTE_PATTERN = re.compile(r'["“”\'‘’]')
DASH_PATTERN = re.compile(r'[—–－]+')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean quotes and dash punctuation in gpt response fields."
    )
    parser.add_argument("--input", required=True, help="Input JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    return parser.parse_args()


def load_data(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_items(data):
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON structure. Expect list or dict with 'items'.")


def clean_response_text(text):
    text = QUOTE_PATTERN.sub("", text)
    text = DASH_PATTERN.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def clean_items(items):
    updated_turns = 0
    updated_items = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        conversations = item.get("conversations")
        if not isinstance(conversations, list):
            continue

        item_changed = False

        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).lower()
            if role not in {"gpt", "assistant", "bot", "model"}:
                continue

            value = turn.get("value")
            if not isinstance(value, dict):
                continue

            response = value.get("response")
            if not isinstance(response, str):
                continue

            cleaned = clean_response_text(response)
            if cleaned != response:
                value["response"] = cleaned
                updated_turns += 1
                item_changed = True

        if item_changed:
            updated_items += 1

    return updated_items, updated_turns


def write_data(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    data = load_data(args.input)
    items = get_items(data)

    total_items = len(items)
    changed_items, changed_turns = clean_items(items)
    write_data(args.output, data)

    print("✅ 清洗完成")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"总条数: {total_items}")
    print(f"改动条目数: {changed_items}")
    print(f"改动回复轮次数: {changed_turns}")


if __name__ == "__main__":
    main()
