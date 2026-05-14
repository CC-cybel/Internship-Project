import argparse
import json
import os
from typing import Any


FIELD_RULES = [
    ("孩子年龄", "年龄"),
    ("孩子性别", "性别"),
    ("孩子姓名", "姓名"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅替换 system 字段里的孩子年龄/性别/姓名具体值。"
    )
    parser.add_argument(
        "--input",
        default="data_remake/outputs/hard/hard_resp.json",
        help="输入 JSON 文件路径，支持 list 或 {'items': [...]} 根结构。",
    )
    parser.add_argument(
        "--output",
        default="data_remake/outputs/hard/hard_resp_system_masked.json",
        help="输出 JSON 文件路径。",
    )
    return parser.parse_args()


def load_items(path: str) -> tuple[Any, list[dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data, [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = [item for item in data["items"] if isinstance(item, dict)]
        return data, items

    raise ValueError("输入根结构必须是 list 或 {'items': list}")


def write_root(path: str, root: Any) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(root, file, ensure_ascii=False, indent=2)


def replace_system_profile(text: str) -> tuple[str, dict[str, int]]:
    updated = text
    counts: dict[str, int] = {}

    for field_name, placeholder in FIELD_RULES:
        replaced_count = updated.count(field_name)
        if replaced_count > 0:
            updated = updated.replace(field_name, placeholder)
        counts[field_name] = replaced_count

    return updated, counts


def main() -> None:
    args = parse_args()
    root, items = load_items(args.input)

    total_samples = 0
    changed_samples = 0
    total_replacements = {field_name: 0 for field_name, _ in FIELD_RULES}

    for item in items:
        total_samples += 1
        system_text = item.get("system")
        if not isinstance(system_text, str) or not system_text:
            continue

        new_system, counts = replace_system_profile(system_text)
        if new_system != system_text:
            item["system"] = new_system
            changed_samples += 1

        for field_name in total_replacements:
            total_replacements[field_name] += counts.get(field_name, 0)

    write_root(args.output, root)

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"total samples: {total_samples}")
    print(f"changed samples: {changed_samples}")
    for field_name, _ in FIELD_RULES:
        print(f"{field_name} replacements: {total_replacements[field_name]}")


if __name__ == "__main__":
    main()
