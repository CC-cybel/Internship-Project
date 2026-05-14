import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="丢弃 system 中包含指定关键词的整条样本")
    parser.add_argument(
        "--input",
        default="data_remake/outputs/normal/normal_s5_dual_full.json",
        help="输入 JSON 文件路径，支持 list 或 {'items': [...]} 根结构",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出 JSON 文件路径，默认在输入文件名后追加 _drop_system_keywords",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["小孩", "娃娃", "家长", "孩子"],
        help="system 命中任一关键词即丢弃，默认：小孩 娃娃 家长",
    )
    return parser.parse_args()


def load_data(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
    else:
        raise ValueError("输入根结构必须是 list 或 {'items': list}")

    valid_items = [item for item in items if isinstance(item, dict)]
    return data, valid_items


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_drop_system_keywords{input_path.suffix}")


def build_output(root: Any, filtered_items: list[dict[str, Any]]) -> Any:
    if isinstance(root, list):
        return filtered_items
    if isinstance(root, dict) and isinstance(root.get("items"), list):
        copied = dict(root)
        copied["items"] = filtered_items
        return copied
    raise ValueError("输入根结构必须是 list 或 {'items': list}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    root, items = load_data(input_path)

    keyword_hits = {keyword: 0 for keyword in args.keywords}
    filtered_items: list[dict[str, Any]] = []
    removed_count = 0

    for item in items:
        system_text = item.get("system")
        if not isinstance(system_text, str):
            filtered_items.append(item)
            continue

        matched = False
        for keyword in args.keywords:
            if keyword in system_text:
                keyword_hits[keyword] += 1
                matched = True

        if matched:
            removed_count += 1
        else:
            filtered_items.append(item)

    output_data = build_output(root, filtered_items)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    total = len(items)
    kept = len(filtered_items)

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"total valid samples: {total}")
    print(f"removed: {removed_count}")
    print(f"kept: {kept}")
    print(f"remove ratio: {removed_count / total:.4%}" if total else "remove ratio: N/A")
    for keyword in args.keywords:
        print(f"system contains {keyword}: {keyword_hits[keyword]}")


if __name__ == "__main__":
    main()
