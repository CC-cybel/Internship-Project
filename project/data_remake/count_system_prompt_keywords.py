import argparse
import json
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 JSON 文件中 system 字段里的关键词出现次数")
    parser.add_argument(
        "--input",
        default="data_remake/outputs/normal/normal_s5_dual_full.json",
        help="输入 JSON 文件路径，支持 list 或 {'items': [...]} 根结构",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["孩子", "家长", "娃娃", "小孩"],
        help="要统计的关键词列表，默认：孩子 家长",
    )
    return parser.parse_args()


def load_items(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
    else:
        raise ValueError("输入根结构必须是 list 或 {'items': list}")

    return [item for item in items if isinstance(item, dict)]


def main() -> None:
    args = parse_args()
    items = load_items(args.input)

    counts = {keyword: 0 for keyword in args.keywords}
    samples_with_system = 0

    for item in items:
        system_text = item.get("system")
        if not isinstance(system_text, str) or not system_text:
            continue

        samples_with_system += 1
        for keyword in args.keywords:
            counts[keyword] += system_text.count(keyword)

    print(f"input: {args.input}")
    print(f"total samples: {len(items)}")
    print(f"samples with system: {samples_with_system}")
    for keyword in args.keywords:
        print(f"{keyword}: {counts[keyword]}")


if __name__ == "__main__":
    main()
