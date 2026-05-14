import argparse
import json
import os
import re


DEFAULT_PLACEHOLDER = "19900000000"


def parse_args():
    parser = argparse.ArgumentParser(description="Mask user phone numbers in JSON datasets.")
    parser.add_argument("--input", required=True, help="Input JSON file path.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument(
        "--placeholder",
        default=DEFAULT_PLACEHOLDER,
        help="Replacement phone number for masked values.",
    )
    parser.add_argument(
        "--roles",
        default="human,user",
        help="Comma-separated roles to mask (default: human,user).",
    )
    parser.add_argument(
        "--replace-landline",
        action="store_true",
        help="Also replace landline numbers (default: only mobile).",
    )
    return parser.parse_args()


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data, data["items"]
    if isinstance(data, list):
        return None, data
    raise ValueError("Unsupported input format: expected list or dict with 'items'.")


def normalize_role(role):
    if not role:
        return None
    r = str(role).lower()
    if r in {"human", "user"}:
        return r
    if r in {"assistant", "gpt", "bot", "model"}:
        return "gpt"
    return r


def build_patterns():
    mobile = re.compile(r"(?<!\d)(?:\+?86[-\s]*)?(1[3-9](?:[-\s]*\d){9})(?!\d)")
    landline = re.compile(r"(?<!\d)(0\d{2,3}[-\s]?\d{7,8})(?!\d)")
    return mobile, landline


def mask_text(text, placeholder, mobile_pat, landline_pat, replace_landline):
    if not isinstance(text, str) or not text:
        return text, 0

    count = 0

    def repl(_match):
        nonlocal count
        count += 1
        return placeholder

    text = mobile_pat.sub(repl, text)
    if replace_landline:
        text = landline_pat.sub(repl, text)
    return text, count


def process_item(item, roles, placeholder, mobile_pat, landline_pat, replace_landline):
    if not isinstance(item, dict):
        return item, 0

    total = 0
    for conv_key in ("conversations", "messages"):
        convs = item.get(conv_key)
        if not isinstance(convs, list):
            continue
        for turn in convs:
            if not isinstance(turn, dict):
                continue
            role = normalize_role(turn.get("from") or turn.get("role"))
            if role not in roles:
                continue

            if "value" in turn and isinstance(turn["value"], str):
                new_text, cnt = mask_text(
                    turn["value"],
                    placeholder,
                    mobile_pat,
                    landline_pat,
                    replace_landline,
                )
                turn["value"] = new_text
                total += cnt
            elif "content" in turn and isinstance(turn["content"], str):
                new_text, cnt = mask_text(
                    turn["content"],
                    placeholder,
                    mobile_pat,
                    landline_pat,
                    replace_landline,
                )
                turn["content"] = new_text
                total += cnt

    return item, total


def write_output(path, wrapper, items):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if wrapper is not None:
        wrapper["items"] = items
        data = wrapper
    else:
        data = items
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    roles = {r.strip().lower() for r in args.roles.split(",") if r.strip()}

    wrapper, items = load_items(args.input)
    mobile_pat, landline_pat = build_patterns()

    total_hits = 0
    for i, item in enumerate(items):
        items[i], cnt = process_item(
            item,
            roles,
            args.placeholder,
            mobile_pat,
            landline_pat,
            args.replace_landline,
        )
        total_hits += cnt

    write_output(args.output, wrapper, items)

    print("✅ 脱敏完成")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"替换次数: {total_hits}")
    print(f"脱敏角色: {', '.join(sorted(roles))}")
    print(f"替换座机: {args.replace_landline}")


if __name__ == "__main__":
    main()
