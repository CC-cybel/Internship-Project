import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple
from xml.sax.saxutils import escape


SLOT_ORDER = [
    "age",
    "gender",
    "name",
    "phone",
    "wechat",
    "symptom",
    "duration",
    "medical_history",
    "patient_relation",
    "relationship",
]

MODE_RESPONSE = "response"
MODE_RESPONSE_SLOT = "response_slot"
MODE_FULL = "thought_slot_response"

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)

OUTPUT_BLOCK_XML_RESPONSE_SLOT = (
    "输出格式规范：\n"
    "Agent 的回复必须是一个 XML 结构，格式如下：\n"
    "<output>\n"
    "  <meta>\n"
    "    <action>...</action>\n"
    "    <slot_values>\n"
    "      <age>0|1</age>\n"
    "      <gender>0|1</gender>\n"
    "      ...\n"
    "    </slot_values>\n"
    "  </meta>\n"
    "  <final_response>面向用户的回复</final_response>\n"
    "</output>\n\n"
    "<meta> 仅用于程序解析，不得输出 JSON。\n"
    "<final_response> 必须遵守语言风格约束。\n"
    "若 User Input 含 <action>...</action>，必须写入 <action> 并在 <final_response> 中执行。\n"
)

OUTPUT_BLOCK_XML_FULL = (
    "输出格式规范：\n"
    "Agent 的回复必须是一个 XML 结构，格式如下：\n"
    "<output>\n"
    "  <meta>\n"
    "    <action>...</action>\n"
    "    <thought>...</thought>\n"
    "    <slot_values>\n"
    "      <age>0|1</age>\n"
    "      <gender>0|1</gender>\n"
    "      ...\n"
    "    </slot_values>\n"
    "  </meta>\n"
    "  <final_response>面向用户的回复</final_response>\n"
    "</output>\n\n"
    "<meta> 仅用于程序解析，不得输出 JSON。\n"
    "<final_response> 必须遵守语言风格约束。\n"
    "若 User Input 含 <action>...</action>，必须写入 <action> 并在 <final_response> 中执行。\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset to XML-tagged assistant output.")
    parser.add_argument("--input", required=True, help="Input JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--mode",
        choices=[MODE_RESPONSE, MODE_RESPONSE_SLOT, MODE_FULL],
        default=MODE_RESPONSE_SLOT,
        help="Conversion mode.",
    )
    return parser.parse_args()


def load_data(path: str) -> Tuple[Any, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data, data["items"]
    if isinstance(data, list):
        return data, data

    raise ValueError("Input root must be a list or a dict with 'items'.")


def write_data(path: str, data: Any) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def strip_spaces(text: Any) -> str:
    return " ".join(str(text).split())


def extract_actions(user_text: str) -> List[str]:
    if not isinstance(user_text, str) or not user_text:
        return []

    match = ACTION_RE.search(user_text)
    if not match:
        return []

    raw = match.group(1).strip()
    parts = re.split(r"[，,、|/\n\r\t ]+", raw)
    return [part for part in parts if part]


def parse_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        text = value.strip()

        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        if "BEGIN_FINAL" in text and "END_FINAL" in text:
            start = text.find("BEGIN_FINAL") + len("BEGIN_FINAL")
            end = text.find("END_FINAL", start)
            response = text[start:end].strip() if end != -1 else text[start:].strip()
            thought = ""
            if "thought=" in text and "END_META" in text:
                meta_start = text.find("thought=") + len("thought=")
                meta_end = text.find("\n", meta_start)
                thought = text[meta_start:meta_end].strip() if meta_end != -1 else text[meta_start:].strip()
            return {"response": response, "thought": thought}

        if "<final_response>" in text and "</final_response>" in text:
            start = text.find("<final_response>") + len("<final_response>")
            end = text.find("</final_response>", start)
            response = text[start:end].strip() if end != -1 else text[start:].strip()
            return {"response": response}

        return {"response": text}

    return {"response": str(value)}


def normalize_slot_lines(slot_values: Any) -> List[Tuple[str, str]]:
    slot_dict = slot_values if isinstance(slot_values, dict) else {}
    ordered_keys = [key for key in SLOT_ORDER if key in slot_dict]
    for key in sorted(slot_dict.keys()):
        if key not in ordered_keys:
            ordered_keys.append(key)

    pairs: List[Tuple[str, str]] = []
    for key in ordered_keys:
        value = slot_dict.get(key)
        if isinstance(value, bool):
            normalized = "1" if value else "0"
        elif value is None:
            normalized = "0"
        else:
            normalized = escape(str(value))
        pairs.append((key, normalized))
    return pairs


def convert_gpt_value(value: Any, last_user_text: str, mode: str) -> str:
    payload = parse_value(value)
    response = escape(str(payload.get("response", "")).strip())

    if mode == MODE_RESPONSE:
        return str(payload.get("response", "")).strip()

    actions = "|".join(extract_actions(last_user_text))
    action_xml = f"<action>{escape(actions)}</action>"

    lines = ["<output>", "  <meta>", f"    {action_xml}"]

    if mode == MODE_FULL:
        thought = escape(strip_spaces(payload.get("thought", "")))
        lines.append(f"    <thought>{thought}</thought>")

    lines.append("    <slot_values>")
    for key, val in normalize_slot_lines(payload.get("slot_values", {})):
        lines.append(f"      <{key}>{val}</{key}>")
    lines.append("    </slot_values>")
    lines.append("  </meta>")
    lines.append(f"  <final_response>{response}</final_response>")
    lines.append("</output>")
    return "\n".join(lines)


def rewrite_output_format(system_text: Any, mode: str) -> Any:
    if not isinstance(system_text, str):
        return system_text

    marker = "输出格式规范："
    prefix = system_text
    if marker in system_text:
        prefix = system_text[: system_text.find(marker)].rstrip()

    if mode == MODE_RESPONSE:
        return prefix

    block = OUTPUT_BLOCK_XML_RESPONSE_SLOT if mode == MODE_RESPONSE_SLOT else OUTPUT_BLOCK_XML_FULL
    return prefix + "\n\n" + block


def process_items(items: List[Dict[str, Any]], mode: str) -> Tuple[int, int]:
    changed_system = 0
    changed_messages = 0

    for item in items:
        if not isinstance(item, dict):
            continue

        if "system" in item:
            new_system = rewrite_output_format(item.get("system"), mode)
            if new_system != item.get("system"):
                item["system"] = new_system
                changed_system += 1

        conversations = item.get("conversations")
        if not isinstance(conversations, list):
            continue

        last_user_text = ""
        for msg in conversations:
            if not isinstance(msg, dict):
                continue

            role = str(msg.get("from", "")).lower()
            if role in {"human", "user"}:
                value = msg.get("value", "")
                last_user_text = value if isinstance(value, str) else str(value)
            elif role in {"gpt", "assistant", "bot", "model"}:
                old_value = msg.get("value")
                new_value = convert_gpt_value(old_value, last_user_text, mode)
                if new_value != old_value:
                    msg["value"] = new_value
                    changed_messages += 1

    return changed_system, changed_messages


def main() -> None:
    args = parse_args()
    root, items = load_data(args.input)

    changed_system, changed_messages = process_items(items, args.mode)
    write_data(args.output, root)

    print("✅ XML 转换完成")
    print(f"模式: {args.mode}")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"样本数: {len(items)}")
    print(f"修改system条数: {changed_system}")
    print(f"修改gpt消息条数: {changed_messages}")


if __name__ == "__main__":
    main()
