import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


META_PATTERN = re.compile(r"BEGIN_META\s*(.*?)\s*END_META", re.DOTALL)
FINAL_PATTERN = re.compile(r"BEGIN_FINAL\s*(.*?)\s*END_FINAL", re.DOTALL)

OUTPUT_BLOCK_THINK = (
    "输出格式规范：\n"
    "Agent 的回复必须包含两部分，顺序固定：\n"
    "<think>\n"
    "action=...\n"
    "slot_age=0/1\n"
    "slot_gender=0/1\n"
    "...\n"
    "</think>\n"
    "(面向用户的最终回复)\n\n"
    "<think> 仅用于程序解析，采用 key=value 的纯文本格式，不得使用 JSON。\n"
    "最终回复为用户可见内容，必须遵守语言风格约束。\n"
    "若 User Input 包含 <action>...</action>，必须在 <think> 中写明，并在最终回复中执行。\n"
)


@dataclass
class ConvertStats:
    sample_count: int = 0
    assistant_count: int = 0
    converted_count: int = 0
    final_only_count: int = 0
    unchanged_count: int = 0
    system_changed_count: int = 0


def _rewrite_output_format_block(system_text: Any) -> Any:
    if not isinstance(system_text, str):
        return system_text

    marker = "输出格式规范："
    prefix = system_text
    if marker in system_text:
        prefix = system_text[: system_text.find(marker)].rstrip()

    if prefix:
        return prefix + "\n\n" + OUTPUT_BLOCK_THINK
    return OUTPUT_BLOCK_THINK


def _to_think_format(content: str) -> tuple[str, bool, bool]:
    meta_match = META_PATTERN.search(content)
    final_match = FINAL_PATTERN.search(content)

    if final_match is None:
        return content, False, False

    final_text = final_match.group(1).strip()
    if meta_match is None:
        return final_text, True, True

    meta_text = meta_match.group(1).strip()
    converted = f"<think>\n{meta_text}\n</think>\n\n{final_text}"
    return converted, True, False


def _convert_example(
    example: dict,
    system_key: str,
    messages_key: str,
    role_key: str,
    content_key: str,
    assistant_roles: set[str],
    stats: ConvertStats,
) -> dict:
    messages = example.get(messages_key)
    stats.sample_count += 1

    system_text = example.get(system_key)
    new_system = _rewrite_output_format_block(system_text)
    if new_system != system_text:
        example[system_key] = new_system
        stats.system_changed_count += 1

    if not isinstance(messages, list):
        return example

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get(role_key)
        if role not in assistant_roles:
            continue

        content = message.get(content_key)
        if not isinstance(content, str):
            continue

        stats.assistant_count += 1
        converted, changed, final_only = _to_think_format(content)
        if changed:
            message[content_key] = converted
            stats.converted_count += 1
            if final_only:
                stats.final_only_count += 1
        else:
            stats.unchanged_count += 1

    return example


def _default_output_path(input_path: Path, output_suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{output_suffix}{input_path.suffix}")


def _convert_json_file(
    input_path: Path,
    output_path: Path,
    system_key: str,
    messages_key: str,
    role_key: str,
    content_key: str,
    assistant_roles: set[str],
) -> ConvertStats:
    stats = ConvertStats()
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for i, example in enumerate(data):
            if isinstance(example, dict):
                data[i] = _convert_example(
                    example=example,
                    system_key=system_key,
                    messages_key=messages_key,
                    role_key=role_key,
                    content_key=content_key,
                    assistant_roles=assistant_roles,
                    stats=stats,
                )
    elif isinstance(data, dict):
        data = _convert_example(
            example=data,
            system_key=system_key,
            messages_key=messages_key,
            role_key=role_key,
            content_key=content_key,
            assistant_roles=assistant_roles,
            stats=stats,
        )
    else:
        raise ValueError(f"Unsupported JSON root type in {input_path}: {type(data)}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return stats


def _convert_jsonl_file(
    input_path: Path,
    output_path: Path,
    system_key: str,
    messages_key: str,
    role_key: str,
    content_key: str,
    assistant_roles: set[str],
) -> ConvertStats:
    stats = ConvertStats()
    with input_path.open("r", encoding="utf-8") as reader, output_path.open("w", encoding="utf-8") as writer:
        for line in reader:
            line = line.strip()
            if not line:
                continue

            example = json.loads(line)
            if not isinstance(example, dict):
                raise ValueError(f"Each line in {input_path} must be a JSON object.")

            example = _convert_example(
                example=example,
                system_key=system_key,
                messages_key=messages_key,
                role_key=role_key,
                content_key=content_key,
                assistant_roles=assistant_roles,
                stats=stats,
            )
            writer.write(json.dumps(example, ensure_ascii=False) + "\n")

    return stats


def convert_file(
    input_path: Path,
    output_path: Path,
    system_key: str,
    messages_key: str,
    role_key: str,
    content_key: str,
    assistant_roles: set[str],
) -> ConvertStats:
    if input_path.suffix == ".json":
        return _convert_json_file(
            input_path=input_path,
            output_path=output_path,
            system_key=system_key,
            messages_key=messages_key,
            role_key=role_key,
            content_key=content_key,
            assistant_roles=assistant_roles,
        )

    if input_path.suffix == ".jsonl":
        return _convert_jsonl_file(
            input_path=input_path,
            output_path=output_path,
            system_key=system_key,
            messages_key=messages_key,
            role_key=role_key,
            content_key=content_key,
            assistant_roles=assistant_roles,
        )

    raise ValueError(f"Unsupported file type: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BEGIN_META/BEGIN_FINAL assistant output to <think> format.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input dataset files (.json/.jsonl).",
    )
    parser.add_argument(
        "--outputs",
        nargs="*",
        default=None,
        help="Optional output files. If omitted, use input stem + output_suffix.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_think",
        help="Suffix used when outputs are not explicitly provided.",
    )
    parser.add_argument("--system-key", default="system", help="Field name for system prompt text.")
    parser.add_argument("--messages-key", default="conversations", help="Field name for message list.")
    parser.add_argument("--role-key", default="from", help="Field name for role, e.g. from/role.")
    parser.add_argument("--content-key", default="value", help="Field name for text content, e.g. value/content.")
    parser.add_argument(
        "--assistant-roles",
        default="gpt,assistant",
        help="Comma-separated assistant role tags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]

    if args.outputs:
        if len(args.outputs) != len(input_paths):
            raise ValueError("--outputs must have the same number of paths as --inputs.")
        output_paths = [Path(path) for path in args.outputs]
    else:
        output_paths = [_default_output_path(path, args.output_suffix) for path in input_paths]

    assistant_roles = {tag.strip() for tag in args.assistant_roles.split(",") if tag.strip()}

    for input_path, output_path in zip(input_paths, output_paths):
        stats = convert_file(
            input_path=input_path,
            output_path=output_path,
            system_key=args.system_key,
            messages_key=args.messages_key,
            role_key=args.role_key,
            content_key=args.content_key,
            assistant_roles=assistant_roles,
        )
        print(
            "[DONE] "
            f"{input_path} -> {output_path} | "
            f"samples={stats.sample_count}, system_changed={stats.system_changed_count}, assistant_msgs={stats.assistant_count}, "
            f"converted={stats.converted_count}, final_only={stats.final_only_count}, unchanged={stats.unchanged_count}"
        )


if __name__ == "__main__":
    main()
