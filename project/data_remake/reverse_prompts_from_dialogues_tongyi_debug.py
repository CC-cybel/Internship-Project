#!/usr/bin/env python3
"""Run a Tongyi reverse-prompt batch for hard_inject_round.json.

This script reuses the v2 LOGIC_REVERSE_TEMPLATE from
reverse_prompts_from_dialogues_v2.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha1
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from reverse_prompts_from_dialogues_v2 import (
    LOGIC_REVERSE_TEMPLATE,
    format_dialogue,
    normalize_system_prompt,
    parse_json_strict,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "raw" / "hard_inject_round.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "runs" / "hard_reverse_tongyi_v2.json"
DEFAULT_RAW_LOG = SCRIPT_DIR / "logs" / "reverse_prompts_hard_v2.txt"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "cache" / "reverse_prompts_hard_v2"
PROCESSED_FLAG = "_reverse_done"

log_lock = threading.Lock()
cache_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSON file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON file for the debug slice.")
    parser.add_argument("--raw-log", default=str(DEFAULT_RAW_LOG), help="Raw model output backup log.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Cache directory for debug calls.")
    parser.add_argument("--offset", type=int, default=0, help="Start index in the input list.")
    parser.add_argument("--limit", type=int, default=20000, help="Number of items to process.")
    parser.add_argument("--workers", type=int, default=40, help="Concurrent API calls.")
    parser.add_argument("--save-every", type=int, default=1000, help="Write partial output every N completed items.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--retry-jitter-seconds", type=float, default=0.3)
    parser.add_argument("--timeout", type=float, default=180.0, help="Single API request timeout in seconds.")
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.getenv("TONGYI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--api-key", default="sk-f76c711b79a24e358d6fa4ca4c69d670")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--force", action="store_true", help="Ignore existing debug cache and call the API again.")
    return parser.parse_args()


def ensure_dirs(args: argparse.Namespace) -> None:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.raw_log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)


def item_key(item: dict[str, Any], original_index: int) -> str:
    for key in ("id", "item_id", "uid", "uuid"):
        if item.get(key) is not None:
            return f"{key}:{item[key]}"
    dialog_text = format_dialogue(item.get("conversations", []))
    digest = sha1(dialog_text.encode("utf-8")).hexdigest()[:12]
    return f"idx:{original_index}-sha1:{digest}"


def cache_path(cache_dir: str, key: str) -> Path:
    digest = sha1(key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def load_cache(cache_dir: str, key: str) -> dict[str, Any] | None:
    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("key") != key:
        return None
    return data


def save_cache(cache_dir: str, key: str, payload: dict[str, Any]) -> None:
    path = cache_path(cache_dir, key)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with cache_lock:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)


def save_raw_log(raw_log: str, original_index: int, content: str) -> None:
    with log_lock:
        with open(raw_log, "a", encoding="utf-8") as f:
            f.write(f"--- INDEX: {original_index} ---\n")
            f.write(content + "\n")
            f.write("-" * 30 + "\n\n")


def call_llm(client: OpenAI, args: argparse.Namespace, final_prompt: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "你是一个数据分析师，直接输出JSON对象。"},
                    {"role": "user", "content": final_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=args.temperature,
                timeout=args.timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt >= args.max_retries:
                break
            sleep_seconds = args.retry_base_seconds * (2 ** (attempt - 1))
            sleep_seconds += random.uniform(0, args.retry_jitter_seconds)
            time.sleep(sleep_seconds)
    assert last_exc is not None
    raise last_exc


def process_one(client: OpenAI, args: argparse.Namespace, item: dict[str, Any], original_index: int) -> tuple[int, dict[str, Any], str | None]:
    key = item_key(item, original_index)
    if not args.force:
        cached = load_cache(args.cache_dir, key)
        if cached and cached.get("status") == "ok" and cached.get("item"):
            return original_index, cached["item"], None

    raw_content = ""
    processed = deepcopy(item)
    try:
        dialog_text = format_dialogue(processed.get("conversations", []))
        final_prompt = LOGIC_REVERSE_TEMPLATE.format(dialog_text=dialog_text)
        raw_content = call_llm(client, args, final_prompt)
        save_raw_log(args.raw_log, original_index, raw_content)

        result = parse_json_strict(raw_content)
        if not result or not result.get("system_prompt"):
            raise ValueError("Invalid JSON response or missing system_prompt")

        processed["system"] = normalize_system_prompt(result["system_prompt"])
        processed[PROCESSED_FLAG] = True
        processed["_debug_original_index"] = original_index
        processed["_debug_model"] = args.model

        save_cache(
            args.cache_dir,
            key,
            {
                "key": key,
                "index": original_index,
                "status": "ok",
                "timestamp": int(time.time()),
                "item": processed,
                "raw_content": raw_content,
            },
        )
        return original_index, processed, None
    except Exception as exc:
        error = f"Error: {exc}"
        if raw_content:
            error += " (Raw content saved to txt)"
        save_cache(
            args.cache_dir,
            key,
            {
                "key": key,
                "index": original_index,
                "status": "error",
                "timestamp": int(time.time()),
                "error": error,
                "raw_content": raw_content,
            },
        )
        processed["_debug_original_index"] = original_index
        processed["_debug_error"] = error
        return original_index, processed, error


def write_output(args: argparse.Namespace, results: list[dict[str, Any] | None], total_input: int) -> None:
    payload = {
        "source_input": str(Path(args.input).resolve()),
        "offset": args.offset,
        "limit": args.limit,
        "model": args.model,
        "base_url": args.base_url,
        "total_input": total_input,
        "items": results,
    }
    output_path = Path(args.output)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, output_path)


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing TONGYI_API_KEY. Please export it or pass --api-key.")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")

    ensure_dirs(args)
    raw_log = Path(args.raw_log)
    if args.force or not raw_log.exists():
        raw_log.write_text("=== TONGYI DEBUG RAW OUTPUT BACKUP ===\n\n", encoding="utf-8")

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data_list = data["items"] if isinstance(data, dict) and "items" in data else data
    if not isinstance(data_list, list):
        raise ValueError("Input must be a list or a dict with an items list.")

    start = max(0, args.offset)
    end = min(len(data_list), start + args.limit)
    selected = data_list[start:end]
    results: list[dict[str, Any] | None] = [None] * len(selected)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    print(f"[INFO] input={input_path} total={len(data_list)} slice=[{start}, {end})")
    print(f"[INFO] model={args.model} base_url={args.base_url}")
    print(f"[INFO] output={args.output}")
    print(f"[INFO] raw_log={args.raw_log}")

    completed_since_save = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pos = {
            executor.submit(process_one, client, args, item, start + pos): pos
            for pos, item in enumerate(selected)
        }
        with tqdm(total=len(selected)) as pbar:
            for future in as_completed(future_to_pos):
                pos = future_to_pos[future]
                original_index, processed, error = future.result()
                results[pos] = processed
                completed_since_save += 1
                if error:
                    print(f"\n[WARN] index={original_index} {error}")
                if completed_since_save >= args.save_every:
                    write_output(args, results, len(data_list))
                    completed_since_save = 0
                pbar.update(1)

    write_output(args, results, len(data_list))
    ok_count = sum(1 for item in results if item and item.get(PROCESSED_FLAG) is True)
    print(f"[OK] done ok={ok_count}/{len(selected)} output={args.output}")


if __name__ == "__main__":
    main()
