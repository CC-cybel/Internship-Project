#!/usr/bin/env python3
"""Build sampled atomic-instruction groups and let an LLM select compatible rules.

Input:
  atomic_instruction_metadata_v1.json

Process:
  1. Randomly sample one instruction from each of the 12 categories.
  2. Send the 12-instruction candidate group to a judge model.
  3. Randomly sample a target selected count from 1-5.
  4. Balanced-sample exactly that many target categories from the 12 categories.
  5. Ask the model to select exactly those target-category instructions.
  6. Write one JSON object per group to JSONL.

The output JSONL is append-only and resume-friendly. Existing ok records are
skipped when rerunning with the same --count/--seed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is convenient but optional.
    tqdm = None


DEFAULT_INPUT = "/data/chengch/project/data_remake/atomic_instruction_metadata_v1.json"
DEFAULT_OUTPUT = (
    "/data/chengch/project/data_remake/outputs/atomic_instruction_groups/"
    "selected_instruction_groups_10k.deepseek_v4_flash.jsonl"
)
DEFAULT_ENV_FILE = "/data/chengch/leadbench-excellent-master/.env"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_COUNT = 10_000
DEFAULT_MAX_WORKERS = 20
DEFAULT_SEED = 20260602
DEFAULT_MAX_RETRIES = 4
DEFAULT_TIMEOUT = 120.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 512
DEFAULT_MIN_SELECTED = 1
DEFAULT_MAX_SELECTED = 5
DEFAULT_BALANCE_CATEGORIES = True

CATEGORY_ORDER = [
    "身份锚定",
    "信息调查",
    "留联触发",
    "分龄策略",
    "强制留联策略",
    "转化借口",
    "时间策略",
    "降级策略",
    "语言风格",
    "格式要求",
    "语言规范要求",
    "特殊要求",
]

SYSTEM_PROMPT = """你是一名严格的指令组筛选器。
你的任务是从候选的 12 条原子指令中，按照用户给出的目标数量，选出恰好 N 条可以同时执行、语义互不冲突、适合组成最终训练 prompt 的指令要求。

筛选原则：
1. 只能选择候选列表中已经给出的 id。
2. selected_ids 数量必须恰好等于用户给出的 target_selected_count。
3. target_selected_categories 是类别覆盖偏好；在不造成冲突的前提下，优先选择这些类别。
4. 互不冲突比命中目标类别更重要。若目标类别中的指令与其他要求冲突，可以换成候选列表中的其他类别。
5. 遇到语言、格式、渠道、语气、阶段或输出结构冲突时，保留更清晰、更通用、更容易同时执行的组合。

只输出 JSON 对象，不输出 markdown，不输出解释文本。"""

USER_PROMPT_TEMPLATE = """请从下面 12 条候选原子指令中，选出互不冲突、适合放在同一个最终指令组里的要求。

本组目标数量 target_selected_count = {target_selected_count}
本组优先目标类别 target_selected_categories = {target_selected_categories_json}
你必须恰好选择 {target_selected_count} 条。
请优先覆盖 target_selected_categories；如果某个目标类别会导致冲突，可以换成其他候选类别，并在 reason 中简要说明。

候选指令：
{candidate_json}

请严格输出如下 JSON：
{{
  "target_selected_count": {target_selected_count},
  "target_selected_categories": {target_selected_categories_json},
  "selected_ids": ["instr_atom_xxxx"],
  "reason": "一句话说明为什么这些指令可以组合；如替换目标类别，说明原因"
}}"""

_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample 12-category atomic instruction groups and select compatible subsets with an LLM."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Atomic instruction metadata JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="Optional .env file containing JUDGE_API_KEY/API_BASE.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of candidate groups to build.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic candidate groups.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="API concurrency.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="API retry count per group.")
    parser.add_argument("--model", default=None, help=f"Judge model. Default: env JUDGE_MODEL_NAME or {DEFAULT_MODEL}.")
    parser.add_argument("--api-key", default=None, help="Judge API key. Default: env/file JUDGE_API_KEY.")
    parser.add_argument("--api-base", default=None, help=f"Judge API base. Default: env/file JUDGE_API_BASE or {DEFAULT_API_BASE}.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="OpenAI client timeout seconds.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Judge sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Judge max output tokens.")
    parser.add_argument("--min-selected", type=int, default=DEFAULT_MIN_SELECTED, help="Minimum sampled selected instruction count.")
    parser.add_argument("--max-selected", type=int, default=DEFAULT_MAX_SELECTED, help="Maximum sampled selected instruction count.")
    parser.add_argument("--balance-categories", dest="balance_categories", action="store_true", default=DEFAULT_BALANCE_CATEGORIES, help="Balanced-sample preferred target categories; judge may replace conflicting categories.")
    parser.add_argument("--no-balance-categories", dest="balance_categories", action="store_false", help="Let the judge freely choose categories after sampling only target count.")
    parser.add_argument("--dry-run", action="store_true", help="Only build and print one candidate group; do not call API.")
    parser.add_argument("--response-format", action="store_true", help="Send response_format={type: json_object} to the API.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output before running.")
    return parser.parse_args()


def load_env_file(path: str) -> Dict[str, str]:
    env_path = Path(path)
    if not path or not env_path.exists():
        return {}

    parsed: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        parsed[key] = value

    # Load only connection settings from the env file by default. The model
    # default stays deepseek-v4-flash unless the shell or --model overrides it.
    for key in ("JUDGE_API_KEY", "JUDGE_API_BASE"):
        if key in parsed and key not in os.environ:
            os.environ[key] = parsed[key]
    return parsed


def resolve_runtime_config(args: argparse.Namespace) -> argparse.Namespace:
    load_env_file(args.env_file)
    args.model = args.model or os.environ.get("JUDGE_MODEL_NAME") or DEFAULT_MODEL
    args.api_key = args.api_key or os.environ.get("JUDGE_API_KEY")
    args.api_base = args.api_base or os.environ.get("JUDGE_API_BASE") or DEFAULT_API_BASE
    if args.min_selected < 1 or args.max_selected > len(CATEGORY_ORDER) or args.min_selected > args.max_selected:
        raise ValueError(
            f"Invalid selected count range: {args.min_selected}-{args.max_selected}. "
            f"Expected 1 <= min <= max <= {len(CATEGORY_ORDER)}."
        )
    if not args.api_key and not args.dry_run:
        raise ValueError(
            "Missing JUDGE_API_KEY. Set it in the environment, pass --api-key, "
            f"or put it in {args.env_file}."
        )
    return args


def load_atomic_instructions(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = data.get("instructions", [])
    else:
        items = data
    if not isinstance(items, list) or not items:
        raise ValueError(f"No instructions found in {path}")
    return items


def group_by_category(items: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        category = item.get("category")
        if category in CATEGORY_ORDER:
            grouped[category].append(item)

    missing = [category for category in CATEGORY_ORDER if not grouped.get(category)]
    if missing:
        raise ValueError(f"Missing categories in atomic metadata: {missing}")
    return grouped


def compact_instruction(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "category": item["category"],
        "subtype": item.get("subtype", ""),
        "instruction": item["instruction"],
    }


def sample_balanced_categories(
    target_count: int,
    usage: Counter,
    rng: random.Random,
) -> List[str]:
    selected: List[str] = []
    for _ in range(target_count):
        available = [category for category in CATEGORY_ORDER if category not in selected]
        min_usage = min(usage[category] for category in available)
        least_used = [category for category in available if usage[category] == min_usage]
        category = rng.choice(least_used)
        selected.append(category)
        usage[category] += 1
    return selected


def build_candidate_groups(
    grouped: Dict[str, List[Dict[str, Any]]],
    count: int,
    seed: int,
    min_selected: int,
    max_selected: int,
    balance_categories: bool,
) -> List[Dict[str, Any]]:
    candidate_rng = random.Random(seed)
    target_count_rng = random.Random(seed + 104_729)
    target_category_rng = random.Random(seed + 209_759)
    target_category_usage: Counter = Counter()
    groups: List[Dict[str, Any]] = []
    for idx in range(count):
        candidates = [
            compact_instruction(candidate_rng.choice(grouped[category]))
            for category in CATEGORY_ORDER
        ]
        target_selected_count = target_count_rng.randint(min_selected, max_selected)
        if balance_categories:
            target_selected_categories = sample_balanced_categories(
                target_selected_count,
                target_category_usage,
                target_category_rng,
            )
        else:
            target_selected_categories = []
        groups.append(
            {
                "group_index": idx,
                "group_id": f"instr_req_group_{idx + 1:05d}",
                "seed": seed,
                "target_selected_count": target_selected_count,
                "target_selected_categories": target_selected_categories,
                "candidate_requirements": candidates,
            }
        )
    return groups


def load_done_records(output_path: str) -> Tuple[Dict[int, Dict[str, Any]], int]:
    path = Path(output_path)
    if not path.exists():
        return {}, 0

    done: Dict[int, Dict[str, Any]] = {}
    ignored_ok_records = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skip invalid JSONL line {line_no}: {path}")
                continue
            if record.get("status") != "ok" or not isinstance(record.get("group_index"), int):
                continue
            target_count = record.get("target_selected_count")
            selected_ids = record.get("selected_ids")
            target_categories = record.get("target_selected_categories")
            if (
                isinstance(target_count, int)
                and isinstance(selected_ids, list)
                and len(selected_ids) == target_count
                and isinstance(target_categories, list)
            ):
                done[record["group_index"]] = record
            else:
                ignored_ok_records += 1
    return done, ignored_ok_records


def get_client(api_key: str, api_base: str, timeout: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for API calls. Install it in the "
            "runtime environment or run this script inside an existing project "
            "environment that already has openai installed."
        ) from exc

    client = getattr(_thread_local, "client", None)
    client_key = getattr(_thread_local, "client_key", None)
    key = (api_key, api_base, timeout)
    if client is None or client_key != key:
        client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        _thread_local.client = client
        _thread_local.client_key = key
    return client


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return stripped


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(cleaned[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Judge output is not a JSON object")
    return obj


def validate_selection(obj: Dict[str, Any], candidate_ids: List[str], target_count: int) -> List[str]:
    raw_ids = obj.get("selected_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("selected_ids is not a list")

    allowed = set(candidate_ids)
    selected: List[str] = []
    for raw_id in raw_ids:
        if not isinstance(raw_id, str):
            continue
        if raw_id in allowed and raw_id not in selected:
            selected.append(raw_id)

    if len(selected) != target_count:
        raise ValueError(
            f"selected_ids length must equal target_selected_count={target_count} "
            f"after validation, got {len(selected)}"
        )
    return selected


def call_judge(group: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    candidate_json = json.dumps(group["candidate_requirements"], ensure_ascii=False, indent=2)
    target_count = group["target_selected_count"]
    target_categories = group.get("target_selected_categories", [])
    target_categories_json = json.dumps(target_categories, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                candidate_json=candidate_json,
                target_selected_count=target_count,
                target_selected_categories_json=target_categories_json,
            ),
        },
    ]
    candidate_ids = [item["id"] for item in group["candidate_requirements"]]
    by_id = {item["id"]: item for item in group["candidate_requirements"]}

    last_error: Optional[str] = None
    raw_content = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            client = get_client(args.api_key, args.api_base, args.timeout)
            request_kwargs = {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            if args.response_format:
                request_kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request_kwargs)
            raw_content = (response.choices[0].message.content or "").strip()
            judge_obj = extract_json_object(raw_content)
            selected_ids = validate_selection(judge_obj, candidate_ids, target_count)
            selected_requirements = [by_id[item_id] for item_id in selected_ids]
            selected_categories = [item["category"] for item in selected_requirements]
            target_category_set = set(target_categories)
            selected_category_set = set(selected_categories)

            return {
                **group,
                "status": "ok",
                "model": args.model,
                "selected_ids": selected_ids,
                "selected_categories": selected_categories,
                "selected_requirements": selected_requirements,
                "target_category_hit_count": len(target_category_set & selected_category_set),
                "target_category_missed": [
                    category for category in target_categories
                    if category not in selected_category_set
                ],
                "non_target_selected_categories": [
                    category for category in selected_categories
                    if category not in target_category_set
                ],
                "judge_result": judge_obj,
                "raw_content": raw_content,
                "processed_at": datetime.now().isoformat(timespec="seconds"),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                sleep_seconds = min(30.0, 1.5 * (2 ** (attempt - 1))) + random.random()
                time.sleep(sleep_seconds)

    return {
        **group,
        "status": "failed",
        "model": args.model,
        "error": last_error,
        "raw_content": raw_content,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_run_meta(
    path: str,
    args: argparse.Namespace,
    category_counts: Counter,
    target_count_distribution: Counter,
    target_category_distribution: Counter,
) -> None:
    meta_path = Path(path).with_suffix(Path(path).suffix + ".meta.json")
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": args.input,
        "output": args.output,
        "count": args.count,
        "seed": args.seed,
        "model": args.model,
        "api_base": args.api_base,
        "max_workers": args.max_workers,
        "max_retries": args.max_retries,
        "min_selected": args.min_selected,
        "max_selected": args.max_selected,
        "selection_count_policy": "uniform_random_exact",
        "category_balance_policy": "soft_target_categories" if args.balance_categories else "none",
        "target_selected_count_distribution": dict(sorted(target_count_distribution.items())),
        "target_selected_category_distribution": dict(target_category_distribution),
        "category_order": CATEGORY_ORDER,
        "category_counts": dict(category_counts),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def progress_iter(futures: Dict[Any, Dict[str, Any]], initial: int, total: int):
    iterator = as_completed(futures)
    if tqdm is None:
        return iterator
    return tqdm(iterator, total=total, initial=initial, dynamic_ncols=True)


def main() -> None:
    args = resolve_runtime_config(parse_args())

    if args.overwrite and Path(args.output).exists():
        Path(args.output).unlink()

    items = load_atomic_instructions(args.input)
    category_counts = Counter(item["category"] for item in items)
    grouped = group_by_category(items)
    groups = build_candidate_groups(
        grouped,
        args.count,
        args.seed,
        args.min_selected,
        args.max_selected,
        args.balance_categories,
    )
    target_count_distribution = Counter(group["target_selected_count"] for group in groups)
    target_category_distribution = Counter(
        category
        for group in groups
        for category in group.get("target_selected_categories", [])
    )

    print(f"Loaded {len(items)} atomic instructions from {args.input}")
    print(f"Built {len(groups)} candidate groups with seed={args.seed}")
    print(f"Target selected count range: {args.min_selected}-{args.max_selected}")
    print(f"Target selected count distribution: {dict(sorted(target_count_distribution.items()))}")
    if args.balance_categories:
        print(f"Preferred target category distribution: {dict(target_category_distribution)}")
    print(f"Model={args.model}, api_base={args.api_base}, max_workers={args.max_workers}")

    if args.dry_run:
        print(json.dumps(groups[0], ensure_ascii=False, indent=2))
        return

    write_run_meta(
        args.output,
        args,
        category_counts,
        target_count_distribution,
        target_category_distribution,
    )
    done, ignored_existing_ok = load_done_records(args.output)
    pending = [group for group in groups if group["group_index"] not in done]

    if ignored_existing_ok:
        print(
            f"Ignored {ignored_existing_ok} existing ok records because they do not have "
            "matching target_selected_count/target_selected_categories. Use --overwrite or a new --output path for a clean file."
        )
    print(f"Existing resumable ok records: {len(done)}")
    print(f"Pending API calls: {len(pending)}")
    if not pending:
        print("All groups are already processed.")
        return

    status_counts = Counter()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_group = {
            executor.submit(call_judge, group, args): group
            for group in pending
        }
        for future in progress_iter(future_to_group, initial=len(done), total=args.count):
            group = future_to_group[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    **group,
                    "status": "failed",
                    "model": args.model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "processed_at": datetime.now().isoformat(timespec="seconds"),
                }
            append_jsonl(args.output, record)
            status_counts[record.get("status", "unknown")] += 1

    print(f"Done. New records: {sum(status_counts.values())}, status={dict(status_counts)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
