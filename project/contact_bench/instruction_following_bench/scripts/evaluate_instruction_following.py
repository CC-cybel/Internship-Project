#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent
_BENCH_ROOT = _HERE.parent
DEFAULT_INPUT = _BENCH_ROOT / "data" / "dataset" / "instruction_following_bench_final300.jsonl"
DEFAULT_OUTPUT_ROOT = _BENCH_ROOT / "output"
DEFAULT_ENV_FILE = Path(
    "/data/chengch/project/verl/recipe/single_turn_reward/v3/"
    "run_grpo_single_turn_4gpu_qwen3_8b_stage5_mid_reward.sh"
)
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
FINAL_RE = re.compile(r"BEGIN_FINAL\s*\n([\s\S]*?)\nEND_FINAL")
META_RE = re.compile(r"BEGIN_META\s*\n([\s\S]*?)\nEND_META")


JUDGE_SYSTEM_PROMPT = """你是一个高精度的“指令遵循评估器”，任务是判断模型回复是否遵守 system 中额外注入的指令。

你只评估 selected_additional_instructions 中的指令遵循程度。不要评价回复是否好听，不要额外奖励医学质量，除非医学或业务安全问题直接影响指令执行。
你必须逐条评估 selected_additional_instructions，不允许漏评，不允许新增指令。

评分标准：
- 2 分：模型明确、完整、自然地遵守该指令。若该指令是禁止类、词汇替换类或格式限制类，当前回复没有违反，也可给 2 分。
- 1 分：模型基本遵守，但存在轻微不完整、表达不稳定、触发条件处理不够精准、或只有部分满足。
- 0 分：模型没有遵守、明显违反、执行了相反行为、遗漏必须动作，或因格式/内容缺失导致无法确认遵守。

触发条件判断：
- 如果指令有明确触发条件，先判断当前回复是否出现触发机会。
- 对“禁止使用 X”“不要出现 X”“不要用某种格式”这类禁止约束，只要回复没有违反，即视为遵守。
- 对“必须用 A 代替 B”这类替换约束，如果回复出现 B 而没有替换为 A，给 0 分；如果当前回复没有提及该概念，给 2 分，但 opportunity 写 not_observed。
- 对“必须询问/必须说明/必须开头/必须结尾/必须包含”这类动作约束，如果当前回复应该执行而没有执行，给 0 分。
- 不要因为模型没有触发某个可选场景就惩罚它，除非该指令本身要求本轮必须执行。

评估范围：
- 优先评估用户可见内容，也就是 BEGIN_FINAL 内的文本。
- 如果附加指令明确要求 thought、BEGIN_META、action 或槽位中体现，则同时检查 BEGIN_META。
- 如果 Candidate Response 没有合法的 BEGIN_FINAL，format_ok=false，并对依赖用户可见回复的指令谨慎降分。
- 不要被模型在 thought 里声称“我遵守了指令”欺骗，必须看实际输出。

输出必须是严格 JSON，不要输出 Markdown，不要解释 JSON 之外的内容。
JSON 字符串内部不要使用英文双引号；如需引用原文，请使用中文引号或直接概括，避免产生非法 JSON。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate instruction-following bench outputs.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--candidate-model", default=None)
    parser.add_argument("--candidate-api-base", default=None)
    parser.add_argument("--candidate-api-key", default=None)
    parser.add_argument("--candidate-max-tokens", type=int, default=768)
    parser.add_argument("--candidate-temperature", type=float, default=0.6)
    parser.add_argument("--candidate-top-p", type=float, default=0.95)
    parser.add_argument("--candidate-timeout", type=float, default=180.0)
    parser.add_argument("--response-field", default=None)
    parser.add_argument("--use-reference", action="store_true")
    parser.add_argument("--history-max-chars", type=int, default=7000)
    parser.add_argument("--system-max-chars", type=int, default=3500)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite-output-dir", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def load_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    parsed: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r':\s*"\$\{([A-Za-z_][A-Za-z0-9_]*):=([^}]*)\}"', line)
        if m:
            parsed[m.group(1)] = m.group(2)
            continue
        if "=" in line and not line.startswith("export "):
            key, value = line.split("=", 1)
            key = key.strip().lstrip(":").strip()
            value = value.strip().strip('"').strip("'")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                parsed[key] = value
    return parsed


def resolve_runtime(args: argparse.Namespace) -> argparse.Namespace:
    env = load_env_file(args.env_file)
    args.api_base = args.api_base or os.environ.get("TONGYI_API_BASE") or env.get("TONGYI_API_BASE") or DEFAULT_API_BASE
    args.api_key = args.api_key or os.environ.get("TONGYI_API_KEY") or env.get("TONGYI_API_KEY")
    args.judge_model = args.judge_model or os.environ.get("JUDGE_MODEL_NAME") or os.environ.get("JUDGE_MODEL") or env.get("JUDGE_MODEL") or DEFAULT_MODEL
    args.candidate_model = args.candidate_model or os.environ.get("CANDIDATE_MODEL_NAME") or env.get("CANDIDATE_MODEL_NAME")
    args.candidate_api_base = args.candidate_api_base or os.environ.get("CANDIDATE_API_BASE") or env.get("CANDIDATE_API_BASE")
    args.candidate_api_key = args.candidate_api_key or os.environ.get("CANDIDATE_API_KEY") or env.get("CANDIDATE_API_KEY")
    if not args.api_key:
        raise ValueError("Missing API key. Set TONGYI_API_KEY, pass --api-key, or provide --env-file.")
    return args


def read_jsonl(path: Path, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset or not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at line {idx + 1}")
            rows.append(obj)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_json_dict(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    left = text.find("{")
    right = text.rfind("}")
    if left < 0 or right <= left:
        return None
    try:
        obj = json.loads(text[left : right + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_final_text(response: str) -> str:
    m = FINAL_RE.search(response or "")
    return m.group(1).strip() if m else ""


def extract_meta_text(response: str) -> str:
    m = META_RE.search(response or "")
    return m.group(1).strip() if m else ""


def get_candidate_id(row: dict[str, Any], idx: int) -> str:
    for key in ("candidate_id", "id", "sample_id", "pair_id"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return f"row_{idx}"



def role_to_openai(role: str) -> str:
    role = (role or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    return ""


def turn_content(turn: dict[str, Any]) -> str:
    value = turn.get("value")
    if value is None:
        value = turn.get("content")
    return "" if value is None else str(value)


def build_candidate_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = str(row.get("system") or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    for turn in row.get("conversations") or []:
        if not isinstance(turn, dict):
            continue
        role = role_to_openai(str(turn.get("from") or turn.get("role") or ""))
        content = turn_content(turn).strip()
        if not role or not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def openai_chat(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    top_p: float | None = None,
    response_format: dict[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    url = api_base.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if response_format is not None:
        payload["response_format"] = response_format
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj["choices"][0]["message"]["content"], obj.get("usage") or {}


def generate_candidate(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if not args.candidate_model or not args.candidate_api_base or not args.candidate_api_key:
        raise ValueError("Missing candidate API config. Set CANDIDATE_MODEL_NAME, CANDIDATE_API_BASE, and CANDIDATE_API_KEY, or provide --response-field/--use-reference.")
    messages = build_candidate_messages(row)
    if not messages or not any(m["role"] == "user" for m in messages):
        raise ValueError("Cannot build candidate messages from sample conversations.")

    token_budgets = []
    for value in [args.candidate_max_tokens, 768, 512, 256]:
        value = int(value)
        if value > 0 and value not in token_budgets:
            token_budgets.append(value)
    last_context_error = ""
    for max_tokens in token_budgets:
        try:
            content, usage = openai_chat(
                api_base=args.candidate_api_base,
                api_key=args.candidate_api_key,
                model=args.candidate_model,
                messages=messages,
                temperature=args.candidate_temperature,
                max_tokens=max_tokens,
                timeout=args.candidate_timeout,
                top_p=args.candidate_top_p,
            )
            usage = dict(usage or {})
            usage["requested_max_tokens"] = max_tokens
            return content, usage
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")[:2000]
            except Exception:
                body = ""
            if exc.code == 400 and ("maximum input length" in body or "context length" in body or "input tokens" in body):
                last_context_error = body or repr(exc)
                continue
            raise RuntimeError(f"candidate_http_error={repr(exc)} body={body}") from exc
    raise RuntimeError(f"candidate_context_length_error_after_retries={last_context_error}")


def get_response(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    if args.use_reference:
        target = row.get("target")
        if isinstance(target, dict) and isinstance(target.get("original_value"), str):
            return target["original_value"].strip(), "target.original_value"
    fields = [args.response_field] if args.response_field else []
    fields.extend(["response", "candidate_response", "model_output", "output", "solution_str"])
    for field in fields:
        if not field:
            continue
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return "", ""


def compact_history(row: dict[str, Any], max_chars: int) -> str:
    lines: list[str] = []
    for i, turn in enumerate(row.get("conversations") or []):
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "")
        value = str(turn.get("value") if turn.get("value") is not None else turn.get("content") or "")
        if role == "gpt" and not value:
            value = "<待评估回复位置>"
        label = {"human": "用户", "user": "用户", "gpt": "客服", "assistant": "客服"}.get(role, role)
        lines.append(f"[{i}] {label}: {value}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return "...以下为截断后的最近对话...\n" + text[-max_chars:]


def compact_system(text: str, max_chars: int) -> str:
    text = text or ""
    append_marker = "【附加指令】"
    append_idx = text.find(append_marker)
    append = text[append_idx:] if append_idx >= 0 else ""
    head = text[:max_chars]
    if append and append not in head:
        head = head[: max(0, max_chars - len(append) - 20)] + "\n...\n" + append
    return head if len(head) <= max_chars else head[:max_chars] + "\n..."


def weight_for_instruction(inst: dict[str, Any]) -> float:
    axis = str(inst.get("axis") or "")
    pool = str(inst.get("pool") or "")
    required = str(inst.get("required_behavior") or "")
    if any(key in required for key in ("危机", "自杀", "监护人", "联系方式", "留联", "安全")):
        return 1.2
    if "format" in axis or "punctuation" in axis:
        return 1.0
    if pool in {"business", "contact", "safety"}:
        return 1.2
    return 1.0


def build_judge_prompt(row: dict[str, Any], response: str, args: argparse.Namespace) -> str:
    instructions = row.get("selected_additional_instructions") or []
    compact_insts = []
    for inst in instructions:
        if not isinstance(inst, dict):
            continue
        compact_insts.append(
            {
                "atom_id": inst.get("atom_id"),
                "trigger_condition": inst.get("trigger_condition"),
                "required_behavior": inst.get("required_behavior"),
                "source": inst.get("source"),
                "pool": inst.get("pool"),
                "axis": inst.get("axis"),
                "sub_axis": inst.get("sub_axis"),
                "check": inst.get("check"),
                "weight": weight_for_instruction(inst),
            }
        )
    payload = {
        "candidate_id": row.get("candidate_id"),
        "system_prompt_excerpt": compact_system(str(row.get("system") or ""), args.system_max_chars),
        "system_append_block": row.get("system_append_block"),
        "conversation_history": compact_history(row, args.history_max_chars),
        "candidate_response": response,
        "candidate_meta": extract_meta_text(response) or "<无META>",
        "candidate_final": extract_final_text(response) or "<无合法BEGIN_FINAL>",
        "selected_additional_instructions": compact_insts,
    }
    schema = {
        "format_ok": True,
        "final_text": "...",
        "instruction_results": [
            {
                "atom_id": "...",
                "score": 0,
                "weight": 1.0,
                "opportunity": "observed|required|not_observed|unclear",
                "evidence": "...",
                "problem": "...",
                "failure_type": "none|missing_required_action|lexical_violation|format_violation|style_violation|content_conflict|unclear",
                "confidence": 0.0,
            }
        ],
        "overall_reason": "...",
    }
    return (
        "请评估下面样本的指令遵循程度。\n\n"
        "重要要求：instruction_results 的数量必须等于 selected_additional_instructions 的数量；"
        "atom_id 必须逐一对应；score 只能是 0、1、2。\n\n"
        "输出 JSON schema 示例：\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "待评估样本：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def chat_completion(args: argparse.Namespace, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    return openai_chat(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.judge_model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        response_format={"type": "json_object"},
    )


def normalize_judge(judge_obj: dict[str, Any] | None, row: dict[str, Any], response: str) -> dict[str, Any]:
    instructions = [x for x in (row.get("selected_additional_instructions") or []) if isinstance(x, dict)]
    expected_ids = [str(x.get("atom_id")) for x in instructions]
    if not isinstance(judge_obj, dict):
        judge_obj = {
            "format_ok": bool(extract_final_text(response)),
            "final_text": extract_final_text(response),
            "instruction_results": [],
            "overall_reason": "judge_parse_failed",
        }
    results_by_id: dict[str, dict[str, Any]] = {}
    for item in judge_obj.get("instruction_results") or []:
        if isinstance(item, dict) and item.get("atom_id") is not None:
            results_by_id[str(item["atom_id"])] = item
    normalized_results: list[dict[str, Any]] = []
    for inst in instructions:
        atom_id = str(inst.get("atom_id"))
        item = dict(results_by_id.get(atom_id) or {})
        try:
            score = int(item.get("score"))
        except Exception:
            score = 0
        score = max(0, min(2, score))
        try:
            confidence = float(item.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        normalized_results.append(
            {
                "atom_id": atom_id,
                "score": score,
                "weight": float(item.get("weight") or weight_for_instruction(inst)),
                "opportunity": str(item.get("opportunity") or "unclear"),
                "evidence": str(item.get("evidence") or "")[:300],
                "problem": str(item.get("problem") or "")[:300],
                "failure_type": ("unclear" if score < 2 and str(item.get("failure_type") or "").strip() in {"", "none"} else str(item.get("failure_type") or ("none" if score == 2 else "unclear"))),
                "confidence": max(0.0, min(1.0, confidence)),
                "instruction": {
                    "trigger_condition": inst.get("trigger_condition"),
                    "required_behavior": inst.get("required_behavior"),
                    "source": inst.get("source"),
                    "pool": inst.get("pool"),
                    "axis": inst.get("axis"),
                    "sub_axis": inst.get("sub_axis"),
                },
            }
        )
    score_sum = sum((x["score"] / 2.0) * x["weight"] for x in normalized_results)
    weight_sum = sum(x["weight"] for x in normalized_results) or 1.0
    unweighted = statistics.mean([x["score"] / 2.0 for x in normalized_results]) if normalized_results else 0.0
    format_ok = bool(judge_obj.get("format_ok")) and bool(extract_final_text(response))
    sample_score = score_sum / weight_sum
    if not format_ok:
        sample_score = min(sample_score, 0.5)
    return {
        "format_ok": format_ok,
        "final_text": str(judge_obj.get("final_text") or extract_final_text(response) or "")[:2000],
        "instruction_results": normalized_results,
        "sample_score": sample_score,
        "sample_score_unweighted": unweighted,
        "all_2_pass": bool(normalized_results) and all(x["score"] == 2 for x in normalized_results),
        "all_ge1_pass": bool(normalized_results) and all(x["score"] >= 1 for x in normalized_results),
        "overall_reason": str(judge_obj.get("overall_reason") or "")[:500],
        "expected_instruction_ids": expected_ids,
    }


def evaluate_one(task: tuple[int, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    idx, row = task
    candidate_id = get_candidate_id(row, idx)
    response, response_source = get_response(row, args)
    base = {
        "row_index": idx,
        "candidate_id": candidate_id,
        "response_source": response_source,
        "instruction_count": len(row.get("selected_additional_instructions") or []),
        "selected_instruction_ids": row.get("selected_instruction_ids") or [],
    }
    candidate_usage: dict[str, Any] = {}
    if not response:
        try:
            response, candidate_usage = generate_candidate(row, args)
            response_source = "generated"
            base["response_source"] = response_source
            base["candidate_model"] = args.candidate_model
            base["candidate_response"] = response
            base["candidate_usage"] = candidate_usage
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")[:2000]
            except Exception:
                body = ""
            return {**base, "status": "generation_error", "error": repr(exc), "error_body": body}
        except Exception as exc:
            return {**base, "status": "generation_error", "error": repr(exc)}
    prompt = build_judge_prompt(row, response, args)
    last_error = ""
    for attempt in range(args.max_retries):
        try:
            raw, usage = chat_completion(
                args,
                [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            )
            judge_obj = extract_json_dict(raw)
            judge = normalize_judge(judge_obj, row, response)
            return {
                **base,
                "status": "ok",
                "judge_model": args.judge_model,
                "judge": judge,
                "usage": usage,
                "raw_judge_content": raw,
            }
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
            time.sleep(min(2**attempt, 20))
    return {**base, "status": "judge_error", "error": last_error}


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.overwrite_output_dir:
        out = Path(args.overwrite_output_dir)
    else:
        stem = Path(args.input_file).stem
        mode = "reference" if args.use_reference else (args.response_field or "response")
        run_name = args.run_name or f"{mode}_{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out = Path(args.output_root) / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def aggregate_report(results: list[dict[str, Any]], args: argparse.Namespace) -> str:
    ok = [r for r in results if r.get("status") == "ok"]
    scores = [r["judge"]["sample_score"] for r in ok]
    unweighted = [r["judge"]["sample_score_unweighted"] for r in ok]
    lines = [
        "# Instruction Following Bench Report",
        "",
        f"- input_file: `{args.input_file}`",
        f"- judge_model: `{args.judge_model}`",
        f"- total_rows: {len(results)}",
        f"- judged_ok: {len(ok)}",
        f"- status_counts: {dict(Counter(r.get('status') for r in results))}",
    ]
    if scores:
        lines.extend(
            [
                f"- mean_score: {statistics.mean(scores):.4f}",
                f"- median_score: {statistics.median(scores):.4f}",
                f"- mean_unweighted_score: {statistics.mean(unweighted):.4f}",
                f"- strict_all_2_pass_rate: {sum(r['judge']['all_2_pass'] for r in ok) / len(ok):.4f}",
                f"- soft_all_ge1_pass_rate: {sum(r['judge']['all_ge1_pass'] for r in ok) / len(ok):.4f}",
                f"- format_ok_rate: {sum(r['judge']['format_ok'] for r in ok) / len(ok):.4f}",
            ]
        )
    lines.append("")
    lines.append("## By Instruction Count")
    by_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        by_count[int(r.get("instruction_count") or 0)].append(r)
    for count in sorted(by_count):
        bucket = by_count[count]
        bucket_scores = [r["judge"]["sample_score"] for r in bucket]
        lines.append(
            f"- {count}: n={len(bucket)}, mean={statistics.mean(bucket_scores):.4f}, "
            f"strict={sum(r['judge']['all_2_pass'] for r in bucket) / len(bucket):.4f}, "
            f"soft={sum(r['judge']['all_ge1_pass'] for r in bucket) / len(bucket):.4f}"
        )
    atom_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    axis_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures = Counter()
    for r in ok:
        for item in r["judge"]["instruction_results"]:
            atom_stats[item["atom_id"]].append(item)
            inst = item.get("instruction") or {}
            axis_stats[str(inst.get("axis") or "unknown")].append(item)
            source_stats[str(inst.get("source") or "unknown")].append(item)
            if item["score"] < 2:
                failures[item.get("failure_type") or "unclear"] += 1
    lines.append("")
    lines.append("## By Source")
    for source in sorted(source_stats):
        vals = source_stats[source]
        lines.append(f"- {source}: n={len(vals)}, mean={statistics.mean([x['score'] / 2 for x in vals]):.4f}")
    lines.append("")
    lines.append("## By Axis")
    for axis in sorted(axis_stats):
        vals = axis_stats[axis]
        lines.append(f"- {axis}: n={len(vals)}, mean={statistics.mean([x['score'] / 2 for x in vals]):.4f}")
    lines.append("")
    lines.append("## Failure Types")
    for key, val in failures.most_common():
        lines.append(f"- {key}: {val}")
    lines.append("")
    lines.append("## Lowest Atom Scores")
    atom_rows = []
    for atom_id, vals in atom_stats.items():
        mean = statistics.mean([x["score"] / 2 for x in vals])
        atom_rows.append((mean, len(vals), atom_id))
    for mean, n, atom_id in sorted(atom_rows)[:25]:
        lines.append(f"- {atom_id}: n={n}, mean={mean:.4f}")
    return "\n".join(lines) + "\n"


def write_summary_json(path: Path, results: list[dict[str, Any]]) -> None:
    ok = [r for r in results if r.get("status") == "ok"]
    scores = [r["judge"]["sample_score"] for r in ok]
    summary = {
        "rows": len(results),
        "status_counts": dict(Counter(r.get("status") for r in results)),
        "judged_ok": len(ok),
        "mean_score": statistics.mean(scores) if scores else None,
        "median_score": statistics.median(scores) if scores else None,
        "instruction_count_distribution": dict(Counter(str(r.get("instruction_count")) for r in results)),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = resolve_runtime(parse_args())
    rows = read_jsonl(Path(args.input_file), limit=args.limit, offset=args.offset)
    out_dir = make_output_dir(args)
    config = {
        "input_file": args.input_file,
        "output_root": args.output_root,
        "judge_model": args.judge_model,
        "api_base": args.api_base,
        "limit": args.limit,
        "offset": args.offset,
        "concurrency": args.concurrency,
        "response_field": args.response_field,
        "use_reference": args.use_reference,
        "candidate_model": args.candidate_model,
        "candidate_api_base": args.candidate_api_base,
        "candidate_max_tokens": args.candidate_max_tokens,
        "candidate_temperature": args.candidate_temperature,
        "candidate_top_p": args.candidate_top_p,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "judge_system_prompt.txt").write_text(JUDGE_SYSTEM_PROMPT, encoding="utf-8")
    try:
        shutil.copy2(Path(args.input_file), out_dir / "input_snapshot.jsonl")
    except Exception as exc:
        (out_dir / "input_snapshot_error.txt").write_text(repr(exc), encoding="utf-8")
    results: list[dict[str, Any]] = []
    total = len(rows)
    completed = 0
    status_counts: Counter[str] = Counter()
    start_time = time.time()

    def emit_progress(final: bool = False) -> None:
        if args.no_progress:
            return
        elapsed = max(0.001, time.time() - start_time)
        rate = completed / elapsed
        pct = (completed / total * 100.0) if total else 100.0
        message = (
            f"\rProgress: {completed}/{total} ({pct:5.1f}%) "
            f"ok={status_counts.get('ok', 0)} "
            f"gen_err={status_counts.get('generation_error', 0)} "
            f"judge_err={status_counts.get('judge_error', 0)} "
            f"missing={status_counts.get('missing_response', 0)} "
            f"elapsed={elapsed:6.1f}s rate={rate:4.2f}/s"
        )
        print(message, end="\n" if final else "", file=sys.stderr, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(evaluate_one, (idx + args.offset, row), args) for idx, row in enumerate(rows)]
        emit_progress()
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            status_counts[str(result.get("status") or "unknown")] += 1
            emit_progress(final=completed == total)
    results.sort(key=lambda x: int(x.get("row_index") or 0))
    write_jsonl(out_dir / "evaluation_results.jsonl", results)
    write_jsonl(
        out_dir / "failed_cases.jsonl",
        [r for r in results if r.get("status") != "ok" or (r.get("judge") and r["judge"]["sample_score"] < 1.0)],
    )
    write_jsonl(out_dir / "excluded_generation_errors.jsonl", [r for r in results if r.get("status") in {"missing_response", "generation_error"}])
    (out_dir / "evaluation_report.md").write_text(aggregate_report(results, args), encoding="utf-8")
    write_summary_json(out_dir / "summary.json", results)
    print(f"Wrote {out_dir}")
    print((out_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
