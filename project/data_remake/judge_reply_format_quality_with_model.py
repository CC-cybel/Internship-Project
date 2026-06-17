#!/usr/bin/env python3
"""Model judge for ShareGPT reply format quality.

Focus: response format / protocol quality for datasets whose assistant replies
should contain BEGIN_META and BEGIN_FINAL blocks. The judge is model-based, with
rule facts supplied as evidence so the model can give detailed reasons and a
0-100 score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

DEFAULT_INPUT = "/data/chengch/project/data_remake/runs/last_turn_value_slots_split_v2/instruction.clean.minimal.jsonl"
DEFAULT_OUTPUT_DIR = "/data1/chengch/project/data_remake/runs/last_turn_value_slots_split_v2/model_judge_format_quality"
DEFAULT_ENV_FILE = "/data/chengch/project/verl/recipe/single_turn_reward/v3/run_grpo_single_turn_4gpu_qwen3_8b_stage5_mid_reward.sh"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4-flash"

REQUIRED_MARKERS = ("BEGIN_META", "END_META", "BEGIN_FINAL", "END_FINAL")
THOUGHT_MARKERS = ("【锚定】", "【分析】", "【检验】", "【策略】")
ANALYSIS_FIELDS = ("留联分层", "用户状态", "生理层", "心理层", "战术层")
REQUIRED_SLOTS = (
    "slot_age",
    "slot_gender",
    "slot_name",
    "slot_phone",
    "slot_wechat",
    "slot_symptom",
    "slot_duration",
    "slot_medical_history",
    "slot_relationship",
    "slot_medical_awareness",
)
META_LEAK_PATTERNS = (
    "BEGIN_META", "END_META", "BEGIN_FINAL", "END_FINAL", "thought=", "slot_",
    "【锚定】", "【分析】", "【检验】", "【策略】", "附加指令", "atom_id",
    "trigger_condition", "required_behavior", "第1条", "第2条",
)
QUESTION_RE = re.compile(r"[？?]")

JUDGE_SYSTEM_PROMPT = """你是一个严格的数据质量审核员，负责审核医疗咨询客服训练数据中的 assistant 回复格式质量。

你只评价格式质量、协议遵循和训练可用性，不评价医学方案是否正确，也不要因为客服话术风格普通而过度扣分。

目标回复协议：
1. 每条 assistant 回复应包含 BEGIN_META/END_META/BEGIN_FINAL/END_FINAL 四个边界。
2. BEGIN_META 内应有 action=、thought= 和 slot_* 槽位。
3. thought 应包含【锚定】【分析】【检验】【策略】四段；【分析】里应包含留联分层、用户状态、生理层、心理层、战术层。
4. BEGIN_FINAL 只能是用户可见话术，不能混入 thought、slot、系统规则、附加指令说明、JSON/key、日志、编号规则解释。
5. 用户可见回复应自然，不应出现明显模板残留、截断、重复边界、空回复、过多问题、过多 <sep>。
6. 如果 system 明确要求每轮最多 2 个问题，超过 2 个问句属于格式/约束问题。

评分标准，输出 0-100：
- 90-100：格式完整，槽位和 thought 结构完整，FINAL 干净，只有很轻微可忽略问题。
- 75-89：总体可训练，有少量轻微格式瑕疵或表达冗余，但不影响学习主要协议。
- 60-74：勉强可用，有若干明显格式问题，需要抽查或低权重使用。
- 40-59：质量较差，有严重结构缺失、FINAL 污染、槽位大量缺失或多处协议不一致。
- 0-39：不可用于训练，如空回复、边界缺失严重、截断、FINAL 大量混入 META、完全不符合协议。

请务必输出合法 JSON 对象，不要输出 Markdown，不要输出多余解释。"""

JUDGE_USER_TEMPLATE = """请审核下面一个训练样本的 assistant 回复格式质量。

审核范围：{turn_scope_desc}

[系统提示摘要]
{system_excerpt}

[规则事实预检]
{rule_facts_json}

[待审核 assistant 回复]
{assistant_blocks}

请输出 JSON，格式固定为：
{{
  "quality_score": 0-100的整数,
  "quality_level": "excellent|good|borderline|poor|bad",
  "is_trainable": true/false,
  "main_problems": ["最主要问题1", "最主要问题2"],
  "format_score": 0-20,
  "meta_thought_score": 0-20,
  "slot_score": 0-15,
  "final_clean_score": 0-20,
  "constraint_score": 0-15,
  "readability_score": 0-10,
  "turn_findings": [
    {{
      "turn_index": assistant轮次编号,
      "score": 0-100,
      "severity": "none|minor|major|critical",
      "problems": ["该轮问题"],
      "evidence": "引用少量证据"
    }}
  ],
  "suggested_action": "keep|review|drop|repair"
}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1800)
    p.add_argument("--turn-scope", choices=["all", "last"], default="all")
    p.add_argument("--max-system-chars", type=int, default=2500)
    p.add_argument("--max-assistant-chars", type=int, default=1800)
    p.add_argument("--fail-score-threshold", type=int, default=75)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_env_file(path: str) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    env: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(": ") and ":=" in line:
            match = re.match(r':\s*"\$\{([A-Za-z_][A-Za-z0-9_]*):=([^}]*)\}"', line)
            if match:
                env[match.group(1)] = match.group(2)
    return env


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    env = load_env_file(args.env_file)
    args.api_base = args.api_base or os.environ.get("TONGYI_API_BASE") or env.get("TONGYI_API_BASE") or DEFAULT_API_BASE
    args.api_key = args.api_key or os.environ.get("TONGYI_API_KEY") or env.get("TONGYI_API_KEY")
    args.model = args.model or os.environ.get("JUDGE_MODEL") or env.get("JUDGE_MODEL") or DEFAULT_MODEL
    if not args.api_key and not args.dry_run:
        raise ValueError("Missing API key. Set --api-key or TONGYI_API_KEY.")
    return args


def role_of(turn: Any) -> str:
    return str(turn.get("from") or turn.get("role") or "") if isinstance(turn, dict) else ""


def value_of(turn: Any) -> str:
    if not isinstance(turn, dict):
        return "" if turn is None else str(turn)
    value = turn.get("value")
    if value is None:
        value = turn.get("content")
    return "" if value is None else str(value)


def load_jsonl(path: Path, limit: int | None, offset: int) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset or not line.strip():
                continue
            rows.append((idx, json.loads(line)))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def extract_between(text: str, begin: str, end: str) -> str:
    b = text.find(begin)
    e = text.rfind(end)
    if b >= 0 and e >= b:
        return text[b + len(begin):e].strip()
    return ""


def extract_meta(text: str) -> str:
    return extract_between(text, "BEGIN_META", "END_META")


def extract_final(text: str) -> str:
    return extract_between(text, "BEGIN_FINAL", "END_FINAL")


def extract_thought(meta: str) -> str:
    m = re.search(r"(?:^|\n)thought=(.*?)(?:\nslot_|$)", meta, re.S)
    return m.group(1).strip() if m else ""


def compact_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def assistant_turns(row: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    turns = []
    for turn_idx, turn in enumerate(row.get("conversations") or []):
        if role_of(turn) == "gpt":
            turns.append({"turn_index": turn_idx, "assistant_ord": len(turns) + 1, "text": value_of(turn)})
    if scope == "last" and turns:
        return [turns[-1]]
    return turns


def rule_facts_for_turn(turn: dict[str, Any]) -> dict[str, Any]:
    text = turn["text"]
    meta = extract_meta(text)
    final = extract_final(text)
    thought = extract_thought(meta)
    missing_markers = [m for m in REQUIRED_MARKERS if m not in text]
    missing_thought_markers = [m for m in THOUGHT_MARKERS if m not in thought]
    missing_analysis_fields = [f for f in ANALYSIS_FIELDS if f not in thought]
    missing_slots = []
    empty_slots = []
    for slot in REQUIRED_SLOTS:
        m = re.search(rf"(?:^|\n){re.escape(slot)}=([^\n]*)", meta)
        if not m:
            missing_slots.append(slot)
        elif not m.group(1).strip():
            empty_slots.append(slot)
    return {
        "turn_index": turn["turn_index"],
        "assistant_ord": turn["assistant_ord"],
        "char_len": len(text),
        "missing_markers": missing_markers,
        "has_action": "action=" in meta,
        "has_thought": "thought=" in meta,
        "missing_thought_markers": missing_thought_markers,
        "missing_analysis_fields": missing_analysis_fields,
        "missing_slots": missing_slots,
        "empty_slots": empty_slots,
        "final_empty": not bool(final.strip()),
        "final_len_compact": compact_len(final),
        "final_question_count": len(QUESTION_RE.findall(final)),
        "final_sep_count": final.count("<sep>"),
        "final_meta_leak_patterns": [p for p in META_LEAK_PATTERNS if p in final],
        "final_preview": final[:180],
    }


def build_prompt(index: int, row: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    turns = assistant_turns(row, args.turn_scope)
    facts = [rule_facts_for_turn(t) for t in turns]
    system_excerpt = str(row.get("system") or "")[:args.max_system_chars]
    blocks = []
    for t in turns:
        text = t["text"]
        if len(text) > args.max_assistant_chars:
            text = text[:args.max_assistant_chars] + "\n...[TRUNCATED_FOR_JUDGE]"
        blocks.append(f"### assistant_ord={t['assistant_ord']} turn_index={t['turn_index']}\n{text}")
    user = JUDGE_USER_TEMPLATE.format(
        turn_scope_desc="全部 assistant 回复" if args.turn_scope == "all" else "最后一条 assistant 回复",
        system_excerpt=system_excerpt,
        rule_facts_json=json.dumps(facts, ensure_ascii=False, indent=2),
        assistant_blocks="\n\n".join(blocks) if blocks else "[NO_ASSISTANT_TURNS]",
    )
    metadata = {"index": index, "rule_facts": facts, "assistant_turn_count": len(turns)}
    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user}], metadata


def call_chat(messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        args.api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    return data["choices"][0]["message"]["content"]


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def normalize_judgement(obj: dict[str, Any]) -> dict[str, Any]:
    score = obj.get("quality_score", 0)
    try:
        score = int(round(float(score)))
    except Exception:
        score = 0
    obj["quality_score"] = max(0, min(100, score))
    if obj.get("quality_level") not in {"excellent", "good", "borderline", "poor", "bad"}:
        s = obj["quality_score"]
        obj["quality_level"] = "excellent" if s >= 90 else "good" if s >= 75 else "borderline" if s >= 60 else "poor" if s >= 40 else "bad"
    if not isinstance(obj.get("main_problems"), list):
        obj["main_problems"] = []
    if not isinstance(obj.get("turn_findings"), list):
        obj["turn_findings"] = []
    if obj.get("suggested_action") not in {"keep", "review", "drop", "repair"}:
        obj["suggested_action"] = "keep" if obj["quality_score"] >= 85 else "review" if obj["quality_score"] >= 70 else "repair" if obj["quality_score"] >= 45 else "drop"
    if not isinstance(obj.get("is_trainable"), bool):
        obj["is_trainable"] = obj["quality_score"] >= 75 and obj["suggested_action"] in {"keep", "review"}
    return obj


def judge_one(index: int, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    messages, meta = build_prompt(index, row, args)
    if args.dry_run:
        judgement = {
            "quality_score": 0,
            "quality_level": "borderline",
            "is_trainable": False,
            "main_problems": ["dry_run_no_model_call"],
            "format_score": 0,
            "meta_thought_score": 0,
            "slot_score": 0,
            "final_clean_score": 0,
            "constraint_score": 0,
            "readability_score": 0,
            "turn_findings": [],
            "suggested_action": "review",
        }
        raw = ""
    else:
        last_exc: Exception | None = None
        for attempt in range(args.max_retries):
            try:
                raw = call_chat(messages, args)
                judgement = normalize_judgement(parse_json_object(raw))
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                if attempt >= args.max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 8))
        else:
            raise RuntimeError(f"judge failed: {last_exc}")
    return {
        "index": index,
        "sample_id": row.get("id") or row.get("candidate_id") or row.get("pair_id") or index,
        "assistant_turn_count": meta["assistant_turn_count"],
        "rule_facts": meta["rule_facts"],
        "judgement": judgement,
        "raw_judge_content": raw,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_existing(path: Path) -> dict[int, dict[str, Any]]:
    existing = {}
    if not path.exists():
        return existing
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("index"), int):
                existing[row["index"]] = row
    return existing


def main() -> None:
    args = resolve_args(parse_args())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "judge_results.jsonl"
    failed_path = out_dir / "low_quality_cases.jsonl"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"

    rows = load_jsonl(Path(args.input), args.limit, args.offset)
    existing = {} if args.overwrite else load_existing(results_path)
    results: dict[int, dict[str, Any]] = dict(existing)
    pending = [(idx, row) for idx, row in rows if idx not in existing]
    print(f"loaded={len(rows)} resumed={len(existing)} pending={len(pending)} output={results_path}")

    errors = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(judge_one, idx, row, args): idx for idx, row in pending}
        iterable = as_completed(futures)
        if tqdm:
            iterable = tqdm(iterable, total=len(futures), desc="model_judge")
        completed_since_write = 0
        for fut in iterable:
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                errors.append({"index": idx, "error": str(exc)})
            completed_since_write += 1
            if completed_since_write >= 50:
                write_jsonl(results_path, [results[k] for k in sorted(results)])
                completed_since_write = 0
    ordered = [results[k] for k in sorted(results)]
    write_jsonl(results_path, ordered)

    low_quality = [r for r in ordered if int(r.get("judgement", {}).get("quality_score", 0)) < args.fail_score_threshold or not r.get("judgement", {}).get("is_trainable", False)]
    write_jsonl(failed_path, low_quality)

    scores = [int(r.get("judgement", {}).get("quality_score", 0)) for r in ordered]
    level_c = Counter(r.get("judgement", {}).get("quality_level", "unknown") for r in ordered)
    action_c = Counter(r.get("judgement", {}).get("suggested_action", "unknown") for r in ordered)
    problem_c = Counter(p for r in ordered for p in r.get("judgement", {}).get("main_problems", []))
    rule_problem_c = Counter()
    for r in ordered:
        for fact in r.get("rule_facts", []):
            for key in ("missing_markers", "missing_thought_markers", "missing_analysis_fields", "missing_slots", "empty_slots", "final_meta_leak_patterns"):
                for item in fact.get(key, []) or []:
                    rule_problem_c[f"{key}:{item}"] += 1
            if fact.get("final_empty"):
                rule_problem_c["final_empty"] += 1
            if fact.get("final_question_count", 0) > 2:
                rule_problem_c["final_question_count_gt2"] += 1
            if fact.get("final_sep_count", 0) > 2:
                rule_problem_c["final_sep_count_gt2"] += 1

    summary = {
        "input": args.input,
        "output_dir": str(out_dir),
        "model": args.model,
        "turn_scope": args.turn_scope,
        "total_judged": len(ordered),
        "pending_errors": len(errors),
        "score_avg": sum(scores) / len(scores) if scores else 0,
        "score_min": min(scores) if scores else 0,
        "score_max": max(scores) if scores else 0,
        "score_lt_threshold_count": len([s for s in scores if s < args.fail_score_threshold]),
        "trainable_count": len([r for r in ordered if r.get("judgement", {}).get("is_trainable", False)]),
        "quality_level_counts": dict(level_c),
        "suggested_action_counts": dict(action_c),
        "top_model_problems": dict(problem_c.most_common(30)),
        "top_rule_facts": dict(rule_problem_c.most_common(30)),
        "errors": errors[:50],
        "paths": {
            "judge_results": str(results_path),
            "low_quality_cases": str(failed_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Model Judge Format Quality Report",
        "",
        f"- Input: `{args.input}`",
        f"- Model: `{args.model}`",
        f"- Turn scope: `{args.turn_scope}`",
        f"- Total judged: **{len(ordered)}**",
        f"- Average score: **{summary['score_avg']:.2f}**",
        f"- Trainable count: **{summary['trainable_count']}**",
        f"- Low quality cases: **{len(low_quality)}**",
        "",
        "## Quality Levels",
        "",
    ]
    for k, v in level_c.most_common():
        report.append(f"- `{k}`: {v}")
    report += ["", "## Suggested Actions", ""]
    for k, v in action_c.most_common():
        report.append(f"- `{k}`: {v}")
    report += ["", "## Top Model Problems", ""]
    for k, v in problem_c.most_common(20):
        report.append(f"- {k}: {v}")
    report += ["", "## Top Rule Facts", ""]
    for k, v in rule_problem_c.most_common(20):
        report.append(f"- `{k}`: {v}")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
