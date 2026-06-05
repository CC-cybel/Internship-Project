from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp


_DEFAULT_DIR = Path(__file__).resolve().parent / "rubrics"
_RUBRIC_VERSION_RE = re.compile(r"contact_rubric_v(\d+)\.json$")


def _load_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_new(path: Path, data: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    l = text.find("{")
    r = text.rfind("}")
    if l < 0 or r < 0 or r <= l:
        return None
    try:
        obj = json.loads(text[l : r + 1])
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _active_rubric_file(rubric_dir: Path, index_path: Path | None) -> Path:
    if index_path is not None:
        index = _load_json(index_path)
        active = str(index.get("active_version", "")).strip()
        if not active:
            raise ValueError(f"Missing active_version in {index_path}")
        return (index_path.parent / active).resolve()

    candidates = []
    for path in rubric_dir.glob("contact_rubric_v*.json"):
        m = _RUBRIC_VERSION_RE.match(path.name)
        if m:
            candidates.append((int(m.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No contact_rubric_v*.json found in {rubric_dir}")
    return sorted(candidates)[-1][1]


def _next_rubric_path(current: Path) -> Path:
    m = _RUBRIC_VERSION_RE.match(current.name)
    if not m:
        raise ValueError(f"Rubric filename must match contact_rubric_vNNN.json: {current}")
    next_n = int(m.group(1)) + 1
    return current.with_name(f"contact_rubric_v{next_n:03d}.json")


def _read_recent_trace(path: Path, window_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    if window_size > 0:
        rows = rows[-window_size:]
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _short_text(value: Any, limit: int = 260) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _case_examples(rows: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: _safe_float(r.get("score"), 0.0))
    examples: list[dict[str, Any]] = []
    for row in ranked[:limit]:
        rubric_results = row.get("rubric_results")
        if isinstance(rubric_results, str):
            try:
                rubric_results = json.loads(rubric_results)
            except Exception:
                rubric_results = []
        if not isinstance(rubric_results, list):
            rubric_results = []

        low_items = []
        for result in rubric_results:
            if not isinstance(result, dict) or not result.get("triggered"):
                continue
            score = result.get("score")
            if score is None or _safe_float(score, 2.0) >= 2.0:
                continue
            low_items.append(
                {
                    "id": result.get("id"),
                    "score": result.get("score"),
                    "deduction": result.get("deduction"),
                    "evidence": _short_text(result.get("evidence"), 80),
                    "reason": _short_text(result.get("reason"), 120),
                }
            )

        examples.append(
            {
                "score": row.get("score"),
                "semantic_score": row.get("semantic_score"),
                "hard_penalty_total": row.get("hard_penalty_total"),
                "question_excerpt": _short_text(row.get("question"), 220),
                "output_excerpt": _short_text(row.get("output"), 360),
                "low_rubric_items": low_items[:6],
                "judge_reason": _short_text(row.get("single_score_reason"), 120),
            }
        )
    return examples


def _aggregate_stats(
    rows: list[dict[str, Any]],
    *,
    max_examples_per_rubric: int = 5,
    max_total_examples: int = 40,
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    score_values: list[float] = []
    status_counts: dict[str, int] = {}
    hard_totals: list[float] = []
    banned_hits: dict[str, int] = {}

    for row in rows:
        score_values.append(_safe_float(row.get("score"), 0.0))
        status = str(row.get("model_judge_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        hard_totals.append(_safe_float(row.get("hard_penalty_total"), 0.0))
        for term in row.get("banned_term_hits") or []:
            term = str(term)
            banned_hits[term] = banned_hits.get(term, 0) + 1

        rubric_results = row.get("rubric_results")
        if isinstance(rubric_results, str):
            try:
                rubric_results = json.loads(rubric_results)
            except Exception:
                rubric_results = []
        if not isinstance(rubric_results, list):
            rubric_results = []

        for result in rubric_results:
            if not isinstance(result, dict):
                continue
            rid = str(result.get("id", "")).strip()
            if not rid:
                continue
            stat = by_id.setdefault(
                rid,
                {
                    "count": 0,
                    "triggered": 0,
                    "score_sum": 0.0,
                    "deduction_sum": 0.0,
                    "deduction_sq_sum": 0.0,
                    "gate_failed": 0,
                    "examples": [],
                },
            )
            stat["count"] += 1
            triggered = bool(result.get("triggered"))
            if triggered:
                stat["triggered"] += 1
                stat["score_sum"] += _safe_float(result.get("score"), 0.0)
                deduction = _safe_float(result.get("deduction"), 0.0)
                stat["deduction_sum"] += deduction
                stat["deduction_sq_sum"] += deduction * deduction
                if result.get("gate_failed"):
                    stat["gate_failed"] += 1
                if len(stat["examples"]) < max_examples_per_rubric:
                    stat["examples"].append(
                        {
                            "score": result.get("score"),
                            "deduction": result.get("deduction"),
                            "evidence": result.get("evidence"),
                            "reason": result.get("reason"),
                        }
                    )

    total = len(rows)
    rubric_stats: dict[str, Any] = {}
    for rid, stat in by_id.items():
        count = max(1, int(stat["count"]))
        triggered = int(stat["triggered"])
        mean_ded = stat["deduction_sum"] / max(1, triggered)
        mean_ded_all = stat["deduction_sum"] / count
        variance = max(0.0, stat["deduction_sq_sum"] / max(1, triggered) - mean_ded * mean_ded)
        rubric_stats[rid] = {
            "count": stat["count"],
            "triggered": triggered,
            "trigger_rate": triggered / count,
            "mean_score_triggered": stat["score_sum"] / max(1, triggered),
            "mean_deduction_triggered": mean_ded,
            "mean_deduction_all": mean_ded_all,
            "deduction_variance_triggered": variance,
            "gate_failed": stat["gate_failed"],
            "examples": stat["examples"],
        }

    return {
        "num_records": total,
        "mean_reward": sum(score_values) / max(1, len(score_values)),
        "mean_hard_penalty": sum(hard_totals) / max(1, len(hard_totals)),
        "status_counts": status_counts,
        "banned_term_hits": banned_hits,
        "rubric_stats": rubric_stats,
        "case_examples_low_score_first": _case_examples(rows, limit=max_total_examples),
        "sampling_config": {
            "max_examples_per_rubric": max_examples_per_rubric,
            "max_total_examples": max_total_examples,
        },
    }


async def _chat(api_base: str, api_key: str, model: str, prompt: str, timeout_s: float) -> str:
    base = api_base.rstrip("/")
    url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4000,
    }
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
            obj = json.loads(text)
    return obj.get("choices", [{}])[0].get("message", {}).get("content", "")


def _build_review_prompt(
    current: dict[str, Any],
    stats: dict[str, Any],
    next_version: str,
    allow_no_change: bool,
) -> str:
    return (
        "你是留联阶段 reward rubric reviewer，负责维护 judge-side semantic rubric buffer。\n"
        "你会看到上一版 rubric、最近窗口的逐规则统计、低分样本摘要和 judge 证据。\n"
        "你的目标不是重新评分，而是产生下一版更有区分度、更稳定、更不容易被模型钻空子的语义 rubric set。\n"
        "输入已经被压缩：全量样本只进入聚合统计，样本正文只保留少量代表案例；不要假设你看到了所有原始对话。\n\n"
        "核心原则，参考 evolving-rubric judge:\n"
        "1) Discriminative and specific: 只保留或新增能区分强/弱回复的标准；描述必须具体到留联任务中的可观察行为。\n"
        "2) Stage-local and non-redundant: 每条 rubric 只负责一个判断点，避免多个 rubric 重复惩罚同一问题。\n"
        "3) Anti-hack: 不要奖励形式服从、空洞自洽、套模板、单纯变长；不要因为 thought 写得好就忽略 final 话术薄弱。\n"
        "4) Evidence-grounded update: 只能依据统计和样本证据调整；不要凭空添加没有在窗口中暴露的偏好。\n"
        "5) Conservative evolution: 小步更新优先。能改 description/score_levels 就不要改 id；能 disable 就不要删除。\n"
        "6) Window-aware: 小窗口下证据不足时可以保持不变；只有统计或代表样本明确支持时才新增/删除/大改 rubric。\n\n"
        "你应按下面步骤思考，但最终只输出 JSON:\n"
        "A. 诊断每条 rubric 的状态: high_signal / needs_clarification / redundant / inactive_but_keep / disable_candidate。\n"
        "B. 对低区分项检查原因: 触发率低、所有样本都满分、描述含糊、与其他项重叠、judge 证据不足。\n"
        "C. 对低分样本检查是否暴露了新失败模式，例如福利泛化、二次套联无承接、槽位虚构、system 指令误判。\n"
        "D. 先决定 update_decision: update 或 no_change；如果 no_change，也要输出完整下一版 JSON，但 rubrics 可与上一版一致。\n"
        "E. 生成完整下一版 rubric JSON。\n\n"
        "严格约束:\n"
        "1) 不要修改硬性机械惩罚，不要加入长度、<sep>、禁用词字面匹配、字符数等本地规则。\n"
        "2) 必须输出完整的新 rubric set JSON，而不是 patch。\n"
        "3) 可以新增、修改、合并、disable 语义 rubric，但必须保留 rubric_set_id/version/source/description/score_scale/rubrics。\n"
        "4) gate 或医疗合规类规则不要轻易删除；如果没有触发但风险高，保留并说明 inactive_but_keep。\n"
        "5) 新 version 必须是给定值。\n"
        "6) 输出中必须包含 update_decision 字段，取值 update 或 no_change。\n"
        "7) 输出中必须包含 reviewer_decisions 数组，记录每条主要修改的 evidence 和 rationale；no_change 时写明证据不足或当前规则仍有效。\n"
        f"8) allow_no_change={allow_no_change}；若为 false，除非输入为空，否则仍应做最小必要更新。\n"
        "9) 只输出 JSON，禁止 Markdown，禁止额外解释。\n\n"
        "输出 JSON 形状:\n"
        "{\n"
        '  "rubric_set_id": "...",\n'
        f'  "version": "{next_version}",\n'
        '  "source": "reviewer",\n'
        '  "description": "这版做了什么调整",\n'
        '  "review_summary": "一句话总结",\n'
        '  "update_decision": "update|no_change",\n'
        '  "reviewer_decisions": [\n'
        '    {"action": "keep|revise|add|disable|merge", "rubric_id": "id", "evidence": "统计或样本依据", "rationale": "为什么这样改"}\n'
        "  ],\n"
        '  "score_scale": {"min": 0, "max": 2, "meaning": {"0": "...", "1": "...", "2": "..."}},\n'
        '  "rubrics": [完整 rubric 列表]\n'
        "}\n\n"
        f"next_version: {next_version}\n\n"
        "当前 rubric:\n"
        f"{json.dumps(current, ensure_ascii=False, indent=2)}\n\n"
        "最近窗口统计:\n"
        f"{json.dumps(stats, ensure_ascii=False, indent=2)}\n"
    )


def _validate_new_rubric(obj: dict[str, Any], next_version: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Reviewer output is not a JSON object")
    rubrics = obj.get("rubrics")
    if not isinstance(rubrics, list) or not rubrics:
        raise ValueError("Reviewer output must contain non-empty rubrics list")
    obj["version"] = next_version
    obj["source"] = "reviewer"
    obj.setdefault("description", "Reviewer-updated semantic rubric set.")
    forbidden = ("长度", "<sep>", "禁用词", "banned", "hard_penalty", "final_char_len")
    for rubric in rubrics:
        if not isinstance(rubric, dict):
            raise ValueError("Each rubric must be an object")
        text = json.dumps(rubric, ensure_ascii=False)
        if any(token in text for token in forbidden):
            raise ValueError(f"Reviewer output appears to include hard penalty rubric: {rubric.get('id')}")
        if not rubric.get("id"):
            raise ValueError("Each rubric must have id")
        rubric.setdefault("enabled", True)
        rubric.setdefault("evaluation_type", "llm")
    return obj


async def _main_async(args: argparse.Namespace) -> None:
    rubric_dir = Path(args.rubric_dir).expanduser().resolve()
    index_path = Path(args.rubric_index).expanduser().resolve() if args.rubric_index else None
    current_path = _active_rubric_file(rubric_dir, index_path)
    next_path = _next_rubric_path(current_path)
    next_version = next_path.stem
    report_path = next_path.with_name(f"review_report_v{_RUBRIC_VERSION_RE.match(next_path.name).group(1)}.json")

    current = _load_json(current_path)
    rows = _read_recent_trace(Path(args.trace_jsonl).expanduser(), args.window_size)
    stats = _aggregate_stats(
        rows,
        max_examples_per_rubric=args.max_examples_per_rubric,
        max_total_examples=args.max_total_examples,
    )

    report: dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "based_on": current_path.name,
        "new_rubric_file": next_path.name,
        "window_size": args.window_size,
        "min_window_size": args.min_window_size,
        "max_examples_per_rubric": args.max_examples_per_rubric,
        "max_total_examples": args.max_total_examples,
        "allow_no_change": args.allow_no_change,
        "trace_jsonl": str(Path(args.trace_jsonl).expanduser()),
        "stats": stats,
        "mode": "copy_without_llm",
        "review_summary": "No reviewer API config supplied; copied current rubric with updated version metadata.",
    }

    enough_rows = len(rows) >= args.min_window_size

    if args.api_base and args.api_key and args.judge_model and enough_rows:
        prompt = _build_review_prompt(current, stats, next_version, args.allow_no_change)
        raw = await _chat(args.api_base, args.api_key, args.judge_model, prompt, args.timeout_s)
        obj = _extract_json_dict(raw)
        if obj is None:
            raise ValueError(f"Reviewer returned non-JSON output: {raw[:300]}")
        new_rubric = _validate_new_rubric(obj, next_version)
        report["mode"] = "llm_reviewer"
        report["raw_reviewer_output"] = raw
        report["update_decision"] = str(new_rubric.get("update_decision", "update"))
        report["review_summary"] = str(new_rubric.get("review_summary", "LLM reviewer generated new rubric."))
    else:
        new_rubric = dict(current)
        new_rubric["version"] = next_version
        new_rubric["source"] = "reviewer_copy"
        new_rubric["created_at_step"] = args.created_at_step
        new_rubric["based_on"] = current_path.name
        new_rubric["update_decision"] = "no_change"
        if not enough_rows:
            report["mode"] = "copy_insufficient_window"
            report["review_summary"] = f"Only {len(rows)} rows available, below min_window_size={args.min_window_size}; copied current rubric."
        report["update_decision"] = "no_change"

    _write_json_new(next_path, new_rubric)
    _write_json_new(report_path, report)
    print(json.dumps({"new_rubric": str(next_path), "report": str(report_path)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new contact rubric JSON version without overwriting old files.")
    parser.add_argument("--trace-jsonl", required=True, help="genrm_io trace JSONL containing rubric_results.")
    parser.add_argument("--rubric-dir", default=str(_DEFAULT_DIR), help="Directory containing contact_rubric_vNNN.json files.")
    parser.add_argument("--rubric-index", default="", help="Optional rubric_index.json. If omitted, latest vNNN is used.")
    parser.add_argument("--window-size", type=int, default=800, help="Number of latest trace rows used for aggregate stats.")
    parser.add_argument("--min-window-size", type=int, default=20, help="Copy current rubric without LLM review when fewer rows are available.")
    parser.add_argument("--max-examples-per-rubric", type=int, default=5, help="Max representative judge snippets retained per rubric.")
    parser.add_argument("--max-total-examples", type=int, default=40, help="Max low-score case summaries shown to the reviewer.")
    parser.add_argument("--allow-no-change", action="store_true", help="Allow reviewer to emit a no_change decision when evidence is weak.")
    parser.add_argument("--created-at-step", type=int, default=-1, help="Training step attached to copied reviewer metadata.")
    parser.add_argument("--api-base", default="", help="OpenAI-compatible API base for LLM reviewer.")
    parser.add_argument("--api-key", default="", help="API key for LLM reviewer.")
    parser.add_argument("--judge-model", default="", help="Reviewer model name.")
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
