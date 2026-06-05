from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from recipe.single_turn_reward.v5.reward_model_stage4_contact_rubric_cloud import score_output_contact_rubric


def _extract_final_block(text: str) -> str:
    import re

    m = re.search(r"BEGIN_FINAL\s*\n([\s\S]*?)\nEND_FINAL", text or "")
    return m.group(1).strip() if m else text.strip()


def _load_records(path: Path, limit: int, offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if limit > 0 and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _record_to_case(record: dict[str, Any], system_prompt_override: str = "") -> dict[str, str] | None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    system_prompt = ""
    turns: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system_prompt = system_prompt_override or ("" if content == "None" else content)
        elif role in {"user", "assistant"}:
            turns.append((role, content))

    if not turns or turns[-1][0] != "assistant":
        return None

    output_answer = turns[-1][1]
    history_turns = turns[:-1]
    history_lines = []
    question = ""
    for role, content in history_turns:
        text = _extract_final_block(content) if role == "assistant" else content
        if role == "user":
            question = content
        history_lines.append(f"{role}: {text}")

    return {
        "question": question,
        "output_answer": output_answer,
        "history_text": "\n".join(history_lines),
        "system_prompt": system_prompt,
    }


async def _score_one(case: dict[str, str], args: argparse.Namespace, sem: asyncio.Semaphore) -> dict[str, Any]:
    async with sem:
        res = await score_output_contact_rubric(
            question=case["question"],
            output_answer=case["output_answer"],
            api_base=args.api_base,
            api_key=args.api_key,
            judge_model=args.judge_model,
            timeout_s=args.timeout_s,
            history_text=case["history_text"],
            system_prompt=case["system_prompt"],
            rubric_path=args.rubric_path,
            rubric_index_path=args.rubric_index_path,
            hard_config_path=args.hard_config_path,
            judge_max_tokens=args.judge_max_tokens,
        )
    return {
        "event": "offline_score",
        "question": case["question"],
        "output": case["output_answer"],
        "history_text": case["history_text"],
        "score": res.get("score"),
        "semantic_score": res.get("semantic_score"),
        "model_judge_score_raw": res.get("model_judge_score_raw"),
        "rubric_version": res.get("rubric_version"),
        "rubric_results": res.get("rubric_results", []),
        "gate_failed": res.get("gate_failed", False),
        "gate_reason": res.get("gate_reason", ""),
        "model_judge_status": res.get("status", ""),
        "single_score_reason": res.get("reason", ""),
        "final_char_len": res.get("final_char_len"),
        "length_penalty": res.get("length_penalty"),
        "sep_penalty": res.get("sep_penalty"),
        "banned_term_penalty": res.get("banned_term_penalty"),
        "banned_term_hits": res.get("banned_term_hits"),
        "hard_penalty_total": res.get("hard_penalty_total"),
        "raw": res.get("raw", ""),
    }


async def _main_async(args: argparse.Namespace) -> None:
    rows = _load_records(Path(args.input).expanduser(), args.limit, args.offset)
    cases = [case for row in rows if (case := _record_to_case(row, args.system_prompt_override)) is not None]
    sem = asyncio.Semaphore(max(1, args.concurrency))
    results = await asyncio.gather(*[_score_one(case, args, sem) for case in cases])

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    scores = [float(r.get("score") or 0.0) for r in results]
    hard = [float(r.get("hard_penalty_total") or 0.0) for r in results]
    ok = sum(1 for r in results if r.get("model_judge_status") == "ok")
    summary = {
        "input": args.input,
        "output": str(out_path),
        "count": len(results),
        "ok": ok,
        "mean_score": sum(scores) / max(1, len(scores)),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "mean_hard_penalty": sum(hard) / max(1, len(hard)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline score high_score_sft.jsonl with v5 contact rubric reward.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--judge-max-tokens", type=int, default=1600)
    parser.add_argument("--rubric-path", default="")
    parser.add_argument("--rubric-index-path", default="/data/chengch/project/verl/recipe/single_turn_reward/v5/rubrics/rubric_index.json")
    parser.add_argument("--hard-config-path", default="/data/chengch/project/verl/recipe/single_turn_reward/v5/contact_reward_hard_config.json")
    parser.add_argument("--system-prompt-override", default="", help="Override system prompt for SFT files whose system message is None.")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
