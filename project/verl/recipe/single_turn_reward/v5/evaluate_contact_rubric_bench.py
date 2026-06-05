#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from recipe.single_turn_reward.v5.reward_model_stage4_contact_rubric_cloud import (  # noqa: E402
    score_output_contact_rubric,
)


DEFAULT_INPUT = (
    _HERE
    / "data"
    / "single_turn_rl_contact_stage_new_sources_12k_age_directed.rubric_bench_300.jsonl"
)
DEFAULT_RUBRIC = _HERE / "rubrics" / "contact_rubric_v001.json"
DEFAULT_HARD_CONFIG = _HERE / "contact_reward_hard_config.json"
DEFAULT_OUTPUT_ROOT = _HERE / "data" / "rubric_eval_outputs"


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_name(value: str) -> str:
    value = str(value or "unknown").strip().replace("/", "_").replace("\\", "_")
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(keep)[:120] or "unknown"


def _normalize_role(role: Any) -> str:
    role = str(role or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "bot", "gpt"}:
        return "assistant"
    if role == "system":
        return "system"
    return ""


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    normalized: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = _normalize_role(msg.get("role", msg.get("from")))
        content = msg.get("content", msg.get("value"))
        if not role or content is None:
            continue
        text = str(content).strip()
        if text:
            normalized.append({"role": role, "content": text})
    return normalized


def _sample_id(sample: dict[str, Any], row_index: int) -> str:
    extra = sample.get("extra_info") if isinstance(sample.get("extra_info"), dict) else {}
    for key in ("index", "id", "key", "sample_id"):
        value = sample.get(key)
        if value not in (None, ""):
            return str(value)
    if isinstance(extra, dict):
        for key in ("sample_id", "index", "id", "conv_id"):
            value = extra.get(key)
            if value not in (None, ""):
                return str(value)
    return f"row_{row_index}"


def _prompt_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("prompt", "messages", "raw_prompt"):
        messages = _normalize_messages(sample.get(key))
        if messages:
            return messages
    extra = sample.get("extra_info") if isinstance(sample.get("extra_info"), dict) else {}
    if isinstance(extra, dict):
        messages = _normalize_messages(extra.get("conversations"))
        if messages:
            system = str(extra.get("original_system_prompt", "") or "").strip()
            return ([{"role": "system", "content": system}] if system else []) + messages
    return []


def _system_prompt(sample: dict[str, Any], messages: list[dict[str, str]]) -> str:
    extra = sample.get("extra_info") if isinstance(sample.get("extra_info"), dict) else {}
    if isinstance(extra, dict):
        for key in ("transformed_system_prompt", "original_system_prompt", "system_prompt"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for msg in messages:
        if msg["role"] == "system":
            return msg["content"]
    return ""


def _question(sample: dict[str, Any], messages: list[dict[str, str]]) -> str:
    extra = sample.get("extra_info") if isinstance(sample.get("extra_info"), dict) else {}
    if isinstance(extra, dict):
        for key in ("question", "instruction", "query"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"].strip()
    return ""


def _strip_final_block(text: str) -> str:
    import re

    m = re.search(r"BEGIN_FINAL\s*\n([\s\S]*?)\nEND_FINAL", text or "")
    return m.group(1).strip() if m else str(text or "").strip()


def _history_text(sample: dict[str, Any], messages: list[dict[str, str]]) -> str:
    extra = sample.get("extra_info") if isinstance(sample.get("extra_info"), dict) else {}
    history = []
    if isinstance(extra, dict):
        history = _normalize_messages(extra.get("conversations"))
    if not history:
        history = [msg for msg in messages if msg["role"] != "system"]

    lines: list[str] = []
    turn_no = 0
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            turn_no += 1
        if role == "assistant":
            content = _strip_final_block(content)
            if not content:
                continue
        prefix = f"[第{turn_no}轮]{role}" if turn_no > 0 else role
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _ground_truth(sample: dict[str, Any]) -> str:
    if isinstance(sample.get("ground_truth"), str) and sample["ground_truth"].strip():
        return sample["ground_truth"].strip()
    reward_model = sample.get("reward_model")
    if isinstance(reward_model, dict):
        value = reward_model.get("ground_truth")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _existing_response(sample: dict[str, Any], response_field: str | None = None) -> str:
    fields = [response_field] if response_field else []
    fields.extend(["response", "candidate_response", "model_output", "output", "solution_str"])
    for field in fields:
        if not field:
            continue
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _openai_url(api_base: str, endpoint: str) -> str:
    base = api_base.rstrip("/")
    endpoint = endpoint.lstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{endpoint}"
    return f"{base}/v1/{endpoint}"


def _needs_candidate_generation(rows: list[dict[str, Any]], args: argparse.Namespace) -> bool:
    if args.use_ground_truth:
        return False
    return any(not _existing_response(row, args.response_field) for row in rows)


async def _preflight_openai_api(
    *,
    name: str,
    api_base: str,
    api_key: str,
    model: str,
    timeout_s: float,
) -> None:
    if not api_base or not api_key or not model:
        missing = []
        if not api_base:
            missing.append("api_base")
        if not api_key:
            missing.append("api_key")
        if not model:
            missing.append("model")
        raise RuntimeError(f"{name} config missing: {', '.join(missing)}")

    url = _openai_url(api_base, "models")
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {text[:240]}")
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    return
                model_ids = []
                for item in data.get("data", []) if isinstance(data, dict) else []:
                    if isinstance(item, dict) and item.get("id"):
                        model_ids.append(str(item["id"]))
                if model_ids and model not in model_ids:
                    print(
                        f"[warn] {name} model `{model}` not listed by {url}; "
                        f"available={model_ids[:8]}",
                        file=sys.stderr,
                    )
    except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
        raise RuntimeError(f"{name} API preflight failed at {url}: {type(exc).__name__}: {exc}") from exc


async def _chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
    retries: int,
) -> dict[str, Any]:
    url = _openai_url(api_base, "chat/completions")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    fatal_statuses = {400, 401, 402, 403, 404, 422}
    attempts: list[str] = []
    total_attempts = max(1, retries)
    for attempt in range(total_attempts):
        attempt_no = attempt + 1
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        error = f"attempt {attempt_no}/{total_attempts} HTTP {resp.status}: {text[:240]}"
                        attempts.append(error)
                        if resp.status in fatal_statuses:
                            raise RuntimeError("; ".join(attempts))
                        if resp.status in retryable_statuses and attempt < total_attempts - 1:
                            await asyncio.sleep(min(20.0, 1.0 * (2**attempt)))
                            continue
                        raise RuntimeError("; ".join(attempts))
                    data = json.loads(text)
                    choice = data.get("choices", [{}])[0].get("message", {})
                    content = str(choice.get("content", "") or "").strip()
                    usage = data.get("usage") or {}
                    return {"content": content, "usage": usage, "raw": data}
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            attempts.append(f"attempt {attempt_no}/{total_attempts} {type(exc).__name__}: {exc}")
            if attempt < total_attempts - 1:
                await asyncio.sleep(min(20.0, 1.0 * (2**attempt)))
                continue
            raise RuntimeError("; ".join(attempts)) from exc
    raise RuntimeError("; ".join(attempts) or "candidate generation failed")


async def _process_one(
    *,
    sample: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        started = time.time()
        sample_id = _sample_id(sample, row_index)
        messages = _prompt_messages(sample)
        system_prompt = _system_prompt(sample, messages)
        question = _question(sample, messages)
        history_text = _history_text(sample, messages)

        generation_status = "existing_response"
        raw_response = ""
        response = _existing_response(sample, args.response_field)
        candidate_usage: dict[str, Any] = {}

        if args.use_ground_truth:
            response = _ground_truth(sample)
            generation_status = "ground_truth"
        elif not response:
            generation_status = "generated"
            if not args.candidate_api_base or not args.candidate_api_key or not args.candidate_model:
                generation_status = "missing_candidate_config"
                raw_response = generation_status
                if args.fail_on_generation_error:
                    raise RuntimeError(generation_status)
                response = ""
            else:
                try:
                    gen = await _chat_completion(
                        api_base=args.candidate_api_base,
                        api_key=args.candidate_api_key,
                        model=args.candidate_model,
                        messages=messages,
                        max_tokens=args.candidate_max_tokens,
                        temperature=args.candidate_temperature,
                        top_p=args.candidate_top_p,
                        timeout_s=args.candidate_timeout_s,
                        retries=args.retries,
                    )
                    response = gen["content"]
                    raw_response = response
                    candidate_usage = gen.get("usage") or {}
                except Exception as exc:  # noqa: BLE001
                    generation_status = f"generation_error:{type(exc).__name__}"
                    raw_response = str(exc)[:500]
                    if args.fail_on_generation_error:
                        raise RuntimeError(f"{generation_status}: {raw_response}") from exc
                    response = ""

        if not response and generation_status in {"generated", "existing_response", "ground_truth"}:
            generation_status = "empty_response"
            raw_response = raw_response or "empty response"
            if args.fail_on_generation_error and not args.allow_empty_response:
                raise RuntimeError(f"{generation_status}: id={sample_id}")

        score_info = await score_output_contact_rubric(
            question=question,
            output_answer=response,
            api_base=args.judge_api_base,
            api_key=args.judge_api_key,
            judge_model=args.judge_model,
            timeout_s=args.judge_timeout_s,
            rubric_path=args.rubric_path,
            hard_config_path=args.hard_config_path,
            history_text=history_text,
            system_prompt=system_prompt,
            judge_max_tokens=args.judge_max_tokens,
            score_max_retries=args.retries,
        )

        rubric_results = score_info.get("rubric_results", [])
        low_rubrics = [
            r
            for r in rubric_results
            if isinstance(r, dict) and r.get("triggered") and float(r.get("score") or 0.0) < 2.0
        ]
        elapsed = time.time() - started
        return {
            "id": sample_id,
            "row_index": row_index,
            "score": score_info.get("score"),
            "semantic_score": score_info.get("semantic_score"),
            "hard_penalty_total": score_info.get("hard_penalty_total"),
            "gate_failed": score_info.get("gate_failed", False),
            "gate_reason": score_info.get("gate_reason", ""),
            "model_judge_status": score_info.get("status"),
            "single_score_reason": score_info.get("reason", ""),
            "rubric_version": score_info.get("rubric_version"),
            "rubric_path": score_info.get("rubric_path"),
            "rubric_results": rubric_results,
            "low_rubrics": low_rubrics,
            "generation_status": generation_status,
            "candidate_model": args.candidate_model if not args.use_ground_truth else "ground_truth",
            "judge_model": args.judge_model,
            "question": question,
            "history_text": history_text if args.include_history else "",
            "response": response,
            "raw_response": raw_response,
            "candidate_usage": candidate_usage,
            "final_char_len": score_info.get("final_char_len"),
            "length_penalty": score_info.get("length_penalty"),
            "sep_penalty": score_info.get("sep_penalty"),
            "banned_term_penalty": score_info.get("banned_term_penalty"),
            "banned_term_hits": score_info.get("banned_term_hits", []),
            "elapsed_seconds": elapsed,
            "original_data": sample if args.include_original else {},
        }


def _score_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _is_generation_excluded(row: dict[str, Any]) -> bool:
    status = str(row.get("generation_status", ""))
    return status.startswith("generation_error") or status in {
        "missing_candidate_config",
        "empty_response",
    }


def _build_report(
    *,
    results: list[dict[str, Any]],
    output_dir: Path,
    input_path: Path,
    rubric_path: Path,
    args: argparse.Namespace,
    total_duration: float,
) -> str:
    scored_results = [r for r in results if not _is_generation_excluded(r)]
    excluded_results = [r for r in results if _is_generation_excluded(r)]
    scores = [_score_float(r, "score") for r in scored_results]
    semantic_scores = [_score_float(r, "semantic_score") for r in scored_results]
    hard_penalties = [_score_float(r, "hard_penalty_total") for r in scored_results]
    status_counts = Counter(str(r.get("model_judge_status", "")) for r in scored_results)
    generation_counts = Counter(str(r.get("generation_status", "")) for r in results)

    rubric_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "triggered": 0,
            "full_score": 0,
            "score_sum": 0.0,
            "deduction_sum": 0.0,
            "gate_failed": 0,
        }
    )
    failed_cases: list[dict[str, Any]] = []
    for row in scored_results:
        sample_failed = bool(row.get("gate_failed")) or _score_float(row, "score") < args.failed_score_threshold
        if sample_failed:
            failed_cases.append(
                {
                    "id": row.get("id"),
                    "score": row.get("score"),
                    "semantic_score": row.get("semantic_score"),
                    "hard_penalty_total": row.get("hard_penalty_total"),
                    "gate_failed": row.get("gate_failed"),
                    "question": row.get("question"),
                    "response": row.get("response"),
                    "low_rubrics": row.get("low_rubrics", []),
                    "reason": row.get("single_score_reason"),
                }
            )

        for result in row.get("rubric_results") or []:
            if not isinstance(result, dict):
                continue
            rid = str(result.get("id", "")).strip()
            if not rid:
                continue
            stat = rubric_stats[rid]
            stat["count"] += 1
            if result.get("triggered"):
                stat["triggered"] += 1
                score = float(result.get("score") or 0.0)
                stat["score_sum"] += score
                stat["deduction_sum"] += float(result.get("deduction") or 0.0)
                if score >= 2.0:
                    stat["full_score"] += 1
                if result.get("gate_failed"):
                    stat["gate_failed"] += 1
                stat["name_cn"] = result.get("name_cn") or stat.get("name_cn") or rid
                stat["rule_type"] = result.get("rule_type") or stat.get("rule_type") or ""

    _write_jsonl(output_dir / "failed_cases.jsonl", failed_cases)
    _write_jsonl(
        output_dir / "excluded_generation_errors.jsonl",
        [
            {
                "id": row.get("id"),
                "row_index": row.get("row_index"),
                "generation_status": row.get("generation_status"),
                "raw_response": row.get("raw_response"),
                "question": row.get("question"),
            }
            for row in excluded_results
        ],
    )

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    lines = [
        "# Contact Rubric Evaluation Report",
        f"- Input: `{input_path}`",
        f"- Rubric: `{rubric_path}`",
        f"- Candidate Model: `{args.candidate_model if not args.use_ground_truth else 'ground_truth'}`",
        f"- Judge Model: `{args.judge_model}`",
        f"- Total Samples: {len(results)}",
        f"- Scored Samples: {len(scored_results)}",
        f"- Excluded Generation Errors: {len(excluded_results)}",
        f"- Mean Score: {mean(scores):.4f}",
        f"- Median Score: {(statistics.median(scores) if scores else 0.0):.4f}",
        f"- Mean Semantic Score: {mean(semantic_scores):.4f}",
        f"- Mean Hard Penalty: {mean(hard_penalties):.4f}",
        f"- Perfect Rate (score >= 0.999): {(sum(1 for s in scores if s >= 0.999) / max(1, len(scores)) * 100):.2f}%",
        f"- Failed Case Rate (score < {args.failed_score_threshold} or gate failed): {(len(failed_cases) / max(1, len(scored_results)) * 100):.2f}%",
        f"- Total Duration: {total_duration:.2f}s",
        f"- Avg Seconds Per Scored Sample: {(total_duration / max(1, len(scored_results))):.2f}",
        "",
        "## Judge Status",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(["", "## Generation Status", "| Status | Count |", "|---|---:|"])
    for status, count in sorted(generation_counts.items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## Rubric Statistics",
            "| Rubric ID | Name | Type | Triggered | Mean Score | Full Score Rate | Mean Deduction | Gate Failed |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rid, stat in sorted(rubric_stats.items()):
        triggered = int(stat["triggered"])
        mean_score = stat["score_sum"] / max(1, triggered)
        full_rate = stat["full_score"] / max(1, triggered) * 100
        mean_deduction = stat["deduction_sum"] / max(1, triggered)
        lines.append(
            f"| {rid} | {stat.get('name_cn', rid)} | {stat.get('rule_type', '')} | "
            f"{triggered} | {mean_score:.3f} | {full_rate:.2f}% | {mean_deduction:.4f} | {stat['gate_failed']} |"
        )

    report = "\n".join(lines) + "\n"
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    return report


def _make_output_dir(args: argparse.Namespace, input_path: Path, rubric_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = _safe_name(args.candidate_model if not args.use_ground_truth else "ground_truth")
    stem = _safe_name(input_path.stem)
    rubric_stem = _safe_name(rubric_path.stem)
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = output_root / f"{candidate}_{stem}_{rubric_stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


async def _amain(args: argparse.Namespace) -> None:
    input_path = Path(args.input_file).expanduser().resolve()
    rubric_path = Path(args.rubric_path).expanduser().resolve()

    rows = _read_jsonl(input_path, args.limit if args.limit > 0 else None)
    if not rows:
        raise ValueError(f"No samples loaded from {input_path}")
    if not args.judge_api_base or not args.judge_api_key or not args.judge_model:
        raise ValueError("Judge config missing: set JUDGE_MODEL_NAME, JUDGE_API_BASE, and JUDGE_API_KEY")
    if _needs_candidate_generation(rows, args) and not args.skip_candidate_preflight:
        await _preflight_openai_api(
            name="candidate",
            api_base=args.candidate_api_base,
            api_key=args.candidate_api_key,
            model=args.candidate_model,
            timeout_s=min(args.candidate_timeout_s, 30),
        )

    output_dir = _make_output_dir(args, input_path, rubric_path)

    shutil.copy2(input_path, output_dir / input_path.name)
    shutil.copy2(rubric_path, output_dir / rubric_path.name)
    if args.hard_config_path and Path(args.hard_config_path).exists():
        shutil.copy2(args.hard_config_path, output_dir / Path(args.hard_config_path).name)

    config = vars(args).copy()
    for secret_key in ("candidate_api_key", "judge_api_key"):
        if config.get(secret_key):
            config[secret_key] = "***"
    config.update({"input_file": str(input_path), "rubric_path": str(rubric_path), "output_dir": str(output_dir)})
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    started = time.time()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        _process_one(sample=sample, row_index=i, args=args, semaphore=semaphore)
        for i, sample in enumerate(rows)
    ]
    results: list[dict[str, Any]] = []
    for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        results.append(result)
        if idx % max(1, args.log_every) == 0 or idx == len(tasks):
            print(f"[progress] {idx}/{len(tasks)} done")

    total_duration = time.time() - started
    results.sort(key=lambda r: (_score_float(r, "score"), str(r.get("id", ""))), reverse=True)
    _write_jsonl(output_dir / "evaluation_results.jsonl", results)
    _build_report(
        results=results,
        output_dir=output_dir,
        input_path=input_path,
        rubric_path=rubric_path,
        args=args,
        total_duration=total_duration,
    )

    print(f"[ok] output_dir={output_dir}")
    print(f"[ok] results={output_dir / 'evaluation_results.jsonl'}")
    print(f"[ok] report={output_dir / 'evaluation_report.md'}")
    print(f"[ok] failed_cases={output_dir / 'failed_cases.jsonl'}")
    print(f"[ok] excluded_generation_errors={output_dir / 'excluded_generation_errors.jsonl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ITBench-style contact rubric benchmark evaluator.")
    parser.add_argument("--input-file", default=os.getenv("INPUT_FILE", str(DEFAULT_INPUT)))
    parser.add_argument("--rubric-path", default=os.getenv("RUBRIC_PATH", str(DEFAULT_RUBRIC)))
    parser.add_argument("--hard-config-path", default=os.getenv("HARD_CONFIG_PATH", str(DEFAULT_HARD_CONFIG)))
    parser.add_argument("--output-root", default=os.getenv("OUTPUT", str(DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--limit", type=int, default=int(os.getenv("EVALUATION_LIMIT", "0") or 0))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", "8") or 8))
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--retries", type=int, default=int(os.getenv("EVAL_RETRIES", "4") or 4))
    parser.add_argument("--response-field", default=os.getenv("RESPONSE_FIELD", ""))
    parser.add_argument("--use-ground-truth", action="store_true")
    parser.add_argument(
        "--allow-generation-errors",
        dest="fail_on_generation_error",
        action="store_false",
        help="Keep legacy behavior: score empty responses when candidate generation fails.",
    )
    parser.add_argument("--skip-candidate-preflight", action="store_true")
    parser.add_argument("--allow-empty-response", action="store_true")
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--failed-score-threshold", type=float, default=0.8)

    parser.add_argument("--candidate-model", default=os.getenv("CANDIDATE_MODEL_NAME", ""))
    parser.add_argument("--candidate-api-base", default=os.getenv("CANDIDATE_API_BASE", ""))
    parser.add_argument("--candidate-api-key", default=os.getenv("CANDIDATE_API_KEY", ""))
    parser.add_argument("--candidate-max-tokens", type=int, default=int(os.getenv("CANDIDATE_MAX_OUTPUT_TOKENS", "2048") or 2048))
    parser.add_argument("--candidate-temperature", type=float, default=float(os.getenv("CANDIDATE_TEMPERATURE", "0.6") or 0.6))
    parser.add_argument("--candidate-top-p", type=float, default=float(os.getenv("CANDIDATE_TOP_P", "0.95") or 0.95))
    parser.add_argument("--candidate-timeout-s", type=float, default=float(os.getenv("CANDIDATE_TIMEOUT_S", "180") or 180))

    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME", os.getenv("JUDGE_MODEL", "qwen-max")))
    parser.add_argument("--judge-api-base", default=os.getenv("JUDGE_API_BASE", os.getenv("API_BASE", "")))
    parser.add_argument("--judge-api-key", default=os.getenv("JUDGE_API_KEY", os.getenv("API_KEY", "")))
    parser.add_argument("--judge-max-tokens", type=int, default=int(os.getenv("JUDGE_MAX_TOKENS", "1600") or 1600))
    parser.add_argument("--judge-timeout-s", type=float, default=float(os.getenv("JUDGE_TIMEOUT_S", "45") or 45))
    parser.set_defaults(fail_on_generation_error=os.getenv("FAIL_ON_GENERATION_ERROR", "1") != "0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
