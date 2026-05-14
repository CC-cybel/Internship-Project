#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


@dataclass
class JudgeResult:
    winner: str
    score_a: float
    score_b: float
    reason: str
    stage: str
    raw: str
    status: str


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


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

    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


async def _chat(
    session: aiohttp.ClientSession,
    api_base: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=text[:400],
                headers=resp.headers,
            )
        return json.loads(text)


class RateLimiter:
    def __init__(self, min_interval_s: float):
        self.min_interval_s = max(0.0, min_interval_s)
        self._lock = asyncio.Lock()
        self._next_ts = 0.0

    async def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            target = max(now, self._next_ts)
            self._next_ts = target + self.min_interval_s
        sleep_s = target - now
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)


def _build_prompt(bench_rules: list[dict[str, Any]], input_text: str, output_text: str, gts_text: str) -> str:
    bench_compact = json.dumps(bench_rules, ensure_ascii=False)
    return (
        "你是严格的A/B评测裁判。\n"
        "任务：基于规则判断两个候选回复哪个更好。\n"
        "A=模型output，B=gts。\n\n"
        "评分规则来源（JSON）：\n"
        f"{bench_compact}\n\n"
        "评测要求：\n"
        "1) 严格以规则为准，关注问诊逻辑、推进性、套电时机与话术、拒绝处理、专业度。\n"
        "2) 给出 A 和 B 的0-1分，允许并列。\n"
        "3) winner 只能是 A/B/tie。\n"
        "4) reason <= 60字，中文。\n"
        "5) 输出必须是 JSON，不要额外文本。\n\n"
        "输出schema:\n"
        "{\n"
        '  "winner": "A|B|tie",\n'
        '  "score_a": 0-1,\n'
        '  "score_b": 0-1,\n'
        '  "stage": "start|mid|contact|unknown",\n'
        '  "reason": "..."\n'
        "}\n\n"
        "以下是待评测样本：\n"
        "[完整上下文input]\n"
        f"{input_text}\n\n"
        "[候选A=output]\n"
        f"{output_text}\n\n"
        "[候选B=gts]\n"
        f"{gts_text}\n"
    )


async def _judge_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    limiter: RateLimiter,
    *,
    api_base: str,
    api_key: str,
    model: str,
    timeout_s: float,
    max_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
    bench_rules: list[dict[str, Any]],
    input_text: str,
    output_text: str,
    gts_text: str,
) -> JudgeResult:
    prompt = _build_prompt(bench_rules, input_text, output_text, gts_text)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 280,
    }

    async with semaphore:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            await limiter.wait()
            try:
                resp = await _chat(session, api_base, api_key, payload, timeout_s)
                text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                obj = _extract_json_dict(text)
                if obj is None:
                    return JudgeResult("tie", 0.5, 0.5, "解析失败", "unknown", text, "parse_failed")

                winner = str(obj.get("winner", "tie")).strip().lower()
                if winner not in {"a", "b", "tie"}:
                    winner = "tie"

                try:
                    score_a = _clip(float(obj.get("score_a", 0.5)))
                except Exception:
                    score_a = 0.5
                try:
                    score_b = _clip(float(obj.get("score_b", 0.5)))
                except Exception:
                    score_b = 0.5

                stage = str(obj.get("stage", "unknown")).strip().lower()
                if stage not in {"start", "mid", "contact", "unknown"}:
                    stage = "unknown"

                reason = str(obj.get("reason", "")).strip()[:120]
                return JudgeResult(winner, score_a, score_b, reason, stage, text, "ok")
            except aiohttp.ClientResponseError as exc:
                last_exc = exc
                if exc.status in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < max_retries - 1:
                    retry_after = None
                    if exc.headers is not None:
                        ra = exc.headers.get("Retry-After")
                        if ra is not None:
                            try:
                                retry_after = float(ra)
                            except Exception:
                                retry_after = None
                    if retry_after is None:
                        retry_after = min(backoff_max_s, backoff_base_s * (2**attempt))
                    retry_after += random.uniform(0.0, 0.4)
                    await asyncio.sleep(retry_after)
                    continue
                return JudgeResult("tie", 0.5, 0.5, f"HTTP{exc.status}", "unknown", "", f"http_error:{exc.status}")
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    sleep_s = min(backoff_max_s, backoff_base_s * (2**attempt)) + random.uniform(0.0, 0.3)
                    await asyncio.sleep(sleep_s)
                    continue
                return JudgeResult("tie", 0.5, 0.5, type(exc).__name__, "unknown", "", f"error:{type(exc).__name__}")

    if last_exc is not None:
        return JudgeResult("tie", 0.5, 0.5, type(last_exc).__name__, "unknown", "", f"error:{type(last_exc).__name__}")
    return JudgeResult("tie", 0.5, 0.5, "unknown", "unknown", "", "unknown")


async def main_async(args: argparse.Namespace) -> int:
    data = _load_json(Path(args.input_file))
    if not isinstance(data, list):
        raise ValueError("input_file must be a JSON array")

    bench = _load_json(Path(args.bench_file))
    if not isinstance(bench, list):
        raise ValueError("bench_file must be a JSON array")

    api_key = args.api_key or os.environ.get("TONGYI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Use --api-key or env TONGYI_API_KEY/DASHSCOPE_API_KEY")

    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))
    limiter = RateLimiter(args.min_interval_s)

    selected = data[: args.max_samples] if args.max_samples > 0 else data

    results: list[dict[str, Any]] = []
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_summary() -> dict[str, Any]:
        wins_a = sum(1 for r in results if r.get("winner") == "a")
        wins_b = sum(1 for r in results if r.get("winner") == "b")
        ties = sum(1 for r in results if r.get("winner") == "tie")
        total = len(results)
        sum_a = sum(float(r.get("score_a_output", 0.5)) for r in results)
        sum_b = sum(float(r.get("score_b_gts", 0.5)) for r in results)
        return {
            "total": total,
            "wins_output_a": wins_a,
            "wins_gts_b": wins_b,
            "ties": ties,
            "avg_score_output_a": (sum_a / total) if total else 0.0,
            "avg_score_gts_b": (sum_b / total) if total else 0.0,
            "model": args.model,
            "api_base": args.api_base,
        }

    def _dump_progress(status: str) -> None:
        payload = {
            "status": status,
            "summary": _build_summary(),
            "results": results,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    _dump_progress("running")

    async with aiohttp.ClientSession() as session:
        tasks: list[asyncio.Task[tuple[int, dict[str, Any], JudgeResult]]] = []
        for i, item in enumerate(selected):
            if not isinstance(item, dict):
                continue
            input_text = str(item.get("input") or item.get("question") or "")
            output_text = str(item.get("output", ""))
            gts_text = str(item.get("gts", ""))

            async def _run_one(
                idx: int,
                src: dict[str, Any],
                input_s: str,
                output_s: str,
                gts_s: str,
            ) -> tuple[int, dict[str, Any], JudgeResult]:
                jr = await _judge_one(
                    session,
                    semaphore,
                    limiter,
                    api_base=args.api_base,
                    api_key=api_key,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    max_retries=args.max_retries,
                    backoff_base_s=args.backoff_base_s,
                    backoff_max_s=args.backoff_max_s,
                    bench_rules=bench,
                    input_text=input_s,
                    output_text=output_s,
                    gts_text=gts_s,
                )
                return idx, src, jr

            tasks.append(asyncio.create_task(_run_one(i, item, input_text, output_text, gts_text)))

        for done in asyncio.as_completed(tasks):
            idx, src, jr = await done
            results.append(
                {
                    "index": idx,
                    "step": src.get("step"),
                    "winner": jr.winner,
                    "score_a_output": jr.score_a,
                    "score_b_gts": jr.score_b,
                    "stage": jr.stage,
                    "reason": jr.reason,
                    "status": jr.status,
                }
            )
            _dump_progress("running")

    results.sort(key=lambda x: int(x.get("index", 0)))
    _dump_progress("done")

    summary = _build_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare output vs gts with qwen-max under bench rules")
    p.add_argument("--input-file", required=True, help="Path to validation_rollouts json")
    p.add_argument(
        "--bench-file",
        default="/data/chengch/project/verl/recipe/single_turn_reward/bench_excellent.json",
        help="Path to bench_excellent.json",
    )
    p.add_argument("--output-file", required=True, help="Path to save evaluation json")
    p.add_argument(
        "--api-base",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible API base",
    )
    p.add_argument("--api-key", default=None, help="API key (or use TONGYI_API_KEY/DASHSCOPE_API_KEY)")
    p.add_argument("--model", default="qwen-max", help="Judge model")
    p.add_argument("--max-samples", type=int, default=0, help="0 means all")
    p.add_argument("--max-concurrency", type=int, default=2)
    p.add_argument("--min-interval-s", type=float, default=0.3)
    p.add_argument("--timeout-s", type=float, default=50.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--backoff-base-s", type=float, default=1.0)
    p.add_argument("--backoff-max-s", type=float, default=20.0)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
