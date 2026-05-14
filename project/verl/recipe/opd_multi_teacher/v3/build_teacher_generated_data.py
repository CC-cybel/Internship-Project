#!/usr/bin/env python3
"""Build teacher-generated data for v3 reverse-KL/top-k distillation.

Input rows should follow the verl RL dataset schema and carry either a top-level
`teacher_route` field or `extra_info.teacher_route`. The script generates a
teacher response with the route-specific teacher model and writes a JSONL file
whose rows include `teacher_response` and `agent_name=teacher_forced_agent`.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_INPUT = "/data/chengch/project/verl/recipe/opd_multi_teacher/data/opd_multi_teacher_10k.train.jsonl"
DEFAULT_OUTPUT_DIR = "/data/chengch/project/verl/recipe/opd_multi_teacher/v3/data"
DEFAULT_CONTACT_TEACHER = "/data1/chengch/models/qwen3_8b_contact_step200"
DEFAULT_MID_TEACHER = "/data1/chengch/models/qwen3_8b_normal_mid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="opd_multi_teacher_v3_teacher_generated.train.jsonl")
    parser.add_argument("--contact-teacher", default=DEFAULT_CONTACT_TEACHER)
    parser.add_argument("--mid-teacher", default=DEFAULT_MID_TEACHER)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-prompt-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def load_rows(path: str, max_samples: int) -> list[dict[str, Any]]:
    if path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=path)["train"]
    elif path.endswith(".json") or path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=path)["train"]
    else:
        raise ValueError(f"Unsupported file format: {path}")
    rows = [dict(row) for row in dataset]
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def get_route(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    route = row.get("teacher_route") or extra_info.get("teacher_route")
    if not route:
        raise ValueError(f"Missing teacher_route in row index={extra_info.get('index')}")
    return str(route)


def build_prompt_text(tokenizer, row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError("Expected row['prompt'] to be a list of chat messages.")
    return tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)


def build_prompt_token_ids(tokenizer, row: dict[str, Any]) -> list[int]:
    prompt_text = build_prompt_text(tokenizer, row)
    return tokenizer(prompt_text, add_special_tokens=False).input_ids


def normalize_row(row: dict[str, Any], teacher_response: str, route: str) -> dict[str, Any]:
    item = dict(row)
    extra_info = dict(item.get("extra_info") or {})
    extra_info["teacher_route"] = route
    extra_info["teacher_response"] = teacher_response
    item["extra_info"] = extra_info
    item["teacher_route"] = route
    item["teacher_response"] = teacher_response
    item["agent_name"] = "teacher_forced_agent"
    reward_model = dict(item.get("reward_model") or {})
    reward_model["ground_truth"] = teacher_response
    item["reward_model"] = reward_model
    return item


def generate_for_route(rows: list[dict[str, Any]], route: str, model_path: str, args: argparse.Namespace) -> list[str | None]:
    from vllm import LLM, SamplingParams

    print(f"[load] route={route} teacher={model_path} rows={len(rows)}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    max_prompt_tokens = args.max_prompt_tokens or (args.max_model_len - args.max_new_tokens)
    max_prompt_tokens = min(max_prompt_tokens, args.max_model_len - 1)
    if max_prompt_tokens <= 0:
        raise ValueError(
            f"Invalid prompt budget: max_prompt_tokens={max_prompt_tokens}, "
            f"max_model_len={args.max_model_len}, max_new_tokens={args.max_new_tokens}"
        )
    prompts: list[dict[str, list[int]]] = []
    prompt_positions: list[int] = []
    responses: list[str | None] = [None] * len(rows)
    dropped = 0
    longest_prompt = 0
    for row_idx, row in enumerate(rows):
        prompt_token_ids = build_prompt_token_ids(tokenizer, row)
        longest_prompt = max(longest_prompt, len(prompt_token_ids))
        if len(prompt_token_ids) > max_prompt_tokens:
            dropped += 1
            continue
        prompts.append({"prompt_token_ids": prompt_token_ids})
        prompt_positions.append(row_idx)
    if dropped:
        print(
            f"[drop] route={route} overlong_prompts={dropped}/{len(rows)} "
            f"longest={longest_prompt} max_prompt_tokens={max_prompt_tokens}"
        )
    if not prompts:
        print(f"[skip] route={route} no prompts left after filtering")
        return responses
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=False,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        for offset, output in enumerate(outputs):
            responses[prompt_positions[start + offset]] = output.outputs[0].text.strip()
        print(f"[generate] route={route} {min(start + args.batch_size, len(prompts))}/{len(prompts)}")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return responses


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    by_route: dict[str, list[tuple[int, dict[str, Any]]]] = {"contact": [], "mid": []}
    generated_rows: list[dict[str, Any] | None] = [None] * len(rows)
    for idx, row in enumerate(rows):
        route = get_route(row)
        if args.reuse_existing and row.get("teacher_response"):
            generated_rows[idx] = normalize_row(row, str(row["teacher_response"]), route)
            continue
        if route not in by_route:
            raise ValueError(f"Unknown teacher_route={route!r}; expected contact or mid.")
        by_route[route].append((idx, row))

    route_to_model = {
        "contact": args.contact_teacher,
        "mid": args.mid_teacher,
    }
    for route, indexed_rows in by_route.items():
        if not indexed_rows:
            continue
        responses = generate_for_route([row for _, row in indexed_rows], route, route_to_model[route], args)
        for (idx, row), response in zip(indexed_rows, responses, strict=True):
            if response is None:
                continue
            generated_rows[idx] = normalize_row(row, response, route)

    written = 0
    dropped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in generated_rows:
            if row is None:
                dropped += 1
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(f"[OK] wrote {output_path} rows={written} dropped={dropped}")


if __name__ == "__main__":
    main()
