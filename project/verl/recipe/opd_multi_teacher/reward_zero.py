"""Neutral reward for pure OPD runs.

The distillation loss is the training signal. This reward keeps the PPO/GRPO
pipeline valid without calling an external judge.
"""

from __future__ import annotations

import json
from pathlib import Path


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _append_jsonl(path: str, record: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_score(*args, **kwargs) -> dict[str, object]:
    extra_info = kwargs.get("extra_info") or {}
    teacher_route = extra_info.get("teacher_route", "unknown")

    if _as_bool(kwargs.get("collect_genrm_io")):
        record = {
            "kind": "opd_rollout",
            "teacher_route": teacher_route,
            "data_source": kwargs.get("data_source"),
            "solution_str": kwargs.get("solution_str"),
            "ground_truth": kwargs.get("ground_truth"),
        }
        if _as_bool(kwargs.get("genrm_io_include_extra_info")):
            record["extra_info"] = extra_info
        _append_jsonl(kwargs.get("genrm_io_path") or "genrm_io.jsonl", record)

    return {
        "score": 0.0,
        "teacher_route": teacher_route,
    }
