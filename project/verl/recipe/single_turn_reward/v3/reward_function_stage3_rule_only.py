from __future__ import annotations

from typing import Any


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    reward_router_address: str | None = None,
    reward_model_tokenizer: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    del data_source
    del solution_str
    del ground_truth
    del extra_info
    del reward_router_address
    del reward_model_tokenizer
    del kwargs

    # v3 is AB-model-judge centric; rule-only fallback returns neutral score.
    score = _clip(0.5)
    return {
        "score": score,
        "rule_score": score,
        "model_judge_score": 0.0,
        "model_judge_status": "disabled_rule_only",
        "rule_weight": 1.0,
        "model_weight": 0.0,
    }
