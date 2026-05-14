from __future__ import annotations

from typing import Any

from recipe.single_turn_reward.v1.reward_rules_first2 import compute_rule_components


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _question(extra_info: dict[str, Any] | None, ground_truth: str) -> str:
    extra_info = extra_info or {}
    for key in ("question", "instruction", "query"):
        v = extra_info.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return (ground_truth or "").strip()


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
    del reward_router_address
    del reward_model_tokenizer
    del kwargs

    q = _question(extra_info, ground_truth)
    a = solution_str or ""

    rule = compute_rule_components(question=q, answer=a, extra_info=extra_info)
    rule_score = _clip(float(rule.get("score", 0.0)))

    return {
        "score": rule_score,
        "rule_score": rule_score,
        "model_judge_score": 0.0,
        "model_judge_status": "disabled_rule_only",
        "rule_weight": 1.0,
        "model_weight": 0.0,
        **{k: v for k, v in rule.items() if k != "score"},
    }
