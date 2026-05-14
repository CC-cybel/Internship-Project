#!/usr/bin/env python3
"""
测试 v6 奖励函数的 LLM-as-judge 逻辑。
由于实际调用需要 API，这里只验证返回格式正确。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "verl"))

from recipe.single_turn_reward.v6.reward_model_stage6_contact import (
    _extract_final_text,
    _normalize_turn,
    _history_text,
)


# ─── 单元测试 ───────────────────────────────────────────────

def test_extract_final_text():
    text = "BEGIN_META\naction=xxx\nEND_META\nBEGIN_FINAL\n这是最终回复\nEND_FINAL"
    assert _extract_final_text(text) == "这是最终回复"
    # 无标签时返回空字符串（让 judge 看到空字符串从而判断无联系方式）
    assert _extract_final_text("无标签回复") == ""
    print("✓ _extract_final_text")


def test_normalize_turn():
    assert _normalize_turn({"role": "user", "content": "你好"}) == ("user", "你好")
    assert _normalize_turn({"from": "human", "value": "你好"}) == ("user", "你好")
    assert _normalize_turn({"role": "assistant", "content": "你好"}) == ("assistant", "你好")
    assert _normalize_turn({"role": "bot", "content": "你好"}) == ("assistant", "你好")
    assert _normalize_turn({"role": "unknown", "content": "你好"}) is None
    assert _normalize_turn(None) is None
    print("✓ _normalize_turn")


def test_history_text():
    extra_info = {
        "conversations": [
            {"role": "user", "content": "我失眠"},
            {"role": "assistant", "content": "多久了？"},
            {"role": "user", "content": "我今年48岁"},
            {"role": "assistant", "content": "好的"},
        ]
    }
    text = _history_text(extra_info)
    assert "48" in text
    assert "失眠" in text
    print("✓ _history_text")


# ─── API mock 测试 ──────────────────────────────────────────

async def test_compute_score_skips_without_api():
    from recipe.single_turn_reward.v6.reward_function_stage6_contact import compute_score

    extra_info = {
        "question": "测试问题",
        "turn_round": 5,
        "rule_contact_round": 5,
        "conversations": [
            {"role": "user", "content": "我今年48岁"},
        ],
        "original_system_prompt": "角色设定：你是一名医疗咨询专家。\n[分龄定向留联策略]：若用户年龄大于35岁，优先索要电话。",
    }

    result = await compute_score(
        data_source="test",
        solution_str="加个微信吧",
        ground_truth="",
        extra_info=extra_info,
        # 不传 api_base/key → 走 skipped 分支
    )
    assert "score" in result
    assert "model_judge_status" in result
    assert "model_judge_score" in result
    assert "rule_score" in result
    print(f"✓ compute_score (no-api) → score={result['score']}, status={result['model_judge_status']}")


async def test_compute_score_format_on_train_data():
    import pandas as pd
    from recipe.single_turn_reward.v6.reward_function_stage6_contact import compute_score

    df = pd.read_parquet(
        "/data/chengch/project/rl_remake/outputs/"
        "single_turn_rl_contact_stage_2k_qwen3_8b_normal_mid_short_step350/"
        "single_turn_rl_contact_stage.train.parquet"
    )

    # 只测前3条（无API，跳过模型评分，验证格式）
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        extra_info = row.get("extra_info")
        if extra_info is None:
            extra_info = {}

        # prompt 最后一轮是 assistant 的回复（作为 solution）
        prompt = row.get("prompt")
        solution_str = ""
        if isinstance(prompt, list):
            # 取最后一个 assistant 的 content
            for item in reversed(prompt):
                if isinstance(item, dict) and item.get("role") == "assistant":
                    solution_str = item.get("content", "")
                    break

        result = await compute_score(
            data_source=str(row.get("data_source", "")),
            solution_str=solution_str,
            ground_truth=str(row.get("ground_truth", "")),
            extra_info=extra_info,
        )
        assert "score" in result
        assert "model_judge_status" in result
        print(f"  [{i}] turn={extra_info.get('turn_round')} contact_round={extra_info.get('rule_contact_round',extra_info.get('contact_round'))} → score={result['score']} status={result['model_judge_status']}")

    print(f"✓ compute_score on {min(3,len(df))} train samples (format only)")


def main():
    print("=== 测试 v6 奖励函数 ===\n")
    test_extract_final_text()
    test_normalize_turn()
    test_history_text()
    asyncio.run(test_compute_score_skips_without_api())
    asyncio.run(test_compute_score_format_on_train_data())
    print("\n✅ 全部测试通过！")


if __name__ == "__main__":
    main()
