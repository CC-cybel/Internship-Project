#!/usr/bin/env python3
"""
测试脚本：验证 reward_model_stage5_mid_cloud.py 的基本功能
"""

import asyncio
import json
from pathlib import Path

# 导入待测试的模块
import sys
sys.path.insert(0, str(Path(__file__).parent))

from reward_model_stage5_mid_cloud import (
    _clip,
    _to_float,
    _extract_json_dict,
    _extract_final_text,
    _final_char_len,
    _length_penalty,
    _load_bench_rules,
    _to_compact_rule_text,
)


def test_helper_functions():
    """测试辅助函数"""
    print("=== 测试辅助函数 ===")

    # 测试 _clip
    assert _clip(0.5) == 0.5
    assert _clip(-0.5) == 0.0
    assert _clip(1.5) == 1.0
    print("✓ _clip 函数正常")

    # 测试 _to_float
    assert _to_float("0.5", 0.0) == 0.5
    assert _to_float("invalid", 0.5) == 0.5
    assert _to_float(None, 0.5) == 0.5
    print("✓ _to_float 函数正常")

    # 测试 _extract_json_dict
    test_json = '{"score": 0.8, "reason": "测试"}'
    result = _extract_json_dict(test_json)
    assert result is not None
    assert result["score"] == 0.8
    print("✓ _extract_json_dict 函数正常")

    # 测试 _extract_final_text
    answer_with_final = "BEGIN_META\nxxx\nEND_META\nBEGIN_FINAL\n这是最终回复\nEND_FINAL"
    final = _extract_final_text(answer_with_final)
    assert final == "这是最终回复"
    print("✓ _extract_final_text 函数正常")

    # 测试 _final_char_len
    assert _final_char_len(answer_with_final) == 5  # "这是最终回复" 字符数
    print("✓ _final_char_len 函数正常")

    # 测试 _length_penalty
    assert _length_penalty(100) == 0.0  # 在理想范围内
    assert _length_penalty(30) > 0  # 太短，应有惩罚
    assert _length_penalty(300) > 0  # 太长，应有惩罚
    print("✓ _length_penalty 函数正常")

    # 测试 bench 规则加载
    rules = _load_bench_rules()
    print(f"✓ 加载了 {len(rules)} 条 bench 规则")

    compact_text = _to_compact_rule_text()
    assert len(compact_text) > 0
    print(f"✓ 生成的 compact rules 文本长度: {len(compact_text)}")


async def test_stage_validation():
    """测试阶段判断逻辑"""
    print("\n=== 测试阶段判断逻辑 ===")

    from reward_model_stage5_mid_cloud import score_output_mid_stage

    # 模拟参数
    base_params = {
        "question": "测试问题",
        "output_answer": "BEGIN_FINAL\n测试回复\nEND_FINAL",
        "api_base": "http://localhost:8000",
        "api_key": "test_key",
        "judge_model": "test_model",
    }

    # 测试：start 阶段（turn_round=1）应该被跳过
    result = await score_output_mid_stage(
        **base_params,
        turn_round=1,
        contact_round=0,
    )
    assert result["status"] == "skipped_wrong_stage"
    assert result["stage"] == "start"
    print("✓ start 阶段正确跳过")

    # 测试：contact 阶段（turn_round=8, contact_round=8）应该被跳过
    result = await score_output_mid_stage(
        **base_params,
        turn_round=8,
        contact_round=8,
    )
    assert result["status"] == "skipped_wrong_stage"
    assert result["stage"] == "contact"
    print("✓ contact 阶段正确跳过")

    # 测试：mid 阶段（turn_round=5, contact_round=8）应该正常评分
    # 注意：因为没有真实的API，会返回 parse_failed 或 error，但不应被跳过
    result = await score_output_mid_stage(
        **base_params,
        turn_round=5,
        contact_round=8,
    )
    # 由于没有真实API，预期会失败，但不应该是 skipped_wrong_stage
    assert result["status"] != "skipped_wrong_stage"
    print(f"✓ mid 阶段进入评分逻辑（status: {result['status']}）")


def main():
    print("开始测试 reward_model_stage5_mid_cloud.py\n")

    try:
        test_helper_functions()
        asyncio.run(test_stage_validation())
        print("\n✅ 所有测试通过！")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
