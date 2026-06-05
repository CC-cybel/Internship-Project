#!/usr/bin/env python3
"""Rewrite contact-age RL JSONL data for Claude distillation.

The transform keeps the original row shape, but:
- replaces the system prompt slot schema module with the lightweight user model
  and new Slot Schema;
- rewrites the conversion excuse line;
- strips assistant targets from BEGIN_META/BEGIN_FINAL format down to the
  user-visible BEGIN_FINAL text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NEW_USER_MODEL_AND_SCHEMA = """

[轻量用户模型]
每轮回复在 thought 的【分析】中必须更新精简用户模型，字段名和顺序固定为：留联分层、用户状态、生理层、心理层、战术层，不能省略。
1. 留联分层：必须按固定结构输出 user_type=...；core_need=...；conversion_barrier=...；lead_strategy=...；fine_label=...。
 - user_type 只能从 [青少年本人, 成人本人, 家长代询, 子女代询, 伴侣代询, 其他家属代询, 朋友代询, 未知] 中选择。
 - core_need 只能从 [病情判断, 治疗方案, 既往治疗不满, 用药安全, 就医路径, 就医决策顾虑, 情绪倾诉, 危机求助, 家属照护无力, 其他] 中选择。
 - conversion_barrier 只能从 [医学认知不足, 路径不清, 费用顾虑, 信任顾虑, 效果顾虑, 用药顾虑, 隐私病耻, 患者不配合, 家庭沟通失效, 时间紧迫, 情绪承载不足, 危机安全风险, 暂无明显障碍, 信息不足] 中选择。
 - lead_strategy 只能从 [低压保密留联, 危机安全回电, 专家评估留联, 家属指导留联, 二次方案评估, 用药风险核对, 到院路径预约, 费用透明解释, 正规资质背书, 情绪承接转评估, 科普判断转留联, 暂不留联继续问诊] 中选择。
 - fine_label 格式固定为 user_type-core_need-conversion_barrier-lead_strategy。
 - user_type/core_need/conversion_barrier 根据当前轮和历史信息填充；若暂无信息或无法推测，必须使用枚举内兜底值：user_type=未知；core_need=其他；conversion_barrier=信息不足。***严禁严禁使用任何未来信息***。lead_strategy 可在首次留联轮次的前一轮或信息足够时填充，若尚不到留联铺垫时机可写 lead_strategy=暂不留联继续问诊。
 - 留联分层一旦写定，后续轮次除非出现严重证据错误或危机风险升级，否则不得随意更改，以保证分层稳定。
2. 用户状态：平静/犹豫/害怕/不信任/对抗/急迫/配合/敷衍/未知，必须选择一个值并说明依据。
3. 生理层：从 [皮质醇, 多巴胺, 内啡肽, 催产素, 血清素] 中选择本轮调控目标，写出激素名↑或激素名↓，并说明理由。
4. 心理层：从 [自主感, 胜任感, 归属感, 优越感, 安全感, 认知节能, 损失厌恶] 中选择本轮满足或利用的心理，并说明理由。
5. 战术层：从 [医学定性, 提问引导, 示弱反衬, 降维打击, 认知重构, 权威借势, 利益诱惑] 中选择本轮战术，并说明具体做法。

[原子化槽位表 Slot Schema]
- age: 患者年龄，输出具体值；未知写“未知”
- gender: 患者性别，输出男/女/未知
- name: 患者称呼或姓名；没有写“暂无”
- relationship: 咨询者与患者关系，本人/母亲/父亲/伴侣/子女/朋友/其他家属/未知
联系方式，互斥必填 Any One：
- phone: 手机号或座机；未获取写“未获取”
- wechat: 微信号；未获取写“未获取”
- symptom: 主诉症状或核心困扰；未知写“未知”
- duration: 病程时长；没有写“暂无”
- medical_history: 既往史、诊断史、用药史、检查史；没有写“暂无”
- medical_awareness: 医学认知水平，未知/小白/半懂/专业/误区明显
""".strip("\n")

NEW_CONVERSION_EXCUSE = "转化借口：根据用户模型中的留联策略，针对用户核心诉求制定留联话术。"

SCHEMA_BLOCK_RE = re.compile(
    r"\n\n(?:\[?原子化槽位表[^\n]*\]?[^\n]*：?)[\s\S]*?(?=\n\n(?:硬性执行指标|\[硬性执行指标))"
)
CONVERSION_EXCUSE_RE = re.compile(r"(?m)^(?P<prefix>\s*-?\s*)转化(?:借口|理由)：[^\n]*")
FINAL_RE = re.compile(r"^\s*BEGIN_META\n[\s\S]*?\nEND_META\nBEGIN_FINAL\n(?P<final>[\s\S]*?)\nEND_FINAL\s*$")


def rewrite_system_prompt(text: str) -> str:
    rewritten, count = SCHEMA_BLOCK_RE.subn("\n\n" + NEW_USER_MODEL_AND_SCHEMA, text, count=1)
    if count == 0:
        raise ValueError("system prompt schema block not found")
    rewritten = CONVERSION_EXCUSE_RE.sub(lambda m: f"{m.group('prefix')}{NEW_CONVERSION_EXCUSE}", rewritten)
    return rewritten


def extract_final(text: str) -> str:
    match = FINAL_RE.match(text or "")
    if not match:
        return text
    return match.group("final").strip()


def rewrite_messages(messages: Any, stats: dict[str, int]) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system" and isinstance(content, str):
            message["content"] = rewrite_system_prompt(content)
            stats["system_prompts"] += 1
        elif role == "assistant" and isinstance(content, str):
            new_content = extract_final(content)
            if new_content != content:
                stats["assistant_contents"] += 1
            message["content"] = new_content


def rewrite_row(row: dict[str, Any], stats: dict[str, int]) -> dict[str, Any]:
    rewrite_messages(row.get("prompt"), stats)

    if isinstance(row.get("ground_truth"), str):
        new_ground_truth = extract_final(row["ground_truth"])
        if new_ground_truth != row["ground_truth"]:
            stats["ground_truth"] += 1
        row["ground_truth"] = new_ground_truth

    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and isinstance(reward_model.get("ground_truth"), str):
        reward_model["ground_truth"] = extract_final(reward_model["ground_truth"])

    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in ("original_system_prompt", "transformed_system_prompt"):
            if isinstance(extra.get(key), str):
                extra[key] = rewrite_system_prompt(extra[key])
        rewrite_messages(extra.get("conversations"), stats)
        rule = extra.get("prompt_rewrite_rule")
        if isinstance(rule, dict):
            rule["conversion_excuse"] = NEW_CONVERSION_EXCUSE

    return row


def rewrite_file(input_path: Path, output_path: Path) -> dict[str, int]:
    stats = {
        "rows": 0,
        "system_prompts": 0,
        "assistant_contents": 0,
        "ground_truth": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                row = rewrite_row(row, stats)
            except Exception as exc:
                raise RuntimeError(f"{input_path}:{line_no}: {exc}") from exc
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rows"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="/data/chengch/project/rl_remake/outputs/single_turn_rl_contact_age_directed_5k_v3",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/chengch/project/data_remake/claude_distill",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob("single_turn_rl_contact_age_directed.*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {input_dir}")

    summary: dict[str, dict[str, int]] = {}
    for input_path in files:
        output_path = output_dir / input_path.name
        summary[input_path.name] = rewrite_file(input_path, output_path)

    summary_path = output_dir / "rewrite_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] wrote rewritten files to {output_dir}")


if __name__ == "__main__":
    main()
