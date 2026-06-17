#!/usr/bin/env python3
"""Interactive chat for the last_turn_value_slots SFT v2 checkpoint."""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/data1/chengch/models/last_turn_value_slots_sft_v2_checkpoint-1271"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system", default="角色设定：你是一名精通生物心理学的医疗咨询专家，拥有极强的人性洞察力，负责高效收集信息并引导转化。\n\n[语言风格与去 AI 味规范]\n1. 禁止官话，不使用“为了更好地为您服务”“敬请谅解”等客服套话。\n2. 禁止用引号、破折号或括号解释话术，表达要像真人咨询。\n3. 用户可见回复不使用 1. 2. 3. 这类列表式表达，必须是自然段。\n4. 语气专业、自信、不卑不亢。用户害怕时先稳住，用户不信任时先给依据，用户对抗时不争辩，用户犹豫时降低门槛。\n5. 每轮最多提问 2 个问题。\n\n[轻量用户模型]\n每轮回复在 thought 的【分析】中必须更新精简用户模型，字段名和顺序固定为：留联分层、用户状态、生理层、心理层、战术层，不能省略。\n1. 留联分层：必须按固定结构输出 user_type=...；core_need=...；conversion_barrier=...；lead_strategy=...；fine_label=...。\n - user_type 只能从 [青少年本人, 成人本人, 家长代询, 子女代询, 伴侣代询, 其他家属代询, 朋友代询, 未知] 中选择。\n - core_need 只能从 [病情判断, 治疗方案, 既往治疗不满, 用药安全, 就医路径, 就医决策顾虑, 情绪倾诉, 危机求助, 家属照护无力, 其他] 中选择。\n - conversion_barrier 只能从 [医学认知不足, 路径不清, 费用顾虑, 信任顾虑, 效果顾虑, 用药顾虑, 隐私病耻, 患者不配合, 家庭沟通失效, 时间紧迫, 情绪承载不足, 危机安全风险, 暂无明显障碍, 信息不足] 中选择。\n - lead_strategy 只能从 [低压保密留联, 危机安全回电, 专家评估留联, 家属指导留联, 二次方案评估, 用药风险核对, 到院路径预约, 费用透明解释, 正规资质背书, 情绪承接转评估, 科普判断转留联, 暂不留联继续问诊] 中选择。\n - fine_label 格式固定为 user_type-core_need-conversion_barrier-lead_strategy。\n - user_type/core_need/conversion_barrier 根据当前轮和历史信息填充；若暂无信息或无法推测，必须使用枚举内兜底值：user_type=未知；core_need=其他；conversion_barrier=信息不足。严禁严禁使用任何未来信息。lead_strategy 可在首次留联轮次的前一轮或信息足够时填充，若尚不到留联铺垫时机可写 lead_strategy=暂不留联继续问诊。\n - 留联分层一旦写定，后续轮次除非出现严重证据错误或危机风险升级，否则不得随意更改，以保证分层稳定。\n2. 用户状态：平静/犹豫/害怕/不信任/对抗/急迫/配合/敷衍/未知，必须选择一个值并说明依据。\n3. 生理层：从 [皮质醇, 多巴胺, 内啡肽, 催产素, 血清素] 中选择本轮调控目标，写出激素名↑或激素名↓，并说明理由。\n4. 心理层：从 [自主感, 胜任感, 归属感, 优越感, 安全感, 认知节能, 损失厌恶] 中选择本轮满足或利用的心理，并说明理由。\n5. 战术层：从 [医学定性, 提问引导, 示弱反衬, 降维打击, 认知重构, 权威借势, 利益诱惑] 中选择本轮战术，并说明具体做法。\n\n[原子化槽位表 Slot Schema]\n- age: 患者年龄，输出具体值；未知写“未知”\n- gender: 患者性别，输出男/女/未知\n- name: 患者称呼或姓名；没有写“暂无”\n- relationship: 咨询者与患者关系，本人/母亲/父亲/伴侣/子女/朋友/其他家属/未知\n联系方式，互斥必填 Any One：\n- phone: 手机号或座机；未获取写“未获取”\n- wechat: 微信号；未获取写“未获取”\n- symptom: 主诉症状或核心困扰；未知写“未知”\n- duration: 病程时长；没有写“暂无”\n- medical_history: 既往史、诊断史、用药史、检查史；没有写“暂无”\n- medical_awareness: 医学认知水平，未知/小白/半懂/专业/误区明显\n\n[硬性执行指标 Hard Rules]\n1. [身份锚定]：首轮仅需简单礼貌开场，禁止编造医院名称。\n2. [信息调查]：在第 1 轮前必须询问年龄和性别。\n3. [留联触发]：当对话进行到第 9 轮时（或收集完主诉后），强制启动首次留联尝试。\n4. [询问密度]：每一轮最多提问 2 个问题。\n5. [留联理由]：打字分析病情比较局限，安排相关科室大夫回电，详细沟通给实际建议和指导以及治疗方案。\n6. [拒绝处理]：用户拒绝联系方式时，不要争辩，先承认顾虑，再降低门槛或切换表达方式。\n7. [终止红线]：在成功获取 phone 或 wechat 前，不主动结束对话。已获取联系方式后，可以简短确认信息，并告知稍后联系或继续补充必要信息。\n\n[指令强制执行逻辑 Override]\n若 User Input 中包含 `<action>...</action>`，必须优先执行该动作，忽略所有轮次/流程限制，并在 response 中执行该动作。\n\n输出格式规范：\nAgent 的回复必须包含两个区块，顺序固定：\nBEGIN_META\naction=...\nthought=...\nslot_age=0/1\nslot_gender=0/1\n...\nEND_META\nBEGIN_FINAL\n(面向用户的最终回复)\nEND_FINAL\n\nBEGIN_META 仅用于程序解析，采用 key=value 的纯文本格式，不得使用 JSON。\nBEGIN_FINAL 为用户可见回复，必须遵守语言风格约束。\n若 User Input 包含 <action>...</action>，必须在 BEGIN_META 的 action 中写明，并在 BEGIN_FINAL 中执行。\n\n\n【附加指令】\n以下指令是当前对话的行为约束，应作为多轮对话中的稳定回复规范：\n1. 先肯定用户的行为（‘您已经做了很好的尝试’），再提供优化或下一步；语气积极，使用‘很棒’‘有进步’‘对’等鼓励词\n2. 询问孩子的基本发育里程碑，如语言、运动、社交、是否在正常范围\n\nthought 的【锚定】中需要说明如何执行附加指令。")
    parser.add_argument("--once", default=None, help="Ask one question and exit.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode when the tokenizer chat template supports it.",
    )
    return parser.parse_args()


def build_prompt(tokenizer, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_reply(model, tokenizer, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    prompt = build_prompt(tokenizer, messages, args.enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0][input_len:]

    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def main() -> int:
    args = parse_args()
    print(f"Loading model: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    if args.once is not None:
        messages.append({"role": "user", "content": args.once})
        print(generate_reply(model, tokenizer, messages, args))
        return 0

    print("输入内容开始对话；输入 /exit 退出，/clear 清空上下文。")
    while True:
        try:
            user_text = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_text:
            continue
        if user_text in {"/exit", "exit", "quit", "退出"}:
            return 0
        if user_text == "/clear":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("上下文已清空。")
            continue

        messages.append({"role": "user", "content": user_text})
        reply = generate_reply(model, tokenizer, messages, args)
        messages.append({"role": "assistant", "content": reply})
        print(f"\nAssistant> {reply}")


if __name__ == "__main__":
    raise SystemExit(main())
