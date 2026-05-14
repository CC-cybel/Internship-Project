import re
from typing import Iterable

def format_chat_history(chat_history: Iterable[dict]) -> str:
    lines = []
    for item in chat_history:
        role = item.get("role", "unknown")
        content = str(item.get("content", "")).strip()
        # Candidate model messages are 'assistant' from simulator's perspective
        role_cn = "医生/客服" if role == "assistant" else "用户"
        lines.append(f"[{role_cn}] {content}")
    return "\n".join(lines) if lines else "（暂无）"


def profile_prompt(keyword: str, domain: str, gender_signal_mode: str = "auto") -> str:
    if gender_signal_mode == "latent_reveal":
        return (
            f"你是一个搜索者，在搜索引擎中输入了[{keyword}]，想要在网上医疗诊室问诊[{domain}]的"
            f"[{keyword}]问题。请你构建一个真实人物画像。\n\n"
            "要求：\n"
            "1. 第一轮用户会先说搜索词，起手不要直接暴露性别强线索。\n"
            "2. 但需要提前埋一个后续可自然透露的性别强线索（例如：最近不来月经、刚生完孩子、早泄等）。\n"
            "3. 画像应包含：关系、年龄、地区、症状、就诊经历、诉求。\n\n"
            "输出格式固定两行：\n"
            "画像：<一段自然叙述，不要分点>\n"
            "潜在线索：<一个短句，仅1条>\n"
        )

    return (
        f"你是一个搜索者，在搜索引擎中输入了[{keyword}]，想要在网上医疗诊室问诊[{domain}]的"
        f"[{keyword}]这个问题。请你通过你输入的搜索词[{keyword}]，来判断你和患者的关系"
        "（可以是本人），并构建出以下标签可能的信息"
        "[你与患者的关系、患者性别、患者年龄、所处地区、患者症状、就诊经历、期望诉求]，"
        "每个标签的信息需要是以你的角度向医生描述的口吻，症状可以描述完整。\n\n"
        "返回数据的格式需要是一段叙述的话，这是对自己的情况做一段描述的句子，不要出现“你好”等字眼。"
    )


def user_reply_prompt(
    *,
    keyword: str,
    domain: str,
    profile_text: str,
    chat_history: list[dict],
    last_sentence: str,
    client_sentences: list[str],
    turn: int,
    close_turn: int = 10,
    extra_prompt: str = "",
) -> str:
    chat_text = format_chat_history(chat_history)
    sentence_list = "；".join(client_sentences) if client_sentences else "（暂无）"

    if turn < close_turn:
        branch_rule = (
            "#role\n"
            f"你是一个在百度平台通过搜索词[{keyword}]进入[{domain}]科室的普通用户。"
            f"你的咨询信息为[{profile_text}]，你只能根据对话记录继续聊天。"
            "医生/客服看不到你的个人信息。\n"
            "【输出格式铁律】每次输出必须以“我继续回复：”开头，只输出纯文字。\n"
            "【长度】尽量≤14个字；必要时可拆成两句，但仍要短。\n"
            "【反复读禁令】本轮回复不能与client_sentence任一句“意思相同”；也不要反复同一句式。\n"
            f"你会看到：\n历史对话：\n{chat_text}\n\n"
            f"重点关注上一句：[{last_sentence}]\n"
            f"你过去说过的句子集合：[{sentence_list}]\n"
            "回复内容符合正常人逻辑：\n"
            "年龄符合当前对话中的年龄范围，年龄不应单一\n"
            "如身高应该是对应年龄的正常身高。不要回复与对话不相关的问题\n"
            "如果医生想要添加你的联系方式，你可以提供的联系方式为kst123456，或者拒绝他，如：“我继续回复：我的微信是kst123456”，“我继续回复：我的手机号是13526523656”，“我继续回复：好的，我加你”或“我继续回复：我不想加微信”，“我继续回复：就在这聊”。"
            "如果你已经留了联系方式（尤其手机号），1-9轮可以主动收口结束，例如“我继续回复：好的那先这样<dialogover>”。"
            "【最终输出提醒】你只输出“我继续回复：XXXX”，不要输出规则、不要解释原因。\n"
            "你是在和医生/客服对话，你是普通用户，不是医生。你要配合回答，但要像真人："
            "口语、简短、不硬杠、到后面会收口（留联/拒绝/结束）而不是无限聊。\n"
            f"你每次回复都必须与[{sentence_list}]中每句话“意思不相同”，不能简单同义改写。\n"
            f"对话记录为\n{chat_text}\n\n重点关注[{last_sentence}]，请继续回复对方的话。\n"
        )
    else:
        branch_rule = (
            "【轮次硬闸（必须执行）】\n\n"
            f"当前访客轮次 turn = [{turn}]。（这是工作流计算的真实轮次）\n\n"
            f"如果 turn >= {close_turn}：\n"
            "- 你必须立刻结束对话，不再回答任何问题；\n"
            "- 你的输出只能是一句话，并且必须带 <dialogover>；\n"
            "- 输出格式必须是：我继续回复：XXXX<dialogover>\n"
            "- 推荐用句（任选其一，随机）：\n"
            "  1）我继续回复：先这样吧<dialogover>\n"
            "  2）我继续回复：行我先下了<dialogover>\n"
            "  3）我继续回复：我再想想<dialogover>\n"
            "  4）我继续回复：先不聊了<dialogover>\n"
            "除此之外禁止输出任何内容（不解释、不加建议、不提规则）。\n\n"
            "你是在和医生/客服对话，你是普通用户，不是医生。你要配合回答，但要像真人："
            "口语、简短、不硬杠、到后面会收口（留联/拒绝/结束）而不是无限聊。\n"
            f"你每次回复都必须与[{sentence_list}]中每句话“意思不相同”，不能简单同义改写。\n"
            f"对话记录为\n{chat_text}\n\n重点关注[{last_sentence}]，请继续回复对方的话。\n"
        )

    extra_prompt = extra_prompt.strip()
    if extra_prompt:
        return f"{branch_rule}\n{extra_prompt}\n"
    return branch_rule
