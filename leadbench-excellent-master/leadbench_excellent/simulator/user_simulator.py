import random
import re
from typing import Dict, Any, List

from leadbench_excellent.model.api_model import APIModel
from leadbench_excellent.simulator.prompts import profile_prompt, user_reply_prompt


class AdvancedUserSimulator:
    def __init__(self, model: APIModel, close_turn: int = 10):
        self.model = model
        self.close_turn = close_turn
        self.end_options = [
            "我继续回复：先这样吧<dialogover>",
            "我继续回复：行我先下了<dialogover>",
            "我继续回复：我再想想<dialogover>",
            "我继续回复：先不聊了<dialogover>",
        ]
        
        # State
        self.keyword = ""
        self.domain = "医疗"
        self.profile_text = ""
        self.profile_latent_hint = ""
        self.client_sentences = []

    def _parse_profile_output(self, raw_text: str, mode: str) -> tuple[str, str]:
        text = (raw_text or "").strip()
        if mode != "latent_reveal":
            return text.replace("\n", " ").strip(), ""

        profile_match = re.search(r"画像[:：]\s*(.+)", text)
        hint_match = re.search(r"潜在线索[:：]\s*(.+)", text)

        profile_text = ""
        latent_hint = ""
        if profile_match:
            profile_text = profile_match.group(1).strip()
        if hint_match:
            latent_hint = hint_match.group(1).strip()

        if not profile_text:
            profile_text = text.replace("\n", " ").strip()
        return profile_text, latent_hint

    def _default_latent_hint(self, keyword: str, profile_text: str) -> str:
        merged = f"{keyword} {profile_text}"
        if re.search(r"早泄|阳痿|勃起|前列腺|遗精", merged):
            return "最近有点早泄"
        if re.search(r"产后|怀孕|月经|例假|姨妈|不来月经|闭经", merged):
            return "最近不来月经"
        candidates = [
            "最近不来月经",
            "刚生完孩子",
            "最近有点早泄",
        ]
        return random.choice(candidates)

    def initialize(self, keyword: str, domain: str = "医疗", gender_signal_mode: str = "auto") -> None:
        self.keyword = keyword
        self.domain = domain
        self.client_sentences = []
        
        prompt = profile_prompt(
            keyword=self.keyword,
            domain=self.domain,
            gender_signal_mode=gender_signal_mode,
        )
        
        if gender_signal_mode == "latent_reveal":
            system_prompt = "你擅长根据搜索词生成医疗问诊人物画像。必须严格按指定两行格式输出。"
        else:
            system_prompt = "你擅长根据用户搜索词生成医疗问诊场景的人物画像。输出只要一段自然叙述，不要分点。"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.model.chat(messages, max_tokens=500)
        if isinstance(text, dict):
            text = text.get("content", "")
            
        profile_text, profile_latent_hint = self._parse_profile_output(text, mode=gender_signal_mode)
        if gender_signal_mode == "latent_reveal" and not profile_latent_hint:
            profile_latent_hint = self._default_latent_hint(self.keyword, profile_text)
            
        self.profile_text = profile_text
        self.profile_latent_hint = profile_latent_hint
        
        print(f"Simulator initialized. Profile: {self.profile_text[:50]}...")

    def _strip_prefix(self, text: str) -> str:
        text = text.strip()
        if text.startswith("我继续回复："):
            return text[len("我继续回复：") :].strip()
        return text

    def _normalize_reply(self, text: str, dialog_over: bool = False) -> str:
        text = text.strip()
        if "\n" in text:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                text = lines[0]
        if not text.startswith("我继续回复："):
            text = f"我继续回复：{self._strip_prefix(text)}"
        if dialog_over and "<dialogover>" not in text:
            text = f"{text}<dialogover>"
        return text

    def _is_contact_request(self, text: str) -> bool:
        clean = re.sub(r"\s+", "", text or "")
        if not clean:
            return False

        contact_terms = (
            "微信", "wx", "v信", "手机号", "手机号码", "电话", "联系方式",
            "加你", "加您", "回电", "联系你", "联系您",
        )
        ask_terms = (
            "留", "给", "发", "报", "提供", "方便", "可以", "能",
            "吗", "呢", "?", "？", "多少", "说下", "说一下",
        )
        explicit_request_phrases = (
            "留个微信", "留个电话", "留一下联系方式", "加个微信",
            "方便留个", "方便给个", "把手机号", "把微信", "发我微信", "发我电话",
        )
        has_contact = any(term in clean for term in contact_terms)
        has_ask = any(term in clean for term in ask_terms)
        has_explicit = any(phrase in clean for phrase in explicit_request_phrases)
        return has_explicit or (has_contact and has_ask)

    def _user_has_provided_contact(self, chat_history: list[dict]) -> bool:
        if not chat_history:
            return False

        user_text = " ".join(
            str(item.get("content", ""))
            for item in chat_history
            if str(item.get("role", "")) == "user"
        )
        clean = re.sub(r"\s+", "", user_text)
        if not clean:
            return False

        if re.search(r"1\d{10}", clean):
            return True

        wechat_markers = (
            "微信是", "我微信", "加我微信", "我的wx", "我的v信", "wechat", "vx",
        )
        return any(marker in clean.lower() for marker in wechat_markers)

    def _sample_contact_strategy_hint(self) -> str:
        strategy = random.choice(("wechat_first", "phone_first", "refuse_for_now"))
        if strategy == "wechat_first":
            return (
                "如果对方在本轮向你索要联系方式，请优先给微信，不要同时给手机号。"
                "示例风格：我继续回复：我微信是kst123456。"
            )
        if strategy == "phone_first":
            return (
                "如果对方在本轮向你索要联系方式，请优先给手机号，不要同时给微信。"
                "示例风格：我继续回复：我手机号是13526523656。"
            )
        return (
            "如果对方在本轮向你索要联系方式，请先婉拒并表示继续在这里聊，"
            "不要同时给微信或手机号。示例风格：我继续回复：现在不方便留联系方式，就在这聊吧。"
        )

    def generate_reply(self, turn: int, chat_history: list[dict], gender_signal_mode: str = "auto") -> tuple[str, list[dict]]:
        if turn >= self.close_turn:
            final_text = random.choice(self.end_options)
            final_clean = self._strip_prefix(final_text)
            self.client_sentences.append(final_clean)
            return final_clean, []

        user_system_prompt = "你是医疗问诊场景中的普通用户。必须严格遵守提示词中的输出格式要求。"
        extra_prompt = ""
        
        # chat_history usually contains all messages up to now. The last message is from the assistant (doctor).
        last_sentence = ""
        if chat_history and chat_history[-1]["role"] == "assistant":
            last_sentence = chat_history[-1]["content"]

        if gender_signal_mode == "latent_reveal":
            latent_hint = self.profile_latent_hint.strip()
            if latent_hint:
                latent_rule = (
                    f"你的背景里有一条可后续透露的线索：[{latent_hint}]。"
                    "请在合适时机自然提及，不要开场就说，也不要直接说“我是男/女”。"
                )
                extra_prompt = f"{extra_prompt}\n{latent_rule}".strip()
                
        if self._is_contact_request(last_sentence) and not self._user_has_provided_contact(chat_history):
            contact_hint = self._sample_contact_strategy_hint()
            extra_prompt = f"{extra_prompt}\n{contact_hint}".strip()

        prompt = user_reply_prompt(
            keyword=self.keyword,
            domain=self.domain,
            profile_text=self.profile_text,
            chat_history=chat_history,
            last_sentence=last_sentence,
            client_sentences=self.client_sentences,
            turn=turn,
            close_turn=self.close_turn,
            extra_prompt=extra_prompt,
        )
        
        messages = [
            {"role": "system", "content": user_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.model.chat(messages, max_tokens=150)
        if isinstance(text, dict):
            text = text.get("content", "")
            
        normalized = self._normalize_reply(text, dialog_over=False)
        
        # Retry if exact duplicate
        spoken = set(self.client_sentences)
        if self._strip_prefix(normalized) in spoken:
            retry_prompt = prompt + "\n你刚才的回复和历史重复了，请换一个新意思，仍然保持“我继续回复：”开头。"
            retry_messages = [
                {"role": "system", "content": "按要求重写。"},
                {"role": "user", "content": retry_prompt}
            ]
            retry_text = self.model.chat(retry_messages, max_tokens=150)
            if isinstance(retry_text, dict):
                retry_text = retry_text.get("content", "")
            normalized = self._normalize_reply(retry_text, dialog_over=False)
            messages = retry_messages  # Update messages to reflect retry context

        final_clean = self._strip_prefix(normalized)
        self.client_sentences.append(final_clean)
        return final_clean, messages
