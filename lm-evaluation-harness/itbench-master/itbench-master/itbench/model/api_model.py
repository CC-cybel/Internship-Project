import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class APIModel:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: str,
        temperature: float = 0.01,
        timeout: float = 300.0
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout
        )
        self.temperature = temperature

    def _get_extra_body(self, enable_thinking: bool) -> Dict[str, Any]:
        """
        Generate extra_body parameters for API call based on model configuration and thinking mode.
        """
        extra_body = {}
        if enable_thinking:
            if self.model_name == "qwen3.5-397b-a17b" and "dashscope.aliyuncs.com" in str(self.client.base_url):
                extra_body = {"enable_thinking": True, "thinking_budget": 500}
            else:
                extra_body = {"chat_template_kwargs": {"thinking": True, "enable_thinking": True}}
        else:
            if self.model_name == "qwen3.5-397b-a17b" and "dashscope.aliyuncs.com" in str(self.client.base_url):
                extra_body = {"enable_thinking": False}
            # Qwen3.5-397B-A17B-FP8 这个默认会开启 thinking，需要关闭
            elif "Qwen3.5-397B-A17B-FP8" in self.model_name:
                extra_body = {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
        
        return extra_body

    def chat(self, messages: List[Dict[str, str]], enable_thinking: bool = False, return_usage: bool = False, max_tokens: int = 2048) -> Any:
        extra_body = self._get_extra_body(enable_thinking)

        try:
            # print(f"DEBUG: Calling API {self.model_name} with {len(messages)} messages...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                extra_body=extra_body if extra_body else None
            )
            # print(f"DEBUG: API {self.model_name} returned successfully.")
            # print(f"DEBUG: API {self.model_name} returned successfully. Response: {response}")
            
            message = response.choices[0].message
            content = message.content.strip()

            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            if enable_thinking:
                reasoning = getattr(message, 'reasoning', '')
                if not reasoning:
                    # 有的在 reasoning_content 中
                    reasoning = getattr(message, 'reasoning_content', '')
                # Also check for <think> tags in content if reasoning_content is empty
                if not reasoning and '<think>' in content:
                    import re
                    match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                    if match:
                        reasoning = match.group(1).strip()
                        # Remove thinking from content to get clean JSON
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
                result = {"content": content, "reasoning": reasoning}
                if return_usage:
                    result["usage"] = usage_info
                return result
            
            if return_usage:
                return {"content": content, "usage": usage_info}

            return content
        except Exception as e:
            logger.error(f"API call failed: {e}")
            print(f"Error calling API {self.model_name}: {e}")
            if return_usage:
                 return {"content": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            return ""
