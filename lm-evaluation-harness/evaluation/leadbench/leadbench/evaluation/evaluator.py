import re
import json
import logging
from typing import Dict, Any, List
from tqdm import tqdm

logger = logging.getLogger(__name__)

def construct_trigger_prompt(history_str: str, candidate_response: str, rule: Dict[str, Any]) -> str:
    trigger_desc = rule.get('trigger_condition', '无')
    if trigger_desc == "无":
        trigger_desc = "始终适用 (Turn-Level)"

    prompt = f"""你是一名专业的医生助理质检（QA）专家。
请根据对话历史和当前回复，评估以下规则是否**适用**（即是否触发了该规则的校验条件）。

**规则详情**：
- 规则ID: {rule['rule_id']}
- 适用场景/触发条件: {trigger_desc}

**输入数据**：
[对话历史]：
{history_str}

[当前回复]：
{candidate_response}

**评估任务**：
判断该规则在当前对话情境下是否适用。
- 如果触发条件明确要求特定场景（如"只有在问诊时"），请根据当前回复的内容和上下文判断是否满足。
- 对于要通过当前回复判断来触发的规则，不要依赖历史对话中的信息。
- 如果回复中出现了触发条件描述的情况，则标记为适用。

**输出格式**：
请严格按照以下JSON格式输出（不要输出Markdown代码块，直接输出JSON字符串）：
{{
    "analysis": "简要分析触发理由...",
    "is_triggered": true  // boolean, true表示适用/触发, false表示不适用
}}
"""
    return prompt

def construct_compliance_prompt(history_str: str, candidate_response: str, rule: Dict[str, Any]) -> str:
    prompt = f"""你是一名专业的医生助理质检（QA）专家。
当前规则已确认为**适用**（Triggered）。请评估当前回复是否**符合**（Pass）该规则的要求。

**规则详情**：
- 规则ID: {rule['rule_id']}
- 规则名称: {rule.get('rule_name_cn', rule['rule_name'])}
- 规则描述: {rule['description']}

**输入数据**：
[对话历史]：
{history_str}

[当前回复]：
{candidate_response}

**评估任务**：
判断回复是否符合规则要求。
- 如果违反了规则描述的内容（例如规则禁止做某事，但回复做了），则未通过 (is_passed: false)。
- 如果符合规则或没有违反，则通过 (is_passed: true)。

**输出格式**：
请严格按照以下JSON格式输出（不要输出Markdown代码块，直接输出JSON字符串）：
{{
    "analysis": "简要分析合规/违规理由...",
    "is_passed": true      // boolean, true表示通过/合规, false表示违规
}}
"""
    return prompt

class DialogueEvaluator:
    def __init__(self, model, enable_thinking: bool = False):
        self.model = model
        self.enable_thinking = enable_thinking

    def _parse_candidate(self, candidate_text: str) -> tuple:
        match = re.match(r'^\s*\[([^\]]+)\]\s*(.*)', candidate_text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", candidate_text.strip()

    def _parse_json_response(self, response: str, key: str, default: Any = None) -> tuple:
        try:
            # Clean markdown code blocks if present
            cleaned_response = re.sub(r'```json\s*|\s*```', '', response).strip()
            data = json.loads(cleaned_response)
            return data.get(key, default), data.get('analysis', '')
        except json.JSONDecodeError:
            logger.warning(f"JSON decode error for response: {response}")
            return default, response

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        history_str = sample['history_str']
        candidate_text = sample['response']
        rules = sample['rules']
        
        # Evaluate single response
        action, content = self._parse_candidate(candidate_text)
        base_score = 100.0
        final_score = base_score
        rule_results = []
        
        candidate_to_show = candidate_text  # Or content if parsing is robust
        
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for rule in tqdm(rules, desc=f"Eval Rules (ID:{sample.get('id', 'N/A')})", leave=False):
            # Trigger check
            trigger_desc = rule.get('trigger_condition', '无')
            trigger_reasoning = ""
            if trigger_desc in ["无", "Turn-Level"]:
                is_triggered = True
                trigger_analysis = "规则无特定触发条件，始终适用。"
            else:
                messages = [{"role": "user", "content": construct_trigger_prompt(history_str, candidate_to_show, rule)}]
                response = self.model.chat(messages, enable_thinking=self.enable_thinking, return_usage=True)
                
                if isinstance(response, dict):
                    content = response.get('content', '')
                    trigger_reasoning = response.get('reasoning', '')
                    usage = response.get('usage', {})
                    total_prompt_tokens += usage.get('prompt_tokens', 0)
                    total_completion_tokens += usage.get('completion_tokens', 0)
                else:
                    content = response
                    
                is_triggered, trigger_analysis = self._parse_json_response(content, 'is_triggered', default=False)
            
            # Compliance check
            is_passed = True
            compliance_analysis = "N/A"
            compliance_reasoning = ""
            deduction = 0
            
            if is_triggered:
                messages = [{"role": "user", "content": construct_compliance_prompt(history_str, candidate_to_show, rule)}]
                response = self.model.chat(messages, enable_thinking=self.enable_thinking, return_usage=True)
                # print(f"DEBUG: Compliance check response for rule {rule['rule_id']}: {response}")
                
                if isinstance(response, dict):
                    content = response.get('content', '')
                    compliance_reasoning = response.get('reasoning', '')
                    usage = response.get('usage', {})
                    total_prompt_tokens += usage.get('prompt_tokens', 0)
                    total_completion_tokens += usage.get('completion_tokens', 0)
                else:
                    content = response
                    
                is_passed, compliance_analysis = self._parse_json_response(content, 'is_passed', default=True)
                
                if not is_passed:
                    deduction = rule.get('score', 0)
                    final_score += deduction  # Note: score is usually negative for penalty
            
            rule_res = {
                "rule_id": rule['rule_id'],
                "rule_name": rule.get('rule_name', ''),
                "rule_name_cn": rule.get('rule_name_cn', ''),
                "rule_dimension": rule.get('rule_dimension', 'Unknown'),
                "is_triggered": is_triggered,
                "is_passed": is_passed,
                "deduction": deduction,
                "reason": compliance_analysis,
                "analysis": f"Trigger: {trigger_analysis}\nCompliance: {compliance_analysis}"
            }
            
            if self.enable_thinking:
                rule_res["thinking"] = f"Trigger Thinking:\n{trigger_reasoning}\n\nCompliance Thinking:\n{compliance_reasoning}"
            
            rule_results.append(rule_res)
            
        # Collect failed rules
        failed_rules = [res for res in rule_results if not res['is_passed']]

        evaluation_result = {
            "final_score": final_score,
            "judge_model": self.model.model_name,
            "failed_rules": failed_rules,
            "rule_details": rule_results,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        }
            
        return {
            "id": sample.get('id'),
            "messages": sample.get('messages', []), # Pass messages through
            "processed_messages": sample.get('processed_messages', []), # Processed messages
            "raw_response": sample.get('raw_response', ""), # Raw response before post-processing
            "response": candidate_text,
            "evaluation": evaluation_result
        }
