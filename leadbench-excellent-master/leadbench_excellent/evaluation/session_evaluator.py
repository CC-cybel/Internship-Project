import json
import logging
import re
import threading
from typing import Dict, Any, List
from tqdm import tqdm

logger = logging.getLogger(__name__)

def construct_trigger_check_prompt(history_str: str, rule: Dict[str, Any]) -> str:
    trigger_condition = rule.get('trigger_condition', '无')
    rule_name = rule.get('rule_name_cn', rule['rule_name'])
    
    prompt = f"""你是一名专业的医生助理质检（QA）专家。
请判断整段对话是否触发了场景 **{rule_name}**。

**场景信息**:
- 场景名称: {rule_name}
- 触发条件: {trigger_condition}

**对话历史**:
{history_str}

**任务**:
判断该场景是否在对话中触发（Triggered）。
- 如果对话内容满足触发条件，返回 true。
- 否则返回 false。

**输出格式**:
请严格按照以下JSON格式输出（不要输出Markdown代码块，直接输出JSON字符串）：
{{
    "analysis": "简短分析触发理由...",
    "is_triggered": true或false // 输出布尔值true或false
}}
"""
    return prompt

def construct_strategy_check_prompt(history_str: str, rule: Dict[str, Any], strategy: Dict[str, Any]) -> str:
    rule_name = rule.get('rule_name_cn', rule['rule_name'])
    strategy_name = strategy['strategy_name']
    strategy_desc = strategy['description']
    
    prompt = f"""你是一名专业的医生助理质检（QA）专家。
已确认对话触发了场景 **{rule_name}**。
请判断对话中是否具体体现了策略 **{strategy_name}**。

**策略信息**:
- 策略名称: {strategy_name}
- 策略描述: {strategy_desc}

**对话历史**:
{history_str}

**任务**:
判断对话中是否使用了该策略。
- 请仔细对照策略描述。
- 只有明确符合描述时才判定为 true。

**输出格式**:
请严格按照以下JSON格式输出（不要输出Markdown代码块，直接输出JSON字符串）：
{{
    "analysis": "简短分析匹配理由...",
    "is_hit": true或false // 输出布尔值true或false
}}
"""
    return prompt

class SessionEvaluator:
    def __init__(self, model, enable_thinking: bool = False):
        self.model = model
        self.enable_thinking = enable_thinking
        self.json_parse_attempts = 0
        self.json_parse_failures = 0
        self.lock = threading.Lock()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        with self.lock:
            self.json_parse_attempts += 1
            
        if not response:
            with self.lock:
                self.json_parse_failures += 1
            return {}

        try:
            # Clean markdown code blocks if present
            cleaned_response = re.sub(r'```json\s*|\s*```', '', response).strip()
            # Try to find the JSON object
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_response = cleaned_response[start_idx:end_idx+1]
            return json.loads(cleaned_response, strict=False)
        except json.JSONDecodeError:
            with self.lock:
                self.json_parse_failures += 1
            logger.warning(f"JSON decode error for response: {response}")
            return {}

    def evaluate_session(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Format history string from messages
        messages = sample.get('messages', [])
        history_str = ""
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            history_str += f"{role}: {content}\n"

        rules = sample.get('rules', [])
        if not rules:
            logger.warning("No rules provided for evaluation.")
            return sample

        rule_results = []
        total_score_change = 0
        scenario_scores_normalized = []

        total_prompt_tokens = 0
        total_completion_tokens = 0

        for rule in tqdm(rules, desc=f"Eval Rules (ID:{sample.get('id', 'N/A')})", leave=False):
            # Step 1: Check Trigger
            is_triggered = False
            trigger_analysis = ""
            trigger_reasoning = "" # New field for thinking process
            
            # Session-Level triggers are always true
            if rule.get('trigger') == 'Session-Level':
                is_triggered = True
                trigger_analysis = "Session-Level trigger (Always True)"
            else:
                # Event-Triggered-Session: Call LLM
                prompt = construct_trigger_check_prompt(history_str, rule)
                messages_payload = [{"role": "user", "content": prompt}]
                response = self.model.chat(messages_payload, enable_thinking=self.enable_thinking, return_usage=True)
                
                content = ""
                if isinstance(response, dict):
                    content = response.get('content', '')
                    trigger_reasoning = response.get('reasoning', '') # Capture thinking
                    usage = response.get('usage', {})
                    total_prompt_tokens += usage.get('prompt_tokens', 0)
                    total_completion_tokens += usage.get('completion_tokens', 0)
                else:
                    content = response
                
                result_data = self._parse_json_response(content)
                is_triggered = result_data.get('is_triggered', False)
                trigger_analysis = result_data.get('analysis', '')

            strategy_evaluations = []
            score_change = 0
            normalized_score = 0.0 # Default if not triggered
            
            # Step 2: Check Strategies (if triggered)
            if is_triggered:
                strategies = rule.get('strategy', [])
                matched_scores = []
                
                for strategy in strategies:
                    prompt = construct_strategy_check_prompt(history_str, rule, strategy)
                    messages_payload = [{"role": "user", "content": prompt}]
                    response = self.model.chat(messages_payload, enable_thinking=self.enable_thinking, return_usage=True)
                    
                    content = ""
                    strategy_reasoning = ""
                    if isinstance(response, dict):
                        content = response.get('content', '')
                        strategy_reasoning = response.get('reasoning', '')
                        usage = response.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0)
                        total_completion_tokens += usage.get('completion_tokens', 0)
                    else:
                        content = response
                        
                    result_data = self._parse_json_response(content)
                    is_hit = result_data.get('is_hit', False)
                    reason = result_data.get('analysis', '')
                    
                    strategy_evaluations.append({
                        "strategy_name": strategy['strategy_name'],
                        "is_hit": is_hit,
                        "reason": reason,
                        "reasoning": strategy_reasoning # Also capturing strategy level reasoning
                    })
                    
                    if is_hit:
                        matched_scores.append(strategy['score'])

                # Score Calculation
                if matched_scores:
                    has_negative = any(s < 0 for s in matched_scores)
                    if has_negative:
                        # "如果有扣分，只取扣分，取最低分（不累加）"
                        score_change = min(s for s in matched_scores if s < 0)
                    else:
                        # "如果只有加分，只取最高分（不累加）"
                        score_change = max(matched_scores)

                # Bench Score Normalization
                positive_scores = [s['score'] for s in strategies if s['score'] > 0]
                max_possible_score = max(positive_scores) if positive_scores else 0
                denominator = max_possible_score if max_possible_score > 0 else 1
                
                normalized_score = score_change / denominator if denominator != 0 else 0
                scenario_scores_normalized.append(normalized_score)
                total_score_change += score_change

            rule_res = {
                "rule_id": rule['rule_id'],
                "rule_name": rule.get('rule_name', ''),
                "rule_name_cn": rule.get('rule_name_cn', ''),
                "is_triggered": is_triggered,
                "trigger_analysis": trigger_analysis,
                "trigger_reasoning": trigger_reasoning, # Added field
                "strategy_evaluations": strategy_evaluations,
                "score_change": score_change,
                "normalized_score": normalized_score if is_triggered else None
            }
            rule_results.append(rule_res)

        # Final Scores
        rl_score = 100 + total_score_change
        bench_score = 0.0
        if scenario_scores_normalized:
            bench_score = sum(scenario_scores_normalized) / len(scenario_scores_normalized)
            
        evaluation_result = {
            "rl_score": rl_score,
            "bench_score": bench_score,
            "judge_model": self.model.model_name,
            "rule_details": rule_results,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        }
        
        return {
            "id": sample.get('id'),
            "messages": sample.get('messages', []),
            "generation_details": sample.get('generation_details', []),
            "evaluation": evaluation_result
        }
