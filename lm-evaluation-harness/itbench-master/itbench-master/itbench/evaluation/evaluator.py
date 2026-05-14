import re
import json
import logging
from typing import Dict, Any, List
from tqdm import tqdm
from itbench.utils.config import config

logger = logging.getLogger(__name__)

def construct_trigger_prompt(history_str: str, candidate_response: str, rule: Dict[str, Any], dynamic_description: str = None) -> str:
    trigger_desc = rule.get('trigger_condition', '无')
    if trigger_desc == "无":
        trigger_desc = "始终适用 (Turn-Level)"
        
    desc_to_use = dynamic_description if dynamic_description else rule['description']

    prompt = f"""你是一名专业的医生助理质检（QA）专家。
请根据对话历史和当前回复，评估以下规则是否**适用**（即是否触发了该规则的校验条件）。

**规则详情**：
- 规则ID: {rule['rule_id']}
- 规则描述: {desc_to_use}
- 原定触发条件: {trigger_desc}

**输入数据**：
[对话历史]：
{history_str}

[当前回复]：
{candidate_response}

**评估任务**：
判断该规则在当前对话情境下是否适用。
**关于对话轮次的特别说明**：
【当前回复】所在的轮次，由[对话历史]中最后一条出现的 `[第X轮]user: ...` 决定。例如，如果历史中最后一条是 `[第5轮]user`，则当前助手生成的回复就是**第5轮回复**，此时当前处于第5轮。请务必基于此准确计算当前所处的轮次。

**特别注意**：
1. 如果原定触发条件中包含“系统提示词中有某某指令”，请注意：必须在系统提示词（[对话历史]中的首条）中**显式包含**了该要求/指令，并且**当前对话轮次**已经达到了规则要求的校验时间窗口（例如“处于第N轮”或最后期限），该规则才应被视为**适用（Triggered）**。如果系统提示词中没有该指令，则不适用。
2. 如果触发条件明确要求特定场景（如"只有在问诊时"），请根据当前回复的内容和上下文判断是否满足。
3. 对于要通过当前回复内容判断来触发的规则，请主要关注当前回复及紧邻的上下文。

**输出格式**：
请严格按照以下JSON格式输出（不要输出Markdown代码块，直接输出JSON字符串）：
{{
    "analysis": "简要分析触发理由，如果是因为到达了校验时间窗口（如第N轮）而触发，请在此说明...",
    "is_triggered": true  // boolean, true表示适用/触发, false表示不适用
}}
"""
    return prompt

def construct_compliance_prompt(history_str: str, candidate_response: str, rule: Dict[str, Any], dynamic_description: str = None) -> str:
    # Use dynamic_description if provided, otherwise fallback to rule['description']
    desc_to_use = dynamic_description if dynamic_description else rule['description']
    prompt = f"""你是一名专业的医生助理质检（QA）专家。
当前规则已确认为**适用**（Triggered）。请评估当前回复是否**符合**（Pass）该规则的要求。

**规则详情**：
- 规则ID: {rule['rule_id']}
- 规则名称: {rule.get('rule_name_cn', rule['rule_name'])}
- 规则描述: {desc_to_use}

**输入数据**：
[对话历史]：
{history_str}

**当前回复**：
{candidate_response}

**评估任务**：
判断【当前回复】是否符合规则要求。

**关于对话轮次的特别说明**：
【当前回复】所在的轮次，由[对话历史]中最后一条出现的 `[第X轮]user: ...` 决定。例如，如果历史中最后一条是 `[第5轮]user`，则当前助手生成的回复就是**第5轮回复**，此时当前处于第5轮。请务必基于此准确计算当前所处的轮次。

**特别注意**：
1. 仅针对【当前回复】的内容进行评估，不要因为对话历史中存在违规行为而判定当前回复不通过。
2. 必须严格、且仅针对当前规则的描述进行评估。如果【当前回复】违反了当前规则描述的内容，则未通过 (is_passed: false)。
3. 如果【当前回复】在当前规则描述的维度上符合要求或没有违反，则通过 (is_passed: true)。即使当前回复违反了**其他规则**（例如连环发问、未问诊等），只要它没有违反**当前正在评估的这条规则**，你也必须判定为通过。不要越权评估当前规则之外的其他违规行为。

**输出格式**：
请严格按照以下JSON格式输出（不要输出Markdown代码块，直接输出JSON字符串）：
{{
    "analysis": "简要分析合规/违规理由...",
    "is_passed": true      // boolean, true表示通过/合规, false表示违规
}}
"""
    return prompt

def construct_extract_params_prompt(history_str: str, rule: Dict[str, Any]) -> str:
    prompt = f"""你是一名专业的医生助理质检（QA）专家。
当前有一个质检规则，该规则的具体约束条件依赖于对话历史中的初始背景设定。
请仔细阅读下面的[对话历史]（特别是第一轮的背景信息），分析出与该规则相关的具体数值、要求或限制，并结合原规则描述，生成一个具体、明确的【新的规则描述】。

**原规则详情**：
- 规则ID: {rule['rule_id']}
- 规则名称: {rule.get('rule_name_cn', rule['rule_name'])}
- 原规则描述: {rule['description']}

**输入数据**：
[对话历史]：
{history_str}

**评估任务**：
1. 分析对话历史的背景设定中对该规则的具体要求（例如：具体的数值N、特定年龄的沟通策略、要求获取的联系方式类型等）。
2. 将分析出的具体要求替换或补充到原规则描述中，形成一条明确的、可直接用于后续评估的【新规则描述】。
3. 如果在对话历史中没有找到相关的具体要求，请尽量保持原规则描述，或指出未找到。

**输出格式**：
请严格按照以下JSON格式输出（不要输出Markdown代码块，直接输出JSON字符串）：
{{
    "analysis": "简要分析过程，说明发现了哪些具体的约束条件...",
    "new_description": "将具体约束条件带入后的明确的新规则描述..."
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
            return data.get(key, default), data.get('analysis', ''), True
        except json.JSONDecodeError:
            logger.warning(f"JSON decode error for response: {response}")
            return default, response, False

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
        total_judge_calls = 0
        failed_parse_attempts = 0

        for rule in tqdm(rules, desc=f"Eval Rules (ID:{sample.get('id', 'N/A')})", leave=False):
            trigger_prompt_text = ""
            extract_prompt_text = ""
            compliance_prompt_text = ""
            
            extract_analysis = ""
            extract_reasoning = ""
            dynamic_description = None
            
            # Trigger check
            trigger_desc = rule.get('trigger_condition', '无')
            trigger_reasoning = ""
            if trigger_desc in ["无", "Turn-Level"]:
                is_triggered = True
                trigger_analysis = "规则无特定触发条件，始终适用。"
            else:
                trigger_prompt_text = construct_trigger_prompt(history_str, candidate_to_show, rule, dynamic_description=rule['description'])
                messages = [{"role": "user", "content": trigger_prompt_text}]
                
                max_retries = 3
                total_judge_calls += 1
                success = False
                for attempt in range(max_retries):
                    response = self.model.chat(messages, enable_thinking=self.enable_thinking, return_usage=True)
                    
                    if isinstance(response, dict):
                        content = response.get('content', '')
                        trigger_reasoning = response.get('reasoning', '')
                        usage = response.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0)
                        total_completion_tokens += usage.get('completion_tokens', 0)
                    else:
                        content = response
                        
                    is_triggered, trigger_analysis, success = self._parse_json_response(content, 'is_triggered', default=False)
                    if success:
                        break
                    if attempt < max_retries - 1:
                        logger.warning(f"Trigger check JSON decode failed, retrying {attempt+1}/{max_retries}...")
                
                if not success:
                    failed_parse_attempts += 1
            
            # Handle parameter extraction only if triggered
            if is_triggered and rule.get('extract_params'):
                extract_prompt_text = construct_extract_params_prompt(history_str, rule)
                extract_msg = [{"role": "user", "content": extract_prompt_text}]
                
                total_judge_calls += 1
                success = False
                for attempt in range(max_retries):
                    extract_response = self.model.chat(extract_msg, enable_thinking=self.enable_thinking, return_usage=True)
                    
                    if isinstance(extract_response, dict):
                        extract_content = extract_response.get('content', '')
                        extract_reasoning = extract_response.get('reasoning', '')
                        usage = extract_response.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0)
                        total_completion_tokens += usage.get('completion_tokens', 0)
                    else:
                        extract_content = extract_response
                        
                    dynamic_description, extract_analysis, success = self._parse_json_response(extract_content, 'new_description', default=rule['description'])
                    if success:
                        break
                    if attempt < max_retries - 1:
                        logger.warning(f"Extract check JSON decode failed, retrying {attempt+1}/{max_retries}...")
                        
                if not success:
                    failed_parse_attempts += 1

            # Compliance check
            is_passed = True
            compliance_analysis = "N/A"
            compliance_reasoning = ""
            deduction = 0
            
            if is_triggered:
                compliance_prompt_text = construct_compliance_prompt(history_str, candidate_to_show, rule, dynamic_description=dynamic_description)
                messages = [{"role": "user", "content": compliance_prompt_text}]
                
                total_judge_calls += 1
                success = False
                for attempt in range(max_retries):
                    response = self.model.chat(messages, enable_thinking=self.enable_thinking, return_usage=True)
                    
                    if isinstance(response, dict):
                        content = response.get('content', '')
                        compliance_reasoning = response.get('reasoning', '')
                        usage = response.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0)
                        total_completion_tokens += usage.get('completion_tokens', 0)
                    else:
                        content = response
                        
                    is_passed, compliance_analysis, success = self._parse_json_response(content, 'is_passed', default=True)
                    if success:
                        break
                    if attempt < max_retries - 1:
                        logger.warning(f"Compliance check JSON decode failed, retrying {attempt+1}/{max_retries}...")
                
                if not success:
                    failed_parse_attempts += 1
                
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
                "trigger_analysis": trigger_analysis,
                "compliance_analysis": compliance_analysis,
                "trigger_prompt": trigger_prompt_text,
                "extract_prompt": extract_prompt_text,
                "compliance_prompt": compliance_prompt_text,
            }
            
            if rule.get('extract_params'):
                rule_res["extract_analysis"] = extract_analysis
                if dynamic_description:
                    rule_res["dynamic_description"] = dynamic_description
            
            if self.enable_thinking:
                rule_res["trigger_thinking"] = trigger_reasoning
                if rule.get('extract_params'):
                    rule_res["extract_thinking"] = extract_reasoning
                rule_res["compliance_thinking"] = compliance_reasoning
            
            rule_results.append(rule_res)
            
        # Collect failed rules
        failed_rules = [res for res in rule_results if not res['is_passed']]
        
        # Collect triggered rules
        triggered_rules = [res for res in rule_results if res['is_triggered']]

        evaluation_result = {
            "final_score": final_score,
            "judge_model": self.model.model_name,
            "candidate_model": config.CANDIDATE_MODEL_NAME,
            "failed_rules": failed_rules,
            "triggered_rules": triggered_rules,
            "rule_details": rule_results,
            "total_judge_calls": total_judge_calls,
            "failed_parse_attempts": failed_parse_attempts,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        }
            
        return {
            "id": sample.get('id'),
            "history_str": history_str,
            "messages": sample.get('messages', []), # Pass messages through
            "processed_messages": sample.get('processed_messages', []), # Processed messages
            "raw_response": sample.get('raw_response', ""), # Raw response before post-processing
            "response": candidate_text,
            "evaluation": evaluation_result
        }