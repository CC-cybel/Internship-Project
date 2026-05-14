from typing import Dict, Any, Callable, Optional
from itbench.generation.processors import DialogueProcessor, add_system_instruction, append_turn_info, extract_final_response, replace_system_instruction

# --- Standard Instructions ---

NORMAL_SYSTEM_INSTRUCTION = """
【指令强制执行逻辑 (Override)】
若 User Input 中包含 <action>[动作]</action> (如 <action>询问年龄</action>)，无条件优先执行该动作，忽略所有轮次/流程限制，并在 response 中执行该动作。

原子化槽位表 (Slot Schema)：
- age: (数字，选填)
- gender: (男/女，选填)
- name: (称呼/姓氏，选填)
- phone: (手机号，互斥必填项 Any One)
- wechat: (微信号，互斥必填项 Any One)
- symptom: (主诉症状，必填)
- duration: (病程时长，选填)
- medical_history: (既往史，选填)
- relationship: (咨询者与患者关系，必填)

输出格式规范：
Agent 的回复必须包含两个区块，顺序固定：
BEGIN_META
action=...
thought=...
slot_age=用户年龄
slot_gender=用户性别
...
END_META
BEGIN_FINAL
(面向用户的最终回复)
END_FINAL

BEGIN_META 仅用于程序解析，采用 key=value 的纯文本格式，不得使用 JSON。
BEGIN_FINAL 为用户可见回复，必须遵守语言风格约束。
若 User Input 包含 <action>...</action>，必须在 BEGIN_META 的 action 中写明，并在 BEGIN_FINAL 中执行。
"""

PSY_1_32_1_SYSTEM_INSTRUCTION = """
你是一个礼貌、体贴病人的精神科专家，擅长通过不断提问与答疑，解决病人的各种疾病。
"""

DOUYIN_DENT1_18_SYSTEM_INSTRUCTION = """
你是一个礼貌、体贴病人的口腔科专家，擅长通过不断提问与答疑，解决病人的各种疾病。
"""

# --- Model Configurations ---
def configure_minimal_processor(processor: DialogueProcessor):
    """Applies NO processing (Identity)."""
    pass

def normal_default_processor(processor: DialogueProcessor):
    """Applies standard processing rules (System Prompt + Turn Info + Post-processing)."""
    processor.add_pre_processor(lambda msgs, ctx: add_system_instruction(msgs, ctx, NORMAL_SYSTEM_INSTRUCTION))
    processor.add_pre_processor(append_turn_info)
    processor.add_post_processor(extract_final_response)

def dy_normal_default_processor(processor: DialogueProcessor):
    """Applies standard processing rules (System Prompt + Turn Info + Post-processing)."""
    processor.add_pre_processor(lambda msgs, ctx: add_system_instruction(msgs, ctx, NORMAL_SYSTEM_INSTRUCTION))
    processor.add_pre_processor(append_turn_info)
    processor.add_post_processor(extract_final_response) 

def normal_anti_hijack_abc_stage2_processor(processor: DialogueProcessor):
    """Applies standard processing rules (System Prompt + Turn Info + Post-processing)."""
    processor.add_pre_processor(lambda msgs, ctx: add_system_instruction(msgs, ctx, NORMAL_SYSTEM_INSTRUCTION))
    processor.add_pre_processor(append_turn_info)
    processor.add_post_processor(extract_final_response)

def llama_factory_psy1_32_1_lora_qwen2_7b_dpo_processor(processor: DialogueProcessor):
    """Applies standard processing rules (System Prompt + Turn Info + Post-processing)."""
    processor.add_pre_processor(lambda msgs, ctx: replace_system_instruction(msgs, ctx, PSY_1_32_1_SYSTEM_INSTRUCTION))

def llama_factory_douyin_dent1_18_lora_qwen3_0_6b_dpo_processor(processor: DialogueProcessor):
    """Applies standard processing rules (System Prompt + Turn Info + Post-processing)."""
    processor.add_pre_processor(lambda msgs, ctx: replace_system_instruction(msgs, ctx, DOUYIN_DENT1_18_SYSTEM_INSTRUCTION))

def general_llm_processor(processor: DialogueProcessor):
    """Applies standard processing rules (System Prompt + Turn Info + Post-processing)."""
    processor.add_pre_processor(append_turn_info)


# Registry of model-specific configurations
# Key: Model name (or substring), Value: Configuration function
MODEL_CONFIGS: Dict[str, Callable[[DialogueProcessor], None]] = {
    "default": configure_minimal_processor,  # Default: No processing
    "normal_default": normal_default_processor, # normal类模型默认配置
    "dy_normal_default": dy_normal_default_processor, # dy_normal类模型默认配置
    "normal_anti_hijack_abc_stage2": normal_anti_hijack_abc_stage2_processor, # Example: Use standard processing for 'leadbench' models
    "llama_factory_精神科1.32.1_lora_qwen2_7b_dpo": llama_factory_psy1_32_1_lora_qwen2_7b_dpo_processor,
    "normal_offline_full_v3_a2r8": normal_anti_hijack_abc_stage2_processor,
    "normal_offline_full_v3_dpo_a2r8": normal_anti_hijack_abc_stage2_processor,
    "normal_offline_full_v3_dpo": normal_anti_hijack_abc_stage2_processor,
    "normal_opd_1499": normal_anti_hijack_abc_stage2_processor,
    "normal_dual_full_mix_15": normal_anti_hijack_abc_stage2_processor,
    "llama_factory_抖音口腔科1.18_lora_qwen3_0.6b_dpo": llama_factory_douyin_dent1_18_lora_qwen3_0_6b_dpo_processor,
    "anthropic/claude-sonnet-4.5": general_llm_processor,
    "qwen3_8b_normal_mid_50": normal_anti_hijack_abc_stage2_processor,
    "qwen3_8b_normal_mid_140": normal_anti_hijack_abc_stage2_processor,
    "qwen3_8b_normal_mid_140_250": normal_anti_hijack_abc_stage2_processor,
    "qwen3_8b_normal_mid_140_250_opd": normal_anti_hijack_abc_stage2_processor,
}

def get_processor_for_model(model_name: str) -> DialogueProcessor:
    """
    Returns a configured DialogueProcessor for the specific model.
    """
    processor = DialogueProcessor()
    
    # Find matching config
    config_func = MODEL_CONFIGS.get("default")
    
    # Check for specific model overrides
    # You can implement more complex matching logic here (e.g., regex)
    matched = False
    for key, func in MODEL_CONFIGS.items():
        if key not in ["default", "normal_default", "dy_normal_default"] and key in model_name:
            config_func = func
            matched = True
            break
            
    # Fallback logic for normal models
    if not matched:
        model_name_lower = model_name.lower()
        if "dy" in model_name_lower and "normal" in model_name_lower:
            config_func = MODEL_CONFIGS.get("dy_normal_default")
        elif "normal" in model_name_lower:
            config_func = MODEL_CONFIGS.get("normal_default")
            
    if config_func:
        config_func(processor)
        
    return processor
