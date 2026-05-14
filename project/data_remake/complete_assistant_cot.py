import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = "experiments/final_dataset.json"           # 您的原始带有 <think> 标签的数据
OUTPUT_FILE = "experiments/final_dataset_rewrite.json"    # 输出文件

API_KEY = "sk-25587b057d5242428bb940d44035b5fd" 
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"
MAX_WORKERS = 5                          

# 强制注入的概率 (例如 30% 的数据会被改写为强制执行模式)
FORCE_INJECTION_RATE = 0.3

# Prompt 模板
COT_COMPLETION_PROMPT = """
# Role
你是一名人类行为分析师。你的任务是还原 Agent 在回复用户时，脑海中真实的、非结构化的**潜意识思维流 (Stream of Consciousness)**。

# Input Data
## Context
System Prompt: 
{system_prompt}

Conversation History:
{history_text}

## Target Turn
User Input: "{user_input}"
Original Label (Intent): "{original_label}"
Actual Response: "{actual_response}"

# Task Requirements
请生成 JSON，包含 `thought`, `slot_values`, `response`。

## 1. response & slot_values
- response: 清洗后的纯文本回复。
- slot_values: 仅提取 System Prompt 定义的原子字段（age, gender, phone, wechat, symptom...）。

## 2. thought (核心：拟人化思维流)
**严禁使用** `[语境]...[策略]...` 这种死板格式！
请使用**第一人称**，模拟一个经验丰富的咨询顾问的心理活动。你的思考应该包含以下特征中的 1-2 种：

- **特征 A (权衡博弈)**: "用户好像有点抗拒，但 SOP 规定第 3 轮必须拿电话。我不能太生硬，得找个软一点的理由..."
- **特征 B (逻辑跳跃)**: "这也是失眠？不对，结合他刚才说的头痛，这可能是神经衰弱。我得确认一下他的病程..."
- **特征 C (紧迫感)**: "坏了，已经第 8 轮了还没拿到联系方式，再聊下去就‘死单’了。不管了，直接上强钩子..."
- **特征 D (动作响应)**: (针对 <action>) "突然收到后台强制指令要我挂号。虽然还在聊病情，但这指令优先级最高，我得赶紧打断一下..."

### 处理逻辑说明：
1. **若包含 <action>xxx</action>**：
   - 表现出**“被打断/接受指令”**的反应。
   - 例如："系统后台突然弹窗让我<xxx>。现在的对话氛围其实不太适合，但我必须执行。我看能不能用‘为了更精准匹配’作为借口强行切过去..."

2. **若无 Action (正常流程)**：
   - 结合 System Prompt 的硬性指标（轮次/必填项）。
   - 例如："看了一眼记录，现在是第一轮age，gender还是空的。如果不问清楚年龄性别，后面没法推专家。趁着他刚说完症状，我顺势问一句吧。"

# Output Examples (Bad vs Good)
- **Bad (Too Robotic)**: "第3轮。检查发现缺少 phone。规则要求第3轮留联。因此决定索要电话。"
- **Good (Human-like)**: "这已经是第三轮了，节奏有点慢了。用户虽然在诉苦，但我还是没拿到他的电话。根据流程，我现在必须发起第一次留联尝试。直接要可能会被拒，不如就说‘打字太慢，电话里讲得清’，这样他比较容易接受。"

# Output Format (JSON Only)
{{
    "thought": "（一段自然、流畅、带有逻辑判断的内心独白）",
    "slot_values": {{ ... }},
    "response": "..."
}}
"""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 辅助函数 =================

def parse_original_response(text):
    """
    分离原始数据中的 <think> 标签和 正文
    返回：(旧标签, 纯净的正文)
    """
    # 提取旧标签（作为 Prompt 的输入参考）
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    old_label = match.group(1).strip() if match else "通用回复"
    
    # 提取纯净正文（这将作为最终的 response，绝对不改动一个字）
    # 注意：这里我们只去掉了 <think> 标签，保留了所有标点、<sep> 等原始痕迹
    # 如果您的原始数据包含 <sep> 且希望保留，这里就不要 replace
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    return old_label, clean_text

def format_history(conversations, current_idx):
    """格式化当前轮次之前的对话历史"""
    history = ""
    for i in range(current_idx):
        turn = conversations[i]
        role = "User" if turn['from'] == 'human' else "Agent"
        # 历史记录里的 Agent 回复也需要清洗一下，只留纯文本，减少干扰
        content = turn['value']
        if role == "Agent":
            _, content = parse_original_response(content)
        history += f"{role}: {content}\n"
    return history

def process_single_item(entry, index):
    try:
        conversations = entry.get("conversations", [])
        system_prompt = entry.get("system", "")
        # 如果有提取的实体信息，也可以拼接到 system_prompt 里辅助生成
        extracted_info = entry.get("extracted_info", {})
        if extracted_info:
             system_prompt += f"\n【参考实体】: {json.dumps(extracted_info, ensure_ascii=False)}"

        new_conversations = []
        round_counter = 0

        i = 0
        while i < len(conversations):
            turn = conversations[i]
            
            if turn['from'] == 'human':
                user_content = turn['value']
                
                # -------------------------------------------------
                # 核心逻辑：判断下一轮 GPT 回复的意图，决定是否注入
                # -------------------------------------------------
                next_gpt_turn = None
                original_label = ""
                actual_response = ""
                
                if i + 1 < len(conversations):
                    next_gpt_turn = conversations[i+1]
                    original_label, actual_response = parse_original_response(next_gpt_turn['value'])
                    round_counter += 1

                # 决定是否进行“强制注入”
                # 条件：有下一轮GPT回复 + 随机概率命中 + 标签不是空的
                is_force_mode = False
                if next_gpt_turn and original_label and random.random() < FORCE_INJECTION_RATE:
                    # 构造 System Action
                    # 直接把原来的 label (例如"问年龄") 变成 <action>问年龄</action>
                    # 拼接到 User Input 后面
                    user_content = f"{user_content}\n<action>{original_label}</action>"
                    is_force_mode = True
                
                # 添加 User 轮次
                new_conversations.append({
                    "from": "human",
                    "value": user_content
                })
                
                # 处理 GPT 轮次 (补全 Thought)
                if next_gpt_turn:
                    original_full_text = next_gpt_turn['value']
            
                    # 1. 提取“不可变”的原始回复文本
                    old_label, immutable_response_text = parse_original_response(original_full_text)    
                    # 准备 Prompt
                    history_text = format_history(conversations, i) # 获取当前之前的历史
                    
                    full_prompt = COT_COMPLETION_PROMPT.format(
                        system_prompt=system_prompt,
                        history_text=history_text,
                        user_input=user_content,
                        original_label=original_label,
                        actual_response=actual_response,
                        round_num=round_counter
                    )

                    # 调用模型
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "你是一个逻辑补全专家。只输出 JSON。"},
                            {"role": "user", "content": full_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.3 # 低温，保证逻辑严谨
                    )
                    
                    # 2. 解析 LLM 返回的 JSON (只取 thought 和 slot_values)
                    llm_output = json.loads(response.choices[0].message.content)

                    generated_thought = llm_output.get("thought", "")
                    generated_slots = llm_output.get("slot_values", {})

                    # 3. 【关键步骤】在代码里组装最终 JSON，强行使用 immutable_response_text
                    final_gpt_json = {
                        "thought": generated_thought,
                        "slot_values": generated_slots,
                        "response": immutable_response_text  # <--- 核心：这里用的是原始变量，不是 LLM 生成的
                    }

                    # 4. 写入
                    new_conversations.append({
                        "from": "gpt",
                        "value": json.dumps(final_gpt_json, ensure_ascii=False)
                    })
                    
                    i += 1 # 跳过原来的 GPT 轮次
            
            i += 1
            
        # 返回处理完的一整条对话
        new_entry = entry.copy()
        new_entry['conversations'] = new_conversations
        return index, new_entry, None

    except Exception as e:
        return index, entry, str(e)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    print(f"📂 读取数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 兼容处理
    if isinstance(data, dict) and 'items' in data: data = data['items']
    
    total = len(data)
    print(f"🚀 开始补全思维链 (FORCE_RATE={FORCE_INJECTION_RATE})，共 {total} 条...")

    results = [None] * total
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_single_item, item, i): i 
            for i, item in enumerate(data)
        }

        for future in tqdm(as_completed(future_to_idx), total=total):
            idx, res, err = future.result()
            if err:
                print(f"⚠️ Item {idx} error: {err}")
                results[idx] = data[idx] # 失败保底
            else:
                results[idx] = res

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 完成！文件已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    # 记得把 COT_COMPLETION_PROMPT 的内容填进去
    main()
