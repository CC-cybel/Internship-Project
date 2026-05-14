import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = "experiments/cleaned_data_sample_5.json"           # 您的原始数据
OUTPUT_FILE = "experiments/dataset_with_system.json"   # 处理后的输出文件

API_KEY = "sk-25587b057d5242428bb940d44035b5fd"   # 您的 API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 以阿里 Qwen 为例
MODEL_NAME = "deepseek-v3.1"                  # 建议用 Plus 或 Max，提取能力更强
MAX_WORKERS = 1                         # 并发线程数

# ================= 提示词模板 =================
EXTRACT_ENTITIES_PROMPT = """
# Role
你是一个高精度的**医疗对话信息抽取员**。你的唯一任务是阅读对话，提取 Agent（医生/客服）方主动透露的**实体信息**。

# Input Data
<conversation>
{dialog_text}
</conversation>

# Extraction Rules (严格执行)
请从对话中提取以下 4 类信息。如果对话中未提及某项信息，**必须返回 null**，严禁根据上下文推测或编造。

1. **hospital_name (医院/机构名称)**:
   - 提取 Agent 自称的医院全称（如“北京华科中西医结合医院”）。
   - **排除**：通用代称（如“本院”、“我们医院”、“专科门诊”、“三甲医院”）。
   - **排除**：用户提到的医院（除非 Agent 确认了）。

2. **agent_name (医生/顾问姓名)**:
   - 提取 Agent 的自称（如“李主任”、“张医生”、“王助理”）。
   - **排除**：通用称呼（如“医生”、“老师”、“助理”）。

3. **official_contact (官方联系方式)**:
   - 提取 Agent **主动提供**给用户的电话号码、微信号、座机号。
   - **严禁提取用户的电话**。

4. **promised_benefits (承诺的福利/钩子)**:
   - 提取 Agent 为了留联而承诺的具体物品或服务。
   - 例如：“《抑郁自查表》”、“专家排班表”、“免挂号费”、“今日名额”。
   - 若只是泛泛地说“方便沟通”，则填 null。

# Output Format (JSON Only)
请直接输出 JSON 对象，不要包含 Markdown 标记：
{{
    "hospital_name": "提取到的名称 OR null",
    "agent_name": "提取到的姓名 OR null",
    "official_contact": "提取到的电话/微信 OR null",
    "promised_benefits": ["福利1", "福利2"] (若无则返回 [])
}}
"""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 核心逻辑 =================

def format_dialogue(conversations):
    """格式化对话为纯文本"""
    text = ""
    for turn in conversations:
        role = "User" if turn['from'] in ['user', 'human'] else "Agent"
        content = turn['value'].replace("<picture>", "").replace("<sep>", " ")
        text += f"{role}: {content}\n"
    return text

def process_single_item(item, index):
    """
    处理单条数据：提取实体 -> 追加到 item -> 返回
    """
    try:
        conversations = item.get('conversations', [])
        if not conversations:
            return index, item, "No conversations"

        dialog_text = format_dialogue(conversations)
        
        # 截断过长文本以节省Token (保留前1000字符和后1000字符通常包含了开头介绍和结尾留联)
        if len(dialog_text) > 2500:
            dialog_text = dialog_text[:1500] + "\n...[skipped]...\n" + dialog_text[-1000:]

        prompt = EXTRACT_ENTITIES_PROMPT.replace("{dialog_text}", dialog_text)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严格的数据提取程序。只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, # 强制 JSON
            temperature=0.0 # 零温度，确保最严格的提取
        )

        extracted_data = json.loads(response.choices[0].message.content)
        
        # === 关键步骤：将提取到的信息追加到原数据中 ===
        # 我们创建一个新字段 'extracted_info'
        item['extracted_info'] = extracted_data
        
        return index, item, None

    except Exception as e:
        # 出错时，给一个空的默认值，保证数据结构一致
        item['extracted_info'] = {
            "hospital_name": None,
            "agent_name": None, 
            "official_contact": None, 
            "promised_benefits": []
        }
        return index, item, str(e)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    print(f"📂 读取数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容处理
    if isinstance(data, dict) and 'items' in data:
        data_list = data['items']
        is_wrapper = True
    elif isinstance(data, list):
        data_list = data
        is_wrapper = False
    else:
        print("❌ 数据格式不支持")
        return

    total = len(data_list)
    print(f"🚀 开始提取实体信息，共 {total} 条 (并发数: {MAX_WORKERS})...")
    
    results = [None] * total
    
    # 线程池并发
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_single_item, item, i): i 
            for i, item in enumerate(data_list)
        }

        for future in tqdm(as_completed(future_to_idx), total=total):
            idx, processed_item, error = future.result()
            results[idx] = processed_item
            if error:
                # 仅打印前几个错误，避免刷屏
                if idx < 5: print(f"⚠️ Item {idx} warning: {error}")

    # 保存
    print(f"💾 正在保存至 {OUTPUT_FILE} ...")
    output_data = data if is_wrapper else results
    if is_wrapper:
        output_data['items'] = results

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("✅ 提取完成！请检查生成的 json 文件中是否多出了 'extracted_info' 字段。")

if __name__ == "__main__":
    main()
