import json
import re
import os

# ================= 配置区域 =================
INPUT_FILE = "experiments/sft_normal_stratified.json"      # 你的原始文件路径
OUTPUT_FILE = "experiments/cleaned_data_normal.json"    # 输出文件路径
# ===========================================

def clean_content(text, role):
    """
    清洗文本的核心函数
    """
    if not isinstance(text, str):
        return text

    # 1. 【通用清洗】移除 "Round X: role: " 前缀
    # 匹配规则：行首 + Round + 空格 + 数字 + 冒号 + 空格 + 角色名 + 冒号 + 空格
    # 例如: " Round 0: human: " 或 "Round 1: assistant: "
    text = re.sub(r"^\s*Round\s+\d+:\s+(human|assistant|gpt):\s*", "", text, flags=re.IGNORECASE)

    # 2. 【模型端清洗】仅针对 gpt/assistant 做特殊处理
    if role in ["gpt", "assistant"]:
        # A. 移除 <picture> 标签
        text = text.replace("<picture>", "")

        # B. 将 [Action] 转换为 <think>Action</think>
        # 正则匹配方括号内的内容
        match = re.search(r"\[(.*?)\]", text)
        if match:
            action_content = match.group(1)
            # 构造 think 标签
            think_tag = f"<think>{action_content}</think>"
            # 替换，只替换第一个匹配项（通常决策都在开头）
            text = text.replace(f"[{action_content}]", think_tag, 1)

    return text.strip()

def main():
    print(f"📂 正在读取文件: {INPUT_FILE} ...")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到文件 '{INPUT_FILE}'，请检查文件名配置。")
        return

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        return

    # 兼容性处理：如果数据包裹在 items 字段里（某些数据集格式），或者直接是 list
    if isinstance(data, dict) and "items" in data:
        data_list = data["items"]
        is_wrapper = True
    elif isinstance(data, list):
        data_list = data
        is_wrapper = False
    else:
        print("❌ 数据格式无法识别，必须是 List 或包含 items 的 Dict")
        return

    print(f"📊 共加载 {len(data_list)} 条数据，开始清洗...")

    count = 0
    for item in data_list:
        if "conversations" not in item:
            continue
            
        for turn in item["conversations"]:
            role = turn.get("from", "")
            original_val = turn.get("value", "")
            
            # 执行清洗
            new_val = clean_content(original_val, role)
            
            turn["value"] = new_val
        
        count += 1
        if count % 2000 == 0:
            print(f"   已处理 {count} 条...")

    # 保存文件
    print(f"💾 正在保存至: {OUTPUT_FILE} ...")
    
    # 如果原数据是 dict 包装的，保持原结构保存
    output_data = data if is_wrapper else data_list
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("✅ 处理完成！")

if __name__ == "__main__":
    main()
