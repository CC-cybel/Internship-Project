import json
import os

# ================= 配置 =================
INPUT_FILE = "experiments/cleaned_data_normal.json"  # 上一步生成的文件名
OUTPUT_FILE = "raw/normal_inject_round.json"    # 最终喂给模型训练的文件

def inject_round_info(data):
    """
    遍历对话，在 Human 输入中注入轮次信息
    """
    total_processed = 0
    
    for entry in data:
        conversations = entry.get('conversations', [])
        
        # 严格按照 (User -> Agent) 为一轮来计算
        # Round 1: index 0 (User), index 1 (Agent)
        # Round 2: index 2 (User), index 3 (Agent)
        # 公式: Current_Round = (Turn_Index // 2) + 1
        
        for i, turn in enumerate(conversations):
            if turn['from'] == 'human':
                current_round = (i // 2) + 1
                
                original_text = turn['value']
                
                # 构造注入标签
                # 格式建议使用换行符隔开，显得更清晰，不干扰原始语义
                round_tag = f"\n【系统数据：当前第 {current_round} 轮】"
                
                # 防止重复注入 (如果跑了两次脚本)
                if "【系统数据：当前第" not in original_text:
                    turn['value'] = original_text + round_tag
        
        total_processed += 1
        
    return data

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    print(f"📂 读取数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容 list 或 dict 格式
    if isinstance(data, dict) and 'items' in data:
        process_list = data['items']
        is_wrapper = True
    else:
        process_list = data
        is_wrapper = False

    print(f"🚀 开始为 {len(process_list)} 条数据注入 User Input 轮次信息...")
    
    new_data = inject_round_info(process_list)
    
    # 保持原文件结构
    output_data = {"items": new_data} if is_wrapper else new_data

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ 注入完成！最终训练数据已保存至: {OUTPUT_FILE}")
    print("✨ 现在，模型只要看到这个 Tag，就知道必须根据 SOP 行动了。")

if __name__ == "__main__":
    main()
