import json
import os
import random
import re

# ================= 配置区域 =================
INPUT_FILE = "/data/chengch/project/data_remake/runs/hard_reverse_tongyi_v2.json"        # 已经注入了轮次标签的文件
OUTPUT_FILE = "/data/chengch/project/data_remake/runs/hard_reverse_tongyi_v2_action.json" # 输出文件

# 注入概率：每一轮 GPT 回复都有 20% 的概率触发
TURN_INJECTION_PROB = 0.2

def extract_think_content(text):
    """
    提取 <think> 标签内的内容。
    支持多行匹配，并去除首尾空格。
    """
    # re.DOTALL 确保 . 能匹配换行符
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        content = match.group(1).strip()
        # 如果标签内容太长（超过50字），可能不适合作为指令，可以选择跳过
        if len(content) > 50: 
            return None
        # 如果内容包含多个逗号（如"暖场,问诊,安慰"），通常取第一个动作作为核心指令最稳健
        # 这里保留原样，您也可以改为 return content.split(',')[0]
        return content
    return None

def inject_actions_robust(data):
    total_dialogues = len(data)
    total_injected_turns = 0
    
    print(f"🔄 开始处理 {total_dialogues} 段对话...")
    
    for entry in data:
        conversations = entry.get('conversations', [])
        
        # 遍历对话，寻找 [Human] -> [GPT] 的结构
        # 从索引 1 开始，检查 i (GPT) 和 i-1 (Human)
        for i in range(1, len(conversations)):
            gpt_turn = conversations[i]
            prev_turn = conversations[i-1]
            
            # 校验角色关系
            if gpt_turn['from'] == 'gpt' and prev_turn['from'] == 'human':
                
                # 1. 概率判定 (20%)
                if random.random() > TURN_INJECTION_PROB:
                    continue
                
                # 2. 提取 GPT 的真实意图 (Ground Truth)
                gpt_text = gpt_turn['value']
                
                # 有些数据可能是 JSON 格式的字符串 (thought/response)，需要先解析
                # 如果您的数据已经是清洗过的纯文本带 <think>，则直接用
                if isinstance(gpt_text, str) and "<think>" in gpt_text:
                    action_content = extract_think_content(gpt_text)
                elif isinstance(gpt_text, dict) and "thought" in gpt_text:
                     # 兼容如果 value 已经是字典的情况
                     action_content = gpt_text.get("thought", "")
                else:
                    action_content = None

                # 3. 执行注入
                if action_content:
                    # 构造 Action 标签
                    action_tag = f"\n<action>{action_content}</action>"
                    
                    original_user_text = prev_turn['value']
                    
                    # 防止重复注入
                    if "<action>" in original_user_text:
                        continue
                        
                    # 4. 插入位置控制：插在 【系统数据...】 之前
                    if "\n【系统数据：" in original_user_text:
                        parts = original_user_text.split("\n【系统数据：")
                        main_text = parts[0]
                        sys_tag = "\n【系统数据：" + parts[1]
                        
                        # 最终形态: 用户文本 \n<action>...</action> \n【系统数据...】
                        new_text = f"{main_text}{action_tag}{sys_tag}"
                    else:
                        # 兜底：如果没有系统标签，直接拼后面
                        new_text = f"{original_user_text}{action_tag}"
                    
                    # 更新数据
                    prev_turn['value'] = new_text
                    total_injected_turns += 1

    print(f"✅ 处理完成！")
    print(f"📊 统计：共 {total_dialogues} 段对话，累计注入了 {total_injected_turns} 个 Action 指令。")
    return data

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return

    print(f"📂 读取数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容列表或字典包装格式
    if isinstance(data, dict) and 'items' in data:
        process_list = data['items']
        is_wrapper = True
    elif isinstance(data, list):
        process_list = data
        is_wrapper = False
    else:
        print("❌ 数据格式错误，根节点必须是 list 或包含 'items' 的 dict")
        return

    # 执行处理
    new_data = inject_actions_robust(process_list)
    
    # 保持原格式输出
    output_data = {"items": new_data} if is_wrapper else new_data

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"💾 结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
