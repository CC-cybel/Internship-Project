
import pandas as pd
import json
import os
import shutil
import random

# Paths
excel_path = '/data1/yezj/gitlab/leadbench/data_prep/data/normal_anti_hijack_abc_stage2_6_49段.xlsx'
output_path = '/data1/yezj/gitlab/leadbench/data_prep/dataset_generation/golden_history_input.jsonl'
backup_path = output_path + '.bak'

# Backup existing if not already backed up or overwrite?
# Since we are iterating, let's keep the original backup if it exists, or create one if not.
if os.path.exists(output_path):
    # Only backup if backup doesn't exist to preserve original? 
    # Or always backup current? Let's backup current.
    shutil.copy2(output_path, backup_path)
    print(f"Backed up {output_path} to {backup_path}")

# Read existing system prompt from backup (which might be the one we just wrote, or the original)
# If we just overwrote it, the backup has the previous version.
# If we want the ORIGINAL system prompt, we should rely on the one from the very first run if possible.
# But the user provided context shows the system prompt is consistent.
# Let's try to read from backup.

system_prompt_content = """角色设定：你是一名精通生物心理学的医疗咨询专家，拥有极强的人性洞察力，负责高效收集信息并引导转化。

[基本信息]医院：厦门仙岳医院 地址：厦门仙岳路137号 上班时间：上午9点到下午6点，周末不休 联系号码：13459015421（微信同号） [语言风格与去 AI 味规范] (Hard Constraints)：

1. 禁止官话：严禁使用“为了更好地为您服务”、“敬请谅解”等客服套话。

2. 禁止使用引号，破折号或者括号进行解释，应该口语化。

3. 拒绝列表：严禁使用 1. 2. 3. 列表式回复，必须使用自然段落。

4. 语气要求：专业、自信、不卑不亢，必要时可适当示弱或施压。

硬性执行指标 (Hard Rules)：

1. [身份锚定]：前面2轮询问和症状相关的信息，首轮如果用户有疑问，需要回复，并吸引用户，3和4轮询问基本信息，闭合式问诊，不要让用户多思考

2. [留联触发]：当对话进行到第 5 轮时（或收集完主诉后），强制启动首次留联尝试。

3. [询问密度]：每一轮最多提问 1 个问题。

4. [终止红线]：严禁在成功获取电话或微信号之前，主动结束对话或引导结束，必须持续寻找话题保持沟通。如果已经获取了联系方式可以主动结束沟通，但需要注意方法。

5. [意向套电]：在聊天过程中，如果用户表达了留联的意愿，必须立即让用户留下响应的微信或者电话联系方式。

获客与拒绝策略：

- 转化借口：根据用户的意图和症状，以“详细讲解成因”、“后期应对方案”及“一对一免费建议指导”为钩子引导留联。

- 留联优先级：优先获取微信联系方式，若用户拒绝或未提供，才考虑获取电话联系方式。

- 降级策略：若均被拒，以“名额保留”或“医疗风险”为由进行最后挽留。"""
# if os.path.exists(backup_path):
#     try:
#         with open(backup_path, 'r', encoding='utf-8') as f:
#             first_line = f.readline()
#             if first_line:
#                 data = json.loads(first_line)
#                 if data.get('messages') and data['messages'][0]['role'] == 'system':
#                     system_prompt_content = data['messages'][0]['content']
#     except Exception as e:
#         print(f"Error reading system prompt from backup: {e}")

if not system_prompt_content:
    # Fallback or empty
    print("Warning: No system prompt found. Using empty string.")

print(f"System Prompt Length: {len(system_prompt_content)}")

# Read Excel
print(f"Reading Excel from {excel_path}...")
df = pd.read_excel(excel_path)

# Map roles
role_map = {
    'SEARCH': 'user',
    'CLIENT': 'user',
    'SERVER': 'assistant'
}

# Group by dialog_id
dialog_ids = df['dialog_id'].unique()
print(f"Found {len(dialog_ids)} dialogues.")

samples = []

for d_id in dialog_ids:
    # Sort by sentence_id to preserve order
    dialog_rows = df[df['dialog_id'] == d_id].sort_values('sentence_id')
    
    # Construct raw message list
    raw_messages = []
    
    for _, row in dialog_rows.iterrows():
        role = role_map.get(row['role'])
        if not role:
            continue
            
        content = str(row['sentence'])
        raw_messages.append({
            "role": role,
            "content": content,
            "original_round": row['round']
        })

    # Merge consecutive messages
    merged_messages = []
    if raw_messages:
        curr = raw_messages[0]
        for next_msg in raw_messages[1:]:
            if next_msg['role'] == curr['role']:
                curr['content'] += "<sep>" + next_msg['content']
                curr['original_round'] = next_msg['original_round']
            else:
                merged_messages.append(curr)
                curr = next_msg
        merged_messages.append(curr)
        
    final_messages_for_dialog = [{"role": "system", "content": system_prompt_content, "turn_id": 0}]
    
    current_turn_id = 0
    
    for msg in merged_messages:
        role = msg['role']
        content = msg['content']
        
        if role == 'user':
            last_role = final_messages_for_dialog[-1]['role']
            if last_role in ['system', 'assistant']:
                current_turn_id += 1
            
        final_messages_for_dialog.append({
            "role": role,
            "content": content,
            "turn_id": current_turn_id
        })

    # Generate samples
    history = []
    # Find max turn_id to exclude last turn
    max_turn_id = 0
    if final_messages_for_dialog:
        max_turn_id = final_messages_for_dialog[-1]['turn_id']

    for i, msg in enumerate(final_messages_for_dialog):
        history.append(msg)
        
        # Check if user message, turn_id >= 2, AND NOT last turn
        if msg['role'] == 'user' and msg['turn_id'] >= 2:
            if msg['turn_id'] == max_turn_id:
                # Skip the last turn
                continue

            raw_id = f"{d_id}_{msg['turn_id']}"
            sample_msgs = [m.copy() for m in history]
            
            # Construct sample WITHOUT top-level turn_id
            sample = {
                "id": "", # Placeholder, will be set sequentially
                "raw_id": raw_id,
                "messages": sample_msgs
            }
            samples.append(sample)

print(f"Generated {len(samples)} samples.")

# Shuffle samples
random.seed(42) # For reproducibility
random.shuffle(samples)

# Assign sequential IDs
for idx, sample in enumerate(samples):
    sample['id'] = f"{idx+1:03d}"

# Save to JSONL
with open(output_path, 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

print(f"Saved to {output_path}")
