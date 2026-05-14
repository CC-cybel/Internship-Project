import pandas as pd
import json
import os

# Input and output paths
excel_path = '/data1/yezj/gitlab/leadbench-excellent/data_prep/data/normal_anti_hijack_abc_stage2_6_49段.xlsx'
example_jsonl_path = '/data1/yezj/gitlab/leadbench-excellent/data/dataset/golden_history_input_v1.jsonl'
output_jsonl_path = '/data1/yezj/gitlab/leadbench-excellent/data/dataset/normal_anti_hijack_abc_stage2.jsonl'

# Read the system prompt from the example file
system_prompt_content = ""
try:
    with open(example_jsonl_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line:
            data = json.loads(first_line)
            for msg in data.get('messages', []):
                if msg.get('role') == 'system':
                    system_prompt_content = msg.get('content', "")
                    break
except Exception as e:
    print(f"Error reading system prompt: {e}")
    # Fallback if needed
    print("Using empty system prompt.")

# Read the Excel file
print(f"Reading Excel file: {excel_path}")
df = pd.read_excel(excel_path)

# Sort by dialog_id, round, and sentence_id if possible
sort_cols = ['dialog_id', 'round']
if 'sentence_id' in df.columns:
    sort_cols.append('sentence_id')
    
df.sort_values(by=sort_cols, inplace=True)

# Group by dialog_id
grouped = df.groupby('dialog_id', sort=False)

output_data = []
processed_count = 0

for dialog_id, group in grouped:
    messages = []
    
    # Add system message first
    messages.append({
        "role": "system",
        "content": system_prompt_content,
        "turn_id": 0
    })
    
    # Iterate through rows sequentially
    for _, row in group.iterrows():
        role_raw = row['role']
        content = str(row['sentence']).strip()
        
        # Skip empty content or NaN
        if not content or content == 'nan':
            continue
            
        # Map roles
        if role_raw in ['SEARCH', 'CLIENT']:
            role = 'user'
        elif role_raw == 'SERVER':
            role = 'assistant'
        else:
            continue # Skip unknown roles
            
        # Check if we should merge with the last message
        last_msg = messages[-1]
        
        if last_msg['role'] == role:
            # Merge
            last_msg['content'] += "<sep>" + content
        else:
            # New message
            messages.append({
                "role": role,
                "content": content,
                "turn_id": 0 # Placeholder, will update later
            })
            
    # NEW LOGIC: Remove last message if it is from user
    if messages and messages[-1]['role'] == 'user':
        messages.pop()
            
    # Re-assign turn_ids
    current_turn = 0
    final_messages = []
    
    for msg in messages:
        if msg['role'] == 'system':
            msg['turn_id'] = 0
        elif msg['role'] == 'user':
            current_turn += 1
            msg['turn_id'] = current_turn
        elif msg['role'] == 'assistant':
            if current_turn == 0:
                current_turn = 1
            msg['turn_id'] = current_turn
            
        final_messages.append(msg)
    
    # Only add if we have actual conversation besides system prompt
    if len(final_messages) > 1:
        entry = {
            "id": f"{processed_count + 1:03d}",
            "raw_id": str(dialog_id),
            "messages": final_messages
        }
        output_data.append(entry)
        processed_count += 1

# Write to JSONL
print(f"Writing {len(output_data)} entries to {output_jsonl_path}")
with open(output_jsonl_path, 'w', encoding='utf-8') as f:
    for entry in output_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("Conversion complete.")
