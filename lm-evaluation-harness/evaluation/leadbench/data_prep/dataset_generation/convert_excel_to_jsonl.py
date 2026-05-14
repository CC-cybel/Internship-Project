import pandas as pd
import json
import os
import shutil

# Paths
excel_path = '/data1/yezj/gitlab/leadbench/tmp/data/normal_anti_hijack_abc_stage2_6_49段.xlsx'
jsonl_path = '/data1/yezj/gitlab/leadbench/data/dataset/golden_history_input.jsonl'
jsonl_bak_path = jsonl_path + '.bak'

# 1. Backup existing JSONL and read System Prompt
system_prompt_content = ""
if os.path.exists(jsonl_path):
    shutil.copy(jsonl_path, jsonl_bak_path)
    print(f"Backed up {jsonl_path} to {jsonl_bak_path}")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                data = json.loads(first_line)
                for msg in data.get('messages', []):
                    if msg['role'] == 'system':
                        system_prompt_content = msg['content']
                        break
    except Exception as e:
        print(f"Error reading system prompt: {e}")

if not system_prompt_content:
    print("Warning: Could not find system prompt in existing file. Using default empty prompt.")
    system_prompt_content = "角色设定：你是一名专业的心理咨询师..." # Fallback? Or just empty?

print(f"System Prompt Length: {len(system_prompt_content)}")

# 2. Read Excel
print(f"Reading Excel from {excel_path}...")
df = pd.read_excel(excel_path)

# 3. Process Dialogues
samples = []
dialog_ids = df['dialog_id'].unique()
print(f"Found {len(dialog_ids)} dialogues.")

for d_id in dialog_ids:
    # Get all rows for this dialog, sorted by sentence_id (to maintain order)
    dialog_rows = df[df['dialog_id'] == d_id].sort_values('sentence_id')
    
    # Identify Rounds
    rounds = sorted(dialog_rows['round'].unique())
    
    history = []
    # Add System Prompt to history start (conceptually, but we construct messages list explicitly)
    
    # We maintain a list of message dicts for the history
    history_messages = [{"role": "system", "content": system_prompt_content, "turn_id": 0}]
    
    # Turn counter (1-based for User turns)
    turn_counter = 1
    
    for r in rounds:
        round_rows = dialog_rows[dialog_rows['round'] == r]
        
        # Extract User Content
        # Role could be SEARCH or CLIENT
        user_rows = round_rows[round_rows['role'].isin(['SEARCH', 'CLIENT'])]
        if user_rows.empty:
            continue # Skip if no user input in this round
            
        # Concatenate user messages if multiple (rare, but safe)
        user_content = "\n".join(user_rows['sentence'].astype(str).tolist())
        
        # Extract Assistant Content
        # Role could be SERVER
        asst_rows = round_rows[round_rows['role'] == 'SERVER']
        asst_content = ""
        if not asst_rows.empty:
             asst_content = "\n".join(asst_rows['sentence'].astype(str).tolist())
        
        # Current Turn ID is turn_counter
        current_turn_id = turn_counter
        
        # Check if we should generate a sample
        # "Start from 2nd turn" -> means start from Turn 2 (Round 1 if Round 0 exists)
        # So if current_turn_id >= 2, we generate a sample
        if current_turn_id >= 2:
            # Construct sample messages: History + Current User
            sample_messages = history_messages.copy()
            sample_messages.append({
                "role": "user",
                "content": user_content,
                "turn_id": current_turn_id
            })
            
            sample = {
                "id": f"{d_id}_{current_turn_id}",
                "messages": sample_messages,
                "turn_id": current_turn_id # Helper for processing
            }
            samples.append(sample)
        
        # Update History for next turn
        history_messages.append({
            "role": "user",
            "content": user_content,
            "turn_id": current_turn_id
        })
        
        if asst_content:
            history_messages.append({
                "role": "assistant",
                "content": asst_content,
                "turn_id": current_turn_id
            })
            
        turn_counter += 1

# 4. Save to JSONL
print(f"Generated {len(samples)} samples.")
with open(jsonl_path, 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"Saved to {jsonl_path}")
