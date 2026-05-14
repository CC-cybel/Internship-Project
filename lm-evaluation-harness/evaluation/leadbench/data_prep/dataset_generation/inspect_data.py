import pandas as pd
import json

# Inspect JSONL
try:
    with open('/data1/yezj/gitlab/leadbench/data/dataset/golden_history_input.jsonl', 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line:
            data = json.loads(first_line)
            print("JSONL Keys:", data.keys())
            print("JSONL Sample:", json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print("JSONL is empty")
except Exception as e:
    print(f"Error reading JSONL: {e}")

print("-" * 20)

# Inspect Excel
try:
    df = pd.read_excel('/data1/yezj/gitlab/leadbench/tmp/data/normal_anti_hijack_abc_stage2_6_49段.xlsx', nrows=3)
    print("Excel Columns:", df.columns.tolist())
    print("Excel Head:", df.head().to_dict(orient='records'))
except Exception as e:
    print(f"Error reading Excel: {e}")
