#!/usr/bin/env python3
import json

OUTPUT_FORMAT = """

输出格式规范：
Agent 的回复必须包含两个区块，顺序固定：
BEGIN_META
action=...
thought=...
slot_age=0/1
slot_gender=0/1
...
END_META
BEGIN_FINAL
(面向用户的最终回复)
END_FINAL

BEGIN_META 仅用于程序解析，采用 key=value 的纯文本格式，不得使用 JSON。
BEGIN_FINAL 为用户可见回复，必须遵守语言风格约束。
若 User Input 包含 <action>...</action>，必须在 BEGIN_META 的 action 中写明，并在 BEGIN_FINAL 中执行。"""

def add_format_to_system(content):
    """检查并添加输出格式规范到 system content 末尾"""
    marker = "BEGIN_META"
    if marker in content:
        # 已有格式规范，跳过
        return content, False
    return content + OUTPUT_FORMAT, True

def process_file(input_path, output_path=None):
    if output_path is None:
        output_path = input_path
    
    modified_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        total_count += 1
        try:
            data = json.loads(line)
            for msg in data.get('messages', []):
                if msg.get('role') == 'system':
                    original_content = msg.get('content', '')
                    new_content, was_modified = add_format_to_system(original_content)
                    if was_modified:
                        msg['content'] = new_content
                        modified_count += 1
                    break
            modified_lines.append(json.dumps(data, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {e}")
            continue
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in modified_lines:
            f.write(line + '\n')
    
    print(f"处理完成: 总计 {total_count} 条数据, 新增格式规范 {modified_count} 条")

if __name__ == '__main__':
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'golden_history_input.jsonl'
    process_file(input_file)
