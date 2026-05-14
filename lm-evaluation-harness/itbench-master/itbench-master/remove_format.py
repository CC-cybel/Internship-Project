#!/usr/bin/env python3
import json

FORMAT_MARKER = "输出格式规范："

def remove_format_from_system(content):
    """删除 output format 块"""
    if FORMAT_MARKER not in content:
        return content, False
    
    # 找到 marker 的位置
    idx = content.find(FORMAT_MARKER)
    # 删除 marker 及之后的所有内容
    new_content = content[:idx].rstrip()
    return new_content, True

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
                    new_content, was_modified = remove_format_from_system(original_content)
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
    
    print(f"处理完成: 总计 {total_count} 条数据, 移除格式规范 {modified_count} 条")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法: python remove_format.py <文件1> [文件2] ...")
        sys.exit(1)
    
    for input_file in sys.argv[1:]:
        print(f"处理: {input_file}")
        process_file(input_file)
