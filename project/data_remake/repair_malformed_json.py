import json
import re

# ==================== 🎛️ 配置区域 (开关) ====================

# 设置为 True  => 🚀 训练模式 (GPT回复转为字符串，适合模型训练)
# 设置为 False => 👓 阅读模式 (GPT回复保持JSON对象，适合人工检查)
FOR_TRAINING = False

# 文件名设置
INPUT_FILE = 'experiments/to_fix.txt'          # 你的原始问题文件
OUTPUT_FILE = 'experiments/to_fix_fixed.json'  # 修复后生成的文件

# ============================================================

def fix_and_convert(input_path, output_path, is_training_mode):
    print(f"🔧 正在处理文件: {input_path}")
    print(f"当前模式: {'🚀 训练模式 (Value -> String)' if is_training_mode else '👓 阅读模式 (Value -> Object)'}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # --- 步骤 1: 基础语法修复 (修复 System 里的引号问题) ---
        # 定位 "system": "..." 之间的内容
        pattern = r'(?<="system": ")([\s\S]*?)(?=",[\s\n]*"extracted_info")'
        
        def escape_inner_quotes(match):
            content = match.group(1)
            # 将未转义的 " 替换为 \"
            return re.sub(r'(?<!\\)"', r'\\"', content)

        fixed_text = re.sub(pattern, escape_inner_quotes, raw_content)

        # --- 步骤 2: 解析为 Python 对象 ---
        try:
            data = json.loads(fixed_text)
        except json.JSONDecodeError as e:
            print(f"❌ 修复引号后仍无法解析 JSON，请检查源文件格式。\n错误信息: {e}")
            return

        # --- 步骤 3: 根据模式转换 GPT 的 Value ---
        process_count = 0
        
        if "conversations" in data:
            for item in data["conversations"]:
                # 只处理 GPT 的回复
                if item.get("from") == "gpt":
                    value = item.get("value")

                    # 模式 A: 训练模式 (我们需要字符串)
                    if is_training_mode:
                        # 如果当前是字典(Object)，就转成字符串
                        if isinstance(value, dict):
                            item["value"] = json.dumps(value, ensure_ascii=False)
                            process_count += 1
                    
                    # 模式 B: 阅读模式 (我们需要对象)
                    else:
                        # 如果当前是字符串，尝试转回字典(Object)方便阅读
                        if isinstance(value, str):
                            try:
                                # 只有当字符串看起来像 JSON 时才转
                                if value.strip().startswith("{"):
                                    item["value"] = json.loads(value)
                                    process_count += 1
                            except:
                                pass # 如果不是JSON字符串，保持原样

        # --- 步骤 4: 保存文件 ---
        with open(output_path, 'w', encoding='utf-8') as f:
            # indent=2 让文件有缩进，好看；ensure_ascii=False 保证中文显示正常
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ 处理完成！(变动了 {process_count} 处 GPT 回复)")
        print(f"📄 结果已保存至: {output_path}")
        
        # --- 步骤 5: 打印一个预览给用户看 ---
        print("\n🔎 [效果预览] 第一条 GPT 回复现在的样子:")
        first_gpt = next((i for i in data["conversations"] if i["from"] == "gpt"), None)
        if first_gpt:
            preview = first_gpt["value"]
            if isinstance(preview, str):
                print(f"类型: String (字符串)\n内容: {preview[:100]}...") 
            else:
                print(f"类型: Object (对象)\n内容: {json.dumps(preview, ensure_ascii=False)[:100]}...")

    except FileNotFoundError:
        print(f"❌ 找不到文件: {input_path}")

if __name__ == "__main__":
    fix_and_convert(INPUT_FILE, OUTPUT_FILE, FOR_TRAINING)
