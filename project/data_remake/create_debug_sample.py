import json
import random
import os

# ================= 配置区域 =================
# 1. 输入的大文件路径 (请改为您修复好的那个文件)
INPUT_FILE = "experiments/sft_hard_samples.json"  # 或者 "/data/wangpf/data/origin_extract_data_fixed.json"

# 2. 输出的小样本文件路径
OUTPUT_FILE = "experiments/sft_hard_samples.json"

# 3. 想抽多少条？
SAMPLE_SIZE = 20000
# ===========================================

def create_sample():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 文件未找到: {INPUT_FILE}")
        return

    print(f"🚀 正在加载大文件: {INPUT_FILE} ...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_count = len(data)
        print(f"📊 原始数据共 {total_count} 条")

        # 确保不会抽多了
        real_sample_size = min(SAMPLE_SIZE, total_count)
        
        # 随机抽取
        print(f"🎲 正在随机抽取 {real_sample_size} 条数据...")
        sampled_data = random.sample(data, real_sample_size)

        # 保存
        print(f"💾 正在保存至: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=4)
            
        print("\n✅ 抽样完成！")
        print(f"👉 接下来可以把 '{OUTPUT_FILE}' 交给后续调试或转换脚本继续处理。")

    except json.JSONDecodeError:
        print("❌ JSON 解析失败：文件格式可能不正确，请检查是否使用了修复后的文件。")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    create_sample()
