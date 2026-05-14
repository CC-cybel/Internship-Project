import random
import os
import json
from typing import Any

# 定义文件名
# 通用数据集
FILE_MIX = "experiments/general.jsonl" 
# 领域数据集
FILE_NOTIME = "outputs/normal/normal_s5_dual_full_drop_system_keywords.json"
FILE_OUTPUT = "normal_dual_full_mix.jsonl"


def sanitize_sharegpt_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item

    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return item

    cleaned_conversations = []
    for turn in conversations:
        if not isinstance(turn, dict):
            return None

        if set(turn.keys()) != {"from", "value"}:
            return None

        role = turn.get("from", turn.get("role"))
        value = turn.get("value")

        if not isinstance(role, str):
            return None

        if not isinstance(value, str):
            return None

        cleaned_conversations.append({"from": role, "value": value})

    cleaned_item = {"conversations": cleaned_conversations}
    if isinstance(item.get("system"), str):
        cleaned_item["system"] = item["system"]
    if "tools" in item:
        cleaned_item["tools"] = item["tools"]

    return cleaned_item


def load_as_jsonl_lines(path: str) -> list[str]:
    """兼容读取 jsonl / json，统一返回 jsonl 行列表（每行一个 JSON 字符串）。"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    dropped_count = 0
    # 优先按 json 解析（可兼容 list 或 {'items': [...]})
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            lines = []
            for item in parsed:
                cleaned = sanitize_sharegpt_item(item)
                if cleaned is None:
                    dropped_count += 1
                    continue
                lines.append(json.dumps(cleaned, ensure_ascii=False))
            if dropped_count:
                print(f"   - 🗑 丢弃异常样本: {dropped_count} 条")
            return lines
        if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
            lines = []
            for item in parsed["items"]:
                cleaned = sanitize_sharegpt_item(item)
                if cleaned is None:
                    dropped_count += 1
                    continue
                lines.append(json.dumps(cleaned, ensure_ascii=False))
            if dropped_count:
                print(f"   - 🗑 丢弃异常样本: {dropped_count} 条")
            return lines
        # 若是单个 dict，也转成一行
        if isinstance(parsed, dict):
            cleaned = sanitize_sharegpt_item(parsed)
            if cleaned is None:
                print("   - 🗑 丢弃异常样本: 1 条")
                return []
            return [json.dumps(cleaned, ensure_ascii=False)]
    except json.JSONDecodeError:
        pass

    # 解析失败则回退为 jsonl（逐行）
    jsonl_lines = []
    for line in content.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed_line = json.loads(raw)
            cleaned = sanitize_sharegpt_item(parsed_line)
            if cleaned is None:
                dropped_count += 1
                continue
            jsonl_lines.append(json.dumps(cleaned, ensure_ascii=False))
        except json.JSONDecodeError:
            dropped_count += 1
            continue
    if dropped_count:
        print(f"   - 🗑 丢弃异常样本: {dropped_count} 条")
    return jsonl_lines

def main():
    final_lines = []

    # 1. 处理 mix.jsonl (随机取 50%)
    if os.path.exists(FILE_MIX):
        print(f"📖 正在读取 {FILE_MIX} ...")
        mix_lines = load_as_jsonl_lines(FILE_MIX)
        
        total_mix = len(mix_lines)
        target_count = total_mix // 5  # 整除2，取一半
        
        print(f"   - 总行数: {total_mix}")
        print(f"   - ✂ 随机采样 50%: {target_count} 条")
        
        # 随机采样一半
        sampled_mix = random.sample(mix_lines, target_count) if target_count > 0 else []
        final_lines.extend(sampled_mix)
    else:
        print(f"⚠ 警告: 找不到 {FILE_MIX}，跳过。")

    # 2. 处理 FILE_NOTIME（json/jsonl 都兼容，全部保留）
    if os.path.exists(FILE_NOTIME):
        print(f"📖 正在读取 {FILE_NOTIME} ...")
        notime_lines = load_as_jsonl_lines(FILE_NOTIME)
        
        print(f"   - ✅ 全部保留: {len(notime_lines)} 条")
        final_lines.extend(notime_lines)
    else:
        print(f"⚠ 警告: 找不到 {FILE_NOTIME}，跳过。")

    # 3. 全局打乱
    print(f"🔄 正在混合并全局打乱共 {len(final_lines)} 条数据...")
    random.shuffle(final_lines)

    # 4. 写入新的 train.jsonl (会自动覆盖旧文件)
    print(f"💾 正在写入 {FILE_OUTPUT} ...")
    with open(FILE_OUTPUT, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(line + '\n')

    print("✨ 完成！你可以去卸载旧数据了。")

if __name__ == "__main__":
    main()
