#!/usr/bin/env python3
"""
从 golden_history_input.jsonl 中提取涉及"分龄定向留联"规则的对话，
按 single_turn_rl_contact_stage 的格式输出为 JSONL。
"""

import json
import os
import random

# 配置路径
INPUT_FILE = "/data/chengch/lm-evaluation-harness/itbench-master/itbench-master/output/v0.1_qwen3_8b_normal_mid_short_step350_golden_history_input_20260417_155221/golden_history_input.jsonl"
OUTPUT_DIR = "/data/chengch/project/rl_remake/outputs/single_turn_rl_contact_stage_2k_qwen3_8b_normal_mid_short_step350"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("从 golden_history_input.jsonl 中提取包含'分龄定向留联'的对话...")

    extracted = []
    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
                key = obj.get("key", "")
                messages = obj.get("messages", [])
                rule_list = obj.get("rule_list", [])

                # 检查是否包含"分龄定向留联"
                has_contact_rule = False
                system_prompt = ""
                conversations = []
                num_turns = 0

                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    turn_id = msg.get("turn_id", 0)

                    if role == "system":
                        system_prompt = content
                        if "分龄定向留联" in content:
                            has_contact_rule = True
                    elif role in ("user", "assistant"):
                        conversations.append({"role": role, "content": content, "turn_id": turn_id})
                        if role == "user":
                            num_turns = max(num_turns, turn_id)

                if not has_contact_rule:
                    continue

                if not conversations:
                    print(f"  [WARN] {key}: 没有对话内容")
                    continue

                # prompt 包含 system + 全部对话（保留 user 和 assistant）
                prompt_msgs = [{"role": "system", "content": system_prompt}]
                for msg in conversations:
                    prompt_msgs.append({"role": msg["role"], "content": msg["content"]})

                # ground_truth 是最后一条 assistant 消息（contact 触发轮的回复）
                ground_truth = ""
                for msg in reversed(conversations):
                    if msg.get("role") == "assistant":
                        ground_truth = msg.get("content", "")
                        break

                if not ground_truth:
                    print(f"  [WARN] {key}: 没有 assistant 回复")
                    continue

                # extra_info
                extra_info = {
                    "sample_id": f"normal_{key}",
                    "source": "normal",
                    "conv_id": key,
                    "turn_id": num_turns,
                    "slice_bucket": "contact_stage",
                    "original_system_prompt": system_prompt,
                    "rule_contact_round": num_turns,
                    "rule_age_gender_deadline_round": 3,
                    "turn_round": num_turns,
                    "question": conversations[-1]["content"] if conversations else "",
                    "conversations": conversations,
                    "exclude_last_turn": False,
                    "truncation_policy": "contact_stage_per_conversation"
                }

                # reward_model
                reward_model = {
                    "style": "contact_stage_rule",
                    "ground_truth": ground_truth,
                    "target_round": num_turns,
                    "contact_round": num_turns,
                    "age_gender_deadline_round": 3
                }

                record = {
                    "prompt": prompt_msgs,
                    "ground_truth": ground_truth,
                    "extra_info": extra_info,
                    "reward_model": reward_model,
                    "data_source": "normal",
                    "agent_name": "single_turn_agent",
                    "index": f"normal_{key}"
                }

                extracted.append(record)
                print(f"  提取: {key}, turn_id={num_turns}, rules={rule_list}")

            except Exception as e:
                print(f"  [ERROR] 处理行失败: {e}")
                continue

    print(f"\n共扫描 {total} 条，提取 {len(extracted)} 条涉及'分龄定向留联'的对话")

    if not extracted:
        print("没有提取到任何数据。")
        return

    # 打乱 + 划分
    random.seed(42)
    random.shuffle(extracted)

    val_size = max(10, int(len(extracted) * 0.05))
    val_data = extracted[:val_size]
    train_data = extracted[val_size:]

    print(f"  train: {len(train_data)}, val: {len(val_data)}")

    # 写入文件
    all_path = os.path.join(OUTPUT_DIR, "single_turn_rl_contact_stage.all.jsonl")
    train_path = os.path.join(OUTPUT_DIR, "single_turn_rl_contact_stage.train.jsonl")
    val_path = os.path.join(OUTPUT_DIR, "single_turn_rl_contact_stage.val.jsonl")

    for path, data in [(all_path, extracted), (train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  写入: {path} ({len(data)} 条)")

    # turn_round 统计
    turn_counts = {}
    for r in extracted:
        tr = str(r["extra_info"]["turn_round"])
        turn_counts[tr] = turn_counts.get(tr, 0) + 1

    # 写入 stats.json
    stats = {
        "seed": 42,
        "targets": {
            "total_samples": len(extracted),
            "val_size": val_size,
            "strict_contact_signal": True
        },
        "selected_counts": {
            "total": len(extracted),
            "train": len(train_data),
            "val": len(val_data),
            "by_source": {"normal": len(extracted)}
        },
        "split_stats": {
            "all": {
                "rows": len(extracted),
                "by_source": {"normal": len(extracted)},
                "unique_conversations": len(extracted),
                "turn_round": dict(sorted(turn_counts.items(), key=lambda x: int(x[0])))
            },
            "train": {
                "rows": len(train_data),
                "by_source": {"normal": len(train_data)},
                "unique_conversations": len(train_data)
            },
            "val": {
                "rows": len(val_data),
                "by_source": {"normal": len(val_data)},
                "unique_conversations": len(val_data)
            }
        },
        "outputs": {
            "all_jsonl": all_path,
            "train_jsonl": train_path,
            "val_jsonl": val_path
        }
    }

    stats_path = os.path.join(OUTPUT_DIR, "single_turn_rl_contact_stage.stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  写入: {stats_path}")

    print("\n完成！")

if __name__ == "__main__":
    main()
