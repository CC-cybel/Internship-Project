import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = "/data/chengch/project/data_remake/runs/hard_rewrite_v2.json"  # 您的重写结果文件
REPORT_FILE = "/data/chengch/project/data_remake/intermediatedialogue_quality_report.json"
BAD_CASE_FILE = "/data/chengch/project/data_remake/outputs/judge/failed_dialogues.json"
STATS_FILE = "/data/chengch/project/data_remake/outputs/judge/dialogue_quality_stats.json"
CACHE_DIR = "/data/chengch/project/data_remake/cache/judge"
SAVE_EVERY = 50
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0
RETRY_JITTER_SECONDS = 0.3
MAX_WORKERS = 40
PROCESSED_FLAG = "_judge_done"

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "your_api_key_here")
BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.environ.get("DASHSCOPE_MODEL_NAME", "deepseek-v4-flash")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
cache_lock = threading.Lock()

# 载入 Prompt
PROMPT_TEMPLATE = """
# Role
你是一名严格的**医疗销售对话质量总监**
你的特长是**“揪出幻觉”**。你不相信 gpt 说的任何信息，除非能在 human 的**历史发言**中找到证据。
你的任务是审核一段**完整的医患对话记录**，判断那个“身披白大褂的销售专家”（GPT）是否合格。

# 审核标准 (The Gold Standard)

## 1. 红线审查 (Hard Fail) - 只要出现一次，整段对话直接 0 分
* **未来泄露**：GPT 是否在前期轮次使用了后期才知道的信息，包括但不限于性别，年龄，姓氏，名字等任何信息？
* **称呼幻觉**：gpt 对用户的称呼，必须能在**上文 user 的输入中找到明确来源**。
* **信息幻觉**：GPT是否使用了根本不存在的信息
* **实体幻觉**：System Prompt 没提供医院名，GPT 却编造了具体的医院/医生名？
* **Action 失效**：User Input 有 `<action>` 指令，但 GPT 实际未执行？
* **严重 AI 味**：在**response**中使用了装饰性双引号（“”）和单引号（‘’）、破折号、或翻译腔。

## 2. 质量验收 (Quality Check) - 1~5 分
## 1-2分：软弱的客服 (Weak)
* **特征**：SOP 流程是对的（问了年龄/要了电话），但是**语气太软、太客气**。
* **典型表现**：使用了太多礼貌用语，用了亲密称呼。
* **评价**：像个淘宝客服，完全没有医生/销冠的气场。
## 3分：机械的机器人 (Mediocre)
* **特征**：没有废话，也没有 AI 味，但**缺乏逻辑连接**。
* **典型表现**：直接扔问题“多大？男的女的？”，但前面没有**医学定性**或**安抚**。
* **评价**：冷漠、生硬，虽然完成了任务，但用户体验不好。
## 4分：合格的医生 (Good)
* **特征**：有医学定性，流程也对。
* **扣分点**：要电话时的**“钩子”不够硬**（比如理由不够充分，或者诱饵不够诱人）。
## 5分：顶级的销冠 (Excellent)
* **特征**：
    1.  **回应感**：精准定性了用户的痛苦。
    2.  **压迫感**：问诊节奏极快，不容置疑。
    3.  **交易感**：要电话时，把“诱饵”包装得天衣无缝。
* thought与response自洽，言行合一

# Input Data
{full_conversation_json}

# Output Format (JSON Only)
请输出一个包含整体评价的 JSON 对象：
{
    "score": 0-5,  // 0=触犯红线，1-2=差，3=及格，4=优，5=完美销冠
    "pass": true/false, // score >= 4 为 true
    "violations": [
            {
                "type": "AI_Punctuation",  // 违规类型
                "evidence": "原文摘录：‘焦虑’不是病" // 必须摘录原文！如果摘不出来，此项作废。
            }
        ], //如果无违规则此项为空列表[]
    "overall_comment": "简短评价整段对话的表现，如：SOP执行完美，但第3轮语气稍软。"
}
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Judge dialogue quality with cache/resume support.")
    parser.add_argument("--input", default=INPUT_FILE, help="Input JSON file path.")
    parser.add_argument("--report", default=REPORT_FILE, help="Report output JSON path.")
    parser.add_argument("--bad-case", default=BAD_CASE_FILE, help="Bad cases output JSON path.")
    parser.add_argument("--stats", default=STATS_FILE, help="Score stats output JSON path.")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Cache directory for resume.")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Max worker threads.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Save report every N items.")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Max retries for API calls.")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name.")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL.")
    parser.add_argument("--api-key", default=API_KEY, help="API key.")
    return parser.parse_args()

def apply_args(args):
    global INPUT_FILE, REPORT_FILE, BAD_CASE_FILE, STATS_FILE, CACHE_DIR
    global MAX_WORKERS, SAVE_EVERY, MAX_RETRIES, MODEL_NAME, BASE_URL, API_KEY, client
    INPUT_FILE = args.input
    REPORT_FILE = args.report
    BAD_CASE_FILE = args.bad_case
    STATS_FILE = args.stats
    CACHE_DIR = args.cache_dir
    MAX_WORKERS = args.max_workers
    SAVE_EVERY = args.save_every
    MAX_RETRIES = args.max_retries
    MODEL_NAME = args.model
    BASE_URL = args.base_url
    API_KEY = args.api_key
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def ensure_dirs():
    ensure_parent_dir(REPORT_FILE)
    ensure_parent_dir(BAD_CASE_FILE)
    ensure_parent_dir(STATS_FILE)
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_item_key(item, index):
    if isinstance(item, dict):
        for key in ("id", "item_id", "uid", "uuid"):
            if key in item and item[key] is not None:
                return f"{key}:{item[key]}"
    payload = {
        "system": item.get("system", ""),
        "conversations": item.get("conversations", []),
    }
    digest = sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"idx:{index}-sha1:{digest}"

def get_cache_path(item_key):
    digest = sha1(item_key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.json")

def load_cache(item_key):
    path = get_cache_path(item_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("key") != item_key:
            return None
        return data
    except Exception:
        return None

def save_cache(item_key, payload):
    path = get_cache_path(item_key)
    tmp_path = path + ".tmp"
    with cache_lock:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, path)

def parse_json_strict(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None

def call_llm(full_prompt):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个严格的数据审计员。请直接输出 JSON。"},
                    {"role": "user", "content": full_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            sleep_seconds = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            sleep_seconds += random.uniform(0, RETRY_JITTER_SECONDS)
            time.sleep(sleep_seconds)
    raise last_exc

def is_item_processed(existing_item):
    if not isinstance(existing_item, dict):
        return False
    if existing_item.get(PROCESSED_FLAG) is True and not existing_item.get("_judge_error"):
        return True
    if existing_item.get("_judge_error"):
        return False
    return "score" in existing_item and "pass" in existing_item and "data_index" in existing_item

def write_report(results):
    ensure_parent_dir(REPORT_FILE)
    tmp_path = REPORT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, REPORT_FILE)

def write_bad_cases(bad_cases):
    ensure_parent_dir(BAD_CASE_FILE)
    tmp_path = BAD_CASE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, BAD_CASE_FILE)

def compute_stats(results):
    total = len(results)
    scores = []
    pass_count = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        score = item.get("score", 0)
        if isinstance(score, bool):
            score = int(score)
        if isinstance(score, (int, float)):
            score = int(score)
        else:
            score = 0
        score = max(0, min(5, score))
        scores.append(score)
        if item.get("pass"):
            pass_count += 1

    score_dist = {str(i): 0 for i in range(6)}
    for s in scores:
        score_dist[str(s)] += 1

    avg_score = sum(scores) / total if total > 0 else 0
    stats = {
        "total": total,
        "pass_count": pass_count,
        "pass_rate": round((pass_count / total) * 100, 2) if total else 0,
        "avg_score": round(avg_score, 4),
        "score_dist": score_dist,
        "count_ge_5": sum(1 for s in scores if s >= 5),
        "count_ge_4": sum(1 for s in scores if s >= 4),
        "count_ge_3": sum(1 for s in scores if s >= 3),
        "count_ge_2": sum(1 for s in scores if s >= 2),
        "count_ge_1": sum(1 for s in scores if s >= 1),
    }
    stats["rate_ge_5"] = round((stats["count_ge_5"] / total) * 100, 2) if total else 0
    stats["rate_ge_4"] = round((stats["count_ge_4"] / total) * 100, 2) if total else 0
    stats["rate_ge_3"] = round((stats["count_ge_3"] / total) * 100, 2) if total else 0
    stats["rate_ge_2"] = round((stats["count_ge_2"] / total) * 100, 2) if total else 0
    stats["rate_ge_1"] = round((stats["count_ge_1"] / total) * 100, 2) if total else 0
    return stats

def write_stats(stats):
    ensure_parent_dir(STATS_FILE)
    tmp_path = STATS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, STATS_FILE)

def clean_json_text(text):
    if "```" in text:
        import re
        pattern = r"```(?:json)?\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            text = text.replace("```json", "").replace("```", "")
    return text.strip()

def evaluate_one_dialogue(entry, index, item_key):
    raw_content = ""
    try:
        # 1. 准备数据
        conversation_str = json.dumps(entry, ensure_ascii=False, indent=2)
        
        # 2. 组装 Prompt
        full_prompt = PROMPT_TEMPLATE.replace("{full_conversation_json}", conversation_str)
        
        # 3. 调用 API
        raw_content = call_llm(full_prompt)
        content = clean_json_text(raw_content)
        result = parse_json_strict(content)
        if not result:
            raise ValueError("Invalid JSON response")
        
        # 注入索引方便追踪
        result['data_index'] = index
        result[PROCESSED_FLAG] = True
        save_cache(item_key, {
            "key": item_key,
            "index": index,
            "status": "ok",
            "timestamp": int(time.time()),
            "result": result,
            "raw_content": raw_content,
        })
        return index, result, None

    except Exception as e:
        error_msg = str(e)
        if raw_content:
            error_msg += " (Raw content saved)"
        error_result = {
            "score": 0,
            "pass": False,
            "red_flags": ["System Error"],
            "overall_comment": error_msg,
            "data_index": index,
            "_judge_error": True,
        }
        save_cache(item_key, {
            "key": item_key,
            "index": index,
            "status": "error",
            "timestamp": int(time.time()),
            "error": error_msg,
            "raw_content": raw_content,
        })
        print(f"⚠️ Eval Error (Index {index}): {error_msg}")
        return index, error_result, error_msg

def main():
    args = parse_args()
    apply_args(args)
    ensure_dirs()
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    process_list = data.get('items', []) if isinstance(data, dict) else data
    total = len(process_list)
    
    results = [None] * total
    
    print(f"🚀 开始整段对话评估 ({total} 条)...")

    # 读取已有报告文件，优先断点续写
    if os.path.exists(REPORT_FILE):
        try:
            with open(REPORT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list) and len(existing) == total:
                for i, item in enumerate(existing):
                    if is_item_processed(item):
                        if isinstance(item, dict) and item.get(PROCESSED_FLAG) is not True:
                            item[PROCESSED_FLAG] = True
                        results[i] = item
                        item_key = get_item_key(process_list[i], i)
                        if not load_cache(item_key):
                            save_cache(item_key, {
                                "key": item_key,
                                "index": i,
                                "status": "ok",
                                "timestamp": int(time.time()),
                                "result": item,
                                "raw_content": "",
                                "seeded_from_report": True,
                            })
            else:
                print("⚠️ 已有报告长度与输入不一致，跳过断点续写加载。")
        except Exception as e:
            print(f"⚠️ 读取已有报告失败，将继续全量处理: {e}")

    # 预加载缓存结果
    for i, item in enumerate(process_list):
        if results[i] is not None:
            continue
        item_key = get_item_key(item, i)
        cached = load_cache(item_key)
        if cached and cached.get("status") == "ok" and cached.get("result"):
            cached_result = cached["result"]
            if isinstance(cached_result, dict) and cached_result.get(PROCESSED_FLAG) is not True:
                cached_result[PROCESSED_FLAG] = True
            results[i] = cached_result

    pending_indices = [i for i, item in enumerate(results) if item is None]
    if not pending_indices:
        print("✅ 所有条目已有缓存/报告，无需重新调用 API。")
    else:
        completed_since_save = 0
        done_count = total - len(pending_indices)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {}
            for i in pending_indices:
                item_key = get_item_key(process_list[i], i)
                future = executor.submit(evaluate_one_dialogue, process_list[i], i, item_key)
                future_to_idx[future] = i

            with tqdm(total=total, initial=done_count) as pbar:
                for future in as_completed(future_to_idx):
                    idx, res, error = future.result()
                    results[idx] = res
                    completed_since_save += 1
                    if completed_since_save >= SAVE_EVERY:
                        write_report(results)
                        completed_since_save = 0
                    pbar.update(1)

        if completed_since_save:
            write_report(results)

    # 生成不合格案例与统计
    bad_cases = []
    for idx, res in enumerate(results):
        if not isinstance(res, dict):
            continue
        if not res.get("pass", False):
            bad_cases.append({
                "eval_result": res,
                "original_data": process_list[idx]
            })

    stats = compute_stats(results)

    print(f"\n📊 评估报告:")
    print(f"总数: {stats['total']}")
    print(f"合格数: {stats['pass_count']} ({stats['pass_rate']}%)")
    print(f"平均分: {stats['avg_score']:.2f}")
    print(f"5分: {stats['count_ge_5']} ({stats['rate_ge_5']}%)")
    print(f"≥4分: {stats['count_ge_4']} ({stats['rate_ge_4']}%)")

    # 保存结果
    write_report(results)
    write_bad_cases(bad_cases)
    write_stats(stats)

    print(f"✅ 详细报告已保存至 {REPORT_FILE}")
    print(f"❌ 不合格案例已保存至 {BAD_CASE_FILE}")
    print(f"📈 得分统计已保存至 {STATS_FILE}")

if __name__ == "__main__":
    main()
