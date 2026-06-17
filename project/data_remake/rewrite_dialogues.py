import argparse
import json
import os
import random
import re
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1

from openai import OpenAI
from tqdm import tqdm

# ================= 配置 =================
INPUT_FILE = "/data/chengch/project/data_remake/runs/debug_reverse_tongyi_10_action.json"
OUTPUT_FILE = "/data/chengch/project/data_remake/runs/normal_dataset_rewrite.json"
RAW_TXT_LOG = "/data/chengch/project/data_remake/logs/raw_rewrite_log.txt"
CACHE_DIR = "/data/chengch/project/data_remake/cache/rewrite3"
SAVE_EVERY = 20
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0
RETRY_JITTER_SECONDS = 0.3
MAX_WORKERS = 2
PROCESSED_FLAG = "_rewrite_done"

API_KEY = "your_api_key_here"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"
TEMPERATURE = 0.8

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
log_lock = threading.Lock()
cache_lock = threading.Lock()

# ================= V12.0 精准单轮重写 Prompt =================
SINGLE_TURN_PROMPT = """
# Role

你是一名**身披白大褂的顶级销售专家**。

你的核心逻辑是：**严格执行 SOP 流程指标**，但用**最犀利的医学/心理学话术**来包装它。

# 核心禁令

1. **严厉禁止利用当前未涉及的信息，只能根据当前轮和历史信息进行回复【high level】**

2. **严禁“AI 标点癖”**：

* **绝对禁止使用装饰性双引号**（如：把‘焦虑’改成焦虑，把‘睡眠不好’改成睡眠不好）。

* **绝对禁止使用破折号（——）**做解释说明。

3. **严禁“AI 逻辑癖”**：

* 绝对禁止说“这说明...”、“这意味着...”。

* 绝对禁止说“为了...请告诉我...”。

* 绝对禁止复述用户的话（如“你说你心情不好...”）。

4. **严禁实体幻觉**：

* System Prompt 没给医院名，就自称“我们”，严禁编造

* User Input 没写 `<action>`，绝不瞎编收到指令。

5. **禁止提及专业名词**，除非用户主动提及。

6. **问诊时尽量少解释，直接询问症状即可**

7. **严禁修改human回复**

8. **禁止在thought和slot_values中输出<sep>**

9. **严禁输出任何回顾原对话的内容，包括但不限于：原对话中...

# 战略思维链 (White List)

在 `thought` 中，必须从以下白名单中选择策略，**严禁发明新词**：

## A. 生理层 (Target)

[皮质醇(降压/施压), 多巴胺(诱饵), 内啡肽(慰藉), 催产素(结盟), 血清素(掌控)]

## B. 心理层 (Target)

[自主感, 胜任感, 归属感, 优越感, 安全感, 认知节能, 损失厌恶]

## C. 战术层 (Strategy)

[医学定性, 提问引导, 示弱反衬, 降维打击, 认知重构, 权威借势, 利益诱惑]

## D. 用户认知 (Medical Cognition)

[未知, 小白, 半懂, 专业, 误区明显]

## E. 用户本轮状态 (User State)

[平静, 犹豫, 害怕, 不信任, 对抗, 急迫, 配合, 敷衍, 未知]

## F. 用户决策障碍 (Decision Barrier)

[暂无, 隐私, 费用, 效果, 正规性, 病耻感, 时间, 家属决策, 反感推销, 未知]

# Task Requirements

请生成 JSON，包含 `thought` 和 `response` 和 `slot_values`。

## 1. thought

请严格按步骤思考：

- 【锚定】：检查有无 `<action>` 有的话是什么action？当前轮次是多少？

- 【分析】：

--1.用户基础信息：基于当前轮和历史信息，概括患者年龄、性别、咨询关系、聊天风格，DISC人格模型偏好(DISC 人格分为四类，D 支配型果决强势、目标导向、爱掌控重结果，行事利落有魄力；I 影响型外向热情、擅长社交表达、感染力强，善于带动氛围与人沟通；S 稳健型温和耐心、踏实稳重、不喜变动，待人谦和有包容心，擅长配合协作；C 谨慎型理性严谨、注重细节逻辑、追求精准完美，做事讲究规则、善于思考分析。)；未知字段必须写未知或未获取，严禁利用未来信息。

--2.用户认知：必须从用户认知白名单中选择一个值，判断用户医学认知水平，并用一句话说明依据。

--3.用户本轮状态：必须从用户本轮状态白名单中选择一个值，判断用户最新输入呈现的状态，并用一句话说明依据。

--4.用户决策障碍：必须从用户决策障碍白名单中选择一个值，判断当前阻碍继续沟通或留联的核心障碍；如果没有明显障碍写暂无。

--5.生理层：本轮要调控用户的什么激素？写出激素名↑或激素名↓（注意这里的激素是我们需要调控的目标，比如需要降低xxx，是xxx↓），一句话解释为什么需要这么调控（例如：血清素↑：通过“神经递质失衡”的专业解释，建立医学权威感，而非简单的客服接待。）,注意，你的调控方法需要结合用户的聊天风格，DISC人格偏好，以及用户本轮状态来定制（例如：对于一个D型用户，可能更适合用直接的权威式话术来调控；对于一个S型用户，可能更适合用温和的关怀式话术来调控）。同样的激素调控方法对于不同类型的用户可能会有不同的效果，所以需要根据用户的具体情况来选择最合适的调控方法。

--6.心理层：当前要利用或者满足用户的什么心理，写出利用的心理，一句话解释为什么，（例如：安全感：用“专业干预中心”替代“医院预约平台”，降低患者对“看病”的排斥和病耻感。），同样需要结合用户的聊天风格，DISC人格偏好，以及用户本轮状态来定制（例如：对于一个I型用户，可能更适合用热情的感染式话术来满足其归属感；对于一个C型用户，可能更适合用理性的数据和分析来满足其胜任感）。同样的心理满足方法对于不同类型的用户可能会有不同的效果，所以需要根据用户的具体情况来选择最合适的方法。

--7.战术层：使用什么战术，写出战术名称，并解释具体做法（例如：权威借势：将心理问题科学化，引导用户从生物学角度正视自己的状态。），同样需要结合用户的聊天风格，DISC人格偏好，以及用户本轮状态来定制（例如：对于一个D型用户，可能更适合用直接的权威式话术来使用权威借势战术；对于一个S型用户，可能更适合用温和的关怀式话术来使用示弱反衬战术）。同样的战术方法对于不同类型的用户可能会有不同的效果，所以需要根据用户的具体情况来选择最合适的方法。

- 【检验】：检验当前的slot_values和当前SOP步骤，比如当前是不是到了请求留联的最终期限，需不需要执行action等等。

- 【策略】：根据【锚定】，【分析】，【检验】，用户的输入，综合分析当前应该做什么。

## 2. response

- **风格**：口语化，问诊时可以**适当**使用书面词，医生的专业感。

- **技巧**：

* **严禁严禁严禁利用**任何**未来信息，包括但不限于性别，姓氏，关系等等，必须只能根据当前轮和当前轮之前的历史信息进行回复！！**

* **第一轮对话需要简单打招呼，比如你好等等**

* **禁止使用引号，破折号或者括号进行解释，应该口语化**

* **基于原response进行修改，必须保证上下文连贯！！！**

* **不要使用一连串的问句（比如：xx吗？xx吗？）应该使用类似有没有xxxx，xxx，或xxx...的语气**

* **禁止复述用户的回答，确认联系方式除外**

* **在一些轮次中适当对用户的症状做出一些浅层的解释，营造必要性和紧迫性，但不要使用太多专业名词，防止用户听不懂，禁止所有轮次都进行解释（过度解释）**

* **可以**适当**使用医学定性制造紧迫感，但禁止直接下定论**

* **可以**适当**使用<sep>为同一轮回复内的分句分隔符（仅限response）**

* **在未获取对方性别前严禁称对方为先生或女士，称你或您，禁止称对方为老师**

* **禁止称对方的姓氏，只称你，您，先生，女士，严禁称呼姓氏，例如王先生等等**

* **在未确定是本人咨询还是代人咨询之前默认是本人咨询**

改写例子：

原句：好的弟弟，根据以上的负面情绪问题来看...你可以留个电话...你看方便

改写：你现在处于情绪高压、认知尚存的阶段，是干预黄金期。文字没法听出你的语气波动，这影响判断深度。<sep>留个电话，我让老师给你做个15分钟深度评估，把调节方案直接告诉你。

## 3. slot_values

填写规则

- 根据 system prompt 中的槽位进行填写，使用 value 形式。

- 已知字段填写具体值，例如 age：38岁，gender：女，symptom：失眠、胸闷，medical_awareness：小白。

- 未知字段按 system prompt 约定填写“未知”“暂无”或“未获取”，例如 phone：未获取，medical_history：暂无。

- 必须根据system prompt进行填写，严禁虚构slot


# Output Format (JSON Only)
请**仅**输出以下结构的 JSON 对象（不要输出 Markdown 标记），格式如下：
{
    "conversations": [
        {
            "from": "human",
            "value": "用户输入（严禁修改）"
        },
        {
            "from": "gpt",
            "value": {
                "thought": "【锚定】...【分析】生理，心理，战术...【检验】...【策略】...",
                "slot_values": { ... },
                "response": "..."
            }
        }...后续对话同理    
    ]
}
"""

USER_INPUT_TEMPLATE = """
请重写以下数据：
{full_conversation_json}
重写完成后需要自我检查，确保满足对话上下文连贯，且没有违反任何禁令。请直接输出重写后的 JSON，不要输出任何多余的文本或者 Markdown 标记。
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite conversations with cache and resume support.")
    parser.add_argument("--input", default=INPUT_FILE, help="Input JSON file path.")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output JSON file path.")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Cache directory for resume.")
    parser.add_argument("--raw-log", default=RAW_TXT_LOG, help="Raw LLM output log file.")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Max worker threads.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Save output every N items.")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Max retries for API calls.")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name.")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL.")
    parser.add_argument("--api-key", default=API_KEY, help="API key.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature for chat completions.")
    parser.add_argument("--force", action="store_true", help="Ignore existing output/cache and call the API again.")
    return parser.parse_args()


def apply_args(args):
    global INPUT_FILE, OUTPUT_FILE, CACHE_DIR, RAW_TXT_LOG
    global MAX_WORKERS, SAVE_EVERY, MAX_RETRIES, MODEL_NAME, BASE_URL, API_KEY, TEMPERATURE, client
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    CACHE_DIR = args.cache_dir
    RAW_TXT_LOG = args.raw_log
    MAX_WORKERS = args.max_workers
    SAVE_EVERY = args.save_every
    MAX_RETRIES = args.max_retries
    MODEL_NAME = args.model
    BASE_URL = args.base_url
    API_KEY = args.api_key
    TEMPERATURE = args.temperature
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def ensure_dirs():
    ensure_parent_dir(RAW_TXT_LOG)
    os.makedirs(CACHE_DIR, exist_ok=True)


def init_raw_log():
    if not os.path.exists(RAW_TXT_LOG):
        ensure_parent_dir(RAW_TXT_LOG)
        with open(RAW_TXT_LOG, "w", encoding="utf-8") as f:
            f.write("=== LLM RAW OUTPUT BACKUP ===\n\n")


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


def parse_json_strict(raw_content):
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_content[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def call_llm(user_payload_json):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SINGLE_TURN_PROMPT},
                    {"role": "user", "content": user_payload_json},
                ],
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
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


def is_item_processed(existing_item, input_item):
    if not isinstance(existing_item, dict):
        return False
    if existing_item.get(PROCESSED_FLAG) is True:
        return True
    out_convs = existing_item.get("conversations")
    in_convs = input_item.get("conversations") if isinstance(input_item, dict) else None
    if out_convs and in_convs and out_convs != in_convs:
        return True
    return False


def save_raw_log(index, content):
    with log_lock:
        ensure_parent_dir(RAW_TXT_LOG)
        with open(RAW_TXT_LOG, "a", encoding="utf-8") as f:
            f.write(f"--- ID: {index} ---\n{content}\n\n")


def process_single_item(original_entry, index, item_key):
    raw_content = ""
    try:
        original_system = original_entry.get("system", "")
        original_convs = original_entry.get("conversations", [])
        user_payload = {
            "instruction": "请重写以下数据。",
            "system": original_system,
            "conversations": original_convs,
        }

        raw_content = call_llm(json.dumps(user_payload, ensure_ascii=False))
        save_raw_log(index, raw_content)

        new_data = parse_json_strict(raw_content)
        if not new_data:
            raise ValueError("Invalid JSON response")
        new_conversations = new_data.get("conversations")

        final_entry = deepcopy(original_entry)
        if new_conversations:
            final_entry["conversations"] = new_conversations
        final_entry[PROCESSED_FLAG] = True

        cache_payload = {
            "key": item_key,
            "index": index,
            "status": "ok",
            "timestamp": int(time.time()),
            "item": final_entry,
            "raw_content": raw_content,
        }
        save_cache(item_key, cache_payload)
        return index, final_entry, None

    except Exception as e:
        error_msg = str(e)
        if raw_content:
            error_msg += " (Raw content saved to txt)"
        cache_payload = {
            "key": item_key,
            "index": index,
            "status": "error",
            "timestamp": int(time.time()),
            "error": error_msg,
            "raw_content": raw_content,
        }
        save_cache(item_key, cache_payload)
        return index, original_entry, error_msg


def main():
    args = parse_args()
    apply_args(args)
    ensure_dirs()
    init_raw_log()
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    process_list = data.get("items", []) if isinstance(data, dict) else data
    total = len(process_list)
    results = [None] * total

    if args.force:
        print("⚠️ 已启用 --force，将忽略已有输出和缓存，重新调用 API。")
    elif os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_list = existing.get("items", []) if isinstance(existing, dict) else existing
            if isinstance(existing_list, list) and len(existing_list) == total:
                for i, item in enumerate(existing_list):
                    if is_item_processed(item, process_list[i]):
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
                                "item": item,
                                "raw_content": "",
                                "seeded_from_output": True,
                            })
            else:
                print("⚠️ 已有输出文件与输入长度不一致，跳过断点续写加载。")
        except Exception as e:
            print(f"⚠️ 读取已有输出文件失败，将继续全量处理: {e}")

    print(f"🚀 开始重写（参考 SOP 但不输出 SOP），共 {total} 条...")

    if not args.force:
        for i, item in enumerate(process_list):
            if results[i] is not None:
                continue
            item_key = get_item_key(item, i)
            cached = load_cache(item_key)
            if cached and cached.get("status") == "ok" and cached.get("item"):
                cached_item = cached["item"]
                if isinstance(cached_item, dict) and cached_item.get(PROCESSED_FLAG) is not True:
                    cached_item[PROCESSED_FLAG] = True
                results[i] = cached_item

    pending_indices = [i for i, item in enumerate(results) if item is None]
    if not pending_indices:
        print("✅ 所有条目已有缓存/输出，无需重新调用 API。")
    else:
        completed_since_save = 0
        done_count = total - len(pending_indices)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {}
            for i in pending_indices:
                item_key = get_item_key(process_list[i], i)
                future = executor.submit(process_single_item, process_list[i], i, item_key)
                future_to_idx[future] = i

            with tqdm(total=total, initial=done_count) as pbar:
                for future in as_completed(future_to_idx):
                    idx, processed_item, error = future.result()
                    results[idx] = processed_item
                    completed_since_save += 1
                    if error:
                        print(f"\n❌ ID {idx} 失败: {error}")
                    if completed_since_save >= SAVE_EVERY:
                        write_output(results, data)
                        completed_since_save = 0
                    pbar.update(1)

        if completed_since_save:
            write_output(results, data)

    write_output(results, data)
    print(f"✅ 处理完成！结果已存至 {OUTPUT_FILE}")


def write_output(results, original_data):
    output_struct = {"items": results} if isinstance(original_data, dict) else results
    ensure_parent_dir(OUTPUT_FILE)
    tmp_path = OUTPUT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output_struct, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, OUTPUT_FILE)


if __name__ == "__main__":
    main()
