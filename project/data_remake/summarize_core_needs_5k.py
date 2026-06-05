import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1

from openai import OpenAI
from tqdm import tqdm


INPUT_FILE = "/data/chengch/project/data_remake/raw/normal_inject_round.json"
OUTPUT_FILE = "/data/chengch/project/data_remake/runs/lead_needs_5k.json"
CACHE_DIR = "/data/chengch/project/data_remake/cache/lead_needs_5k"
RAW_LOG = "/data/chengch/project/data_remake/logs/raw_lead_needs_5k.txt"

API_KEY = ""
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.6-plus"

LIMIT = 5000
MAX_WORKERS = 2
MAX_RETRIES = 3
REQUEST_TIMEOUT = 180
SAVE_EVERY = 50
TEMPERATURE = 0.2

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT)
cache_lock = threading.Lock()
log_lock = threading.Lock()


SYSTEM_PROMPT = """
你是医疗线上咨询转化分层专家。你的任务不是做医学诊断，而是阅读一段线上咨询对话，判断用户最适合哪种留联策略。

请严格按照“用户类型 + 核心诉求 + 转化障碍 + 套联策略”四层输出稳定 JSON。分类要服务于后续电话/微信留联，不要把标签拆得过细。

# 一、优先级规则
按下面顺序判断，命中高优先级时不要被表面问题带偏：
1. 只要出现明确自杀、自残、攻击、幻觉、严重失控、活不下去等高风险表达，优先归为 core_need=危机求助。
2. 如果咨询者是青少年本人，且主要表达压抑、崩溃、无人理解、不敢告诉家人、想找人说，优先归为 core_need=情绪倾诉；有自伤自杀则仍归为危机求助。
3. 如果用户替孩子、老人、伴侣、父母等家属咨询，且表达管不了、劝不动、不配合、没办法、家庭冲突，优先归为 core_need=家属照护无力。
4. 如果用户已有检查、诊断、用药、住院、心理咨询等经历且效果不好，优先归为 core_need=既往治疗不满。
5. 如果对话重心是问费用、医保、公立私立、正规吗、地址、挂号、电话、微信、预约，优先判断是否属于就医决策顾虑或就医路径。

# 二、用户类型 user_type
只能从下列枚举中选择一个：
1. 青少年本人：未成年人或学生本人咨询自己的情绪、心理、行为问题。
2. 成人本人：成年人本人咨询自己的症状、治疗、用药或就医。
3. 家长代询：父母/监护人替孩子咨询。
4. 子女代询：子女替父母或老人咨询。
5. 伴侣代询：丈夫/妻子/恋人替伴侣咨询。
6. 其他家属代询：兄弟姐妹、亲戚等替家属咨询。
7. 朋友代询：替朋友咨询。
8. 未知：无法判断咨询者身份。

# 三、核心诉求 core_need
只能从下列枚举中选择一个：
1. 病情判断：用户想知道症状、检查报告或表现意味着什么，包括原因、是否正常、是否严重、是不是某种病。合并原来的症状解释、严重性判断、诊断确认、检查报告解读。
2. 治疗方案：用户想知道怎么治、有什么方法、是否需要心理疏导/药物/检查/住院、下一步怎么干预。适合首次或初步自查用户。
3. 既往治疗不满：用户已经检查、诊断、吃药、住院、心理咨询或尝试过治疗，但效果不好、有副作用、反复发作，想换方案或找更专业方法。
4. 用药安全：用户重点询问药物功效、副作用、依赖、剂量、停药、能不能吃、是否适合孩子或老人。
5. 就医路径：用户主要想知道去哪家医院、挂什么科、找什么医生、地址、电话、微信、预约时间、如何到院。
6. 就医决策顾虑：用户主要在评估是否值得去，重点关注费用、医保、公立私立、正规性、医生资质、会不会被骗。
7. 情绪倾诉：用户主要是在表达痛苦、压抑、崩溃、孤立、没人理解、不敢说，希望有人接住情绪。
8. 危机求助：用户或患者存在自杀、自残、攻击、幻觉、严重失控等安全风险，需要优先安全干预。
9. 家属照护无力：家属面对患者不配合、劝不动、管不了、家庭关系紧张、长期照护疲惫，想要外部专业介入。
10. 其他：以上都不匹配时使用。

# 四、转化障碍 conversion_barrier
只能从下列枚举中选择一个最影响留联的障碍：
1. 医学认知不足：不知道症状意味着什么，不知道该不该就医。
2. 路径不清：不知道挂什么科、找谁、下一步怎么做。
3. 费用顾虑：担心价格、收费、经济压力、医保报销。
4. 信任顾虑：担心医院正规性、医生水平、公立私立、被推销或被骗。
5. 效果顾虑：担心治不好、复发、治疗无效、白花钱。
6. 用药顾虑：担心副作用、依赖、剂量、长期吃药。
7. 隐私病耻：害怕别人知道，不敢告诉家人/学校/朋友，羞耻或回避。
8. 患者不配合：患者本人不愿就医、不听劝、不承认问题。
9. 家庭沟通失效：家属已反复劝说、争吵、管教失败，关系紧张。
10. 时间紧迫：开学、考试、发作、明天就诊、症状急性加重等带来时间压力。
11. 情绪承载不足：用户需要先被安抚、理解、接住，再能进入留联。
12. 危机安全风险：存在自杀、自残、攻击、幻觉等安全风险。

# 五、套联策略 lead_strategy
只能从下列枚举中选择一个：
1. 低压保密留联：适合青少年本人、隐私病耻、害怕家人知道的用户；强调保密、先文字/微信、低压力沟通。
2. 危机安全回电：适合自杀、自残、攻击、幻觉、严重失控；强调安全风险、需要尽快电话评估。
3. 专家评估留联：适合病情复杂、家属拿不准、需要专业判断；强调专科老师/医生一对一评估。
4. 家属指导留联：适合家长或家属照护无力、患者不配合；强调先教家属怎么沟通、怎么带动患者配合。
5. 二次方案评估：适合既往治疗不满；强调带既往诊断/用药/检查，由专科老师判断方案是否需要调整。
6. 用药风险核对：适合用药安全；强调电话/微信核对药名、剂量、年龄、病史，避免盲目用药。
7. 到院路径预约：适合已准备就医；强调地址、科室、时间、预约名额、少走弯路。
8. 费用透明解释：适合费用顾虑；强调先了解情况再估算检查/治疗范围，避免盲目花钱。
9. 正规资质背书：适合信任顾虑；强调医院资质、专科属性、医生/流程规范。
10. 情绪承接转评估：适合情绪倾诉但未达到危机；先接住情绪，再引导留微信/电话做进一步评估。
11. 科普判断转留联：适合病情判断或医学认知不足；先解释可能方向，再说明需要进一步评估。

# 六、输出要求
只输出 JSON 对象，不要 Markdown，不要解释。
字段固定如下：
{
  "user_type": "枚举值",
  "core_need": "枚举值",
  "conversion_barrier": "枚举值",
  "lead_strategy": "枚举值",
  "fine_label": "用户类型-核心诉求-转化障碍-套联策略",
  "confidence": 0.0到1.0之间的小数,
  "evidence": "引用或概括对话中最能支持判断的用户表达，尽量简短",
  "reason": "一句话说明为什么这样判断，重点说明它为什么适合这个套联策略"
}

注意：
- 你要判断“如何更容易留联”，不是只做医学分类。
- 青少年本人咨询时，优先识别隐私、孤立、情绪承载和危机风险。
- 家属代询时，优先识别照护压力、患者不配合、家庭沟通失效。
- 问“能治好吗、多久好、会不会复发”通常不要单独成类，优先视为效果顾虑。
- 问“多少钱、公立私立、医保、正规不正规”通常属于就医决策顾虑。
"""


USER_TEMPLATE = """
请分析下面这段线上咨询对话。

【对话】
{dialogue}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize core user needs for first N dialogues.")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--raw-log", default=RAW_LOG)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key", default=API_KEY)
    return parser.parse_args()


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def atomic_write_json(path, data):
    ensure_parent_dir(path)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def cache_path(cache_dir, index, item):
    payload = json.dumps(item.get("conversations", []), ensure_ascii=False, sort_keys=True)
    digest = sha1(f"{index}:{payload}".encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{digest}.json")


def strip_round_tag(text):
    text = re.sub(r"\n?【系统数据：当前第\s*\d+\s*轮】", "", text)
    return text.strip()


def format_dialogue(item, max_chars=9000):
    lines = []
    for conv in item.get("conversations", []):
        role = conv.get("from", "")
        value = conv.get("value", "")
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        value = strip_round_tag(value).replace("<sep>", " / ")
        value = re.sub(r"<think>.*?</think>", "", value).strip()
        if role == "human":
            lines.append(f"用户：{value}")
        elif role == "gpt":
            lines.append(f"咨询师：{value}")
    dialogue = "\n".join(lines)
    if len(dialogue) > max_chars:
        dialogue = dialogue[:max_chars] + "\n...（后文截断）"
    return dialogue


def parse_json(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def save_raw_log(path, index, raw):
    with log_lock:
        ensure_parent_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"--- ID: {index} ---\n{raw}\n\n")


def call_llm(args, dialogue):
    global client
    prompt = USER_TEMPLATE.format(dialogue=dialogue)
    last_exc = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=args.temperature,
                timeout=args.timeout,
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_exc = exc
            if attempt >= args.max_retries:
                break
            time.sleep((2 ** (attempt - 1)) + random.uniform(0, 0.5))
    raise last_exc


def validate_result(obj):
    required = [
        "user_type",
        "core_need",
        "conversion_barrier",
        "lead_strategy",
        "fine_label",
        "confidence",
        "evidence",
        "reason",
    ]
    if not isinstance(obj, dict):
        raise ValueError("result is not a JSON object")
    for key in required:
        if key not in obj:
            raise ValueError(f"missing field: {key}")
    return obj


def process_one(args, index, item):
    path = cache_path(args.cache_dir, index, item)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    dialogue = format_dialogue(item)
    raw = call_llm(args, dialogue)
    save_raw_log(args.raw_log, index, raw)
    result = validate_result(parse_json(raw))

    record = {
        "index": index,
        "status": "ok",
        "result": result,
        "first_user": next(
            (
                strip_round_tag(c.get("value", "")).replace("<sep>", " / ")
                for c in item.get("conversations", [])
                if c.get("from") == "human"
            ),
            "",
        ),
    }
    with cache_lock:
        atomic_write_json(path, record)
    return record


def load_existing_output(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data if isinstance(data, list) else [])
    return {record.get("index"): record for record in records if isinstance(record, dict)}


def main():
    global client
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    os.makedirs(args.cache_dir, exist_ok=True)
    ensure_parent_dir(args.output)
    ensure_parent_dir(args.raw_log)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data[:args.limit]
    output_by_index = load_existing_output(args.output)
    pending = [(i, item) for i, item in enumerate(items) if i not in output_by_index]

    print(f"开始总结核心诉求，共 {len(items)} 条，待处理 {len(pending)} 条。")
    completed_since_save = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_one, args, index, item): index
            for index, item in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            index = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "index": index,
                    "status": "failed",
                    "error": str(exc),
                    "first_user": next(
                        (
                            strip_round_tag(c.get("value", "")).replace("<sep>", " / ")
                            for c in items[index].get("conversations", [])
                            if c.get("from") == "human"
                        ),
                        "",
                    ),
                }
                print(f"\nID {index} 失败: {exc}")
            output_by_index[index] = record
            completed_since_save += 1
            if completed_since_save >= args.save_every:
                save_output(args, output_by_index)
                completed_since_save = 0

    save_output(args, output_by_index)
    print(f"处理完成，结果已保存到 {args.output}")


def save_output(args, output_by_index):
    records = [output_by_index[i] for i in sorted(output_by_index)]
    payload = {
        "input_file": args.input,
        "model": args.model,
        "limit": args.limit,
        "schema": "user_type + core_need + conversion_barrier + lead_strategy",
        "records": records,
    }
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
