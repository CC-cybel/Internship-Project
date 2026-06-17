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
from check_hard_rewrite_v2_object_quality import IssueCollector, check_item, SEVERE_ISSUES
# ================= 配置 =================
INPUT_FILE = "/data/chengch/project/data_remake/runs/hard_reverse_tongyi_v2_action.json"
OUTPUT_FILE = "/data/chengch/project/data_remake/runs/hard_rewrite_v2.json"
RAW_TXT_LOG = "/data/chengch/project/data_remake/logs/hard_rewrite_v2.txt"           # 原始输出备份
CACHE_DIR = "/data/chengch/project/data_remake/cache/hard_rewrite_v2"                    # 缓存目录（断点续写）
SAVE_EVERY = 20                                  # 每处理 N 条写一次输出文件
MAX_RETRIES = 3                                  # 请求失败重试次数
MAX_QUALITY_RETRIES = 3                          # 质量检查失败后的重写次数
REQUEST_TIMEOUT_SECONDS = 180                      # 单次 API 请求超时时间
RETRY_BASE_SECONDS = 1.0                         # 重试基础等待
RETRY_JITTER_SECONDS = 0.3                       # 重试抖动
MAX_WORKERS = 20
PROCESSED_FLAG = "_rewrite_done"

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "your_api_key_here")
BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.environ.get("DASHSCOPE_MODEL_NAME", "deepseek-v4-flash")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS)
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

## D. 用户本轮状态 (User State)

[平静, 犹豫, 害怕, 不信任, 对抗, 急迫, 配合, 敷衍, 未知]

## E. 留联分层 (Lead Segmentation)

user_type 只能从以下枚举选择：
[青少年本人, 成人本人, 家长代询, 子女代询, 伴侣代询, 其他家属代询, 朋友代询, 未知]

core_need 只能从以下枚举选择：
[病情判断, 治疗方案, 既往治疗不满, 用药安全, 就医路径, 就医决策顾虑, 情绪倾诉, 危机求助, 家属照护无力, 其他]

conversion_barrier 只能从以下枚举选择：
[医学认知不足, 路径不清, 费用顾虑, 信任顾虑, 效果顾虑, 用药顾虑, 隐私病耻, 患者不配合, 家庭沟通失效, 时间紧迫, 情绪承载不足, 危机安全风险, 暂无明显障碍, 信息不足]

lead_strategy 只能从以下枚举选择：
[低压保密留联, 危机安全回电, 专家评估留联, 家属指导留联, 二次方案评估, 用药风险核对, 到院路径预约, 费用透明解释, 正规资质背书, 情绪承接转评估, 科普判断转留联, 暂不留联继续问诊]

fine_label 格式固定为：
user_type-core_need-conversion_barrier-lead_strategy

# Task Requirements

请生成 JSON，包含 `thought` 和 `response` 和 `slot_values`。

## 1. thought

***请严格按步骤输出（重要）***：

- 【锚定】：检查有无 `<action>` 有的话是什么action？当前轮次是多少？

- 【分析】 必须按照以下 5 个固定字段逐条分析，字段名和顺序不能改变，不能省略：

--1.留联分层：按固定结构输出 user_type=...；core_need=...；conversion_barrier=...；lead_strategy=...；fine_label=...。user_type/core_need/conversion_barrier 要尽早根据当前轮和历史信息填充；若暂无信息或无法推测，必须使用枚举内兜底值：user_type=未知；core_need=其他；conversion_barrier=信息不足。***严禁严禁使用任何未来信息***。lead_strategy 可在首次留联轮次的前一轮或信息足够时填充，若尚不到留联铺垫时机可写 lead_strategy=暂不留联继续问诊。留联分层一旦写定，后续轮次除非出现***严重证据错误或危机风险升级***，否则不得随意更改，以保证分层稳定。
优先级规则
按下面顺序判断，命中高优先级时不要被表面问题带偏：
1. 只要出现明确自杀、自残、攻击、幻觉、严重失控、活不下去等高风险表达，优先归为 core_need=危机求助。
2. 如果咨询者是青少年本人，且主要表达压抑、崩溃、无人理解、不敢告诉家人、想找人说，优先归为 core_need=情绪倾诉；有自伤自杀则仍归为危机求助。
3. 如果用户替孩子、老人、伴侣、父母等家属咨询，且表达管不了、劝不动、不配合、没办法、家庭冲突，优先归为 core_need=家属照护无力。
4. 如果用户已有检查、诊断、用药、住院、心理咨询等经历且效果不好，优先归为 core_need=既往治疗不满。
5. 如果对话重心是问费用、医保、公立私立、正规吗、地址、挂号、电话、微信、预约，优先判断是否属于就医决策顾虑或就医路径。

用户类型 user_type
从下列枚举中选择一个：
1. 青少年本人：未成年人或学生本人咨询自己的情绪、心理、行为问题。
2. 成人本人：成年人本人咨询自己的症状、治疗、用药或就医。
3. 家长代询：父母/监护人替孩子咨询。
4. 子女代询：子女替父母或老人咨询。
5. 伴侣代询：丈夫/妻子/恋人替伴侣咨询。
6. 其他家属代询：兄弟姐妹、亲戚等替家属咨询。
7. 朋友代询：替朋友咨询。
8. 未知：无法判断咨询者身份。

核心诉求 core_need
从下列枚举中选择一个：
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

转化障碍 conversion_barrier
从下列枚举中选择一个最影响留联的障碍：
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
13. 暂无明显障碍：当前没有明显阻碍。
14. 信息不足：缺少必要信息，无法判断主要障碍。

套联策略 lead_strategy
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
12. 暂不留联继续问诊：信息不足、尚未到留联铺垫时机，先继续补充关键问题。
注意：
- 你要判断“如何更容易留联”，不是只做医学分类。
- 青少年本人咨询时，优先识别隐私、孤立、情绪承载和危机风险。
- 家属代询时，优先识别照护压力、患者不配合、家庭沟通失效。
- 问“能治好吗、多久好、会不会复发”通常不要单独成类，优先视为效果顾虑。
- 问“多少钱、公立私立、医保、正规不正规”通常属于就医决策顾虑。

--2.用户状态：必须从用户本轮状态白名单中选择一个值，判断用户最新输入呈现的状态，并用一句话说明依据。

--3.生理层：本轮要调控用户的什么激素？写出激素名↑或激素名↓，并用一句话解释为什么需要这么调控。调控方法必须结合留联分层、用户状态和当前轮信息。

--4.心理层：当前要利用或者满足用户的什么心理，写出利用的心理，并用一句话解释为什么。心理选择必须服务于当前 lead_strategy。

--5.战术层：使用什么战术，写出战术名称，并解释具体做法。战术必须服务于当前 lead_strategy 和本轮 SOP/action。

- 【检验】：检验当前的slot_values和当前SOP步骤，比如当前是不是到了请求留联的最终期限，需不需要执行action等等。

- 【策略】：根据【锚定】，【分析】，【检验】，用户的输入，综合分析当前应该做什么。

-- 【分析】内部必须按固定字段顺序逐项输出，字段名必须完整保留，不能省略任何字段；除非发生严重信息不足，否则都应通过推测来填充，若出现严重无法推测的情况也必须写“未知”“暂无法判断”或“信息不足”。
--  固定字段顺序如下：
留联分层；用户状态；生理层；心理层；战术层。

- 输出格式必须严格按照以下示范格式，包含【锚定】，【分析】，【检验】，【策略】***严禁严禁使用任何未来信息***：
【锚定】第1轮，无action。需介绍医院全称电话并询问年龄性别。【分析】留联分层：user_type=未知；core_need=就医路径；conversion_barrier=路径不清；lead_strategy=暂不留联继续问诊；fine_label=未知-就医路径-路径不清-暂不留联继续问诊。用户状态：平静，用户主动询问心理疾病医院，当前没有明显对抗或急迫。生理层：血清素↑，用明确医院信息和官方电话建立秩序感与掌控感。心理层：安全感，用户在筛选可信机构，先给确定信息能降低不确定感。战术层：权威借势，先介绍医院资质和电话，再用低门槛问题收集年龄性别。【检验】需完成身份锚定和年龄性别询问。【策略】先介绍医院权威信息，再收集基础信息；暂不急于套联，等主诉和关系更清楚后稳定留联分层。

## 2. response

- **风格**：口语化，问诊时可以**适当**使用书面词，医生的专业感。

- **技巧**：

* **严禁严禁严禁利用**任何**未来信息，包括但不限于性别，姓氏等等，必须只能根据当前轮和历史信息进行回复！！

* **第一轮对话需要简单打招呼，比如你好等等**

* **禁止使用引号，破折号或者括号进行解释，应该口语化**

* **基于原response进行修改，必须保证上下文连贯**

* **不要使用一连串的问句（比如：xx吗？xx吗？）应该使用类似有没有xxxx，xxx，或xxx...的语气**

* **禁止复述用户的回答，确认联系方式除外**

* **在一些轮次中适当对用户的症状做出一些浅层的解释，营造必要性和紧迫性，但不要使用太多专业名词，防止用户听不懂，禁止所有轮次都进行解释（过度解释）**

* **可以**适当**使用医学定性制造紧迫感，但禁止直接下定论**

* **可以**适当**使用<sep>为同一轮回复内的分句分隔符（仅限response）**

* **在未获取对方性别前严禁称对方为先生或女士，称你或您，禁止称对方为老师

* **禁止称对方的姓氏，只称你，您，先生，女士，严禁称呼姓氏，例如王先生等等

* **在未确定是本人咨询还是代人咨询之前默认是本人咨询

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
                "thought": "【锚定】...【分析】留联分层：user_type=...；core_need=...；conversion_barrier=...；lead_strategy=...；fine_label=...。用户状态：...。生理层：...。心理层：...。战术层：...。【检验】...【策略】...",
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
    parser.add_argument("--quality-retries", type=int, default=MAX_QUALITY_RETRIES, help="Max rewrite attempts when output quality validation fails.")
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT_SECONDS, help="Per-request API timeout seconds.")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name.")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL.")
    parser.add_argument("--api-key", default=API_KEY, help="API key.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N items after offset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N input items before processing.")
    return parser.parse_args()

def apply_args(args):
    global INPUT_FILE, OUTPUT_FILE, CACHE_DIR, RAW_TXT_LOG
    global MAX_WORKERS, SAVE_EVERY, MAX_RETRIES, MAX_QUALITY_RETRIES, REQUEST_TIMEOUT_SECONDS, MODEL_NAME, BASE_URL, API_KEY, client
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    CACHE_DIR = args.cache_dir
    RAW_TXT_LOG = args.raw_log
    MAX_WORKERS = args.max_workers
    SAVE_EVERY = args.save_every
    MAX_RETRIES = args.max_retries
    MAX_QUALITY_RETRIES = args.quality_retries
    REQUEST_TIMEOUT_SECONDS = args.request_timeout
    MODEL_NAME = args.model
    BASE_URL = args.base_url
    API_KEY = args.api_key
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS)

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
    if not isinstance(item, dict):
        return f"idx:{index}-empty"
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

def summarize_quality_errors(records, limit=12):
    parts = []
    for record in records[:limit]:
        issue = record.get("issue", "unknown")
        message_index = record.get("message_index")
        detail = record.get("detail", "")
        if message_index is None:
            part = issue
        else:
            part = f"{issue}@message_{message_index}"
        if detail:
            part += f"({detail})"
        parts.append(part)
    if len(records) > limit:
        parts.append(f"...and {len(records) - limit} more")
    return "; ".join(parts)


def validate_rewritten_item(original_entry, rewritten_entry):
    collector = IssueCollector(sample_limit=3)
    check_item(rewritten_entry, 0, collector, min_response_chars=2)
    errors = [record for record in collector.records if record.get("issue") in SEVERE_ISSUES]

    original_convs = original_entry.get("conversations") if isinstance(original_entry, dict) else None
    rewritten_convs = rewritten_entry.get("conversations") if isinstance(rewritten_entry, dict) else None
    if isinstance(original_convs, list) and isinstance(rewritten_convs, list):
        original_humans = [m.get("value") for m in original_convs if isinstance(m, dict) and m.get("from") == "human"]
        rewritten_humans = [m.get("value") for m in rewritten_convs if isinstance(m, dict) and m.get("from") == "human"]
        if len(original_humans) != len(rewritten_humans):
            errors.append({"issue": "human_turn_count_changed", "detail": f"original={len(original_humans)}, rewritten={len(rewritten_humans)}"})
        else:
            for turn_index, (old, new) in enumerate(zip(original_humans, rewritten_humans), start=1):
                if old != new:
                    errors.append({"issue": "human_value_changed", "detail": f"human_turn={turn_index}"})
                    break
        original_gpt_count = sum(1 for m in original_convs if isinstance(m, dict) and m.get("from") == "gpt")
        rewritten_gpt_count = sum(1 for m in rewritten_convs if isinstance(m, dict) and m.get("from") == "gpt")
        if original_gpt_count != rewritten_gpt_count:
            errors.append({"issue": "gpt_turn_count_changed", "detail": f"original={original_gpt_count}, rewritten={rewritten_gpt_count}"})
    return errors


def build_retry_feedback(errors):
    summary = summarize_quality_errors(errors)
    return (
        "上一次改写质量检查不合格，请重新输出完整 JSON。"
        f" 检查失败项：{summary}。"
        " 必须保持 human 原文和轮次不变；conversations 必须严格 human/gpt 交替；"
        "每个 gpt.value 必须是对象，且包含非空 thought、slot_values、response；"
        "slot_values 必须包含 age、gender、name、relationship、phone、wechat、symptom、duration、medical_history、medical_awareness；"
        "response 必须是非空字符串，不能混入系统轮次、<action>、<think> 或 JSON 片段。"
    )


def make_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        return make_json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return make_json_safe(value.dict())
    return str(value)


def add_usage(total, usage):
    if not isinstance(usage, dict):
        return total
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value
    return total


def format_usage(usage):
    if not usage:
        return "{}"
    keys = ["prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"]
    parts = []
    for key in keys:
        if key in usage:
            parts.append(f"{key}={usage[key]}")
    for key in sorted(usage):
        if key not in keys and isinstance(usage[key], int):
            parts.append(f"{key}={usage[key]}")
    return ", ".join(parts)


def call_llm(user_payload_json):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SINGLE_TURN_PROMPT},
                    {"role": "user", "content": user_payload_json}
                ],
                response_format={"type": "json_object"},
                temperature=0.8
            )
            return response.choices[0].message.content, make_json_safe(response.usage)
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
    if not isinstance(input_item, dict):
        return existing_item.get(PROCESSED_FLAG) is True
    if existing_item.get(PROCESSED_FLAG) is not True:
        return False
    return not validate_rewritten_item(input_item, existing_item)
def save_raw_log(index, content):
    with log_lock:
        ensure_parent_dir(RAW_TXT_LOG)
        with open(RAW_TXT_LOG, "a", encoding="utf-8") as f:
            f.write(f"--- ID: {index} ---\n{content}\n\n")

def save_usage_log(index, attempt, usage):
    if not usage:
        return
    usage_path = RAW_TXT_LOG + ".usage.jsonl"
    record = {"index": index, "attempt": attempt, "usage": usage, "timestamp": int(time.time())}
    with log_lock:
        ensure_parent_dir(usage_path)
        with open(usage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_single_item(original_entry, index, item_key):
    raw_content = ""
    try:
        if not isinstance(original_entry, dict):
            return index, original_entry, "Skipped empty/non-dict item", {}
        # 1. 提取原有的 system 和对话
        original_system = original_entry.get('system', '')
        original_convs = original_entry.get('conversations', [])

        # 2. 构造发送给模型的 User 内容
        # 将 system 放在这里作为背景资料
        user_payload = {
            "instruction": "请重写以下数据。重写完成后需要自我检查，确保满足对话上下文连贯，且没有违反任何禁令。请直接输出重写后的 JSON，不要输出任何多余的文本或者 Markdown 标记。",
            "system": original_system,
            "conversations": original_convs
        }

        final_entry = None
        quality_errors = []
        retry_feedback = None
        usage_total = {}
        usage_attempts = []
        for quality_attempt in range(1, MAX_QUALITY_RETRIES + 1):
            current_payload = deepcopy(user_payload)
            if retry_feedback:
                current_payload["retry_feedback"] = retry_feedback
                current_payload["instruction"] += "\n" + retry_feedback

            # 3. 调用 API
            raw_content, usage = call_llm(json.dumps(current_payload, ensure_ascii=False))
            usage_attempts.append({"attempt": quality_attempt, "usage": usage})
            add_usage(usage_total, usage)
            save_usage_log(index, quality_attempt, usage)
            save_raw_log(f"{index}/attempt_{quality_attempt}", raw_content)

            # 4. 解析并拼接
            new_data = parse_json_strict(raw_content)
            if not new_data:
                quality_errors = [{"issue": "invalid_json_response"}]
                retry_feedback = build_retry_feedback(quality_errors)
                if quality_attempt < MAX_QUALITY_RETRIES:
                    continue
                raise ValueError("Invalid JSON response after quality retries")
            new_conversations = new_data.get("conversations")

            # 使用深拷贝保留原有的 system, extracted_info 以及其他元数据
            candidate_entry = deepcopy(original_entry)
            candidate_entry["conversations"] = new_conversations
            quality_errors = validate_rewritten_item(original_entry, candidate_entry)
            if not quality_errors:
                final_entry = candidate_entry
                break

            retry_feedback = build_retry_feedback(quality_errors)
            if quality_attempt < MAX_QUALITY_RETRIES:
                continue

        if final_entry is None:
            error_summary = summarize_quality_errors(quality_errors)
            raise ValueError(f"Quality validation failed after {MAX_QUALITY_RETRIES} attempts: {error_summary}")

        final_entry[PROCESSED_FLAG] = True

        cache_payload = {
            "key": item_key,
            "index": index,
            "status": "ok",
            "timestamp": int(time.time()),
            "item": final_entry,
            "raw_content": raw_content,
            "quality_attempts": quality_attempt,
            "token_usage": usage_total,
            "token_usage_attempts": usage_attempts,
        }
        save_cache(item_key, cache_payload)
        
        return index, final_entry, None, usage_total

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
            "token_usage": locals().get("usage_total", {}),
            "token_usage_attempts": locals().get("usage_attempts", []),
        }
        save_cache(item_key, cache_payload)
        return index, original_entry, error_msg, locals().get("usage_total", {})
    

def main():
    args = parse_args()
    apply_args(args)
    ensure_dirs()
    init_raw_log()
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    process_list = data.get('items', []) if isinstance(data, dict) else data
    if not isinstance(process_list, list):
        print("❌ 输入文件必须是 list，或包含 items 列表的 dict。")
        return
    if args.offset or args.limit is not None:
        start = max(args.offset, 0)
        end = None if args.limit is None else start + args.limit
        process_list = process_list[start:end]
        data = {"items": process_list} if isinstance(data, dict) else process_list
    total = len(process_list)
    results = [None] * total

    # 读取已有输出文件，优先用作断点续写
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_list = existing.get("items", []) if isinstance(existing, dict) else existing
            if isinstance(existing_list, list) and len(existing_list) == total:
                for i, item in enumerate(existing_list):
                    if process_list[i] is None:
                        results[i] = None
                        continue
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

    skipped_empty = 0
    for i, item in enumerate(process_list):
        if item is None:
            results[i] = None
            skipped_empty += 1
    if skipped_empty:
        print(f"⚠️ 输入中有 {skipped_empty} 条空占位（None/null），将原位跳过。")

    # 预加载缓存结果
    for i, item in enumerate(process_list):
        if results[i] is not None:
            continue
        if item is None:
            continue
        item_key = get_item_key(item, i)
        cached = load_cache(item_key)
        if cached and cached.get("status") == "ok" and cached.get("item"):
            cached_item = cached["item"]
            if isinstance(cached_item, dict) and cached_item.get(PROCESSED_FLAG) is not True:
                cached_item[PROCESSED_FLAG] = True
            if is_item_processed(cached_item, item):
                results[i] = cached_item

    run_usage_total = {}
    pending_indices = [i for i, item in enumerate(results) if item is None and process_list[i] is not None]
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
                    idx, processed_item, error, usage = future.result()
                    add_usage(run_usage_total, usage)
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

    # 保存最终结果
    write_output(results, data)

    print(f"✅ 处理完成！结果已存至 {OUTPUT_FILE}")
    print(f"📊 本次新调用 token 消耗: {format_usage(run_usage_total)}")

def write_output(results, original_data):
    output_struct = {"items": results} if isinstance(original_data, dict) else results
    ensure_parent_dir(OUTPUT_FILE)
    tmp_path = OUTPUT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output_struct, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, OUTPUT_FILE)

if __name__ == "__main__":
    main()
