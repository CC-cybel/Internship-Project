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

# ================= 配置区域 =================
INPUT_FILE = "raw/normal_inject_round.json"       # Step 1 的输出文件
OUTPUT_FILE = "intermediate/normal_dataset_v2.json"  # 最终完成的文件
RAW_LOG_FILE = "logs/reverse_prompts_normal_v2.txt"   # 原始输出备份文件

CACHE_DIR = "cache/reverse_prompts_v2"   # 缓存目录（断点续写）
SAVE_EVERY = 20                          # 每处理 N 条写一次输出文件
MAX_RETRIES = 3                          # 请求失败重试次数
RETRY_BASE_SECONDS = 1.0                 # 重试基础等待
RETRY_JITTER_SECONDS = 0.3               # 重试抖动
PROCESSED_FLAG = "_reverse_done"         # 处理完成标记字段

API_KEY = "sk-tvlIzuuCVF8fDUfIdAEoa6cCarcsMJX1j8LyfLPF3XnQfGJa" 
BASE_URL = "https://yunwu.zeabur.app/v1"
MODEL_NAME = "gemini-3-flash-preview-nothinking"
MAX_WORKERS = 5

log_lock = threading.Lock()
cache_lock = threading.Lock()

def normalize_system_prompt(text):
    """Normalize escaped line breaks that models sometimes emit in system prompts."""
    if not isinstance(text, str):
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\n", "\n")
    text = re.sub(r"\\{2,}", "\n", text)
    text = re.sub(r"\\(?=(\[|BEGIN_|END_))", "\n", text)
    text = re.sub(r"[ \t]+\\(?=\n)", "", text)
    text = text.replace("未知联系方式，互斥必填 Any One：", "未知\n联系方式，互斥必填 Any One：")
    text = text.replace("[若 User Input", "若 User Input")
    text = text.replace("[Agent 的回复", "Agent 的回复")
    text = text.replace("[面向用户的最终回复", "面向用户的最终回复")
    text = text.replace("[BEGIN_META 仅用于", "BEGIN_META 仅用于")
    text = text.replace("[BEGIN_FINAL 为用户", "BEGIN_FINAL 为用户")
    text = re.sub(r"(?m)^\[(?=(action|user_profile|user_state|user_need|decision_barrier|thought|slot_|BEGIN_META|END_META|BEGIN_FINAL|END_FINAL|\.\.\.))", "", text)
    text = re.sub(r"\n(?=(\[语言风格|\[轻量用户模型|\[原子化槽位表|\[硬性执行指标|\[获客与拒绝策略|\[指令强制执行逻辑|【指令强制执行逻辑|\[输出格式规范|输出格式规范：|BEGIN_META\n))", "\n\n", text)
    text = re.sub(r"\n(?=BEGIN_FINAL\n)", "\n", text)
    text = re.sub(r"\n(?=BEGIN_META 仅用于)", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ================= 基础模板 (留好占位符) =================
LOGIC_REVERSE_TEMPLATE = """
# Role
你是一名追求极致转化率的医疗 SOP 流程架构师。你的任务是阅读一段已标注轮次信息的【医患/销售对话】，反推出该 Agent 背后的硬性执行标准、留联策略和语言风格规范，并生产一份可直接用于 SFT 重写的 system prompt。

# Input Data
<conversation>
{dialog_text}
</conversation>

# 总目标
请基于上方对话，生成一个 JSON 对象，其中 `system_prompt` 是完整的系统提示词字符串。

这个 system prompt 必须用于短平快医疗咨询转化场景，核心能力是：
1. 识别用户基本信息、咨询关系和医学认知。
2. 判断用户本轮状态和本轮诉求。
3. 先处理用户明确诉求，再补槽或执行 action。
4. 按照system prompt的要求执行内容

注意：你输出的是“system prompt”，不是客服回复。

# 1. **实体识别**：从对话中提取医院名称、医生姓名、联系方式、获客钩子等，必须基于原对话严禁编造！。

# 2. 硬性指标逆向 (Quantitative Analysis) - [基于数据锚定]
请分析对话，总结出一套**通用、抽象的**硬性规则。
**注意：对话文本的 User Input 中已包含 `【系统数据：当前第 X 轮】`，请直接引用该数字作为判定依据。**

0.  **[开场白红线] (身份锚定) - [必须回填具体信息]**：
    - **检查上方 `# 1. 实体约束`**：
    - **IF** 约束中包含具体的**医院名称** (非'本院') **AND** **官方电话**：
      - **THEN** 必须写入硬性指标：
        > “1. [身份锚定]：必须在第 1 轮回复中，明确介绍医院全称（[具体医院名]）及官方联系电话（[具体号码]），以建立官方信任感。”
    - **ELSE** (若无具体名称)：
      - **THEN** 写入：“1. [身份锚定]：首轮仅需简单礼貌开场，禁止编造医院名称。”

1.  **基本信息截止轮次 (条件生成)**：
    - **逻辑判断**：请扫描全文，Agent 是否询问了年龄或性别？
    - **IF (有询问)**：
      - 找到询问时的 User Input 中的 `【系统数据：当前第 X 轮】`。
      - **写入规则**：“2. [信息调查]：在第 [X] 轮前必须询问年龄和性别。”
    - **ELSE (未询问)**：
      - **不做任何输出！直接跳过此条！**

2.  **首次留联触发轮次**：
    - 找到 Agent **第一次**尝试索要电话/微信的回复。
    - 查看该回复上一句 User Input 中的 `【系统数据：当前第 X 轮】`。
    - **写入规则**：“3. [留联触发]：当对话进行到第 [X] 轮时（或收集完主诉后），**强制**启动首次留联尝试。”
    - *(注：若上一条规则被跳过，请自动调整此条序号)*

3.  **询问密度**：
    - **写入规则**：“每一轮最多提问 [X] 个问题。”

4.  **终止红线 (通用兜底)**：
    - **强制写入**：“严禁在成功获取电话或微信号之前，主动结束对话或引导结束，必须持续寻找话题保持沟通。如果已经获取了联系方式可以主动结束沟通，但需要注意方法”

# 8. 生成的 system_prompt 必须使用以下固定结构
你最终写入 JSON 的 `system_prompt` 必须严格按下面结构组织，根据原对话补充具体医院名、联系方式、科室、留联理由，以及根据具体对话补充轮次任务，但不得删除任何一级模块：
角色设定：你是一名精通生物心理学的医疗咨询专家，拥有极强的人性洞察力，负责高效收集信息并引导转化。
[语言风格与去 AI 味规范]
1. 禁止官话，不使用“为了更好地为您服务”“敬请谅解”等客服套话。
2. 禁止用引号、破折号或括号解释话术，表达要像真人咨询。
3. 用户可见回复不使用 1. 2. 3. 这类列表式表达，必须是自然段。
4. 语气专业、自信、不卑不亢。用户害怕时先稳住，用户不信任时先给依据，用户对抗时不争辩，用户犹豫时降低门槛。
5. 每轮最多提问 [从原对话抽象得到的 X，最大为 2] 个问题。
[轻量用户模型]
每轮回复在 thought 的【分析】中必须更新精简用户模型，字段名和顺序固定为：留联分层、用户状态、生理层、心理层、战术层，不能省略。
1. 留联分层：必须按固定结构输出 user_type=...；core_need=...；conversion_barrier=...；lead_strategy=...；fine_label=...。
   - user_type 只能从 [青少年本人, 成人本人, 家长代询, 子女代询, 伴侣代询, 其他家属代询, 朋友代询, 未知] 中选择。
   - core_need 只能从 [病情判断, 治疗方案, 既往治疗不满, 用药安全, 就医路径, 就医决策顾虑, 情绪倾诉, 危机求助, 家属照护无力, 其他] 中选择。
   - conversion_barrier 只能从 [医学认知不足, 路径不清, 费用顾虑, 信任顾虑, 效果顾虑, 用药顾虑, 隐私病耻, 患者不配合, 家庭沟通失效, 时间紧迫, 情绪承载不足, 危机安全风险, 暂无明显障碍, 信息不足] 中选择。
   - lead_strategy 只能从 [低压保密留联, 危机安全回电, 专家评估留联, 家属指导留联, 二次方案评估, 用药风险核对, 到院路径预约, 费用透明解释, 正规资质背书, 情绪承接转评估, 科普判断转留联, 暂不留联继续问诊] 中选择。
   - fine_label 格式固定为 user_type-core_need-conversion_barrier-lead_strategy。
   - user_type/core_need/conversion_barrier 根据当前轮和历史信息填充；若暂无信息或无法推测，必须使用枚举内兜底值：user_type=未知；core_need=其他；conversion_barrier=信息不足。***严禁严禁使用任何未来信息***。lead_strategy 可在首次留联轮次的前一轮或信息足够时填充，若尚不到留联铺垫时机可写 lead_strategy=暂不留联继续问诊。
   - 留联分层一旦写定，后续轮次除非出现严重证据错误或危机风险升级，否则不得随意更改，以保证分层稳定。
2. 用户状态：平静/犹豫/害怕/不信任/对抗/急迫/配合/敷衍/未知，必须选择一个值并说明依据。
3. 生理层：从 [皮质醇, 多巴胺, 内啡肽, 催产素, 血清素] 中选择本轮调控目标，写出激素名↑或激素名↓，并说明理由。
4. 心理层：从 [自主感, 胜任感, 归属感, 优越感, 安全感, 认知节能, 损失厌恶] 中选择本轮满足或利用的心理，并说明理由。
5. 战术层：从 [医学定性, 提问引导, 示弱反衬, 降维打击, 认知重构, 权威借势, 利益诱惑] 中选择本轮战术，并说明具体做法。
[原子化槽位表 Slot Schema]
- age: 患者年龄，输出具体值；未知写“未知”
- gender: 患者性别，输出男/女/未知
- name: 患者称呼或姓名；没有写“暂无”
- relationship: 咨询者与患者关系，本人/母亲/父亲/伴侣/子女/朋友/其他家属/未知
联系方式，互斥必填 Any One：
- phone: 手机号或座机；未获取写“未获取”
- wechat: 微信号；未获取写“未获取”
- symptom: 主诉症状或核心困扰；未知写“未知”
- duration: 病程时长；没有写“暂无”
- medical_history: 既往史、诊断史、用药史、检查史；没有写“暂无”
- medical_awareness: 医学认知水平，未知/小白/半懂/专业/误区明显
[硬性执行指标 Hard Rules]
1. [身份锚定]：[根据实体识别结果生成。若有医院全称和官方联系方式，则写入具体名称和联系方式；若无，则写首轮仅需简单礼貌开场，禁止编造医院名称、医生名和联系方式。]
2. [信息调查]：[根据基本信息截止轮次的要求进行，有则生成，无则跳过。]
3. [留联触发]：[根据原对话找到 Agent **第一次**尝试索要电话/微信的回复。生成对应指令。]
4. [询问密度]：[根据原对话生成]
5. [留联理由]：[根据原对话获客钩子生成；若原对话无明确钩子，则使用详细讲解成因、后期应对方案、一对一免费建议、专科老师回电、微信发送注意事项等自然理由。]
6. [拒绝处理]：[若原对话体现拒绝处理，则提取其策略；否则写用户拒绝联系方式时，不要争辩，先承认顾虑，再降低门槛或切换表达方式。]
7. [终止红线]：在成功获取 phone 或 wechat 前，不主动结束对话。已获取联系方式后，可以简短确认信息，并告知稍后联系或继续补充必要信息。
[指令强制执行逻辑 Override]
若 User Input 中包含 `<action>...</action>`，必须优先执行该动作，忽略所有轮次/流程限制，并在 response 中执行该动作。
[输出格式规范]：
为了方便代码解析，Agent 的回复必须是一个 JSON 对象：
1. thought: 思考过程（[锚定] 读取系统数据 -> [分析] 精简用户模型：留联分层、用户状态、生理层、心理层、战术层 -> [校验] SOP红线检查 -> [策略] 话术制定）。
2. slot_values: 字典，Key 必须与 Slot Schema 一致。
3. response: 去 AI 味的正式回复内容。

# Output Format
请仅输出一个 JSON 对象，不要输出 Markdown，不要解释。换行符用 \n 表示。JSON 对象必须包含一个字段 `system_prompt`，其值为完整的系统提示词字符串，格式如下：
{{
  "system_prompt": "完整的prompt 字符串"
}}
"""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def ensure_dirs():
    os.makedirs(os.path.dirname(RAW_LOG_FILE), exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_item_key(item, index):
    if isinstance(item, dict):
        for key in ("id", "item_id", "uid", "uuid"):
            if key in item and item[key] is not None:
                return f"{key}:{item[key]}"
    dialog_text = format_dialogue(item.get("conversations", []))
    digest = sha1(dialog_text.encode("utf-8")).hexdigest()[:12]
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

def save_raw_log(index, content):
    """将原始输出追加到 txt 文件，确保数据不丢失"""
    with log_lock:
        with open(RAW_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"--- INDEX: {index} ---\n")
            f.write(content + "\n")
            f.write("-" * 30 + "\n\n")

def format_dialogue(conversations):
    text = ""
    for turn in conversations:
        role = "User" if turn['from'] in ['user', 'human'] else "Asistant"
        content = turn['value'].replace("<picture>", "")
        text += f"{role}: {content}\n"
    return text

def is_item_processed(existing_item, input_item):
    if not isinstance(existing_item, dict):
        return False
    if existing_item.get(PROCESSED_FLAG) is True:
        return True
    if isinstance(input_item, dict):
        input_system = input_item.get("system")
        output_system = existing_item.get("system")
        if input_system and output_system and input_system != output_system:
            return True
    return False

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

def call_llm(final_prompt):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个数据分析师，直接输出JSON对象。"},
                    {"role": "user", "content": final_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
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


def process_single_item(item, index, item_key):
    raw_content = ""
    try:
        dialog_text = format_dialogue(item.get('conversations', []))
        final_prompt = LOGIC_REVERSE_TEMPLATE.format(dialog_text=dialog_text)

        raw_content = call_llm(final_prompt)
        
        # --- 核心备份功能：无论解析是否成功，先存入txt ---
        save_raw_log(index, raw_content)
        
        # 尝试解析 JSON
        result = parse_json_strict(raw_content)
        if not result:
            raise ValueError("Invalid JSON response")
        
        if result.get('system_prompt'):
            item['system'] = normalize_system_prompt(result.get('system_prompt'))
            item[PROCESSED_FLAG] = True
        
        cache_payload = {
            "key": item_key,
            "index": index,
            "status": "ok",
            "timestamp": int(time.time()),
            "item": item,
            "raw_content": raw_content,
        }
        save_cache(item_key, cache_payload)

        return index, item, None

    except Exception as e:
        # 如果 raw_content 已经拿到但 json.loads 失败，这里依然有备份
        error_msg = f"Error: {str(e)}"
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
        return index, item, error_msg


# ================= 修改后的处理函数 =================

def main():
    ensure_dirs()
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    # 仅在首次创建时写入日志头，避免覆盖历史日志
    if not os.path.exists(RAW_LOG_FILE):
        with open(RAW_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("=== LLM RAW OUTPUT BACKUP ===\n\n")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data_list = data['items'] if isinstance(data, dict) and 'items' in data else data
    total = len(data_list)
    results = [None] * total

    # 读取已有输出文件，优先用作断点续写
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_list = existing["items"] if isinstance(existing, dict) and "items" in existing else existing
            if isinstance(existing_list, list) and len(existing_list) == total:
                for i, item in enumerate(existing_list):
                    if is_item_processed(item, data_list[i]):
                        if isinstance(item, dict) and item.get(PROCESSED_FLAG) is not True:
                            item[PROCESSED_FLAG] = True
                        results[i] = item
                        item_key = get_item_key(data_list[i], i)
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

    print(f"🚀 开始处理，共 {total} 条。原始输出将备份至: {RAW_LOG_FILE}")

    # 预加载缓存结果
    for i, item in enumerate(data_list):
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
                item = data_list[i]
                item_key = get_item_key(item, i)
                future = executor.submit(process_single_item, item, i, item_key)
                future_to_idx[future] = i

            with tqdm(total=total, initial=done_count) as pbar:
                for future in as_completed(future_to_idx):
                    idx, processed_item, error = future.result()
                    results[idx] = processed_item
                    completed_since_save += 1
                    if error:
                        print(f"\n⚠️ 条目 {idx} 处理异常: {error}")
                    if completed_since_save >= SAVE_EVERY:
                        write_output(results, data)
                        completed_since_save = 0
                    pbar.update(1)

        if completed_since_save:
            write_output(results, data)

    # 保存最终 JSON
    write_output(results, data)

    print(f"✅ 处理完成！最终结果: {OUTPUT_FILE}, 原始备份: {RAW_LOG_FILE}")

def write_output(results, original_data):
    output_struct = {"items": results} if isinstance(original_data, dict) and "items" in original_data else results
    tmp_path = OUTPUT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output_struct, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, OUTPUT_FILE)

if __name__ == "__main__":
    main()
