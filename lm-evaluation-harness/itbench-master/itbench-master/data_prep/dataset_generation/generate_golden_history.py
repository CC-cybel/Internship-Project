import json
import random
import os
import sys
from pathlib import Path
import time
import argparse
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from itbench.utils.config import config
from itbench.model.api_model import APIModel
from itbench.generation.model_configs import get_processor_for_model
from itbench.generation.generator import ResponseGenerator

def get_judge_model():
    return APIModel(
        model_name=config.JUDGE_MODEL_NAME,
        api_key=config.JUDGE_API_KEY,
        api_base=config.JUDGE_API_BASE,
        temperature=0.7
    )

def get_candidate_generator():
    candidate_model = APIModel(
        model_name=config.CANDIDATE_MODEL_NAME,
        api_key=config.CANDIDATE_API_KEY,
        api_base=config.CANDIDATE_API_BASE,
        temperature=0.7
    )
    processor = get_processor_for_model(config.CANDIDATE_MODEL_NAME)
    # limit max_tokens to avoid exceeding the 4096 context window (2048 is too high, set to 512)
    return ResponseGenerator(model=candidate_model, processor=processor, max_tokens=512)

def load_rules():
    rule_file = Path('data/rules/itbench_rule.json')
    with open(rule_file, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    return rules

def load_search_words(domain):
    word_file = Path(f'data_prep/data/{domain}/search_words.txt')
    with open(word_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return words

def load_system_prompt_demo(domain):
    prompt_file = Path(f'data_prep/data/{domain}/system_prompt_demo.txt')
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_prompt_template(domain):
    template_file = Path(f'data_prep/dataset_generation/prompt/{domain}/system_prompt_generation.txt')
    with open(template_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def _strip_json_fence(text):
    text = (text or "").strip()
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    return match.group(1).strip() if match else text

def _extract_jsonish_string(text, field, next_field):
    field_pat = re.compile(rf'["\']{re.escape(field)}["\']\s*:', re.DOTALL)
    match = field_pat.search(text)
    if not match:
        return None

    pos = match.end()
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] not in {'"', "'"}:
        return None

    quote = text[pos]
    start = pos + 1
    next_pat = re.compile(rf'{re.escape(quote)}\s*,\s*["\']{re.escape(next_field)}["\']\s*:', re.DOTALL)
    next_match = next_pat.search(text, start)
    if not next_match:
        return None
    return text[start:next_match.start()]

def _parse_system_prompt_response(response):
    json_str = _strip_json_fence(response)
    try:
        data = json.loads(json_str)
        return data
    except Exception:
        pass

    # The prompt text often contains unescaped quotes, so fall back to field-boundary extraction.
    thinking = _extract_jsonish_string(json_str, "thinking", "new_system_prompt") or ""
    new_system_prompt = _extract_jsonish_string(json_str, "new_system_prompt", "end_turn")
    end_turn_match = re.search(r'["\']end_turn["\']\s*:\s*["\']?(\d+)["\']?', json_str)
    if new_system_prompt and end_turn_match:
        return {
            "thinking": thinking,
            "new_system_prompt": new_system_prompt,
            "end_turn": int(end_turn_match.group(1)),
            "json_repaired_by": "field_boundary_parser",
        }

    raise ValueError("Could not parse system prompt response as JSON or json-like fields")

def generate_system_prompt(judge_model, base_prompt, selected_rules, search_word, prompt_template):
    rules_desc = "\n".join([f"- 规则ID: {r['rule_id']}，名称: {r['rule_name_cn']}，描述: {r['description']}" for r in selected_rules])
    
    prompt = prompt_template.format(
        base_prompt=base_prompt,
        num_rules=len(selected_rules),
        rules_desc=rules_desc,
        search_word=search_word
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    max_retries = 3
    last_error = None
    last_response = ""
    
    for attempt in range(max_retries):
        response = judge_model.chat(messages, enable_thinking=False, max_tokens=40960)
        
        try:
            data = _parse_system_prompt_response(response)
            return data['new_system_prompt'], data['end_turn'], data, prompt
        except Exception as e:
            last_error = e
            last_response = response
            print(f"Attempt {attempt + 1} failed to parse JSON for system prompt generation. Error: {e}")
            time.sleep(1) # Add a slight delay before retrying
            
    print(f"All {max_retries} attempts failed for system prompt generation.")
    print(f"Raw response: {last_response}")
    # Indicate failure via the result dict
    return base_prompt + "\n\n[附加规则]：\n" + rules_desc, 5, {"error": str(last_error), "raw_response": last_response}, prompt

def simulate_user_reply(judge_model, search_word, history):
    history_str = ""
    for msg in history[1:]:
        role = "医生/客服" if msg['role'] == 'assistant' else "患者"
        history_str += f"[{role}]: {msg['content']}\n"
        
    prompt = f"""你是一个在网上寻求医疗咨询的普通患者，你的初始搜索词是："{search_word}"。
以下是目前的对话历史：
{history_str}

请根据医生/客服的最后一条回复，给出一句简短、自然的回答（一般不超过15个字）。
要求：
1. 不要显得太配合，也不要太生硬，符合普通患者的心理。
2. 医生问你问题你就简短回答，医生如果不问问题你可以反问。
3. 如果医生要求留电话或微信，你可以选择拒绝（“不方便”、“不想加”）或者同意（提供类似13800138000或微信号wx123）。
4. 只需输出你的回复内容，不要包含其他任何分析、角色前缀或引号。
"""
    messages = [{"role": "user", "content": prompt}]
    reply = judge_model.chat(messages, enable_thinking=False)
    return reply.strip(' "''\n'), prompt

def generate_single_sample(
    idx,
    num_samples,
    search_words,
    available_rules,
    base_system_prompt,
    prompt_template,
    include_rule_ids=None,
    num_rules_min=4,
    num_rules_max=5,
):
    print(f"[{idx+1}/{num_samples}] Starting sample generation...")
    
    # Instantiate models per thread to ensure thread-safety with HTTP connections
    judge_model = get_judge_model()
    candidate_generator = get_candidate_generator()
    
    search_word = random.choice(search_words)
    include_rule_ids = include_rule_ids or []
    include_rules = [r for r in available_rules if r.get("rule_id") in include_rule_ids]
    include_rule_id_set = {r.get("rule_id") for r in include_rules}
    optional_rules = [r for r in available_rules if r.get("rule_id") not in include_rule_id_set]

    num_rules = random.randint(num_rules_min, num_rules_max)
    fill_count = max(0, min(num_rules, len(available_rules)) - len(include_rules))
    selected_rules = include_rules + random.sample(optional_rules, min(fill_count, len(optional_rules)))
    random.shuffle(selected_rules)
    
    print(f"[{idx+1}/{num_samples}] Generating system prompt and end_turn...")
    new_system_prompt, end_turn, system_prompt_generation_result, system_prompt_generation_prompt = generate_system_prompt(judge_model, base_system_prompt, selected_rules, search_word, prompt_template)
    end_turn = max(2, min(8, int(end_turn)))
    print(f"[{idx+1}/{num_samples}] Target end_turn: {end_turn}")
    
    # messages list for output JSON and generating user simulator prompt
    messages = [
        {"role": "system", "content": new_system_prompt, "turn_id": 0},
        {"role": "user", "content": search_word, "turn_id": 1}
    ]
    
    generation_details = []
    
    # Use candidate history without 'turn_id' keys for API compatibility if needed
    candidate_history = [
        {"role": "system", "content": new_system_prompt},
        {"role": "user", "content": search_word}
    ]
    
    print(f"[{idx+1}/{num_samples}] Simulating interaction...")
    
    for turn in range(1, end_turn):
        # Assistant generates response for the current `turn`
        gen_sample = {
            'messages': candidate_history.copy(),
            'turn_id': turn
        }
        
        try:
            # Generate response via CANDIDATE_MODEL
            candidate_reply = candidate_generator.generate_response(gen_sample)
            candidate_reply_raw = gen_sample.get('raw_response', candidate_reply)
            candidate_processed_messages = gen_sample.get('processed_messages', candidate_history)
            
            if isinstance(candidate_reply_raw, dict):
                candidate_reply_raw = candidate_reply_raw.get('content', str(candidate_reply_raw))
                
        except Exception as e:
            print(f"[{idx+1}/{num_samples}] API error during candidate turn: {e}")
            candidate_reply = "你好，请问有什么可以帮您？"
            candidate_reply_raw = candidate_reply
            candidate_processed_messages = candidate_history
            
        messages.append({
            "role": "assistant",
            "content": candidate_reply.strip(),
            "turn_id": turn
        })
        candidate_history.append({"role": "assistant", "content": candidate_reply_raw})
        
        # Simulate user reply for the NEXT turn
        user_reply, user_generation_prompt = simulate_user_reply(judge_model, search_word, messages)
        messages.append({
            "role": "user",
            "content": user_reply,
            "turn_id": turn + 1
        })
        candidate_history.append({"role": "user", "content": user_reply})
        
        # Save intermediate generation detail
        generation_details.append({
            'turn': turn,
            'candidate_processed_messages': candidate_processed_messages,
            'candidate_reply_raw': candidate_reply_raw,
            'candidate_reply': candidate_reply,
            'user_generation_prompt': user_generation_prompt,
            'user_reply': user_reply
        })
    
    output_item = {
        "key": f"gen_test_{int(time.time())}_{idx+1:03d}",
        "models": {
            "judge_model": config.JUDGE_MODEL_NAME,
            "candidate_model": config.CANDIDATE_MODEL_NAME
        },
        "messages": messages,
        "rule_list": [r['rule_name_cn'] for r in selected_rules],
        "system_prompt_generation_prompt": system_prompt_generation_prompt,
        "system_prompt_generation_result": system_prompt_generation_result, # Added system prompt generation result
        "generation_details": generation_details # Include details as in ref/lead_playground
    }
    
    print(f"[{idx+1}/{num_samples}] Finished sample.")
    return output_item

def main():
    parser = argparse.ArgumentParser(description="Generate golden history test cases.")
    parser.add_argument('--domain', type=str, default='psychiatry', help="The domain name for data directories (e.g., psychiatry)")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of samples to generate")
    parser.add_argument('--num_workers', type=int, default=5, help="Number of concurrent workers")
    parser.add_argument('--include_rule_ids', type=str, default="", help="Comma-separated rule IDs that must appear in every sample, e.g. 15")
    parser.add_argument('--exclude_rule_ids', type=str, default="", help="Comma-separated rule IDs to exclude from the optional pool, e.g. 14")
    parser.add_argument('--num_rules_min', type=int, default=4, help="Minimum number of rules per generated sample")
    parser.add_argument('--num_rules_max', type=int, default=5, help="Maximum number of rules per generated sample")
    parser.add_argument('--output_file', type=str, default="", help="Optional output JSONL path. Defaults to data_prep/data/{domain}/generated_golden_history.jsonl")
    parser.add_argument('--append', action='store_true', help="Append to output_file instead of overwriting it")
    args = parser.parse_args()

    domain = args.domain
    num_samples = args.num_samples
    num_workers = args.num_workers
    include_rule_ids = [int(x.strip()) for x in args.include_rule_ids.split(",") if x.strip()]
    exclude_rule_ids = [int(x.strip()) for x in args.exclude_rule_ids.split(",") if x.strip()]
    num_rules_min = min(args.num_rules_min, args.num_rules_max)
    num_rules_max = max(args.num_rules_min, args.num_rules_max)

    all_rules = load_rules()
    if domain == 'douyin_dentistry':
        target_rule_ids = [3, 4, 5, 9, 10, 11, 12, 13, 16]
    else:
        target_rule_ids = [3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16]
    target_rule_ids = [rid for rid in target_rule_ids if rid not in exclude_rule_ids]
    for rid in include_rule_ids:
        if rid not in target_rule_ids:
            target_rule_ids.append(rid)
    available_rules = [r for r in all_rules if r.get('rule_id') in target_rule_ids]
    
    search_words = load_search_words(domain)
    base_system_prompt = load_system_prompt_demo(domain)
    prompt_template = load_prompt_template(domain)
    
    output_file = Path(args.output_file) if args.output_file else Path(f'data_prep/data/{domain}/generated_golden_history.jsonl')
    os.makedirs(output_file.parent, exist_ok=True)
    
    # Thread-safe writing lock
    write_lock = threading.Lock()

    print(f"Starting generation of {num_samples} samples with {num_workers} workers...")
    print(f"include_rule_ids={include_rule_ids or '<none>'}, exclude_rule_ids={exclude_rule_ids or '<none>'}")
    print(f"rules_per_sample={num_rules_min}-{num_rules_max}")
    print(f"output_file={output_file}")
    
    success_count = 0
    failed_count = 0
    
    open_mode = 'a' if args.append else 'w'
    with open(output_file, open_mode, encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    generate_single_sample, 
                    i,
                    num_samples,
                    search_words,
                    available_rules,
                    base_system_prompt,
                    prompt_template,
                    include_rule_ids,
                    num_rules_min,
                    num_rules_max,
                ): i for i in range(num_samples)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    output_item = future.result()
                    
                    # Track failures by checking if 'error' exists in the generation result dict
                    if "error" in output_item.get("system_prompt_generation_result", {}):
                        failed_count += 1
                    else:
                        success_count += 1
                        
                    with write_lock:
                        f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                        f_out.flush()
                except Exception as e:
                    print(f"[{idx+1}/{num_samples}] Error occurred during generation: {e}")
                    failed_count += 1

    print("\n" + "="*40)
    print("Generation Stats Summary")
    print("="*40)
    print(f"Total samples requested: {num_samples}")
    print(f"Successfully generated (JSON parsed): {success_count}")
    print(f"Failed JSON parsing (used fallback):  {failed_count}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
