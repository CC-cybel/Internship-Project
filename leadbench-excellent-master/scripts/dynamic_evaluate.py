import argparse
import json
import os
import sys
import time
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path to allow importing leadbench package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leadbench_excellent.utils.config import config
from leadbench_excellent.utils.dataset import DialogueEvaluationDataset
from leadbench_excellent.model.api_model import APIModel
from leadbench_excellent.evaluation.session_evaluator import SessionEvaluator
from leadbench_excellent.generation.generator import ResponseGenerator
from leadbench_excellent.generation.model_configs import get_processor_for_model
from leadbench_excellent.simulator.user_simulator import AdvancedUserSimulator

def process_sample_session(sample, evaluator, candidate_generator, user_simulator_model):
    """
    Process a single sample (dynamic dialogue generation + session evaluation).
    """
    try:
        initial_messages = sample.get('messages', [])
        if len(initial_messages) < 2:
            print(f"Skipping sample {sample.get('id')} - not enough initial messages.")
            return None
            
        system_msg = initial_messages[0]
        user_first_msg = initial_messages[1]
        
        # Build candidate history
        candidate_history = [
            {"role": "system", "content": system_msg["content"]},
            {"role": "user", "content": user_first_msg["content"]}
        ]
        
        max_turns = 10
        turn = 1
        
        # Build Advanced Simulator
        simulator = AdvancedUserSimulator(model=user_simulator_model, close_turn=max_turns)
        simulator.initialize(keyword=user_first_msg["content"], domain="医疗")
        
        simulator_history = [
            {"role": "user", "content": user_first_msg["content"]}
        ]
        
        final_messages = [
            {"role": "system", "content": system_msg["content"], "turn_id": 0},
            {"role": "user", "content": user_first_msg["content"], "turn_id": 1}
        ]
        
        generation_details = []
        
        while turn <= max_turns:
            # Candidate Model Replies
            gen_sample = {
                'messages': candidate_history.copy(),
                'turn_id': turn
            }
            candidate_reply = candidate_generator.generate_response(gen_sample)
            candidate_reply_raw = gen_sample.get('raw_response', candidate_reply)
            if isinstance(candidate_reply_raw, dict):
                candidate_reply_raw = candidate_reply_raw.get('content', str(candidate_reply_raw))
            candidate_processed_messages = gen_sample.get('processed_messages', candidate_history)
            
            # If candidate fails to reply or API error, stop interaction to avoid empty loops
            if not candidate_reply or not candidate_reply.strip():
                print(f"Warning: Candidate model returned empty reply at turn {turn}. Ending dialogue.")
                break
                
            final_messages.append({"role": "assistant", "content": candidate_reply, "turn_id": turn})
            candidate_history.append({"role": "assistant", "content": candidate_reply_raw})
            
            # User Simulator Replies
            simulator_history.append({"role": "assistant", "content": candidate_reply})
            sim_input_messages = list(simulator_history)
            
            sim_reply, sim_messages = simulator.generate_reply(turn=turn, chat_history=simulator_history)
            if isinstance(sim_reply, dict):
                sim_reply = sim_reply.get('content', '')
            
            # If user simulator fails to reply, stop interaction
            if not sim_reply or not sim_reply.strip():
                print(f"Warning: User simulator model returned empty reply at turn {turn}. Ending dialogue.")
                break
                
            is_dialog_over = False
            if "<dialogover>" in sim_reply:
                sim_reply = sim_reply.replace("<dialogover>", "").strip()
                is_dialog_over = True
                
            # Record generation details for this turn
            generation_details.append({
                'turn': turn,
                'candidate_processed_messages': candidate_processed_messages,
                'candidate_reply_raw': candidate_reply_raw,
                'candidate_reply': candidate_reply,
                'simulator_messages': sim_messages, # Log the actual messages sent to the simulator model
                'simulator_history': sim_input_messages, # Log the history context
                'simulator_reply': sim_reply
            })
                
            if is_dialog_over:
                if sim_reply:
                    final_messages.append({"role": "user", "content": sim_reply, "turn_id": turn + 1})
                break
                
            final_messages.append({"role": "user", "content": sim_reply, "turn_id": turn + 1})
            candidate_history.append({"role": "user", "content": sim_reply})
            simulator_history.append({"role": "user", "content": sim_reply})
            
            turn += 1
            
        sample['messages'] = final_messages
        sample['generation_details'] = generation_details
        
        # Evaluate
        return evaluator.evaluate_session(sample)
    except Exception as e:
        print(f"Error processing sample ID {sample.get('id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_report(results, output_dir, execution_time_seconds, stats=None):
    """
    Generates a markdown report from the evaluation results.
    """
    report_path = os.path.join(output_dir, 'evaluation_report.md')
    
    total_candidates = len(results)
    total_triggered_checks = 0
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # Rule stats: {rule_id: {'name': '', 'dimension': '', 'triggered': 0, 'normalized_scores': []}}
    rule_stats = defaultdict(lambda: {'name': '', 'dimension': 'Unknown', 'triggered': 0, 'normalized_scores': []})

    dialogue_scores = []

    for res in results:
        eval_res = res.get('evaluation', {})
        usage = eval_res.get('usage', {})
        total_prompt_tokens += usage.get('prompt_tokens', 0)
        total_completion_tokens += usage.get('completion_tokens', 0)
        total_tokens += usage.get('total_tokens', 0)

        current_dialogue_normalized_scores = []
        rule_details = eval_res.get('rule_details', [])
        for rule in rule_details:
            rule_id = rule.get('rule_id')
            rule_name = rule.get('rule_name_cn', rule.get('rule_name', 'Unknown'))
            
            # Update name if not set
            if not rule_stats[rule_id]['name']:
                rule_stats[rule_id]['name'] = rule_name
            
            # Dimension is unknown for now
            dimension = 'Unknown'
            # Heuristic for dimension (optional)
            if 'logic' in rule_name.lower():
                dimension = 'logic'
            elif 'compliance' in rule_name.lower():
                dimension = 'compliance'
            
            rule_stats[rule_id]['dimension'] = dimension

            if rule.get('is_triggered', False):
                rule_stats[rule_id]['triggered'] += 1
                total_triggered_checks += 1
                
                # normalized_score is already calculated in session_evaluator.py
                # It is score_change / max_possible_score
                normalized_score = rule.get('normalized_score', 0)
                
                # If normalized_score is None (shouldn't happen if triggered), default to 0
                if normalized_score is None:
                    normalized_score = 0
                    
                rule_stats[rule_id]['normalized_scores'].append(normalized_score)
                current_dialogue_normalized_scores.append(normalized_score)

        # 2. 计算单个对话得分：触发的场景平均得分
        if current_dialogue_normalized_scores:
            avg_dialogue_score = sum(current_dialogue_normalized_scores) / len(current_dialogue_normalized_scores)
            dialogue_scores.append(avg_dialogue_score)

    
    # 3. 计算总平均得分：所有对话的得分平均
    overall_avg_score = sum(dialogue_scores) / len(dialogue_scores) if dialogue_scores else 0
    
    avg_prompt = total_prompt_tokens / total_candidates if total_candidates > 0 else 0
    avg_completion = total_completion_tokens / total_candidates if total_candidates > 0 else 0
    avg_total = total_tokens / total_candidates if total_candidates > 0 else 0
    
    # Format execution time
    hours, remainder = divmod(execution_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s ({execution_time_seconds:.2f}s)"
    
    avg_time_per_sample = execution_time_seconds / total_candidates if total_candidates > 0 else 0

    # Generate Markdown
    md = []
    md.append("# 评估统计报告 (Evaluation Report)")
    md.append(f" - **总候选回复数 (Total Candidates)**: {total_candidates}")
    md.append(f" - **总触发规则次数 (Total Triggered Checks)**: {total_triggered_checks}")
    md.append(f" - **平均得分**: {overall_avg_score:.4f} (Base 1.0)")
    md.append("   > **计算公式**：\n   > 1. **计算场景得分**：场景得分 = 该场景实际得分 / 该场景可能获得的最大得分（未触发不计入）\n   > 2. **计算单个对话得分**：单个对话得分 = 该对话中所有已触发场景得分的平均值\n   > 3. **计算总平均得分**：总平均得分 = 所有对话得分的平均值")
    md.append(f" - **平均Token消耗 (Average Tokens per Sample)**: Input: {avg_prompt:.2f}, Output: {avg_completion:.2f}, Total: {avg_total:.2f}")
    md.append(f" - **总评估耗时 (Total Evaluation Time)**: {time_str}")
    md.append(f" - **平均单条评估耗时 (Average Time per Sample)**: {avg_time_per_sample:.2f}s")
    
    if stats:
        md.append("\n## API 调用与解析可靠性 (API & Parsing Reliability)")
        md.append("如果失败率过高，可能会影响最终评估结果的准确性。")
        md.append("\n| 模型角色 | 模型名称 | 总调用数 | 失败数 | 失败率 |")
        md.append("| --- | --- | --- | --- | --- |")
        
        for role, role_name in [('candidate_model', '待评估模型 (Candidate)'), ('user_simulator_model', '用户模拟模型 (User Sim)'), ('judge_model', '评委模型 (Judge)')]:
            if role in stats:
                model_stats = stats[role]
                calls = model_stats.get('api_calls', 0)
                failures = model_stats.get('api_failures', 0)
                rate = f"{(failures / calls * 100):.2f}%" if calls > 0 else "0.00%"
                md.append(f"| {role_name} | {model_stats.get('name', 'N/A')} | {calls} | {failures} | {rate} |")
                
        judge_stats = stats.get('judge_model', {})
        parse_attempts = judge_stats.get('json_parse_attempts', 0)
        parse_failures = judge_stats.get('json_parse_failures', 0)
        parse_rate = f"{(parse_failures / parse_attempts * 100):.2f}%" if parse_attempts > 0 else "0.00%"
        md.append(f"\n- **Judge 模型 JSON 解析统计**: 总尝试 {parse_attempts} 次, 失败 {parse_failures} 次, 解析失败率 {parse_rate}")

    # Removed Dimension Statistics as requested

    md.append("\n ## 各规则详情 (Per-Rule Statistics)")
    md.append("| Rule ID | Rule Name | Triggered | Average Score |")
    md.append("|---|---|---|---|")
    
    # Data for bar chart
    chart_labels = []
    chart_scores = []
    
    # Sort by ID if integer, else string
    try:
        sorted_rules = sorted(rule_stats.items(), key=lambda x: int(x[0]))
    except:
        sorted_rules = sorted(rule_stats.items(), key=lambda x: str(x[0]))
        
    for rid, stats in sorted_rules:
        if stats['triggered'] == 0:
            avg_score_str = "N/A"
            avg_score = 0
        else:
            avg_score = sum(stats['normalized_scores']) / len(stats['normalized_scores'])
            avg_score_str = f"{avg_score:.4f}"
            
        md.append(f"| {rid} | {stats['name']} | {stats['triggered']} | {avg_score_str} |")
        
        # Add to chart data if it was triggered
        if stats['triggered'] > 0:
            # Use a short version of the name or ID+name for the chart
            label = f"[{rid}] {stats['name']}"
            if len(label) > 15:
                label = label[:12] + "..."
            chart_labels.append(label)
            chart_scores.append(avg_score)

    # Generate Bar Chart
    if chart_labels:
        plt.figure(figsize=(12, 6))
        
        # Support Chinese characters in matplotlib
        import matplotlib.font_manager as fm
        
        # Try to find a valid Chinese font
        font_path = None
        # Priority list of fonts to check
        possible_fonts = [
            '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
            '/usr/share/fonts/google-droid/DroidSansFallback.ttf',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        ]
        
        # Also check matplotlib's font manager for system fonts by name if file path not found
        system_font_names = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'PingFang SC']
        
        for p in possible_fonts:
            if os.path.exists(p):
                font_path = p
                break
        
        my_font = None
        if font_path:
            # Load font from file path
            my_font = fm.FontProperties(fname=font_path)
            fm.fontManager.addfont(font_path)
            print(f"Loaded font from: {font_path}")
        else:
            # Fallback to system font name search
            print("Warning: Chinese font file not found. Trying system font names.")
            for name in system_font_names:
                try:
                    # Check if font name is valid
                    if fm.findfont(name):
                         # If found, just set family name, FontProperties will be created later or use default
                         # But better to find the path
                         path = fm.findfont(name)
                         if path:
                             my_font = fm.FontProperties(fname=path)
                             print(f"Found system font '{name}' at {path}")
                             break
                except:
                    continue

        if my_font:
             # Set font for all text elements explicitly
             # This is more robust than rcParams
             plt.rcParams['font.family'] = my_font.get_name() 
        else:
             print("Warning: No suitable Chinese font found. Charts may have garbled text.")
             # Fallback
             plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Droid Sans Fallback', 'SimHei', 'Arial Unicode MS']
            
        plt.rcParams['axes.unicode_minus'] = False
        
        y_pos = np.arange(len(chart_labels))
        bars = plt.bar(y_pos, chart_scores, align='center', alpha=0.7, color='skyblue')
        
        # Apply font to xticks
        plt.xticks(y_pos, chart_labels, rotation=45, ha='right', fontproperties=my_font)
        
        # Apply font to labels and title
        plt.ylabel('Average Score (Normalized)', fontproperties=my_font)
        plt.title('各场景平均得分 (Average Score per Scenario)', fontproperties=my_font)
        
        # Add values on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=9, fontproperties=my_font)
            
        plt.ylim(min(0, min(chart_scores) - 0.2) if chart_scores else 0, max(1.1, max(chart_scores) + 0.2) if chart_scores else 1.1)
        plt.tight_layout()
        
        chart_filename = "scenario_scores_chart.png"
        chart_path = os.path.join(output_dir, chart_filename)
        plt.savefig(chart_path, dpi=300)
        plt.close()

        
        md.append("\n## 各场景平均得分可视化")
        md.append(f"![各场景平均得分]({chart_filename})")

    content = "\n".join(md)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Report saved to {report_path}")

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="LeadBench Session Evaluation Pipeline")
    args = parser.parse_args()

    # Load version info
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'version'), 'r') as f:
            version = f.read().strip()
    except:
        version = "unknown"

    print(f"Initializing Judge Model: {config.JUDGE_MODEL_NAME}...")
    judge_model = APIModel(
        model_name=config.JUDGE_MODEL_NAME,
        api_key=config.JUDGE_API_KEY,
        api_base=config.JUDGE_API_BASE,
        temperature=0.01 
    )
    
    print(f"Initializing Candidate Model: {config.CANDIDATE_MODEL_NAME}...")
    candidate_model = APIModel(
        model_name=config.CANDIDATE_MODEL_NAME,
        api_key=config.CANDIDATE_API_KEY,
        api_base=config.CANDIDATE_API_BASE,
        temperature=0.7 
    )
    
    # Initialize Generator for Candidate Model
    print(f"Configuring processor for model: {config.CANDIDATE_MODEL_NAME}...")
    processor = get_processor_for_model(config.CANDIDATE_MODEL_NAME)
    candidate_generator = ResponseGenerator(
        model=candidate_model,
        processor=processor,
        max_tokens=config.CANDIDATE_MAX_OUTPUT_TOKENS,
    )
    
    print(f"Initializing User Simulator Model: {config.USER_SIMULATOR_MODEL_NAME}...")
    user_simulator_model = APIModel(
        model_name=config.USER_SIMULATOR_MODEL_NAME,
        api_key=config.USER_SIMULATOR_API_KEY,
        api_base=config.USER_SIMULATOR_API_BASE,
        temperature=0.7 
    )
    
    evaluator = SessionEvaluator(
        model=judge_model,
        enable_thinking=config.JUDGE_ENABLE_THINKING
    )

    # Initialize Dataset
    print(f"Loading dataset from {config.INPUT_FILE}...")
    print(f"Using rules from {config.RULES_FILE}...")
    try:
        dataset = DialogueEvaluationDataset(
            file_path=config.INPUT_FILE,
            rules_file=config.RULES_FILE
        )
        print(f"Loaded {len(dataset)} samples.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    # Determine subset
    indices = range(len(dataset))
    if config.EVALUATION_LIMIT:
        print(f"Limiting evaluation to first {config.EVALUATION_LIMIT} samples.")
        indices = range(min(len(dataset), config.EVALUATION_LIMIT))

    # Run Evaluation
    results = []
    print(f"Starting evaluation with concurrency {config.CONCURRENCY}...")
    
    # Generate Output Directory and Filename
    # Format: output/v<version>_<candidate_model>_<user_sim_model>_<judge_model>_<input_filename_stem>_<timestamp>/evaluation_results.jsonl
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    judge_model_safe = config.JUDGE_MODEL_NAME.replace('/', '_').replace(':', '_')
    candidate_model_safe = config.CANDIDATE_MODEL_NAME.replace('/', '_').replace(':', '_')
    sim_model_safe = config.USER_SIMULATOR_MODEL_NAME.replace('/', '_').replace(':', '_')
    input_stem = Path(config.INPUT_FILE).stem
    
    # Use version read at the beginning of main()
    output_dir_name = f"v{version}_cand_{candidate_model_safe}_sim_{sim_model_safe}_judge_{judge_model_safe}_{input_stem}_{timestamp}"
    
    # Base output directory
    if config.OUTPUT_FILE:
        base_output_dir = os.path.dirname(config.OUTPUT_FILE)
    else:
        # Default output directory relative to project root
        base_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
        
    final_output_dir = os.path.join(base_output_dir, output_dir_name)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    
    output_filename = "evaluation_results.jsonl"
    output_path = os.path.join(final_output_dir, output_filename)
    
    print(f"Results will be saved to: {output_path}")

    # Use ThreadPoolExecutor for concurrency
    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as executor:
        # Submit all tasks
        futures = {executor.submit(process_sample_session, dataset[i], evaluator, candidate_generator, user_simulator_model): i for i in indices}
        
        # Use tqdm for the main progress bar (samples)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Samples"):
            result = future.result()
            if result:
                results.append(result)

    for sample in results:
        eval_res = sample.get('evaluation', {})
        eval_res['judge_model'] = config.JUDGE_MODEL_NAME
        eval_res['candidate_model'] = config.CANDIDATE_MODEL_NAME
        eval_res['user_simulator_model'] = config.USER_SIMULATOR_MODEL_NAME
        # Ensure it's placed before rule_details
        if 'rule_details' in eval_res:
            rule_details = eval_res.pop('rule_details')
            eval_res['rule_details'] = rule_details
        sample['evaluation'] = eval_res

    # Save Results
    with open(output_path, 'w', encoding='utf-8') as f:
        for res in results:
            # Create a new dict with the desired order
            ordered_res = {}
            for k, v in res.items():
                ordered_res[k] = v
                if k == 'id':
                    # Add rl_score right after id
                    ordered_res['rl_score'] = res.get('evaluation', {}).get('rl_score', 0)
            f.write(json.dumps(ordered_res, ensure_ascii=False) + '\n')
    print(f"Evaluation complete. Results saved to {output_path}")
    
    # Save Config
    config_output_path = os.path.join(final_output_dir, "config.json")
    config_data = {
        "JUDGE_MODEL_NAME": config.JUDGE_MODEL_NAME,
        "CANDIDATE_MODEL_NAME": config.CANDIDATE_MODEL_NAME,
        "USER_SIMULATOR_MODEL_NAME": config.USER_SIMULATOR_MODEL_NAME,
        "INPUT_FILE": config.INPUT_FILE,
        "RULES_FILE": config.RULES_FILE,
        "EVALUATION_LIMIT": config.EVALUATION_LIMIT,
        "CONCURRENCY": config.CONCURRENCY,
        "JUDGE_ENABLE_THINKING": config.JUDGE_ENABLE_THINKING,
        "JUDGE_API_BASE": config.JUDGE_API_BASE,
        "CANDIDATE_API_BASE": config.CANDIDATE_API_BASE,
        "USER_SIMULATOR_API_BASE": config.USER_SIMULATOR_API_BASE
    }
    with open(config_output_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)
    print(f"Configuration saved to {config_output_path}")
    
    # Save Rule-based JSONL Output
    try:
        from collections import defaultdict
        rule_groups = defaultdict(lambda: {"rule_id": 0, "rule_name_cn": "", "rule_name": "", "evaluations": []})
        for res in results:
            dialogue_id = res.get('id')
            messages = res.get('messages', [])
            evaluation = res.get('evaluation', {})
            rule_details = evaluation.get('rule_details', [])
            
            for rule in rule_details:
                if not rule.get('is_triggered', False):
                    continue
                    
                rule_id = rule.get('rule_id')
                if rule_groups[rule_id]["rule_id"] == 0:
                    rule_groups[rule_id]["rule_id"] = rule_id
                    rule_groups[rule_id]["rule_name_cn"] = rule.get('rule_name_cn')
                    rule_groups[rule_id]["rule_name"] = rule.get('rule_name')
                    
                record = {
                    "id": dialogue_id,
                    "normalized_score": rule.get('normalized_score', 0),
                    "score_change": rule.get('score_change', 0),
                    "trigger_analysis": rule.get('trigger_analysis'),
                    "evaluation_analysis": rule.get('evaluation_analysis'),
                    "strategy_evaluations": rule.get('strategy_evaluations', []),
                    "messages": messages
                }
                rule_groups[rule_id]["evaluations"].append(record)

        sorted_rules = sorted(rule_groups.values(), key=lambda x: x['rule_id'])
        for rule_group in sorted_rules:
            rule_group['evaluations'].sort(key=lambda x: x['normalized_score'], reverse=True)
            
        rule_output_filename = "evaluation_results_by_rule_sorted.jsonl"
        rule_output_path = os.path.join(final_output_dir, rule_output_filename)
        with open(rule_output_path, 'w', encoding='utf-8') as f:
            for rule_group in sorted_rules:
                f.write(json.dumps(rule_group, ensure_ascii=False) + '\n')
        print(f"Rule-based evaluation results saved to {rule_output_path}")
    except Exception as e:
        print(f"Failed to generate rule-based results: {e}")
    
    # Copy Input File and Rules File to output directory for traceability
    try:
        if config.INPUT_FILE and os.path.exists(config.INPUT_FILE):
            shutil.copy(config.INPUT_FILE, final_output_dir)
            print(f"Copied input file {config.INPUT_FILE} to {final_output_dir}")
        if config.RULES_FILE and os.path.exists(config.RULES_FILE):
            shutil.copy(config.RULES_FILE, final_output_dir)
            print(f"Copied rules file {config.RULES_FILE} to {final_output_dir}")
    except Exception as e:
        print(f"Failed to copy input/rules files: {e}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Collect statistics
    stats = {
        'candidate_model': {
            'name': config.CANDIDATE_MODEL_NAME,
            'api_calls': candidate_model.api_calls,
            'api_failures': candidate_model.api_failures
        },
        'user_simulator_model': {
            'name': config.USER_SIMULATOR_MODEL_NAME,
            'api_calls': user_simulator_model.api_calls,
            'api_failures': user_simulator_model.api_failures
        },
        'judge_model': {
            'name': config.JUDGE_MODEL_NAME,
            'api_calls': judge_model.api_calls,
            'api_failures': judge_model.api_failures,
            'json_parse_attempts': evaluator.json_parse_attempts,
            'json_parse_failures': evaluator.json_parse_failures
        }
    }
    
    # Generate Report
    generate_report(results, final_output_dir, execution_time, stats)
    
    # Calculate Bench Scores Summary
    if results:
        rl_scores = [r['evaluation']['rl_score'] for r in results]
        bench_scores = [r['evaluation']['bench_score'] for r in results]
        
        avg_rl = sum(rl_scores) / len(rl_scores) if rl_scores else 0
        avg_bench = sum(bench_scores) / len(bench_scores) if bench_scores else 0
        
        print("\n=== Evaluation Summary ===")
        print(f"Total Samples Evaluated: {len(results)}")
        print(f"Average RL Score (Base 100): {avg_rl:.2f}")
        print(f"Average Bench Score (0-1): {avg_bench:.4f}")
        print(f"Total Execution Time: {execution_time:.2f}s")

if __name__ == "__main__":
    main()
