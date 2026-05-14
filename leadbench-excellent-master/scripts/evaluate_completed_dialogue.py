import argparse
import json
import os
import sys
import time
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

def process_sample_session(sample, evaluator):
    """
    Process a single sample (completed dialogue) through session evaluation.
    """
    try:
        return evaluator.evaluate_session(sample)
    except Exception as e:
        print(f"Error processing sample ID {sample.get('id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_report(results, output_dir, execution_time_seconds):
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
    # Format: output/v<version>_<model_name>_<input_filename_stem>_<timestamp>/evaluation_results.jsonl
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = config.JUDGE_MODEL_NAME.replace('/', '_').replace(':', '_')
    input_stem = Path(config.INPUT_FILE).stem
    
    # Use version read at the beginning of main()
    output_dir_name = f"v{version}_{model_name_safe}_{input_stem}_{timestamp}"
    
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
        futures = {executor.submit(process_sample_session, dataset[i], evaluator): i for i in indices}
        
        # Use tqdm for the main progress bar (samples)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Samples"):
            result = future.result()
            if result:
                results.append(result)

    # Save Results
    with open(output_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    print(f"Evaluation complete. Results saved to {output_path}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Generate Report
    generate_report(results, final_output_dir, execution_time)
    
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
