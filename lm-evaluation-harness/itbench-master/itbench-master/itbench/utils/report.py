from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from matplotlib.font_manager import FontProperties
from itbench.utils.config import config

class ReportGenerator:
    def __init__(self, output_dir, rules_file=None, version="unknown"):
        self.output_dir = output_dir
        self.version = version
        # rule_stats: {rule_id: {'triggered': 0, 'passed': 0, 'name': '...', 'dimension': '...'}}
        self.rule_stats = defaultdict(lambda: {'triggered': 0, 'passed': 0, 'name': '', 'dimension': 'Unknown'})
        
        # Pre-load rule names if rules file is provided
        if rules_file and os.path.exists(rules_file):
            try:
                with open(rules_file, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                    for rule in rules:
                        rule_id = rule.get('rule_id')
                        if rule_id is not None:
                            self.rule_stats[rule_id]['name'] = rule.get('rule_name_cn', rule.get('rule_name', f'Rule {rule_id}'))
                            self.rule_stats[rule_id]['dimension'] = rule.get('rule_dimension', 'Unknown')
            except Exception as e:
                print(f"Warning: Failed to load rules file: {e}")

        self.total_candidates = 0
        self.total_triggered_checks = 0
        self.total_passed_checks = 0
        self.failed_cases = [] # Store triggered but not passed cases
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        # New metrics
        self.total_score = 0
        self.perfect_samples = 0
        self.hard_passed_samples = 0
        self.total_judge_calls = 0
        self.total_failed_parse_attempts = 0

    def add_result(self, result):
        if not result or 'evaluation' not in result:
            return

        self.total_candidates += 1
        
        evaluation = result['evaluation']
        if 'usage' in evaluation:
            self.total_prompt_tokens += evaluation['usage'].get('prompt_tokens', 0)
            self.total_completion_tokens += evaluation['usage'].get('completion_tokens', 0)
            
        self.total_judge_calls += evaluation.get('total_judge_calls', 0)
        self.total_failed_parse_attempts += evaluation.get('failed_parse_attempts', 0)

        # Calculate score metrics
        final_score = evaluation.get('final_score', 0)
        self.total_score += final_score
        if final_score == 100:
            self.perfect_samples += 1

        if 'rule_details' not in evaluation:
            # If no rules checked, assume hard pass (no violations)
            self.hard_passed_samples += 1
            return
            
        is_hard_fail = False
        for rule_res in evaluation['rule_details']:
            rule_id = rule_res['rule_id']
            
            # Check for hard failure condition: not passed AND deduction > 1
            if not rule_res.get('is_passed', False):
                deduction = rule_res.get('deduction', 0)
                if abs(deduction) > 1:
                    is_hard_fail = True
            
            # Update rule metadata if missing
            if not self.rule_stats[rule_id]['name']:
                self.rule_stats[rule_id]['name'] = rule_res.get('rule_name_cn', rule_res.get('rule_name', 'Unknown'))
            
            # Update dimension if still Unknown and available in result
            if self.rule_stats[rule_id]['dimension'] == 'Unknown':
                self.rule_stats[rule_id]['dimension'] = rule_res.get('rule_dimension', 'Unknown')

            if rule_res['is_triggered']:
                self.rule_stats[rule_id]['triggered'] += 1
                self.total_triggered_checks += 1
                if rule_res['is_passed']:
                    self.rule_stats[rule_id]['passed'] += 1
                    self.total_passed_checks += 1
                else:
                    # Record failed case
                    failed_case = {
                        "id": result.get('id'),
                        "rule_id": rule_id,
                        "rule_name": self.rule_stats[rule_id]['name'],
                        "dimension": self.rule_stats[rule_id]['dimension'],
                        "messages": result.get('messages', []),
                        "response": result.get('response'),
                        "history_str": result.get('history_str', ''),
                        "reason": rule_res.get('reason', 'No reason provided'),
                        "trigger_analysis": rule_res.get('trigger_analysis', ''),
                        "extract_analysis": rule_res.get('extract_analysis', ''),
                        "compliance_analysis": rule_res.get('compliance_analysis', ''),
                        "trigger_prompt": rule_res.get('trigger_prompt', ''),
                        "extract_prompt": rule_res.get('extract_prompt', ''),
                        "compliance_prompt": rule_res.get('compliance_prompt', ''),
                        "dynamic_description": rule_res.get('dynamic_description', ''),
                        "trigger_thinking": rule_res.get('trigger_thinking', ''),
                        "extract_thinking": rule_res.get('extract_thinking', ''),
                        "compliance_thinking": rule_res.get('compliance_thinking', ''),
                        "judge_model": evaluation.get('judge_model', ''),
                        "candidate_model": evaluation.get('candidate_model', '')
                    }
                    self.failed_cases.append(failed_case)
        
        if not is_hard_fail:
            self.hard_passed_samples += 1

    def generate_report(self, total_duration_seconds=None):
        print("Generating statistical report...")
        
        # Calculate dimension stats
        dim_stats = defaultdict(lambda: {'triggered': 0, 'passed': 0, 'pass_rates': []})
        
        # Prepare data for plotting
        rule_ids = []
        pass_rates = []
        rule_names = []
        
        # Calculate per-rule pass rates first
        valid_rules_count = 0
        total_pass_rate_sum = 0.0

        sorted_rules = sorted(self.rule_stats.items(), key=lambda x: x[0])
        
        table_lines = [
            "\n## 各规则详情 (Per-Rule Statistics)",
            "| Rule ID | Rule Name | Dimension | Triggered | Passed | Pass Rate |",
            "|---|---|---|---|---|---|"
        ]

        for rule_id, stats in sorted_rules:
            triggered = stats['triggered']
            passed = stats['passed']
            dimension = stats['dimension']
            
            rate = (passed / triggered * 100) if triggered > 0 else 0.0
            
            table_lines.append(f"| {rule_id} | {stats['name']} | {dimension} | {triggered} | {passed} | {rate:.2f}% |")
            
            if triggered > 0:
                rule_ids.append(f"ID {rule_id}")
                pass_rates.append(rate)
                rule_names.append(stats['name'])
                
                # Accumulate for average calculation
                total_pass_rate_sum += rate
                valid_rules_count += 1
                
                # Update dimension stats
                dim_stats[dimension]['triggered'] += triggered
                dim_stats[dimension]['passed'] += passed
                dim_stats[dimension]['pass_rates'].append(rate)
        
        # Calculate overall pass rate as the average of individual rule pass rates
        overall_pass_rate = (total_pass_rate_sum / valid_rules_count) if valid_rules_count > 0 else 0.0
        
        # Calculate new metrics
        avg_score = self.total_score / self.total_candidates if self.total_candidates > 0 else 0
        sample_pass_rate = (self.perfect_samples / self.total_candidates * 100) if self.total_candidates > 0 else 0
        hard_pass_rate = (self.hard_passed_samples / self.total_candidates * 100) if self.total_candidates > 0 else 0
        
        # Calculate token usage stats
        avg_prompt_tokens = self.total_prompt_tokens / self.total_candidates if self.total_candidates > 0 else 0
        avg_completion_tokens = self.total_completion_tokens / self.total_candidates if self.total_candidates > 0 else 0
        avg_total_tokens = avg_prompt_tokens + avg_completion_tokens

        report_lines = [
            f"# 评估统计报告 (Evaluation Report) - LeadBench v{self.version}",
            f"- **Candidate Model (被测模型)**: {config.CANDIDATE_MODEL_NAME}",
            f"- **Judge Model (评判模型)**: {config.JUDGE_MODEL_NAME}",
            f"- **总候选回复数 (Total Candidates)**: {self.total_candidates}",
            f"- **总触发规则次数 (Total Triggered Checks)**: {self.total_triggered_checks}",
            f"- **总通过次数 (Total Passed Checks)**: {self.total_passed_checks}",
            f"- **整体通过率 (Average of Rules Pass Rate)**: {overall_pass_rate:.2f}%",
            f"- **Average Score (平均分)**: {avg_score:.2f}",
            f"- **Hard Pass Rate (硬性通过率)**: {hard_pass_rate:.2f}%",
            f"- **Sample Pass Rate (完美率)**: {sample_pass_rate:.2f}%",
            f"- **平均Token消耗 (Average Tokens per Sample)**: Input: {avg_prompt_tokens:.2f}, Output: {avg_completion_tokens:.2f}, Total: {avg_total_tokens:.2f}",
            f"- **调用Judge总次数 (Total Judge Calls)**: {self.total_judge_calls}",
            f"- **JSON解析失败次数 (Failed Parse Attempts)**: {self.total_failed_parse_attempts}",
        ]

        # Add time statistics if available
        if total_duration_seconds is not None:
            avg_duration = total_duration_seconds / self.total_candidates if self.total_candidates > 0 else 0.0
            hours, remainder = divmod(total_duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            report_lines.append(f"- **总评估耗时 (Total Evaluation Time)**: {time_str} ({total_duration_seconds:.2f}s)")
            report_lines.append(f"- **平均单条评估耗时 (Average Time per Sample)**: {avg_duration:.2f}s")


        # Add Dimension Statistics Section
        dim_table_lines = [
            "\n## 维度统计 (Dimension Statistics)",
            "| Dimension | Triggered | Passed | Macro Pass Rate | Micro Pass Rate |",
            "|---|---|---|---|---|"
        ]
        
        dim_names = []
        dim_macro_rates = []

        for dim, stats in sorted(dim_stats.items()):
            dim_triggered = stats['triggered']
            dim_passed = stats['passed']
            dim_micro_rate = (dim_passed / dim_triggered * 100) if dim_triggered > 0 else 0.0
            
            # Macro Average: Average of pass rates of rules within this dimension
            dim_rule_rates = stats['pass_rates']
            dim_macro_rate = (sum(dim_rule_rates) / len(dim_rule_rates)) if dim_rule_rates else 0.0
            
            dim_table_lines.append(f"| {dim} | {dim_triggered} | {dim_passed} | {dim_macro_rate:.2f}% | {dim_micro_rate:.2f}% |")
            
            dim_names.append(dim)
            dim_macro_rates.append(dim_macro_rate)

        report_lines.extend(dim_table_lines)
        report_lines.extend(table_lines)

        # Save Markdown Report
        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"Report saved to {report_path}")

        # Save Failed Cases JSONL
        failed_cases_path = os.path.join(self.output_dir, "failed_cases.jsonl")
        
        # Group by rule_id
        grouped_cases = {}
        
        for case in self.failed_cases:
            rid = case['rule_id']
            if rid not in grouped_cases:
                grouped_cases[rid] = {
                    "rule_id": rid,
                    "rule_name": case['rule_name'],
                    "dimension": case['dimension'],
                    "failed_cases": []
                }
            
            # Remove redundant fields from individual case
            case_detail = {k: v for k, v in case.items() if k not in ['rule_id', 'rule_name', 'dimension']}
            grouped_cases[rid]['failed_cases'].append(case_detail)
            
        with open(failed_cases_path, 'w', encoding='utf-8') as f:
            for rid in sorted(grouped_cases.keys()):
                json.dump(grouped_cases[rid], f, ensure_ascii=False)
                f.write('\n')
        print(f"Failed cases saved to {failed_cases_path}")

        # Generate Chart
        self._plot_chart(rule_ids, pass_rates, rule_names, dim_names, dim_macro_rates)
        
        # Add report path to return value
        return report_path

    def _plot_chart(self, rule_ids, pass_rates, rule_names, dim_names=None, dim_macro_rates=None):
        if not rule_ids:
            return

        # Use absolute path to the confirmed available font
        font_path = '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc'
        font_prop = None
        if os.path.exists(font_path):
            font_prop = FontProperties(fname=font_path)
        else:
            # Fallback for local environments or different paths
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False

        # If dimension data is available, plot two subplots
        if dim_names and dim_macro_rates:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Subplot 1: Dimension Pass Rates
            bars1 = ax1.bar(range(len(dim_names)), dim_macro_rates, color='lightgreen')
            if font_prop:
                ax1.set_xlabel('Dimension (维度)', fontproperties=font_prop)
                ax1.set_ylabel('Pass Rate (%)', fontproperties=font_prop)
                ax1.set_title('Dimension Pass Rates (维度通过率)', fontproperties=font_prop)
                ax1.set_xticks(range(len(dim_names)))
                ax1.set_xticklabels(dim_names, rotation=0, fontproperties=font_prop)
            else:
                ax1.set_xlabel('Dimension (维度)')
                ax1.set_ylabel('Pass Rate (%)')
                ax1.set_title('Dimension Pass Rates (维度通过率)')
                ax1.set_xticks(range(len(dim_names)))
                ax1.set_xticklabels(dim_names, rotation=0)
            
            ax1.set_ylim(0, 100)
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

            # Subplot 2: Rule Pass Rates (same as before)
            x_labels = [f"{rid} {name}" for rid, name in zip(rule_ids, rule_names)]
            bars2 = ax2.bar(range(len(x_labels)), pass_rates, color='skyblue')
            
            if font_prop:
                ax2.set_xlabel('Rule (规则)', fontproperties=font_prop)
                ax2.set_ylabel('Pass Rate (%)', fontproperties=font_prop)
                ax2.set_title('Rule Pass Rates (规则通过率)', fontproperties=font_prop)
                ax2.set_xticks(range(len(x_labels)))
                ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontproperties=font_prop)
            else:
                ax2.set_xlabel('Rule (规则)')
                ax2.set_ylabel('Pass Rate (%)')
                ax2.set_title('Rule Pass Rates (规则通过率)')
                ax2.set_xticks(range(len(x_labels)))
                ax2.set_xticklabels(x_labels, rotation=45, ha='right')
            
            ax2.set_ylim(0, 100)
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        else:
            # Fallback to single plot if no dimension data
            plt.figure(figsize=(12, 6)) 
            # ... (existing code for single plot could go here, but for simplicity we assume dimensions are always calculated now)
            # Re-implementing single plot logic just in case
            x_labels = [f"{rid} {name}" for rid, name in zip(rule_ids, rule_names)]
            bars = plt.bar(range(len(x_labels)), pass_rates, color='skyblue')
            
            if font_prop:
                plt.xlabel('Rule (规则)', fontproperties=font_prop)
                plt.ylabel('Pass Rate (%)', fontproperties=font_prop)
                plt.title('Rule Pass Rates (规则通过率)', fontproperties=font_prop)
                plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right', fontproperties=font_prop)
            else:
                plt.xlabel('Rule (规则)')
                plt.ylabel('Pass Rate (%)')
                plt.title('Rule Pass Rates (规则通过率)')
                plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
            
            plt.ylim(0, 100)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, "rule_pass_rates.png")
        plt.savefig(chart_path)
        print(f"Chart saved to {chart_path}")
