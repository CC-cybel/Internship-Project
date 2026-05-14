import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.font_manager import FontProperties

def parse_md_table(filepath, start_line, end_line):
    rule_names = []
    pass_rates = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines[start_line-1:end_line]:
        line = line.strip()
        if line.startswith('|') and not line.startswith('| Rule ID') and not line.startswith('|---'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 6:
                rule_id = parts[0]
                rule_name = parts[1]
                pass_rate_str = parts[5].replace('%', '')
                try:
                    pass_rate = float(pass_rate_str)
                    rule_names.append(f"{rule_id}. {rule_name}")
                    pass_rates.append(pass_rate)
                except ValueError:
                    pass
                    
    return rule_names, pass_rates

def main():
    file1 = "/data1/yezj/gitlab/ITBench/output/v0.1_anthropic_claude-sonnet-4.5_golden_history_input_20260414_204916/evaluation_report.md"
    file2 = "/data1/yezj/gitlab/ITBench/output/v0.1_normal_anti_hijack_abc_stage2_golden_history_input_20260414_163839/evaluation_report.md"
    
    # 25 to 43 lines (1-indexed) are 24:43 in slice, but our parse function uses 1-indexed start/end
    names1, rates1 = parse_md_table(file1, 25, 43)
    names2, rates2 = parse_md_table(file2, 25, 43)
    
    if names1 != names2:
        print("Warning: Rule names do not match perfectly. Using the first list.")
        
    x = np.arange(len(names1))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Check for Chinese font
    font_path = '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc'
    font_prop = None
    if os.path.exists(font_path):
        font_prop = FontProperties(fname=font_path)
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
    rects1 = ax.bar(x - width/2, rates1, width, label='Claude 3.5 Sonnet', color='#4C72B0')
    rects2 = ax.bar(x + width/2, rates2, width, label='Normal Anti-Hijack ABC Stage2', color='#55A868')
    
    if font_prop:
        ax.set_ylabel('Pass Rate (%)', fontproperties=font_prop, fontsize=12)
        ax.set_title('Model Comparison: Per-Rule Pass Rates', fontproperties=font_prop, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(names1, rotation=45, ha='right', fontproperties=font_prop, fontsize=10)
    else:
        ax.set_ylabel('Pass Rate (%)', fontsize=12)
        ax.set_title('Model Comparison: Per-Rule Pass Rates', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(names1, rotation=45, ha='right', fontsize=10)
        
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0)
    ax.set_ylim(0, 110) # Give some space for labels on top of bars
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    output_path = "/data1/yezj/gitlab/ITBench/output/model_comparison_chart.png"
    plt.savefig(output_path, dpi=300)
    print(f"Chart successfully saved to {output_path}")

if __name__ == "__main__":
    main()