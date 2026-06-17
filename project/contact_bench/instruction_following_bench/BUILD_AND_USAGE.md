# Instruction Following Bench 构建与使用说明

本文档总结 `/data/chengch/project/contact_bench/instruction_following_bench` 的构建思路、数据来源、评测流程、运行方式和结果解读方式。

## 1. 核心目标

这个 benchmark 的目标不是评估普通医疗咨询回复质量，而是评估模型在真实多轮咨询上下文中，对额外注入 system 指令的遵循能力。

每条样本都包含：

- 原始医疗咨询 system prompt
- 多轮对话历史
- 当前待回复位置
- 一组额外注入的指令原子 `selected_additional_instructions`
- candidate model 生成的回复
- judge model 对每条注入指令的逐项评分

核心评测链路是：

```text
bench 样本 -> candidate model 生成回复 -> judge model 逐条指令评分 -> 聚合报告与可视化
```

因此正式评测时必须调用 candidate model 生成回复，不能只拿已有 reference 答案打分。`USE_REFERENCE=1` 只用于 judge sanity check。

## 2. Bench 数据构建思路

最终使用的数据文件是：

```text
data/dataset/instruction_following_bench_final300.jsonl
```

它来源于 instruction experiment 下筛选出的 300 条高质量指令遵循 bench 样本：

```text
/data/chengch/project/data_remake/outputs/instruction_exp/bench_instruction_eval_seed20260611_simple/bench_simple_adapted_system_candidates.seed20260611.final300.jsonl
```

构建过程的大体思路如下。

### 2.1 原子来源

指令原子由三类组成：

- `legacy_quality_atom`：历史高质量业务/格式/记忆类指令，偏真实业务约束
- `paraphrase_atom`：对已有指令语义做改写，用于评估同义泛化能力
- `novel_atom`：新奇、OOD、形式化或风格化指令，用于拉开模型指令遵循能力差异

早期尝试过单原子、两两组合、三三组合等更复杂构造，但最终回归更稳定的 rough group 逻辑。

### 2.2 Rough group 组合

简单版 rough group 的设计是每组 5 个原子：

```text
1 个 novel 原子
2 个 paraphrase_and_novel_atoms.seed20260609.jsonl 中的原子
2 个 legacy_quality_atoms.seed20260611.jsonl 中的原子
```

这样既保留真实业务指令，又保留泛化和 OOD 挑战。

相关构建脚本：

```text
/data/chengch/project/data_remake/outputs/instruction_exp/scripts/build_simple_bench_rough_groups.py
/data/chengch/project/data_remake/outputs/instruction_exp/scripts/adapt_simple_bench_instruction_groups.py
```

### 2.3 API 选择可注入指令

每条候选对话会给多个 rough group，让 LLM 根据当前对话上下文选择适合注入的一组，并从组内筛选真正适合当前样本的指令。

选择原则：

- novel/paraphrase 是 benchmark 挑战目标，不因为“不自然”就轻易拒绝
- 只有在硬冲突、医疗安全风险、明显无法执行时才拒绝 novel/paraphrase
- legacy/business 类指令需要有当前对话证据，不能只因为“不冲突”就强行注入
- stable style / surface 类指令可以弱触发，只要可检查且不冲突即可

### 2.4 最终 300 条筛选

中间曾得到 500 条重平衡数据，分布为：

```text
1 条指令: 100
2 条指令: 200
3 条指令: 200
```

最终从中筛出约 300 条高质量数据，同时保持指令多样性，最终分布为：

```text
1 条指令: 60
2 条指令: 120
3 条指令: 120
```

筛选时优先保留：

- judge confidence 较高的样本
- 指令和对话上下文匹配度高的样本
- atom 覆盖更均匀的样本
- novel/paraphrase/business 指令都有覆盖的样本

最终数据复制到当前 bench 项目中，作为独立 benchmark 数据集。

## 3. 项目结构

```text
instruction_following_bench/
├── data/
│   └── dataset/
│       └── instruction_following_bench_final300.jsonl
├── instructionbench/
│   └── evaluation/
├── output/
│   ├── claude/
│   ├── qwen3_8b_sft_v2/
│   ├── qwen3_8b_sft_v2_instruction_full300/
│   └── comparison_*/
├── scripts/
│   ├── evaluate_instruction_following.py
│   ├── visualize_instruction_following.py
│   └── compare_instruction_following_runs.py
├── env.example
├── README.md
├── requirements.txt
├── run_eval.sh
└── BUILD_AND_USAGE.md
```

核心文件：

- `run_eval.sh`：主入口，负责读取环境变量、预检 candidate API、调用 evaluator
- `scripts/evaluate_instruction_following.py`：candidate 生成、judge 评分、报告聚合
- `scripts/visualize_instruction_following.py`：单个 run 的可视化
- `scripts/compare_instruction_following_runs.py`：多个 run 的横向对比可视化

## 4. 评测流程

默认正式流程：

```text
1. 读取 instruction_following_bench_final300.jsonl
2. 对每条样本构造 OpenAI-compatible chat messages
3. 调用 candidate model 生成回复
4. 从回复中解析 BEGIN_META / BEGIN_FINAL
5. 调用 judge model 逐条评估 selected_additional_instructions
6. 每条指令输出 0/1/2 分
7. 将每条指令分数归一化为 score / 2
8. 对单条样本做加权平均
9. 聚合 overall、1/2/3 指令桶、source、axis、atom、failure_type 等报告
```

注意：正式模型能力评测时，`RESPONSE_FIELD` 为空，脚本会在线调用 candidate model。只有离线复评已有模型输出时才设置 `RESPONSE_FIELD`。

## 5. Judge 评分标准

每个注入指令单独评分：

```text
0 = 完全不遵循、明显违反、遗漏必须动作、内容相反、无法验证
1 = 基本遵循但不完整，或只有弱触发/弱表达
2 = 完全遵循，或者禁止类约束没有被违反
```

样本分数计算：

```text
instruction_score_normalized = raw_score / 2
sample_score = weighted_mean(instruction_score_normalized)
```

所以报告中的 `mean_score` 满分是 `1.0`，不是 `2.0`。

目前权重逻辑较轻：

- 默认指令权重为 `1.0`
- 涉及危机、自杀、监护人、联系方式、留联、安全等业务关键行为时权重为 `1.2`

## 6. 运行方式

### 6.1 准备 candidate 服务

默认 `run_eval.sh` 使用：

```bash
CANDIDATE_MODEL_NAME=qwen3_8b_sft_v2
CANDIDATE_API_BASE=http://127.0.0.1:8001/v1
CANDIDATE_API_KEY=111
```

运行前需要确保 vLLM 或其他 OpenAI-compatible 服务已启动，并且：

```bash
curl -H "Authorization: Bearer 111" http://127.0.0.1:8001/v1/models
```

能返回目标模型。

### 6.2 小规模 smoke test

建议先跑 10 条：

```bash
EVALUATION_LIMIT=10 RUN_NAME=smoke10 \
bash /data/chengch/project/contact_bench/instruction_following_bench/run_eval.sh
```

### 6.3 全量 300 条评测

例如并发 10：

```bash
CONCURRENCY=10 RUN_NAME=qwen3_8b_sft_v2_full300 \
bash /data/chengch/project/contact_bench/instruction_following_bench/run_eval.sh
```

输出目录：

```text
output/<RUN_NAME>/
```

### 6.4 切换 candidate model

```bash
CANDIDATE_MODEL_NAME=your_model_name \
CANDIDATE_API_BASE=http://127.0.0.1:8002/v1 \
CANDIDATE_API_KEY=your_key \
CONCURRENCY=10 \
RUN_NAME=your_model_full300 \
bash /data/chengch/project/contact_bench/instruction_following_bench/run_eval.sh
```

### 6.5 Judge-only sanity check

只用于检查 judge 逻辑，不代表模型能力：

```bash
USE_REFERENCE=1 EVALUATION_LIMIT=10 RUN_NAME=reference_probe \
bash /data/chengch/project/contact_bench/instruction_following_bench/run_eval.sh
```

### 6.6 离线已有输出评测

如果已经有模型输出 JSONL，并且里面有字段 `model_output`：

```bash
INPUT_FILE=/path/to/model_outputs.jsonl \
RESPONSE_FIELD=model_output \
RUN_NAME=my_model_offline_eval \
bash /data/chengch/project/contact_bench/instruction_following_bench/run_eval.sh
```

## 7. 输出文件说明

每次 run 输出：

```text
config.json
input_snapshot.jsonl
judge_system_prompt.txt
evaluation_results.jsonl
failed_cases.jsonl
excluded_generation_errors.jsonl
evaluation_report.md
summary.json
```

含义：

- `config.json`：本次运行参数
- `input_snapshot.jsonl`：输入数据快照，方便复现
- `judge_system_prompt.txt`：judge 使用的系统提示词快照
- `evaluation_results.jsonl`：每条样本完整结果，包括 candidate response、usage、judge 明细
- `failed_cases.jsonl`：非满分样本和异常样本
- `excluded_generation_errors.jsonl`：candidate 生成失败样本
- `evaluation_report.md`：聚合报告
- `summary.json`：机器可读摘要

## 8. 分数解读

报告中的关键字段：

- `total_rows`：总样本数
- `judged_ok`：成功生成且成功 judge 的样本数
- `status_counts`：ok / generation_error / judge_error 等状态数量
- `mean_score`：只在 `ok` 样本上计算的平均分
- `median_score`：只在 `ok` 样本上计算的中位数
- `strict_all_2_pass_rate`：所有注入指令都得 2 分的样本比例
- `soft_all_ge1_pass_rate`：所有注入指令都至少得 1 分的样本比例
- `format_ok_rate`：回复格式可解析比例

需要特别注意：

```text
mean_score 默认不把 generation_error 算进去。
```

如果要更保守地把生成失败按 0 分计入，可使用：

```text
full_mean = sum(ok sample_score) / total_rows
```

可视化脚本和 comparison 脚本会同时报告：

```text
judged_only_mean_score
full_mean_score_generation_errors_as_zero
```

## 9. 可视化

### 9.1 单个 run 可视化

```bash
/data/wangpf/project/miniconda3/envs/lm-evaluation-harness/bin/python \
  /data/chengch/project/contact_bench/instruction_following_bench/scripts/visualize_instruction_following.py \
  /data/chengch/project/contact_bench/instruction_following_bench/output/qwen3_8b_sft_v2_instruction_full300
```

输出：

```text
<run_dir>/visualization/
  01_overview.png
  02_score_distribution.png
  03_by_instruction_count.png
  04_by_source.png
  05_by_axis.png
  06_failure_types.png
  07_lowest_atoms.png
  visualization_summary.md
```

### 9.2 多模型对比可视化

```bash
/data/wangpf/project/miniconda3/envs/lm-evaluation-harness/bin/python \
  /data/chengch/project/contact_bench/instruction_following_bench/scripts/compare_instruction_following_runs.py \
  --run claude /data/chengch/project/contact_bench/instruction_following_bench/output/claude \
  --run qwen3_sft_instruction /data/chengch/project/contact_bench/instruction_following_bench/output/qwen3_8b_sft_v2_instruction_full300 \
  --run qwen3_sft_v2 /data/chengch/project/contact_bench/instruction_following_bench/output/qwen3_8b_sft_v2 \
  --output-dir /data/chengch/project/contact_bench/instruction_following_bench/output/comparison_claude_vs_qwen3_instruction_vs_qwen3_sft_v2
```

输出：

```text
01_overall_comparison.png
02_status_counts.png
03_by_instruction_count.png
04_by_source.png
05_by_axis.png
06_failure_types.png
07_lowest_atoms_comparison.png
comparison_report.md
comparison_summary.json
```

## 10. 当前已有结果示例

截至当前，已有三个完整 run：

```text
output/claude
output/qwen3_8b_sft_v2_instruction_full300
output/qwen3_8b_sft_v2
```

三方对比结果摘要：

```text
claude
judged_mean_score: 0.5816
full_mean_score_generation_errors_as_zero: 0.5816
strict_all_2_pass_rate: 0.3167
soft_all_ge1_pass_rate: 0.3933

qwen3_sft_instruction
judged_mean_score: 0.6763
full_mean_score_generation_errors_as_zero: 0.6583
strict_all_2_pass_rate: 0.4281
soft_all_ge1_pass_rate: 0.4897

qwen3_sft_v2
judged_mean_score: 0.4217
full_mean_score_generation_errors_as_zero: 0.4105
strict_all_2_pass_rate: 0.1541
soft_all_ge1_pass_rate: 0.1952
```

当前结果说明：

- `qwen3_sft_instruction` 在 judged-only 和 generation-error-as-zero 两种口径下都最好
- `claude` 没有 generation error，但整体指令遵循分低于 `qwen3_sft_instruction`
- `qwen3_sft_v2` 明显较弱，尤其 novel 指令和 strict pass rate 较低

## 11. 常见问题

### 11.1 candidate generation 出现 400 Bad Request

常见原因是：

```text
prompt tokens + max output tokens > model max context length
```

当前脚本默认：

```bash
CANDIDATE_MAX_OUTPUT_TOKENS=768
```

并在遇到上下文长度错误时自动降级重试：

```text
768 -> 512 -> 256
```

如果仍失败，可以进一步降低：

```bash
CANDIDATE_MAX_OUTPUT_TOKENS=512
```

### 11.2 为什么 full300 里有 generation_error

`generation_error` 是 candidate model 没成功产出回复，不进入 judged-only mean，但会写入：

```text
excluded_generation_errors.jsonl
```

正式比较时建议同时报告：

```text
judged_only_mean_score
full_mean_score_generation_errors_as_zero
```

### 11.3 为什么 novel 原子分数普遍更低

这是预期现象。novel 原子用于评估 OOD 指令遵循能力，很多要求是模型平时不常见的表达、格式、风格或结构化约束，因此更容易拉开模型差距。

### 11.4 为什么有些禁止类指令没触发也给 2 分

禁止类或替换类指令的逻辑是：只要当前回复没有违反，即可视为遵守。例如“不要使用某个词”在回复中没有出现该词，就是合规。

## 12. 后续建议

后续可以继续增强三个方向：

1. 增加 judge calibration：抽样人工复核低分和高分样本，检查 judge 是否过严或过松。
2. 增加 per-atom 稳定性分析：找出 judge 不稳定或本身不适合医疗场景的 atom。
3. 增加 candidate response 缓存：避免同一个模型重复跑时反复调用 candidate API。

这个 benchmark 当前已经可以作为一个独立的 instruction-following 评测基线使用。
