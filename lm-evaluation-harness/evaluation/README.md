# 这里描述对模型进行评估

使用api的方法进行评估

## 评估包括：

### 通用能力
- IF-EVAl: 指令跟随，使用 lm-evaluation-harness  
- MMLU: 通用知识，使用 lm-evaluation-harness
- GSM8K: 数学推理，使用 lm-evaluation-harness
- HumanEval: 代码生成，使用 lm-evaluation-harness
- TruthfulQA: 真实性/幻觉，使用 lm-evaluation-harness
- MT-Bench：多轮对话能力，使用 FastChat （待定）

### 专业能力
- LeadBench: 使用 Leadbench
- ITBench: 使用 ITBench

### 定义公共变量

```bash
domain=douyin_dentistry # 评估领域，抖音口腔科 或 精神科 douyin_dentistry 或 psychiatry
model_name=dy_dent_normal_dual_full_turn_m15_ah10_stage1_14500
base_url=http://36.248.221.141:18585/v1
tokenizer_path=/data/yezj/trained_model/dy_dent_normal_dual_full_turn_m15_ah10_stage1_14500
output=./eval_results
```

## lm-evaluation-harnesss使用

### lm-evaluation-harnesss安装

```
conda create -n lm-evaluation-harness python=3.11
conda activate lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install "lm_eval[api]"
pip install transformers
pip install langdetect
pip install immutabledict
```

### ifeval, gsm8k
```bash
conda activate lm-evaluation-harness
HF_ENDPOINT=https://hf-mirror.com lm_eval --model local-chat-completions --model_args model=${model_name},base_url=${base_url}/chat/completions,num_concurrent=20 --tasks ifeval,gsm8k --batch_size 1  --apply_chat_template --output ${output}/lm-evaluation-harness
```

示例（硬编码原版）：
```bash
conda activate lm-evaluation-harness
HF_ENDPOINT=https://hf-mirror.com lm_eval --model local-chat-completions --model_args model=dy_dent_normal_dual_full_turn_m15_ah10_stage1_14500,base_url=http://36.248.221.141:18585/v1/chat/completions,num_concurrent=20 --tasks ifeval,gsm8k --batch_size 1  --apply_chat_template --output ./eval_results/lm-evaluation-harness
```

其中，`model` 是模型的名称，`base_url` 是模型的api地址，`--tasks` 是要评估的任务，`--batch_size` 是批量大小，`--apply_chat_template` 是是否应用聊天模板的。

耗时：
ifeval: 2分钟左右
gsm8k: 3-4分钟

### humaneval

```bash
conda activate lm-evaluation-harness
HF_ENDPOINT=https://hf-mirror.com HF_ALLOW_CODE_EVAL="1" lm_eval --model local-completions --model_args model=${model_name},base_url=${base_url}/completions,num_concurrent=20,tokenizer=${tokenizer_path},tokenized_requests=False --tasks humaneval --batch_size 1 --confirm_run_unsafe_code --output ${output}/lm-evaluation-harness
```

示例（硬编码原版）：
```bash
conda activate lm-evaluation-harness
HF_ENDPOINT=https://hf-mirror.com HF_ALLOW_CODE_EVAL="1" lm_eval --model local-completions --model_args model=dy_dent_normal_dual_full_turn_m15_ah10_stage1_14500,base_url=http://36.248.221.141:18585/v1/completions,num_concurrent=20,tokenizer=/data/yezj/trained_model/dy_dent_normal_dual_full_turn_m15_ah10_stage1_14500,tokenized_requests=False --tasks humaneval --batch_size 1 --confirm_run_unsafe_code --output ./eval_results/lm-evaluation-harness
```

其中，`--confirm_run_unsafe_code` 是是否确认运行不安全的代码。

耗时：
humaleval: 1分钟以内


### mmlu, truthfulqa

```bash
conda activate lm-evaluation-harness
HF_ENDPOINT=https://hf-mirror.com  lm_eval --model local-completions --model_args model=${model_name},base_url=${base_url}/completions,num_concurrent=20,tokenizer=${tokenizer_path},tokenized_requests=False --tasks truthfulqa,mmlu --batch_size 1 --output ${output}/lm-evaluation-harness
```

示例（硬编码原版）：
```bash
conda activate lm-evaluation-harness
HF_ENDPOINT=https://hf-mirror.com  lm_eval --model local-completions --model_args model=dy_dent_normal_dual_full_turn_m15_ah10_stage1_14500,base_url=http://36.248.221.141:18585/v1/completions,num_concurrent=20,tokenizer=/data/yezj/trained_model/dy_dent_normal_dual_full_turn_m15_ah10_stage1_14500,tokenized_requests=False --tasks truthfulqa,mmlu --batch_size 1 --output ./eval_results/lm-evaluation-harness
```

其中，`model` 是模型的名称，`base_url` 是模型的api地址，`--tasks` 是要评估的任务，`--batch_size` 是批量大小，`--tokenizer` 是分词器的路径，`--tokenized_requests` 是是否对请求进行分词。

耗时：
truthfulqa: 5分钟不到
mmlu: 11分钟左右
humaleval: 15分钟左右

**few-shot**

mmlu测试 5-shot，可以加 --num_fewshot 5
耗时：2个半小时

## Leadbench使用

### leadbench安装

```
conda create -n leadbench python=3.11
conda activate leadbench
cd leadbench
pip install -r requirements.txt
```

### leadbench评估

```bash
conda activate leadbench
DOMAIN=${domain} \
CANDIDATE_MODEL_NAME=${model_name} \
CANDIDATE_API_KEY=tokenabc123 \
CANDIDATE_API_BASE=${base_url} \
CANDIDATE_MAX_OUTPUT_TOKENS=512 \
OUTPUT=${output}/leadbench \
python evaluation/leadbench/scripts/evaluate_golden_history.py
```

示例（硬编码原版）：
```bash
conda activate leadbench
DOMAIN=douyin_dentistry CANDIDATE_MODEL_NAME=my_custom_model CANDIDATE_API_KEY=tokenabc123 CANDIDATE_API_BASE=http://123.181.192.120:18581/v1 CANDIDATE_MAX_OUTPUT_TOKENS=512 OUTPUT=./my_custom_output python evaluation/leadbench/scripts/evaluate_golden_history.py
```

指标主要看 `evaluation_report.md` 中的统计报告。
```
- **整体通过率 (Average of Rules Pass Rate)**
- **Average Score (平均分)**
- **Hard Pass Rate (硬性通过率)**:
- **Sample Pass Rate (完美率)**
```

耗时： 1-2个小时
