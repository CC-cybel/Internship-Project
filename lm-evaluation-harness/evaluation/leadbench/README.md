# LeadBench 

LeadBench 是一个基于规则的对话质量评估框架，支持通过 API 调用待评估模型（Candidate Model）生成回复，并使用裁判模型（Judge Model）进行自动化评估。

## 项目结构

```
.
├── leadbench/                # 核心代码包
│   ├── evaluation/           # 评测核心逻辑 (evaluator.py)
│   ├── generation/           # 生成核心逻辑 (generator.py, model_configs.py, processors.py)
│   ├── model/                # 模型接口封装 (api_model.py)
│   └── utils/                # 通用工具 (config.py, report.py, dataset.py)
├── data/                     # 核心数据目录
│   └── dataset/              # 标准评测集 (golden_history_input.jsonl)
├── data_prep/                # 数据处理工作区
│   └── dataset_generation/   # 数据处理脚本与中间产物 (Excel转JSONL等)
├── output/                   # 评测结果输出目录
├── scripts/                  # 启动脚本
├── requirements.txt          # 项目依赖
└── README.md                 # 项目说明
```

## 环境准备

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **配置文件**:
    复制 `.env.example` 为 `.env` 并填入你的配置信息。
    ```bash
    cp .env.example .env
    ```
    
    在 `.env` 中配置：
    *   **Candidate Model**: 用于生成回复的模型（如你的微调模型，使用vllm部署）。
    *   **Judge Model**: 用于评估回复质量的模型（通常是能力较强的模型，这个不要改，使用默认就好，方便对比）。
    *   **EVALUATION_LIMIT**: (可选) 设置评估样本数量上限，用于快速调试 (e.g., `EVALUATION_LIMIT=10`，这个一般也不改)。

## 数据说明

项目包含不同领域的输入数据集，无需修改，通过 `DOMAIN` 环境变量进行切换：
*   **输入数据**: `data/dataset/${DOMAIN}/golden_history_input_v1.jsonl`
*   **规则文件**: `data/rules/leadbench_rule.json`

示例：
- Psychiatry 领域：`data/dataset/psychiatry/golden_history_input_v1.jsonl`
- Douyin Dentistry 领域：`data/dataset/douyin_dentistry/golden_history_input_v1.jsonl`

## 运行评估

运行完整的生成与评估流水线，你可以通过设置环境变量来动态指定模型配置和运行哪个领域的数据集（例如 `psychiatry` 或 `douyin_dentistry`）。

除了 `DOMAIN` 以外，你还可以动态传入候选模型相关的配置参数：
- `CANDIDATE_MODEL_NAME`: 候选模型名称
- `CANDIDATE_API_KEY`: API Key
- `CANDIDATE_API_BASE`: API Base URL
- `CANDIDATE_MAX_OUTPUT_TOKENS`: 最大输出长度
- `OUTPUT`: 结果输出目录 (默认: `./output`)

**方式一：使用 `export` 设置环境变量**

```bash
export DOMAIN=psychiatry
export CANDIDATE_MODEL_NAME=my_custom_model
export CANDIDATE_API_KEY=tokenabc123
export CANDIDATE_API_BASE=http://123.181.192.120:18581/v1
export CANDIDATE_MAX_OUTPUT_TOKENS=512
export OUTPUT=./my_custom_output

python scripts/evaluate_golden_history.py
```

**方式二：在一行命令中直接传入（推荐，.env复制后就不用再改文件里面的配置了）**

```bash
DOMAIN=psychiatry CANDIDATE_MODEL_NAME=qwen3_8b_normal_short_step450 CANDIDATE_API_KEY=tokenabc123 CANDIDATE_API_BASE=http://127.0.0.1:8000/v1 CANDIDATE_MAX_OUTPUT_TOKENS=512 OUTPUT=./my_custom_output python scripts/evaluate_golden_history.py
```

如果未设置环境变量，程序将默认使用 `.env` 文件中配置的值。

## 输出结果

程序运行结束后，将在配置的输出目录（默认为 `output/`）生成：
1.  **evaluation_results.jsonl**: 包含生成回复及详细评估结果的 JSONL 文件。
2.  **evaluation_report.md**: Markdown 格式的统计报告。
3.  **rule_pass_rates.png**: 各规则通过率的柱状图。
4.  **failed_cases.jsonl**: 仅包含未通过规则校验的失败案例，方便快速定位问题。

指标主要看 `evaluation_report.md` 中的统计报告。
```
- **整体通过率 (Average of Rules Pass Rate)**
- **Average Score (平均分)**
- **Hard Pass Rate (硬性通过率)**:
- **Sample Pass Rate (完美率)**
```


## 模型特定处理 (Model-Specific Processing)

LeadBench 支持针对不同模型配置特定的前处理（Pre-processing）和后处理（Post-processing）逻辑，例如设置 System Prompt、追加轮次信息或提取特定格式的回复。

配置文件位于：`leadbench/generation/model_configs.py`

### 如何添加新模型配置

1.  在 `leadbench/generation/model_configs.py` 中定义一个新的配置函数：

    ```python
    def configure_my_model(processor: DialogueProcessor):
        # 添加前处理：设置 System Prompt
        processor.add_pre_processor(lambda msgs, ctx: add_system_instruction(msgs, ctx, "你的自定义 System Prompt"))
        
        # 添加前处理：追加轮次信息
        processor.add_pre_processor(append_turn_info)
        
        # 添加后处理：提取最终回复
        processor.add_post_processor(extract_final_response)
    ```

2.  在 `MODEL_CONFIGS` 字典中注册你的模型名称关键字：

    ```python
    MODEL_CONFIGS = {
        "default": configure_identity_processor, # 默认配置：不做任何处理 (Identity)
        "leadbench": configure_standard_processor,  # 只要模型名称包含 "leadbench"，就会使用标准处理
    }
    ```

程序运行时会自动根据 `CANDIDATE_MODEL_NAME` 匹配最合适的配置。默认情况下（如果未匹配到任何关键字），**不会应用任何额外的前处理或后处理**，直接使用原始数据进行生成。

### 特别说明

目前针对以下模型已经做了特定的前处理/后处理配置，其他模型请根据需要自行配置：
- 包含 `dy` 且包含 `normal` 的模型名称 (抖音口腔科的指令跟随模型，Fallback 至 `dy_normal_default` 配置)
- 包含 `normal` 的模型名称 (指令跟随模型，Fallback 至 `normal_default` 配置)
- `normal_anti_hijack_abc_stage2`
- `llama_factory_精神科1.32.1_lora_qwen2_7b_dpo`
- `normal_offline_full_v3_a2r8`
- `normal_offline_full_v3_dpo_a2r8`
- `normal_offline_full_v3_dpo`
- `normal_opd_1499`
- `normal_dual_full_mix_15`
