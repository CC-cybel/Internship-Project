# LeadBench-Excellent

优化话术Bench-Excellent

## 成本

使用 qwen3.5-397b-a17b （500个思维预算，每个对话0.30元）


## 项目结构

```
.
├── leadbench_excellent/      # 核心代码包
│   ├── evaluation/           # 评测核心逻辑 (evaluator.py, session_evaluator.py)
│   ├── generation/           # 候选模型生成逻辑 (generator.py, model_configs.py, processors.py)
│   ├── model/                # 模型接口封装 (api_model.py)
│   ├── simulator/            # 用户模拟器逻辑 (user_simulator.py, prompts.py)
│   └── utils/                # 通用工具 (config.py, report.py, dataset.py)
├── data/                     # 核心数据目录
│   ├── dataset/              # 评测种子集 (search_word_input_v1.jsonl)
│   └── rules/                # 评测规则定义 (leadbench_excellent_rule.json)
├── data_prep/                # 数据处理工作区
│   └── dataset_generation/   # 数据处理脚本与中间产物
├── output/                   # 评测结果输出目录 (每次运行按时间戳生成独立子目录)
├── scripts/                  # 启动脚本
│   └── dynamic_evaluate.py   # 动态对话评估主脚本
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
    
    在 `.env` 中配置三大模型及运行参数：
    *   **Candidate Model**: 待评估的候选模型（例如你的微调模型，支持 vLLM 部署的 OpenAI 兼容接口）。
    *   **Judge Model**: 用于评估对话质量的裁判模型（通常是能力较强的模型，如 qwen3.5-397b-a17b, deepseek-chat 等）。
    *   **User Simulator Model**: 用于模拟真实访客行为的用户模拟器模型（如 qwen-max-latest, claude-sonnet-4.5 等）。
    *   **EVALUATION_LIMIT**: (可选) 设置评估样本数量上限，用于快速调试 (e.g., `EVALUATION_LIMIT=10`)。

## 数据说明

项目已包含固定的输入数据集与规则文件：
*   **输入数据**: `data/dataset/search_word_input_v1.jsonl`
*   **规则文件**: `data/rules/leadbench_excellent_rule.json`

`search_word_input_v1.jsonl` 包含了用于触发不同业务场景的初始搜索词（Search Words）和用户画像，模拟器将根据这些信息与候选模型展开动态对话。

## 运行评估

本项目支持两种不同的评估方式：**动态互动评估** 和 **已完成对话评估**。

### 1. 动态互动评估 (Dynamic Evaluation)

使用 `dynamic_evaluate.py` 脚本。此方式下，**用户模拟器**将根据种子集中的搜索词和用户画像，与**候选模型**展开实时的多轮互动，生成完整的对话过程，随后交由 **Judge模型** 进行评估。

运行完整的“模拟器互动生成 -> Judge模型评估”流水线：

```bash
python scripts/dynamic_evaluate.py
```

### 2. 已完成对话评估 (Static Evaluation)

使用 `evaluate_completed_dialogue.py` 脚本。此方式无需模拟器参与，直接对预先准备好的、已包含完整多轮对话历史的静态数据集进行规则打分。适用于回测线上真实对话日志或评估第三方生成的固定对话样本。

运行静态评估流水线：

```bash
python scripts/evaluate_completed_dialogue.py
```

> **注意**: 运行静态评估前，需确保 `.env` 中的 `INPUT_FILE` 指向包含完整 `messages` 对话记录的数据集文件（例如 `data/dataset/completed_dialogue_input_v1.jsonl`）。

## 输出结果

程序运行结束后，将在 `output/` 目录下生成一个包含模型名和时间戳的独立子目录（例如：`output/v0.1_cand_XXX_sim_XXX_judge_XXX_20260323_140000/`），其中包含：
1.  **evaluation_results.jsonl**: 包含完整的模拟对话记录（messages）及详细的策略/规则评估结果的 JSONL 文件。
2.  **evaluation_report.md**: Markdown 格式的统计报告，包含各规则得分及平均得分。
3.  **evaluation_results_by_rule_sorted.jsonl**: 按照规则分类并按得分排序的 JSONL 结果，方便分析每个规则的具体表现。
4.  **scenario_scores_chart.png**: 各场景/规则得分对比的柱状图。
5.  **failed_cases.jsonl**: 仅包含未通过规则校验或得分为负的失败案例。
6.  **config.json**: 记录本次运行的各项配置参数。

## 模型特定处理 (Model-Specific Processing)

LeadBench-Excellent 支持针对不同的 Candidate Model 配置特定的前处理（Pre-processing）和后处理（Post-processing）逻辑，例如设置 System Prompt、追加轮次信息或提取特定格式的回复。

配置文件位于：`leadbench_excellent/generation/model_configs.py`

### 如何添加新模型配置

1.  在 `leadbench_excellent/generation/model_configs.py` 中定义一个新的配置函数：

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
        "default": configure_minimal_processor, # 默认配置：不做任何处理 (Identity)
        "my_custom_model": configure_my_model,  # 只要模型名称包含 "my_custom_model"，就会使用该处理
    }
    ```

程序运行时会自动根据 `.env` 中的 `CANDIDATE_MODEL_NAME` 匹配最合适的配置。默认情况下（如果未匹配到任何关键字），**不会应用任何额外的前处理或后处理**，直接使用原始数据进行生成。

### 特别说明

目前针对 `normal_anti_hijack_abc_stage2`、`normal_offline_full_v3` 系列以及 `llama_factory_精神科1.32.1_lora_qwen2_7b_dpo` 等模型已经内置了特定的系统指令与处理配置，其他模型请根据需要进行配置。
