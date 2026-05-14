# CC Single-Turn RL

本仓库用于单轮对话强化学习实验，主要包含两部分：

1. `rl_remake/`：从多轮对话数据中抽取单轮 RL 训练样本；
2. `verl/recipe/`：基于 verl 框架的 GRPO 训练 recipe 和多专家蒸馏实验。

当前重点实验方向是面向销售/咨询类对话的阶段化训练：

- **contact stage**：套联阶段，训练模型在合适时机完成联系方式获取；
- **mid stage**：中间阶段，训练模型在套联前完成价值建立、需求挖掘和自然推进。

## 目录结构

```text
.
├── rl_remake/
│   ├── prepare_single_turn_rl_dataset_contact_stage.py
│   ├── prepare_single_turn_rl_dataset_contact_stage_v2.py
│   ├── prepare_single_turn_rl_dataset_mid_stage.py
│   ├── prepare_single_turn_rl_dataset_first2.py
│   └── stat_final_length.py
└── verl/
    └── recipe/
        ├── single_turn_reward/
        │   ├── v4/
        │   ├── v5/
        │   └── README.md
        └── opd_multi_teacher/
            ├── build_opd_multi_teacher_data.py
            ├── run_opd_multi_teacher_qwen3_8b.sh
            └── README.md
```

大数据、训练输出、日志、wheel 包、Python 缓存等文件不会上传到 git。
这些内容需要在本地重新生成或放在外部存储路径中。

## 数据构建

`rl_remake` 负责把多轮 SFT 格式对话切成 verl 可用的单轮 RL 数据。
每条样本主要包含：

- `prompt`：目标 assistant 回复之前的对话上下文；
- `ground_truth`：参考回复；
- `extra_info`：阶段信息、轮次信息和 reward 所需的辅助字段；
- `reward_model`、`data_source`、`agent_name`、`index`：verl 训练兼容字段。

### 套联阶段数据

```bash
cd /data/chengch/project
python rl_remake/prepare_single_turn_rl_dataset_contact_stage.py
```

这个脚本会优先抽取系统提示中标记为套联轮次附近的 assistant 回复，
并检查回复是否符合 `BEGIN_META / BEGIN_FINAL` 的格式要求。

### 中间阶段数据

```bash
cd /data/chengch/project
python rl_remake/prepare_single_turn_rl_dataset_mid_stage.py
```

这个脚本会抽取第 3 轮以后、正式套联之前的 assistant 回复，
用于训练模型在中间阶段进行价值建立和自然推进。

默认输出目录是：

```text
rl_remake/outputs/
```

该目录已被 `.gitignore` 排除，因为生成的数据文件通常比较大。

## Single-Turn Reward Recipe

主要 recipe 位于：

```text
verl/recipe/single_turn_reward/
```

目前主要使用：

- `v4`：套联阶段 reward；
- `v5`：中间阶段 reward。

这两个版本都通过 verl 的 custom reward function 接入：

```text
reward.custom_reward_function.path=...
reward.custom_reward_function.name=compute_score
```

## V4：套联阶段训练

入口脚本：

```text
verl/recipe/single_turn_reward/v4/run_grpo_single_turn_4gpu_qwen3_8b_stage4_reward.sh
```

推荐运行方式：

```bash
cd /data/chengch/project/verl
TONGYI_API_KEY=your_key \
MODEL_PATH=/path/to/base_or_sft_model \
TRAIN_FILE=/path/to/single_turn_rl_contact_stage.train.parquet \
VAL_FILE=/path/to/single_turn_rl_contact_stage.val.parquet \
bash recipe/single_turn_reward/v4/run_grpo_single_turn_4gpu_qwen3_8b_stage4_reward.sh
```

核心配置：

- reward 文件：`recipe/single_turn_reward/v4/reward_function_stage4_contact_cloud.py`
- 默认 GPU：`CUDA_VISIBLE_DEVICES=1,2,3,4`
- 默认每个 prompt 采样数：`ROLLOUT_N=4`
- 默认输出目录：`/data1/chengch/verl_outputs/grpo_single_turn/<exp_name>/`

V4 支持收集模型裁判的输入输出，便于后续分析或训练 reward model：

```bash
COLLECT_GENRM_IO=True
GENRM_IO_PATH=/path/to/genrm_io.jsonl
```

也支持导出高分样本，用于后续 SFT：

```bash
SAVE_HIGH_SCORE_SFT=True
SFT_SCORE_THRESHOLD=0.9
SFT_OUTPUT_PATH=/path/to/high_score_sft.jsonl
```

## V5：中间阶段训练

入口脚本：

```text
verl/recipe/single_turn_reward/v5/run_grpo_single_turn_4gpu_qwen3_8b_stage5_mid_reward.sh
```

推荐运行方式：

```bash
cd /data/chengch/project/verl
TONGYI_API_KEY=your_key \
MODEL_PATH=/path/to/contact_stage_checkpoint \
TRAIN_FILE=/path/to/single_turn_rl_random_rounds_mid_stage.train.parquet \
VAL_FILE=/path/to/single_turn_rl_random_rounds_mid_stage.val.parquet \
bash recipe/single_turn_reward/v5/run_grpo_single_turn_4gpu_qwen3_8b_stage5_mid_reward.sh
```

核心配置：

- reward 文件：`recipe/single_turn_reward/v5/reward_function_stage5_mid_cloud.py`
- 默认 GPU：`CUDA_VISIBLE_DEVICES=4,5,6,7`
- 默认最大 prompt 长度：`MAX_PROMPT_LENGTH=2040`
- 默认输出目录：`/data1/chengch/verl_outputs/grpo_single_turn/<exp_name>/`

常用覆盖参数示例：

```bash
TOTAL_EPOCHS=1 SAVE_FREQ=25 TEST_FREQ=25 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash recipe/single_turn_reward/v5/run_grpo_single_turn_4gpu_qwen3_8b_stage5_mid_reward.sh
```

## 多专家蒸馏实验

多专家蒸馏实验位于：

```text
verl/recipe/opd_multi_teacher/
```

这个 recipe 主要用于尝试 multi-teacher on-policy distillation：

- contact 样本路由到套联阶段 teacher；
- mid 样本路由到中间阶段 teacher；
- student 在同一次训练中学习不同阶段 teacher 的行为。

数据构建脚本会混合两类样本，并在每条样本中写入：

```text
teacher_route=contact
teacher_route=mid
```

构建或重建混合数据：

```bash
cd /data/chengch/project/verl
python recipe/opd_multi_teacher/build_opd_multi_teacher_data.py --overwrite
```

运行多专家蒸馏：

```bash
cd /data/chengch/project/verl
MODEL_PATH=/path/to/student_model \
CONTACT_TEACHER=/path/to/contact_teacher \
MID_TEACHER=/path/to/mid_teacher \
bash recipe/opd_multi_teacher/run_opd_multi_teacher_qwen3_8b.sh
```

常用覆盖参数：

```bash
REBUILD_DATA=true TEACHER_TP=2 TRAIN_BATCH_SIZE=8 \
bash recipe/opd_multi_teacher/run_opd_multi_teacher_qwen3_8b.sh
```

该实验开启了 distillation 配置：

```text
distillation.enabled=True
distillation.distillation_loss.use_policy_gradient=True
distillation.distillation_loss.use_task_rewards=False
```

其中 `reward_zero.py` 用于关闭任务 reward，让训练主要由 distillation loss 驱动。

## 训练输出

默认训练输出位于：

```text
/data1/chengch/verl_outputs/
```

常见输出包括：

- checkpoint；
- validation rollout；
- SwanLab 日志；
- `genrm_io.jsonl`；
- 高分样本导出文件。

这些文件通常比较大，不建议提交到 git。

## 注意事项

- 运行 verl recipe 时，建议工作目录切到 `/data/chengch/project/verl`。
- API key、SwanLab key、模型路径建议通过环境变量传入，不要提交到仓库。
- `rl_remake/outputs/`、`verl/recipe/opd_multi_teacher/data/` 等生成数据目录已被忽略。
- 本仓库将 upstream verl 代码放在 `verl/` 下，并在其上增加了当前项目所需的 recipe、reward function 和训练脚本。
