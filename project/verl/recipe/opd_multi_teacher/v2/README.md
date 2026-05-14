# OPD Multi-Teacher v2: Teacher-Generated Forward-KL Top-k

v2 是一个独立实验分支，不影响上一级目录里的 v1 reverse-KL OPD。

核心区别：

- v1：student 先 rollout，teacher 对 student token 计算 logprob，使用 reverse-KL estimator。
- v2：teacher 先生成回复，训练时把 teacher 回复作为 rollout response，student 对 teacher top-k 分布做 forward KL。

## 文件

```text
recipe/opd_multi_teacher/v2/
├── build_teacher_generated_data.py
├── run_forward_kl_topk_qwen3_8b.sh
├── teacher_forced_agent_loop.py
├── teacher_forced_agent_loop.yaml
├── reward_zero.py
└── README.md
```

## 第一步：构建 Teacher-Generated 数据

默认读取 v1 的 mixed OPD 数据：

```text
recipe/opd_multi_teacher/data/opd_multi_teacher_10k.train.jsonl
```

按 `teacher_route` 调用不同 teacher 生成回复，并写入：

```text
recipe/opd_multi_teacher/v2/data/opd_multi_teacher_v2_teacher_generated.train.jsonl
```

运行：

```bash
cd /data/chengch/project/verl
python recipe/opd_multi_teacher/v2/build_teacher_generated_data.py \
  --input recipe/opd_multi_teacher/data/opd_multi_teacher_10k.train.jsonl \
  --contact-teacher /data1/chengch/models/qwen3_8b_contact_step200 \
  --mid-teacher /data1/chengch/models/qwen3_8b_normal_mid
```

调试时可以先少量生成：

```bash
python recipe/opd_multi_teacher/v2/build_teacher_generated_data.py --max-samples 128
```

输出数据会包含：

- `teacher_route`
- `teacher_response`
- `agent_name=teacher_forced_agent`
- `reward_model.ground_truth=teacher_response`

## 第二步：Forward-KL Top-k 蒸馏

运行：

```bash
cd /data/chengch/project/verl
bash recipe/opd_multi_teacher/v2/run_forward_kl_topk_qwen3_8b.sh
```

脚本默认配置：

```text
distillation.distillation_loss.loss_mode=forward_kl_topk
distillation.distillation_loss.use_policy_gradient=False
distillation.distillation_loss.use_task_rewards=False
distillation.distillation_loss.topk=64
```

也就是说，训练目标是直接最小化 teacher top-k 分布和 student 分布之间的 forward KL。

常用覆盖参数：

```bash
DISTILL_TOPK=128 \
TRAIN_BATCH_SIZE=4 \
TEACHER_GPU_MEMORY_UTIL=0.35 \
TEACHER_MAX_NUM_SEQS=2 \
bash recipe/opd_multi_teacher/v2/run_forward_kl_topk_qwen3_8b.sh
```

如果 teacher logprob OOM，优先降低：

```bash
DISTILL_TOPK=32
TRAIN_BATCH_SIZE=4
TEACHER_MAX_NUM_SEQS=1
```

## 输出

默认输出目录：

```text
/data1/chengch/verl_outputs/opd_multi_teacher/v2/<exp_name>/
```

其中：

- `validation_rollouts/`：验证样本 JSONL；
- `genrm_io.jsonl`：teacher-forced 样本记录；
- `swanlog/`：SwanLab 本地日志；
- checkpoint：verl 默认 checkpoint。

## 重要说明

v2 通过自定义 `teacher_forced_agent` 实现 teacher-forced rollout。它不会调用 student rollout 生成回复，而是直接使用数据中的 `teacher_response` 作为 response tokens。后续 teacher top-k logprob 和 actor 更新仍然由 verl 原生 distillation 逻辑完成。
