# OPD Multi-Teacher v3: Teacher-Generated Reverse-KL Top-k

v3 基于 v2，仍然使用 teacher-generated response 和 teacher top-k 分布。
唯一核心区别是 KL 方向从 `forward_kl_topk` 改为 `reverse_kl_topk`：

- v2：在 teacher top-k 支撑集上最小化 `KL(teacher || student)`。
- v3：在 teacher top-k 支撑集上最小化 `KL(student || teacher)`。

## 文件

```text
recipe/opd_multi_teacher/v3/
├── build_teacher_generated_data.py
├── run_reverse_kl_topk_qwen3_8b.sh
├── teacher_forced_agent_loop.py
├── teacher_forced_agent_loop.yaml
├── reward_zero.py
└── README.md
```

## 第一步：构建 Teacher-Generated 数据

默认读取：

```text
recipe/opd_multi_teacher/data/opd_multi_teacher_10k.train.jsonl
```

并写入：

```text
recipe/opd_multi_teacher/v3/data/opd_multi_teacher_v3_teacher_generated.train.jsonl
```

运行：

```bash
cd /data/chengch/project/verl
python recipe/opd_multi_teacher/v3/build_teacher_generated_data.py \
  --input recipe/opd_multi_teacher/data/opd_multi_teacher_10k.train.jsonl \
  --contact-teacher /data1/chengch/models/qwen3_8b_contact_step200 \
  --mid-teacher /data1/chengch/models/qwen3_8b_normal_mid
```

## 第二步：Reverse-KL Top-k 蒸馏

```bash
cd /data/chengch/project/verl
bash recipe/opd_multi_teacher/v3/run_reverse_kl_topk_qwen3_8b.sh
```

关键配置：

```text
distillation.distillation_loss.loss_mode=reverse_kl_topk
distillation.distillation_loss.use_policy_gradient=False
distillation.distillation_loss.use_task_rewards=False
distillation.distillation_loss.topk=64
```

输出目录：

```text
/data1/chengch/verl_outputs/opd_multi_teacher/v3/<exp_name>/
```
