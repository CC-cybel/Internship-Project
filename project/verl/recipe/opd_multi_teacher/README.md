# Multi-Teacher OPD Recipe

This recipe routes samples to different teachers during on-policy distillation:

- `teacher_route=contact` uses `/data1/chengch/models/qwen3_8b_merged`
- `teacher_route=mid` uses `/data1/chengch/models/qwen3_8b_mid_short_step500`

The student defaults to `/data/chengch/normal_stage2_exp7_qwen3_8b_full_sft_t4`.

Run:

```bash
bash recipe/opd_multi_teacher/run_opd_multi_teacher_qwen3_8b.sh
```

The script builds a 10k mixed JSONL dataset on first run, with 5k contact-stage
samples and 5k mid-stage samples. It keeps student rollout on-policy, routes each
sample to the configured teacher, and uses teacher top-k reverse-KL distillation
with `distillation.distillation_loss.loss_mode=reverse_kl_topk`, `topk=20`, and
no task reward.

Useful overrides:

```bash
TEACHER_TP=2 TRAIN_BATCH_SIZE=8 bash recipe/opd_multi_teacher/run_opd_multi_teacher_qwen3_8b.sh
REBUILD_DATA=true bash recipe/opd_multi_teacher/run_opd_multi_teacher_qwen3_8b.sh
DISTILL_TOPK=32 bash recipe/opd_multi_teacher/run_opd_multi_teacher_qwen3_8b.sh
```
