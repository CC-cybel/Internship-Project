# Single-Turn Hybrid Reward Template

This folder provides a Reward Loop custom reward template for:
- generative judge: lead reward
- generative judge: instruction following
- rule-based reward: output format checks

## Files
- `reward_function.py`: main async reward function entry (`compute_score`)
- `lead_reward.py`: integrated lead reward implementation (loaded by default as `compute_lead_score`)
- `lead_reward_template.py`: collaborator template for lead reward (`compute_lead_score`)
- `instruction_reward_template.py`: collaborator template for instruction reward (`compute_instruction_score`)

## What You Need To Fill
In `reward_function.py`, fill:
- `LEAD_JUDGE_PROMPT_TEMPLATE`
- `INSTRUCTION_JUDGE_PROMPT_TEMPLATE`

Both templates support placeholders:
- `{question}`
- `{answer}`
- `{ground_truth}`
- `{extra_info_json}`

## How To Plug Into verl
Use Hydra overrides in your run command:

```bash
custom_reward_function.path=recipe/single_turn_reward/reward_function.py \
custom_reward_function.name=compute_score \
reward_model.use_reward_loop=True \
reward_model.enable=True \
reward_model.model.path=/path/to/your/genrm-or-judge-model \
reward_model.reward_manager=naive \
+custom_reward_function.reward_kwargs.judge_model=/path/to/your/genrm-or-judge-model \
+custom_reward_function.reward_kwargs.lead_weight=0.4 \
+custom_reward_function.reward_kwargs.instruction_weight=0.4 \
+custom_reward_function.reward_kwargs.format_weight=0.2 \
+custom_reward_function.reward_kwargs.hard_format_gate=true \
+custom_reward_function.reward_kwargs.format_gate_threshold=1.0 \
+custom_reward_function.reward_kwargs.format_fail_score=0.0
```

## PPO 8-GPU Config + Script
- Config file: `recipe/single_turn_reward/config/ppo_single_turn_reward.yaml`
- Launch script: `recipe/single_turn_reward/train_ppo_single_turn_reward_8gpu.sh`

Example:

```bash
cd /data/wangpf/project/verl
SWANLAB_API_KEY=your_key \
REWARD_ROUTER_ADDRESS=127.0.0.1:8000 \
JUDGE_MODEL=gpt-5.2 \
TOTAL_EPOCHS=3 \
ROLLOUT_EVERY_N_EPOCHS=1 \
bash recipe/single_turn_reward/train_ppo_single_turn_reward_8gpu.sh
```

Script features:
- 8-GPU PPO startup (`trainer.n_gpus_per_node=8`)
- SwanLab logger enabled (`trainer.logger=["console","swanlab"]`)
- Reward API/URL/model and key are configurable by env vars
- Auto export validation rollout samples every `N` epochs:
  - sets `trainer.test_freq = steps_per_epoch * ROLLOUT_EVERY_N_EPOCHS`
  - dumps samples to `trainer.validation_data_dir`

SwanLab metrics for three rewards (step-level):
- `reward_components/lead_score/mean`
- `reward_components/instruction_follow_score/mean`
- `reward_components/format_score/mean`

## Team Collaboration (two people developing separately)
Recommended way:
- Person A implements lead reward in one file.
- Person B implements instruction-follow reward in another file.
- Main `reward_function.py` loads both functions and merges scores.

Configure:

```bash
+custom_reward_function.reward_kwargs.instruction_reward_func_path=recipe/single_turn_reward/instruction_reward_template.py \
+custom_reward_function.reward_kwargs.instruction_reward_func_name=compute_instruction_score
```

Notes:
- `reward_function.py` will load `recipe/single_turn_reward/lead_reward.py` by default.
- If you want to override lead reward module path/name, set:
  - `+custom_reward_function.reward_kwargs.lead_reward_func_path=...`
  - `+custom_reward_function.reward_kwargs.lead_reward_func_name=...`
- `instruction_reward.py` enables rule mapper by default, but it only makes the extra mapper call when `extra_info.system_prompt` is non-empty.

Lead reward mode switches:

```bash
+custom_reward_function.reward_kwargs.lead_eval_mode=single_call \
+custom_reward_function.reward_kwargs.lead_single_call_protocol=violations_only \
+custom_reward_function.reward_kwargs.lead_single_call_fallback_to_batch=true
```

Supported values:
- `lead_eval_mode`: `dimension_batch` (default) or `single_call`
- `lead_single_call_protocol`: `violations_only` (default) or `full_results`
- `lead_single_call_fallback_to_batch`: whether single-call parse/request failures fall back to the old dimension-batch path

Expected external function signature (sync/async both supported):

```python
async def compute_lead_score(
    question: str,
    answer: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str | None = None,
    **kwargs,
):
    return {"score": 0.0, "raw": "", "status": "ok"}
```

Return can be:
- number in `[0, 1]`, or
- dict with key `score` in `[0, 1]`

If your judge calls external APIs with strict QPS limits, consider:

```bash
reward_model.reward_manager=rate_limited \
+reward_model.max_concurrent=8 \
+reward_model.max_rpm=300 \
+reward_model.timeout=60
```

## Collect GenRM I/O (for RM training data)
You can collect all generative RM inputs/outputs into one JSONL file:

```bash
+custom_reward_function.reward_kwargs.collect_genrm_io=true \
+custom_reward_function.reward_kwargs.genrm_io_path=/path/to/genrm_io.jsonl \
+custom_reward_function.reward_kwargs.genrm_io_include_extra_info=false
```

Notes:
- The file is JSONL (one JSON object per line), append-safe for training runs.
- It includes records from:
  - `lead_reward` per dimension-batch or single-call judge call + summary
  - `instruction_reward` rule-mapper call (when enabled and system prompt exists) + single-call / per-rule judge call + summary
  - `reward_function` final merged summary

## Format Reward Rules (optional)
You can set rules in either:
- dataset `extra_info` (recommended)
- `custom_reward_function.reward_kwargs`

Supported keys:
- `hard_format_gate` (default `true`)
- `format_gate_threshold` (default `1.0`)
- `format_fail_score` (default `0.0`)

Format checker is fixed to `hard_dual_s5_full` and enforces:
- exactly one `BEGIN_META/END_META` and one `BEGIN_FINAL/END_FINAL`
- strict block order: `META -> FINAL`
- first line in `META` is `action=...`
- second line in `META` is `thought=...`
- all meta lines are `key=value` with ASCII key name
- final block is non-empty

## Return Fields
`compute_score` returns:
- `score` (final reward, required)
- `lead_score`
- `instruction_follow_score`
- `format_score`
- debug fields (`*_status`, `*_judge_raw`, `format_detail`, weights)
