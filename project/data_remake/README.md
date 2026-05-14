# data_remake 交接说明

## 目录定位

`data_remake/` 是当前仓库里最重要的数据重制目录。

它承担了三类工作：

1. 把历史医疗对话改造成统一训练格式。
2. 为 anti-hijack、strict dual、LlamaFactory、verl 等不同训练口径生成数据。
3. 下载 / 清洗 / 去重外部 prompt pool，并准备 OPD / teacher rollout 数据。

## 一句话理解主线

主流水线是：

`prompt reverse -> 脱敏 -> action 注入 -> 重写 -> judge -> 再转成各种训练格式`

入口脚本是：

- `run_pipeline.py`

目前这个目录的脚本命名统一按“动词 + 对象”风格整理，优先从文件名直接判断用途。

## 重要提醒

- 这个目录里混合了“当前主线脚本”和“历史实验脚本”。
- 很多旧脚本仍使用文件头常量配置输入输出路径，不是统一 CLI。
- 部分旧脚本直接写了 `API_KEY`、`BASE_URL`、`MODEL_NAME` 常量。交接后建议优先改成环境变量，并清理敏感信息。

## 目录结构

- `raw/`
  主线原始输入。
- `intermediate/`
  中间产物目录，目前更多中间结果改写到 `runs/` 中。
- `outputs/`
  阶段性成品。
- `runs/`
  每次 `run_pipeline.py` 的完整执行结果。
- `cache/`
  断点续跑缓存。
- `logs/`
  原始日志和 LLM 输出日志。
- `experiments/`
  草稿、实验和历史样本，不是当前标准入口。

## 当前推荐的主线入口

### 1. 跑 normal / hard 数据主线

```bash
python data_remake/run_pipeline.py \
  --input data_remake/raw/normal_inject_round.json
```

### 2. 从中间步骤续跑

```bash
python data_remake/run_pipeline.py \
  --start rewrite \
  --stop judge \
  --input data_remake/runs/normal_dataset/normal_dataset_action.json
```

### 3. 严格要求只放行完成前一步的样本

```bash
python data_remake/run_pipeline.py \
  --input data_remake/runs/normal_dataset/normal_dataset_reverse.json \
  --start mask \
  --require-prev-done
```

## `run_pipeline.py` 负责什么

`run_pipeline.py` 只编排主流水线，不负责所有辅助脚本。

它串起来的步骤是：

1. `reverse`
   调 `reverse_prompts_from_dialogues.py`
2. `mask`
   调 `mask_user_phone_numbers.py`
3. `inject`
   调 `inject_random_actions.py`
4. `rewrite`
   调 `rewrite_dialogues.py`
5. `judge`
   调 `judge_rewrite_quality.py`

额外还会在 `rewrite` 后调用：

- `clean_response_symbols.py`

## 顶层文件逐项说明

下面这一节按文件逐项说明，是这个目录最重要的交接内容。

### 主流水线脚本

- `run_pipeline.py`
  主编排器。统一处理路径命名、步骤顺序、断点续跑、`*_pipeline_report.json` 生成。
- `reverse_prompts_from_dialogues.py`
  读取历史对话，反推该对话背后的 system prompt / SOP 约束，给后续重写提供统一 system。
- `mask_user_phone_numbers.py`
  对 `human/user` 侧手机号做脱敏，可选是否连座机一起替换。
- `inject_random_actions.py`
  从 assistant 的 `<think>` 或结构化标签里抽 action，按概率注入到上一轮用户输入。
- `rewrite_dialogues.py`
  主重写脚本。把 assistant 回复改写成结构化 `thought + slot_values + response` 形式。
- `clean_response_symbols.py`
  对重写结果做表层清洗，去掉引号、破折号等不希望保留的符号。
- `judge_rewrite_quality.py`
  对整段重写后的对话做质量审核，输出分数、通过标记、违规证据和统计摘要。

### 格式转换与训练集准备

- `convert_to_dual_channel_format.py`
  把数据转成 strict dual 文本格式：
  `BEGIN_META / BEGIN_FINAL`
  支持 response-only、response+slot、full 三种模式。
- `convert_to_xml_channel_format.py`
  把数据转成 XML 风格双通道格式，适合测试另一套结构化输出协议。
- `convert_meta_final_to_think_format.py`
  把 strict dual 的 `BEGIN_META / BEGIN_FINAL` 转成 `<think>...</think> + final` 风格。
- `prepare_llamafactory_sft.py`
  把 ShareGPT 风格数据整理成更适合 LlamaFactory SFT 使用的格式，并控制 assistant 序列化方式。
- `build_anti_hijack_dataset.py`
  从 ShareGPT 对话切片构造 anti-hijack A/B/C 三类训练集。
- `prepare_single_turn_rl_dataset.py`
  从 hard / normal dual 数据中抽单轮样本，生成 20k 级别的 RL 数据集。
- `convert_opd_jsonl_to_parquet.py`
  把 OPD JSONL 转成 `verl` 更易读取的 parquet 结构。

### OPD / Prompt Pool 相关

- `download_opd_prompt_pool.py`
  从公开 Hugging Face 数据集下载并采样 50k OPD prompt pool。
- `prepare_opd_dataset_from_pool.py`
  把下载回来的 prompt pool 清洗、去重并转换成 OPD-ready 数据。
- `download_teacher_rollout_prompt_pool.py`
  下载 teacher rollout 用的通用 prompt pool，并根据首条 user prompt 与 OPD pool 去重。
- `download_opd_prompt_pool.sh`
  `download_opd_prompt_pool.py` 的 shell 包装脚本。
- `download_teacher_rollout_prompt_pool.sh`
  `download_teacher_rollout_prompt_pool.py` 的 shell 包装脚本。

### 分析、过滤和一次性工具脚本

- `analyze_length_cutoffs.py`
  统计多轮样本 token 长度，并给出 cutoff 建议。
- `count_system_prompt_keywords.py`
  统计 system prompt 中特定关键词的命中次数。
- `filter_system_keyword_samples.py`
  丢弃 system prompt 里含某些关键词的整条样本，常用于去掉“孩子/家长”等口径。
- `drop_abnormal_samples.py`
  清理结构异常的 ShareGPT 样本，例如奇数轮、字段缺失、`from/value` 不规范。
- `clean_dialogue_text.py`
  历史清洗脚本，用于去掉 `Round X:` 前缀、`<picture>`、把 `[动作]` 转成 `<think>`。
- `mix_training_datasets.py`
  把通用数据和领域数据混合成一个 JSONL 输出。
- `extract_dialogue_entities.py`
  用 LLM 从对话中提取医院名、客服名、联系方式、福利钩子等实体信息。
- `complete_assistant_cot.py`
  给已有回复补拟人化 thought，用于把旧数据补成带 CoT 的版本。
- `inject_round_tags.py`
  给用户输入注入 `【系统数据：当前第 N 轮】` 标签，是更早期的预处理步骤。
- `repair_malformed_json.py`
  修某些半坏不坏的 JSON / 文本文件，偏一次性补丁工具。
- `create_debug_sample.py`
  从大样本中随机抽 debug 子集。
- `replace_child_profile_fields.py`
  把 system 中的“孩子年龄/性别/姓名”等具体字段替换成更通用占位词。

### 根目录工件

- `normal_dual_full_mix.jsonl`
  一份已经混合好的 JSONL 数据工件，用途偏下游训练输入，不是生成脚本。
- `README.md`
  当前目录交接文档。

## 什么时候该用哪些脚本

### 场景 1：我要从原始 normal / hard 对话重新做一遍主线

优先看：

- `raw/`
- `run_pipeline.py`
- `runs/`

### 场景 2：我要给 LlamaFactory 准备 SFT 数据

通常会用：

- `convert_to_dual_channel_format.py`
- `convert_meta_final_to_think_format.py`
- `prepare_llamafactory_sft.py`

### 场景 3：我要给 verl 准备单轮 RL 数据

通常会用：

- `prepare_single_turn_rl_dataset.py`
- `convert_opd_jsonl_to_parquet.py`

### 场景 4：我要下载通用 prompt pool 或 teacher rollout pool

通常会用：

- `download_opd_prompt_pool.py`
- `prepare_opd_dataset_from_pool.py`
- `download_teacher_rollout_prompt_pool.py`

## 重点子目录

- `raw/README.md`
  原始输入说明。
- `outputs/README.md`
  阶段性成品说明。
- `runs/README.md`
  单次流水线结果说明。
- `experiments/README.md`
  实验区说明。

## 接手建议

- 现在要维护主线时，优先相信：
  `run_pipeline.py`
  `runs/`
  `outputs/`
- 现在要理解历史口径时，再回头读：
  `clean_dialogue_text.py`
  `complete_assistant_cot.py`
  `extract_dialogue_entities.py`
  `experiments/`
- 别把 `experiments/` 里的文件当成生产真值。
