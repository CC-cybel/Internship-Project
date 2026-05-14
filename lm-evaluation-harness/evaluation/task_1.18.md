模型评估任务

## 要求
1. 任务开始前，把你对任务的理解放出来，让我确认，应包括要执行的脚本，我确认后可以开始，一旦确认后，你就运行到结束，中间不用我确认。
2. 任务开始后，要每隔几分钟汇报进度，现在是哪个阶段，预计还剩多少时间
3. 完成后，写一份任务报告，包括任务的执行时间，任务的结果，任务的分析等，报告放在 output_dir 中
4. 报告中要包含以下内容：
    - 评估任务的执行时间
    - 评估任务的结果
    - 评估任务的分析
    - 评估任务时的命令（方便我复现）
    - 各个评估的原始结果数据（用绝对路径，方便核对）
5. GPU只用用4,5,6,7中可用的GPU，没有资源的话，告诉我
6. 过程中重要节点或者需要我协助的发邮件通知我

## 任务

评估以下模型：

/data/yezj/trained_model/llama_factory_抖音口腔科1.18_lora_qwen3_0.6b_dpo

output_dir为 ./output/{model_name}

### 模型启动方式

```bash
gpu_id=6
model_name=llama_factory_开口全科1.2_lora_qwen2.5_0.5b （名字取模型路径中的模型名称）
model_path=/data/yezj/gitlab/llama_factory_forge/ckpt/开口全科1.2/qwen2.5_0.5b_it/lora/sft/llama_factory_开口全科1.2_lora_qwen2.5_0.5b
port=18585  （使用端口前先检查是否被占用，被占用的话，换一个）

conda activate vllm_0.17.0
CUDA_VISIBLE_DEVICES=${gpu_id} python -m vllm.entrypoints.openai.api_server \
    --served-model-name ${model_name} \
    --model ${model_path} \
    --port ${port} \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 4096 \
    --dtype float16
```

其中 `CUDA_VISIBLE_DEVICES=${gpu_id}` 表示使用指定的 GPU 进行测试，可根据实际情况修改，一个GPU即可
`--model` 表示模型路径，需要和合并模型时的路径保持一致
`--served-model-name` 和 `--model` 中的模型名称需要保持一致

### 评估任务

读取 evaluation/README.md 中的评估任务说明
此次评估 

- IF-EVAl: 指令跟随，使用 lm-evaluation-harness （报告中应包括 inst_level_strict_acc inst_level_loose_acc prompt_level_strict_acc prompt_level_loose_acc） 
- MMLU: 通用知识，使用 lm-evaluation-harness （zero shot就行）
- GSM8K: 数学推理，使用 lm-evaluation-harness
- HumanEval: 代码生成，使用 lm-evaluation-harness
- TruthfulQA: 真实性/幻觉，使用 lm-evaluation-harness （报告中应包括 mc1 acc mc2 acc gen bleu_acc）

lm-evaluation-harness环境已经安装了，直接激活即可：
conda activate lm-evaluation-harness