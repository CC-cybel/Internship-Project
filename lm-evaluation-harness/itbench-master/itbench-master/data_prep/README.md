用于前期数据处理，直接使用Bench可以不用运行该脚本，如果要用的话，自己安装所需依赖。

## 黄金历史测试集生成 (Golden History Generation)

`dataset_generation/generate_golden_history.py` 脚本用于动态生成测试用例（基于用户提供的前置条件和随机选取的规则）。
它通过调用指定的 Judge 模型生成新的提示词，并与候选模型（Candidate Model）对战以生成多轮历史对话。

### 环境依赖 (.env 配置)

该脚本的运行强依赖于项目根目录下的 `.env` 文件中的配置，具体来说：

*   **`JUDGE_MODEL_NAME` (及 `JUDGE_API_KEY`, `JUDGE_API_BASE`)**:
    *   **作用 1**：充当 Prompt 工程师，将随机选中的规则无缝融合进基础系统提示词中，并决定当前测试对话的终止轮次 (`end_turn`)。
    *   **作用 2**：充当患者（User Simulator），根据对话上下文模拟真实患者的语气进行简短回复。
*   **`CANDIDATE_MODEL_NAME` (及 `CANDIDATE_API_KEY`, `CANDIDATE_API_BASE`)**:
    *   **作用**：充当医生/客服（Assistant），加载融合好规则的新系统提示词，与患者进行多轮对战回复。其输出会经过 `itbench` 中对应模型配置的前处理和后处理。

### 准备数据

在运行生成脚本前，需要确保在 `data_prep/data/{domain}/` 目录下已准备好以下两个输入文件：

1.  **`search_words.txt`**：用户搜索词列表文件，每行一个搜索词（如：“阳光抑郁的症状表现”）。脚本会从中随机选取作为每轮模拟对话的开场白。
2.  **`system_prompt_demo.txt`**：基础的系统提示词模板。大模型会以此为基础，将选中的测试规则无缝融合进去，形成最终对话使用的系统指令。

### 使用方法

可以在终端中运行该脚本，并使用命令行参数来控制生成的领域（`domain`）和生成的样本数量（`num_samples`）。

```bash
# 生成默认 psychiatry 领域的 10 条测试数据
python data_prep/dataset_generation/generate_golden_history.py

# 指定生成领域和生成数量
python data_prep/dataset_generation/generate_golden_history.py --domain psychiatry --num_samples 10
```

### 参数说明

*   `--domain` (str): 控制读取文件和输出文件的领域路径变量，默认是 `psychiatry`。
    脚本会去以下路径读取文件：
    *   搜索词文件：`data_prep/data/{domain}/search_words.txt`
    *   基础提示词：`data_prep/data/{domain}/system_prompt_demo.txt`
    生成结果也会保存在：`data_prep/data/{domain}/generated_golden_history.jsonl`
*   `--num_samples` (int): 要生成的样本数量，默认是 `10`。
