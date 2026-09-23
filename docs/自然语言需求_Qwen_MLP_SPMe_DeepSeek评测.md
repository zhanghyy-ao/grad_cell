# 自然语言需求到SPMe仿真与DeepSeek验收

该流程用于评估“DeepSeek生成的用户需求 → Qwen语义表示 → MLP电池设计 → PyBaMM SPMe仿真 → DeepSeek逐项验收”。它是推理与验收流程，不会继续训练Qwen或MLP。

## 证据边界

当前MLP只输出Chen2020体系下的7个微观设计参数。SPMe给出1C、5C和6C下的容量、能量与能量保持率，但不能验证产品级额定容量、化学体系选择、封装、尺寸、质量、循环寿命、安全、成本或低温性能。DeepSeek必须把缺少直接证据的要求标为`not_evaluable`，不能凭常识判定满足。

MLP原始输出与进入SPMe的输出会同时保存。如果正/负极的“孔隙率+活性材料体积分数”超过1，脚本会按比例投影到`1 - 1e-4`，并记录投影幅度。最终报告分别给出原始可行率、投影后可行率和投影发生率，防止投影掩盖模型自身的越界问题。

## A100运行

先在`.env`中设置：

```bash
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

指定数据、第二阶段模型和GPU后运行：

```bash
CUDA_DEVICE=2 \
QWEN_MODEL_NAME="$PWD/models/Qwen3-8B" \
APPLICATION_PROMPT_DATA=data/application_prompts/application_battery_prompts_deepseek_s7.jsonl \
DESIGN_CHECKPOINT=results/deepseek_2158_three_stage/seed_7/stage2_spme_online/best_model.pt \
APPLICATION_EVAL_OUTPUT_DIR=results/application_requirements_spme_deepseek \
bash scripts/run_application_requirements_spme_deepseek_server.sh
```

首次建议只检查4条：

```bash
APPLICATION_EVAL_MAX_RECORDS=4 \
CUDA_DEVICE=2 \
QWEN_MODEL_NAME="$PWD/models/Qwen3-8B" \
APPLICATION_PROMPT_DATA=data/application_prompts/application_battery_prompts_deepseek_s7.jsonl \
DESIGN_CHECKPOINT=results/deepseek_2158_three_stage/seed_7/stage2_spme_online/best_model.pt \
bash scripts/run_application_requirements_spme_deepseek_server.sh
```

逐条结果保存到`evaluations.jsonl`，汇总保存到`report.json`。重复运行时，已经成功的DeepSeek判定会被复用，避免重复消耗API。

## 如何阅读结果

- `raw_structural_feasibility_rate`：MLP原始输出自身的物理可行率。
- `projection_rate`：需要修正后才能送入SPMe的比例。
- `spme_success_rate`：1C、5C和6C三种工况全部成功的比例。
- `deepseek_success_rate`：评判API成功返回合格JSON的比例。
- `verdict_counts`：整体判定分布。
- `requirement_status_counts`：逐项需求的满足、不满足和不可验证数量。

对于70Ah、尺寸、循环寿命等应用级需求，大量`not_evaluable`是正确的能力边界报告，不是API失败。若要真正验收这些需求，需要增加电芯尺度/并联缩放模型、热模型、老化模型、结构与质量模型以及成本和安全约束。
