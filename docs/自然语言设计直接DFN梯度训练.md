# 自然语言设计直接 DFN 梯度训练

## 训练链路

```text
电池自然语言描述
→ 冻结的 Qwen embedding
→ SingleDesignPhysicsMLP
→ 7 个设计参数
→ PyBaMM DFN：1C、5C、6C
→ 电压轨迹及 forward sensitivities
→ 容量、能量和倍率保持率 loss
→ 自定义 PyTorch backward
→ 更新设计 MLP
```

该路径不使用 DFN 性能代理网络。PyBaMM 在每个训练 batch 中真实求解三次 DFN，并返回电压轨迹对输入参数的 sensitivity。`DifferentiablePhysicsLayer` 在反向传播中计算向量－雅可比积，将性能损失的梯度传回设计 MLP。

Qwen 仍然只负责预先生成冻结 embedding。训练时更新的是设计 MLP；PyBaMM 没有可训练权重，但它提供物理梯度。

训练中的截止电压采用平滑 sigmoid 门计算容量和能量。原因是硬截止事件本身不可微；电压轨迹仍由真实 DFN 计算，反向梯度仍来自 PyBaMM sensitivity。最终验收应继续使用物理硬截止 DFN 重放，检查平滑训练指标与严格指标的偏差。

## 七个输入参数

```text
Positive electrode porosity
Negative electrode porosity
Separator porosity
Positive electrode active material volume fraction
Negative electrode active material volume fraction
Positive particle diffusivity multiplier
Negative particle diffusivity multiplier
```

每个倍率还会增加一个运行时电流输入，但电流不是 MLP 的设计输出。

## A100 运行

第三张 GPU：

```bash
CUDA_DEVICE=2 \
QWEN_MODEL_NAME="$PWD/models/Qwen3-8B" \
DFN_BATCH_SIZE=2 \
DFN_EPOCHS=10 \
bash scripts/run_deepseek_2158_direct_dfn_server.sh
```

只训练一个随机种子时，可以直接运行：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_battery_description_direct_dfn.py \
  --data data/multiset_dfn_language/deepseek_2158_strict_physics.jsonl \
  --embeddings data/multiset_dfn_language/deepseek_2158_physics_qwen_embeddings.npz \
  --output-dir results/deepseek_2158_direct_dfn/seed_7 \
  --batch-size 2 \
  --epochs 10 \
  --learning-rate 1e-4 \
  --time-points 151 \
  --rtol 1e-6 \
  --atol 1e-8 \
  --seed 7 \
  --device cuda
```

## 计算开销

DFN sensitivity 求解主要使用 CPU，A100 只负责 MLP 前向和反向。每条训练样本需要进行 1C、5C、6C 三次 DFN 求解，因此该训练会比代理模型训练慢很多。建议先进行小规模冒烟运行：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_battery_description_direct_dfn.py \
  --data data/multiset_dfn_language/deepseek_2158_strict_physics.jsonl \
  --embeddings data/multiset_dfn_language/deepseek_2158_physics_qwen_embeddings.npz \
  --output-dir results/direct_dfn_smoke \
  --batch-size 1 \
  --epochs 1 \
  --time-points 51 \
  --seed 7 \
  --device cuda
```

正式实验建议记录：

- `train_dfn_success_rate`；
- `validation_dfn_success_rate`；
- `train_dfn_runtime_s`；
- `gradient_qa.design_model_gradient_norm`；
- 1C、5C、6C 的性能误差；
- 不同随机种子之间的结果差异。

输出 checkpoint 中：

```json
{
  "gradient_source": "PyBaMM_DFN_forward_sensitivities",
  "uses_performance_surrogate": false
}
```

这两个字段用于区分直接 DFN 实验和原来的冻结代理实验。

## 应用级自然语言的真实 DFN 测试

直接 DFN checkpoint 使用以下脚本测试，不再传入代理模型：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/test_application_prompts_direct_dfn.py \
  --data data/application_prompts/application_battery_prompts_deepseek_s7.jsonl \
  --model-name "$PWD/models/Qwen3-8B" \
  --checkpoint results/deepseek_2158_direct_dfn/seed_7/best_model.pt \
  --output results/application_prompt_direct_dfn/predictions.jsonl \
  --report results/application_prompt_direct_dfn/report.json \
  --batch-size 1 \
  --local-files-only
```

该测试直接返回 `direct_dfn.solver_success`、真实 DFN 计算得到的八项性能和每条样本的求解时间。对于 70Ah LFP 等超出 Chen2020 单体范围的需求，DFN失败或容量误差较大是能力边界的真实反映，不能解释为已经满足产品需求。
