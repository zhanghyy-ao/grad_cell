# DeepSeek 2158 单设计物理引导训练

## 1. 目标

本版本不再训练 Top-K 多候选输出，而是将主线改为：

```text
自然语言电池描述
→ 冻结的本地 Qwen3-8B embedding
→ MLP 生成一套 7 参数设计
→ 冻结的可微 DFN 性能代理模型
→ 容量、能量、5C/6C 性能
→ 与自然语言对应的 verified_performance 计算 loss
→ 梯度穿过代理模型回传到单设计 MLP
```

代理模型的训练标签来自严格 DFN 仿真，但在单设计训练时代理模型的权重被冻结。因此：

- 代理模型参数不更新；
- 代理模型对输入设计的导数仍然保留；
- 性能 loss 可以反向更新设计 MLP；
- Qwen 仍然冻结，只产生文本 embedding。

## 2. 为什么使用性能代理模型

严格 DFN 每个 batch 都需要执行多次 PyBaMM 求解，直接放在几百个 epoch 中成本很高，并且电压截止事件会导致梯度不连续。

本版本先用 DFN archive 训练一个小型、可微的性能代理模型，用它提供稳定的反向梯度。训练结束后，再用真实 DFN 回放测试集设计。

## 3. 严格数据边界

代理模型先按 `physical_design_id` 对 2158 条语言记录去重，得到：

| Split | 独立 DFN 设计数 |
|---|---:|
| Train | 576 |
| Validation | 72 |
| Test | 72 |

代理模型只使用 576 个训练设计更新权重。验证和测试设计不参与代理模型训练，避免将测试 DFN 性能泄漏进物理梯度。

代理模型还有强制质量门：验证集全局 MAPE 默认不能超过 10%，任意单个性能字段的 MAPE 不能超过 20%。未通过时一键脚本会终止，不会继续使用不可信的性能梯度。

当前数据仍然只支持：

```text
parameter set = Chen2020
generation mode = regular
```

## 4. 代理模型输入输出

输入是七个参数的 log-multiplier：

1. Positive electrode porosity
2. Negative electrode porosity
3. Separator porosity
4. Positive electrode active material volume fraction
5. Negative electrode active material volume fraction
6. Positive particle diffusivity multiplier
7. Negative particle diffusivity multiplier

默认输出八个 log-performance：

1. `capacity_1c_ah`
2. `energy_1c_wh`
3. `capacity_5c_ah`
4. `energy_5c_wh`
5. `energy_retention_5c`
6. `capacity_6c_ah`
7. `energy_6c_wh`
8. `energy_retention_6c`

设计和性能都先在 log 空间中标准化，使 loss 更接近相对误差，避免数值尺度较大的性能指标支配训练。

## 5. 单设计训练 loss

总 loss 为：

\[
L=\lambda_dL_{design}+\lambda_pL_{performance}+\lambda_fL_{feasibility}
+\lambda_sL_{support}
\]

### 5.1 Design anchor

\[
L_{design}=\operatorname{SmoothL1}(\hat x,x_{teacher})
\]

它防止设计 MLP 为了降低代理模型 loss 而离开已知 DFN 数据分布。默认权重为 0.25。

### 5.2 Performance loss

\[
L_{performance}=\operatorname{SmoothL1}(S(\hat x),p_{target})
\]

`S` 是冻结的可微 DFN 性能代理模型。此 loss 的梯度穿过 `S` 返回设计 MLP。前 20 个 epoch 线性增加性能 loss 权重，避免随机初始设计立即进入代理模型边界。

### 5.3 Structural feasibility

训练中直接惩罚：

\[
\max(\varepsilon_p+\phi_p-1,0)^2+
\max(\varepsilon_n+\phi_n-1,0)^2
\]

同时设计头经过 `tanh` 约束在代理模型训练范围内，降低利用代理模型外推误差的风险。

### 5.4 Regular Mode support loss

当前 2158 条数据全部属于 Regular Mode，每个 teacher 主要只改变一个参数。为防止设计 MLP 同时大幅改变多个参数、进入代理模型没见过的交互区域，加入：

\[
L_{support}=\sum_j z_j^2-\max_j z_j^2
\]

其中 \(z_j\) 是第 \(j\) 个参数的 log-multiplier。它保留最大的一个参数变化，惩罚其余参数同时偏离标称值。当以后加入 Extreme Mode 多参数数据后，可将 `--support-weight` 降低或设为 0。

## 6. 一键训练

使用第三张 A100：

```bash
cd /data/yuwj/zhanghaoyu/grad_cell

CUDA_DEVICE=2 \
QWEN_MODEL_NAME="$PWD/models/Qwen3-8B" \
DEEPSEEK_SOURCE_DATA="$PWD/data/multiset_dfn_language/battery_description_modes_v3.jsonl" \
bash scripts/run_deepseek_2158_physics_guided_server.sh
```

默认顺序：

```text
严格筛选2158条记录
→ 提取本地Qwen embedding
→ 训练DFN性能代理模型
→ 检查代理模型输入梯度
→ seed 7/17/27训练单设计MLP
→ 检查性能loss是否真正回传到MLP
→ 输出测试集单设计
```

## 7. 真实 DFN 回放

默认不在训练后自动执行 216 条测试记录的严格 DFN 回放。如需一键训练后立即复核 seed 7：

```bash
CUDA_DEVICE=2 \
RUN_DFN_VERIFY=1 \
DFN_VERIFY_SEED=7 \
QWEN_MODEL_NAME="$PWD/models/Qwen3-8B" \
bash scripts/run_deepseek_2158_physics_guided_server.sh
```

DFN 复核会报告：

- 结构可行率；
- DFN 求解成功率；
- 八个性能指标的平均、中位和 P90 相对误差；
- 全部指标同时落在 5% 容差内的比例。

## 8. 任意自然语言推理

训练完成后：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/predict_battery_design_physics_guided.py \
  --description "该电池在1C下容量和能量较高，5C和6C下仍保持较好的能量输出。" \
  --model-name "$PWD/models/Qwen3-8B" \
  --design-checkpoint results/deepseek_2158_physics_guided/seed_7/best_model.pt \
  --surrogate-checkpoint results/deepseek_2158_physics_guided/dfn_surrogate/best_model.pt \
  --output results/deepseek_2158_physics_guided/example_design.json \
  --local-files-only
```

输出包含：

- 一套七参数 multiplier；
- 对应的物理参数值；
- 代理模型预测的八个性能指标；
- 结构可行性标记。

## 9. 输出文件

```text
results/deepseek_2158_physics_guided/
├── dfn_surrogate/
│   ├── best_model.pt
│   ├── history.json
│   └── metrics.json
├── seed_7/
│   ├── best_model.pt
│   ├── history.json
│   ├── metrics.json
│   ├── test_predictions.jsonl
│   ├── test_predictions_dfn.jsonl   # 可选
│   └── dfn_replay_metrics.json      # 可选
├── seed_17/
└── seed_27/
```

## 10. 验收顺序

1. 先查看 `dfn_surrogate/metrics.json`，确认 validation/test 的各性能 MAPE 和 R²；
2. 查看每个 seed 的 `gradient_qa`，必须满足 MLP gradient norm > 0 且 surrogate parameter gradient = false；
3. 比较三个 seed 的 surrogate performance MAPE；
4. 对最佳 seed 执行真实 DFN 回放；
5. 只有 DFN 回放指标达标后，才能声称“自然语言生成的设计在物理性能上贴合描述”。

代理模型指标不能替代真实 DFN 指标。
