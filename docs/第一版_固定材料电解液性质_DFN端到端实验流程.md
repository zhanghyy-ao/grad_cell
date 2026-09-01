# 第一版重构：电解液倍率的 PyBaMM DFN 纯物理端到端实验

## 1. 实验定义

当前正式支线不再训练“CALiSol 电导率监督回归 + DFN 辅助损失”，而是验证一个单参数
DFN 逆问题：

```text
CALiSol-23 配方特征
→ 标准化特征
→ MLP
→ bounded log_conductivity_scale
→ PyBaMM Chen2020 DFN
→ 电压轨迹
→ physics voltage loss
```

训练中 **Property Loss 已删除**。CALiSol-23 的实测电导率只在离线数据准备阶段用于
生成合成的 DFN 目标电压，以及在训练后作为诊断；它不参与反向传播、早停或 checkpoint
选择。因此本实验应称为“合成闭环下的单参数 DFN 逆问题”，不能称为真实电芯电压监督。

## 2. 输入、输出与物理接口

网络输入包括温度、盐浓度、38 种溶剂摩尔分数，以及盐类型和浓度单位的 one-hot 特征。
特征归一化统计量仅由 DOI-group train split 计算。

网络只输出一个无量纲参数：

```text
log_conductivity_scale ∈ [log(0.5), log(2.0)]
```

输出层使用 `tanh` 映射到上述硬边界，并直接传给 PyBaMM：

```text
kappa_DFN(c_e, T) = kappa_Chen2020(c_e, T) * exp(log_conductivity_scale)
```

它不是“标准化 log 电导率”，也不需要目标均值/标准差逆变换。第一版只学习一个整体
倍率，以降低多参数不可辨识风险。

## 3. 数据和划分

数据先按来源 DOI 分为 train/validation/test，再在各分区内独立选取稳定范围样本。
默认生成 32 个训练、8 个验证和 8 个测试物理样本：

1. 用实测电导率和 10 mS/cm 参考值计算离线 `log_scale`；
2. 用 PyBaMM Chen2020 DFN 生成 5 A、0–600 s 合成目标电压；
3. 训练仅读取 train 电压轨迹；
4. 早停和 checkpoint 仅使用 validation voltage RMSE；
5. test 只在最终评估时使用。

这些电压是同一 DFN 生成的模型域标签，不是 CALiSol-23 对应电芯的实验电压。600 s 是
公共梯度探针窗口，物理截止事件仍保留；提前截止或求解失败不能用零轨迹替代。

## 4. 损失与梯度

唯一训练目标为：

```text
L_total = Huber(V_DFN(predicted_log_scale) / 0.05,
                V_target / 0.05)
```

没有 `L_property`，也没有 `physics_weight`。PyBaMM IDAKLU 给出：

```text
J = d voltage(t) / d [log_conductivity_scale, current]
```

自定义 autograd backward 计算 `Jᵀ·dL/dV`，PyTorch 再把梯度传回 MLP。模型按验证集
`voltage_rmse_scaled` 早停，而不是按电导率误差选模。

## 5. 执行方法

安装：

```bash
pip install -e ".[physics,dev]"
```

一键运行真实 DFN：

```bash
python scripts/run_electrolyte_v1_pipeline.py \
  --physics-backend dfn \
  --physics-samples 32 \
  --physics-validation-samples 8 \
  --physics-test-samples 8 \
  --epochs 100 \
  --seed 7 \
  --output-dir results/electrolyte_physics_only_dfn_s7
```

服务器入口：

```bash
SEED=7 PHYSICS_SAMPLES=32 PHYSICS_VALIDATION_SAMPLES=8 \
PHYSICS_TEST_SAMPLES=8 EPOCHS=100 bash scripts/run_electrolyte_v1_server.sh
```

低成本软件闭环先使用 `--physics-backend analytic --epochs 3`。真实实验前必须执行 pytest、
Ruff 和 DFN 多 epsilon 方向导数检查。

## 6. 输出与验收

`metrics.json` 必须显示：

- `training_objective = physics_voltage_only`；
- `property_loss_enabled = false`；
- `physics_model = PyBaMM DFN`；
- `model_output = bounded_log_conductivity_scale`；
- train/validation/test 物理样本量；
- DFN success rate 和 voltage RMSE。

`conductivity_diagnostics_not_used_for_training` 只帮助判断合成逆问题是否恢复了构造目标，
不能作为训练目标或选模依据。

## 7. 结论边界与下一步

该流程成功只能说明“配方网络—有界参数—PyBaMM DFN—电压损失”的一阶端到端链路可用。
由于目标电压仍由同一 DFN 和实测电导率合成，它不能证明从真实电芯电压辨识出了真实电解液
性质。下一步应依次增加分层物理样本、多倍率、把样本温度真正传入 DFN、灵敏度过滤和真实
匹配电芯曲线；单参数稳定之前不扩展迁移数或扩散系数等多输出。
