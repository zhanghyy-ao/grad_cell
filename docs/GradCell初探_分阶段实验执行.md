# GradCell 初探：分阶段实验执行

本实验不再把“预测电导率”当作主任务。主链路是：

```text
性能偏好 λ（1C 比能量/3C 容量保持率权衡）
→ GradCell 初始化器
→ 5 维可行电芯结构
→ PyBaMM SPMe 可微仿真
→ 1C 比能量、3C 容量保持率与多目标 Loss
→ 网络参数更新 / K 步物理梯度修正
→ 硬截止 SPMe 评估
→ 少量候选由 DFN 复核
```

所有命令都在仓库根目录执行。每次只运行一个阶段，确认结果后再进入下一阶段。

## 0. 环境检查

```bash
python scripts/run_gradcell_exploration.py --stage checks
```

若服务器没有 pytest，先安装开发依赖：

```bash
python -m pip install -e ".[physics,dev]"
```

## 1. 建立独立 SPMe 参考数据

```bash
python scripts/run_gradcell_exploration.py \
  --stage reference-data \
  --reference-samples 5000 \
  --reference-seed 101
```

这里用随机 latent 采样和真实电压截止生成 1C/3C 指标。该数据不训练 GradCell，
只作为独立随机搜索基线和目标量纲校准。先用 200 样本试跑，确认成功率和运行时间后再用
5000；正式比较时固定样本数和 seed。

## 2. 构建参考 Pareto 前沿

```bash
python scripts/run_gradcell_exploration.py \
  --stage reference-front \
  --reference-samples 5000 \
  --reference-seed 101
```

输出 `results/gradcell_exploration/reference/pareto_front.npz`。其中的 ideal/nadir
用于统一归一化比能量与容量保持率，避免某一个目标仅因数值尺度更大而主导 Loss。
脚本会拒绝少于 3 个非支配点的前沿，防止在不存在有效权衡时继续训练。
该阶段还会从低倍率标定容量与解析容量的比值估计 `capacity_multiplier`。训练、checkpoint
和评估统一读取这个系数，确保训练所称的 1C/3C 与参考数据使用相同容量基准。

## 3. 训练 K=0 初始化器

```bash
python scripts/run_gradcell_exploration.py \
  --stage train-k0 \
  --reference-samples 5000 \
  --training-steps 1000 \
  --batch-size 4 \
  --model-seed 7
```

K=0 只训练“偏好 → 设计”的初始化网络，但梯度仍通过 SPMe 物理层返回。它是判断
amortized initializer 本身是否有效的基础实验。

## 4. 用硬截止 SPMe 评估 K=0

```bash
python scripts/run_gradcell_exploration.py \
  --stage evaluate-k0 \
  --reference-samples 5000 \
  --preference-points 11 \
  --model-seed 7
```

主要观察：`success_rate`、`mean_scalarized_regret` 和
`candidate_beats_nominal_fraction`。训练期固定时域的软指标不能替代这里的硬截止结果。

## 5. 训练 K=3 Learned Refiner

```bash
python scripts/run_gradcell_exploration.py \
  --stage train-refiner \
  --reference-samples 5000 \
  --training-steps 1000 \
  --batch-size 4 \
  --refinement-steps 3 \
  --model-seed 7
```

## 6. 用硬截止 SPMe 评估 K=3

```bash
python scripts/run_gradcell_exploration.py \
  --stage evaluate-refiner \
  --reference-samples 5000 \
  --refinement-steps 3 \
  --preference-points 11 \
  --model-seed 7
```

只有 K=3 相比 K=0 在相同 seed、相同偏好点和相同硬截止口径下稳定降低 regret，
才能说明 PyBaMM 梯度修正有增益。

## 7. DFN 复核候选

```bash
python scripts/run_gradcell_exploration.py \
  --stage verify-dfn \
  --reference-samples 5000 \
  --refinement-steps 3 \
  --dfn-candidates 11 \
  --model-seed 7
```

DFN 在本阶段不是训练标签生成器，而是高保真验证器。重点查看 SPMe 与 DFN 的比能量、
3C 容量保持率相对误差和联合求解成功率。如果偏差很大，应先限定设计空间或改用多保真训练，
而不是直接扩大网络规模。

## 正式实验顺序

先完成 seed 7 的全链路，再将 `--model-seed` 依次改为 17、27。不要在看到单个 seed
后调整边界或样本筛选规则。最终报告 K=0 与 K=3 的均值、标准差、硬截止失败率，以及
SPMe→DFN 的模型偏差。
