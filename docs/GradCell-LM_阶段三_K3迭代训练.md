# GradCell-LM 阶段三：K=3 物理迭代与最佳设计选择

## 目标

阶段三加载阶段二K=0模型并冻结语言到初始设计的映射，只训练GradCell的
`DiagonalPhysicsRefiner`。每个任务依次产生：

```text
u0 → physics/gradient → u1 → physics/gradient → u2 → physics/gradient → u3
```

最终不是无条件采用`u3`，而是在`u0...u3`中按每个样本的物理loss选择最佳设计，并输出
该设计的严格JSON、1C比能量、5C/6C保持率、solver status和被选中的step。

## 冻结策略

- Qwen、LoRA、projector、continuous head、level head：全部冻结；
- K=0初始设计保持不变；
- 只训练physics refiner；
- PyBaMM sensitivity提供每一步的设计梯度，不需要二阶sensitivity。

## 损失

```text
L3 = min(L(u0), L(u1), L(u2), L(u3))
   + 0.1 mean(L(u0), L(u1), L(u2))
   + 0.1 monotonic_penalty
   + 1e-3 step_penalty
```

`best-step`目标允许refiner探索，同时monotonic penalty抑制明显退化。每一步更新仍受现有
L2范数上限约束。

## 训练

```bash
python scripts/train_language_stage3_k3_refiner.py \
  --stage1-dir results/gradcell_lm/stage1_s7 \
  --stage2-checkpoint results/gradcell_lm/stage2_k0_s7.pt \
  --reference-front results/gradcell_exploration/reference/pareto_front_1c5c6c.npz \
  --output results/gradcell_lm/stage3_k3_s7.pt \
  --backend pybamm --physics-model SPMe --load-in-4bit \
  --steps 300 --refinement-steps 3
```

## 输出设计

```bash
python scripts/generate_language_material_design.py \
  --stage1-dir results/gradcell_lm/stage1_s7 \
  --checkpoint results/gradcell_lm/stage3_k3_s7.pt \
  --reference-front results/gradcell_exploration/reference/pareto_front_1c5c6c.npz \
  --preference 0.7 --refinement-steps 3 --load-in-4bit
```

## 验收

- K=3相对同一模型K=0的平均/中位regret改善；
- 最佳step分布，避免所有样本机械选择同一步；
- 每一步求解成功率、退化率和更新范数；
- 相同物理调用预算下与exact-gradient、原GradCell refiner比较；
- 最终JSON严格可解析且其中数值与被选物理解码设计一致。
