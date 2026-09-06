# GradCell-LM 阶段一：结构化语义与合格 JSON 生成

## 目标

阶段一训练 Qwen3-8B 理解带标签的电池设计任务，并直接生成固定 schema 的材料/结构设计
JSON。本阶段不调用 PyBaMM，避免昂贵物理求解干扰基础语法与数值语义学习。

## 数据

输入是现有偏好任务的结构化 token：

```text
<TASK>
<OBJECTIVE_A>ENERGY_1C</OBJECTIVE_A>
<OBJECTIVE_B>MIN_RETENTION_5C_6C</OBJECTIVE_B>
<PREFERENCE><LEVEL_178></PREFERENCE>
</TASK>
<DESIGN>
```

监督输出是严格 JSON：

```json
{"schema":"gradcell.material_design.v1","design":{"positive_electrode_porosity":0.30,"negative_electrode_porosity":0.31,"separator_porosity":0.45,"positive_active_material_fraction":0.57,"negative_to_positive_capacity_ratio":1.10}}
```

训练标签由现有 K=0 checkpoint 生成。`target_json`、teacher latent 和量化 level 同时保存，
从而让自回归输出与后续连续物理通道对齐。

## 模型与损失

Qwen 使用 4-bit NF4 和 LoRA。总损失为：

```text
L1 = L_causal_json
   + 2.0 L_schema
   + 1.0 L_latent
   + 0.25 L_level
   + 10.0 L_feasibility
```

`L_schema` 对 JSON 括号、引号、键名、冒号和逗号对应 token 额外加权；`L_latent` 和
`L_level` 使同一隐藏状态能够恢复教师设计；`L_feasibility` 检查孔隙率、活性相、N/P 和
容量平衡约束。最终部署还会把物理解码后的设计重新序列化，所以不会把非法自由文本当作
正式设计。

## 命令

```bash
python scripts/generate_language_design_data.py \
  --checkpoint results/gradcell_exploration/k0_s7/model.pt \
  --reference-front results/gradcell_exploration/reference/pareto_front_1c5c6c.npz \
  --samples 4096 --output data/gradcell_lm/k0_distillation_s7.jsonl

python scripts/train_language_stage1_semantic.py \
  --data data/gradcell_lm/k0_distillation_s7.jsonl \
  --output-dir results/gradcell_lm/stage1_s7 \
  --load-in-4bit
```

## 验收

- validation JSON严格解析率为100%；
- schema和字段顺序正确率为100%；
- JSON物理边界与耦合约束通过率为100%；
- held-out preference 的连续latent MSE显著优于常数设计。

阶段一只能说明Qwen学会了任务语义和设计格式，不能说明设计已经达到最佳物理性能。
