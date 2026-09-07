# GradCell-LM 阶段一：结构化语义与合格 JSON 生成

## 目标

阶段一训练 Qwen3-8B 理解带标签的电池设计任务，并直接生成固定 schema 的材料/结构设计
JSON。本阶段不调用 PyBaMM，避免昂贵物理求解干扰基础语法与数值语义学习。

## 数据

输入是现有偏好任务的结构化 token：

```text
<TASK>
<MATERIAL_CONTEXT>
<MATERIAL_PARAMETER_SET>Chen2020</MATERIAL_PARAMETER_SET>
<MATERIAL_PROPERTIES_MODE>FIXED</MATERIAL_PROPERTIES_MODE>
</MATERIAL_CONTEXT>
<OPERATING_CONDITION>
<TEMPERATURE_K>298.15</TEMPERATURE_K>
<DISCHARGE_PROTOCOL>1C,5C,6C_CONSTANT_CURRENT</DISCHARGE_PROTOCOL>
</OPERATING_CONDITION>
<PERFORMANCE_REQUIREMENTS>
<OBJECTIVE_A>SPECIFIC_ENERGY_1C_WH_KG</OBJECTIVE_A>
<OBJECTIVE_B>ENERGY_RETENTION_5C_6C</OBJECTIVE_B>
<TARGET_ENERGY>155.0</TARGET_ENERGY>
<MIN_R5>0.50</MIN_R5>
<MIN_R6>0.44</MIN_R6>
</PERFORMANCE_REQUIREMENTS>
<PREFERENCE_PROFILE>
<PREFERENCE><LEVEL_178></PREFERENCE>
<ENERGY_PRIORITY>HIGH</ENERGY_PRIORITY>
<RATE_PRIORITY>LOW</RATE_PRIORITY>
</PREFERENCE_PROFILE>
<DESIGN_CONSTRAINTS>
<POSITIVE_POROSITY_RANGE>0.20,0.42</POSITIVE_POROSITY_RANGE>
<NEGATIVE_POROSITY_RANGE>0.20,0.42</NEGATIVE_POROSITY_RANGE>
<SEPARATOR_POROSITY_RANGE>0.35,0.60</SEPARATOR_POROSITY_RANGE>
<NP_RATIO_RANGE>1.02,1.25</NP_RATIO_RANGE>
<CAPACITY_BALANCE>ANALYTIC_NEGATIVE_ACTIVE_FRACTION</CAPACITY_BALANCE>
<FEASIBILITY_POLICY>HARD_FEASIBLE_DECODER</FEASIBILITY_POLICY>
</DESIGN_CONSTRAINTS>
<OUTPUT_CONTRACT>
<OUTPUT_SCHEMA>gradcell.material_design.v1</OUTPUT_SCHEMA>
<SELECTION_POLICY>MINIMUM_PHYSICS_LOSS</SELECTION_POLICY>
</OUTPUT_CONTRACT>
</TASK>
<DESIGN>
```

这些字段分为四类：材料体系与是否开放材料属性、实际仿真的工况、参与loss的性能需求、
以及由decoder强制执行的制造/容量平衡约束。当前温度和材料体系明确标为固定上下文；模型
不会把尚未接入PyBaMM的条件伪装成可优化变量。

监督输出是严格 JSON，并紧跟显式设计结束标签和Qwen EOS token。完整监督序列为
`JSON + \n</DESIGN> + EOS`，避免模型在生成一个对象后继续重复：

```json
{"schema":"gradcell.material_design.v1","design":{"positive_electrode_porosity":0.30,"negative_electrode_porosity":0.31,"separator_porosity":0.45,"positive_active_material_fraction":0.57,"negative_to_positive_capacity_ratio":1.10}}
```

```text
</DESIGN><EOS>
```

训练标签由现有 K=0 checkpoint 生成。`target_json`、teacher latent 和量化 level 同时保存，
从而让自回归输出与后续连续物理通道对齐。

## 模型与损失

Qwen 使用 LoRA；8GB显存环境可以使用4-bit NF4，A100 40GB直接使用BF16。总损失为：

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

## 🚀 A100 40GB运行指令

A100 40GB直接加载Qwen3-8B BF16权重，不传`--load-in-4bit`。首次运行先完成环境检查：

```bash
git pull origin main
cd grad_cell
source .venv/bin/activate

bash scripts/setup_language_gpu.sh
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("bf16:", torch.cuda.is_bf16_supported())
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
PY
```

若要使用服务器上的第三张物理GPU，将可见卡改为索引`2`。进程内它仍显示为`cuda:0`，这是CUDA的正常重编号行为：

```bash
export CUDA_VISIBLE_DEVICES=2
nvidia-smi -i 2
python -c 'import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.mem_get_info())'
```

确认K=0 checkpoint和参考前沿存在：

```bash
test -f results/gradcell_exploration/k0_s7/model.pt
test -f results/gradcell_exploration/reference/pareto_front_1c5c6c.npz
mkdir -p data/gradcell_lm results/gradcell_lm/logs
```

生成结构化训练集：

```bash
python scripts/generate_language_design_data.py \
  --checkpoint results/gradcell_exploration/k0_s7/model.pt \
  --reference-front results/gradcell_exploration/reference/pareto_front_1c5c6c.npz \
  --samples 4096 --bins 256 --seed 7 \
  --output data/gradcell_lm/k0_distillation_s7.jsonl

wc -l data/gradcell_lm/k0_distillation_s7.jsonl
head -n 1 data/gradcell_lm/k0_distillation_s7.jsonl
```

后台启动S1。A100 40GB先使用稳妥配置`batch-size=1`、梯度累积16次，不使用4-bit。当前实现已启用梯度检查点，并复用语言模型单次前向的输入末端隐藏状态计算设计头损失：

```bash
nohup python scripts/train_language_stage1_semantic.py \
  --data data/gradcell_lm/k0_distillation_s7.jsonl \
  --output-dir results/gradcell_lm/stage1_s7 \
  --model-name Qwen/Qwen3-8B \
  --epochs 3 --batch-size 1 --gradient-accumulation 16 \
  --learning-rate 1e-4 --max-length 512 --bins 256 \
  --lora-rank 16 --lora-alpha 32 \
  --schema-penalty-weight 2.0 \
  --latent-weight 1.0 --level-weight 0.25 \
  --feasibility-penalty-weight 10.0 --seed 7 \
  > results/gradcell_lm/logs/stage1_s7.log 2>&1 &

echo $! > results/gradcell_lm/logs/stage1_s7.pid
tail -f results/gradcell_lm/logs/stage1_s7.log
```

另开终端监控：

```bash
watch -n 2 nvidia-smi
```

若训练前显存不是接近空闲，先定位占用者：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ps -fp <PID>
```

只有确认PID属于自己且是已失效的旧训练任务时，才终止它：

```bash
kill <PID>
# 等待并再次检查；只有进程无法正常退出时才使用：
kill -9 <PID>
```

报错中若只剩几十MiB显存，`expandable_segments`不能代替释放占用显存的进程。若A100必须与其他任务共享，可改用4-bit加载以进一步降低模型权重和优化器占用：

```bash
nohup python scripts/train_language_stage1_semantic.py \
  --data data/gradcell_lm/k0_distillation_s7.jsonl \
  --output-dir results/gradcell_lm/stage1_s7_qlora \
  --model-name Qwen/Qwen3-8B --load-in-4bit \
  --epochs 3 --batch-size 1 --gradient-accumulation 16 \
  --learning-rate 1e-4 --max-length 512 --bins 256 \
  --lora-rank 16 --lora-alpha 32 \
  --schema-penalty-weight 2.0 \
  --latent-weight 1.0 --level-weight 0.25 \
  --feasibility-penalty-weight 10.0 --seed 7 \
  > results/gradcell_lm/logs/stage1_s7_qlora.log 2>&1 &
```

S1结束后检查产物：

```bash
test -d results/gradcell_lm/stage1_s7/qwen_adapter
test -f results/gradcell_lm/stage1_s7/language_heads.pt
test -f results/gradcell_lm/stage1_s7/metrics.json
cat results/gradcell_lm/stage1_s7/metrics.json
```

## 验收

- validation JSON严格解析率为100%；
- `validation_design_stop_rate`为100%，即生成显式`</DESIGN>`后结束；
- schema和字段顺序正确率为100%；
- JSON物理边界与耦合约束通过率为100%；
- held-out preference 的连续latent MSE显著优于常数设计。

阶段一只能说明Qwen学会了任务语义和设计格式，不能说明设计已经达到最佳物理性能。
