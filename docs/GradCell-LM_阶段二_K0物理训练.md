# GradCell-LM 阶段二：K=0 可微物理训练

## 目标

阶段二加载阶段一的Qwen LoRA、projector和设计head，让同一Qwen隐藏表示直接产生连续
GradCell latent。设计经硬可行解码器进入1C、5C和6C PyBaMM-SPMe，以物理目标反向训练
语言到初始设计的映射。

本阶段对应原GradCell initializer的K=0：每个任务只提出一个设计，不执行refiner。

## 可微路径

```text
tagged task
→ frozen Qwen/LoRA
→ projector + continuous design head
→ hard-feasible decoder
→ PyBaMM SPMe sensitivities
→ energy/retention objective
→ Jᵀv
→ continuous head and projector
```

JSON自回归采样是离散操作，不能接收PyBaMM梯度。因此物理训练使用同一个Qwen隐藏状态的
连续head；训练完成后的连续设计被序列化为与阶段一完全相同的JSON schema。

## 冻结策略

- Qwen和阶段一LoRA：冻结；
- projector：训练；
- continuous design head：训练；
- JSON level head：冻结；
- GradCell refiner：冻结；
- PyBaMM：不含可训练权重，只提供forward sensitivity。

## 命令

```bash
python scripts/train_language_stage2_k0_physics.py \
  --stage1-dir results/gradcell_lm/stage1_s7 \
  --reference-front results/gradcell_exploration/reference/pareto_front_1c5c6c.npz \
  --output results/gradcell_lm/stage2_k0_s7.pt \
  --backend pybamm --physics-model SPMe --load-in-4bit \
  --steps 1000 --batch-size 1
```

## 🚀 A100 40GB运行指令

本阶段Qwen和LoRA冻结，A100直接使用BF16，不传`--load-in-4bit`。PyBaMM通常运行在CPU，
因此建议先检查CPU核心数和内存：

```bash
cd grad_cell
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

nvidia-smi
lscpu | head -n 20
free -h
```

检查S1输入和参考前沿：

```bash
test -d results/gradcell_lm/stage1_s7/qwen_adapter
test -f results/gradcell_lm/stage1_s7/language_heads.pt
test -f results/gradcell_exploration/reference/pareto_front_1c5c6c.npz
mkdir -p results/gradcell_lm/logs
```

建议先运行10步SPMe短任务，确认PyBaMM反向链和checkpoint写入正常：

```bash
python scripts/train_language_stage2_k0_physics.py \
  --stage1-dir results/gradcell_lm/stage1_s7 \
  --reference-front results/gradcell_exploration/reference/pareto_front_1c5c6c.npz \
  --output results/gradcell_lm/stage2_k0_smoke_s7.pt \
  --backend pybamm --physics-model SPMe \
  --steps 10 --batch-size 1 --learning-rate 3e-5 \
  --validation-interval 5
```

短任务通过后启动正式S2：

```bash
nohup python scripts/train_language_stage2_k0_physics.py \
  --stage1-dir results/gradcell_lm/stage1_s7 \
  --reference-front results/gradcell_exploration/reference/pareto_front_1c5c6c.npz \
  --output results/gradcell_lm/stage2_k0_s7.pt \
  --backend pybamm --physics-model SPMe \
  --steps 1000 --batch-size 1 --learning-rate 3e-5 \
  --validation-interval 25 \
  > results/gradcell_lm/logs/stage2_k0_s7.log 2>&1 &

echo $! > results/gradcell_lm/logs/stage2_k0_s7.pid
tail -f results/gradcell_lm/logs/stage2_k0_s7.log
```

运行中同时观察GPU和进程：

```bash
watch -n 2 nvidia-smi
ps -fp "$(cat results/gradcell_lm/logs/stage2_k0_s7.pid)"
```

完成后检查checkpoint：

```bash
test -s results/gradcell_lm/stage2_k0_s7.pt
ls -lh results/gradcell_lm/stage2_k0_s7.pt
tail -n 20 results/gradcell_lm/logs/stage2_k0_s7.log
```

## 验收与对照

- 三倍率求解成功率和硬约束满足率；
- 21点及离网格偏好的scalarized regret；
- 与原Fourier-MLP K=0、阶段一未物理微调模型、随机设计比较；
- preference到能量/保持率的单调性；
- DFN只用于最终复核，不能用于训练结论替代真实数据。
