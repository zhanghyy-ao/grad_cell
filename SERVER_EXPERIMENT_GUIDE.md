# GradCell 服务器实验参考

本轮新增三条实验入口：完整梯度链验证、Chen2020 随机仿真数据生成、监督代理模型训练。正式结论使用 `pybamm` 后端；`toy` 只用于快速排查环境和代码。

## 环境安装

```bash
git clone https://github.com/zhanghyy-ao/grad_cell.git
cd grad_cell
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[physics,dev]"
```

建议记录 Python、PyTorch、PyBaMM、CPU、Git commit 和随机种子。

## 1. 完整梯度传递验证

```bash
python scripts/validate_end_to_end_gradients.py \
  --backend toy --samples 5 --directions 5 --eps 1e-4

for eps in 1e-3 3e-4 1e-4 3e-5; do
  python scripts/validate_end_to_end_gradients.py \
    --backend pybamm --samples 3 --directions 3 --eps "$eps" \
    --output "results/gradient_chain_${eps}.json"
done
```

该脚本覆盖 `latent -> hard-feasible decoder -> 1C/3C physics -> soft metrics -> scalar loss`。重点观察 median relative error 随步长是否形成稳定平台，不要只依据单个方向判断。

## 2. 生成 Chen2020 监督数据

先做冒烟测试：

```bash
python scripts/generate_supervised_data.py \
  --backend pybamm --samples 20 --batch-size 2 \
  --output data/chen2020_smoke.npz
```

确认失败率和耗时后再提交正式任务：

```bash
mkdir -p results/logs
nohup python scripts/generate_supervised_data.py \
  --backend pybamm --samples 10000 --batch-size 4 --seed 7 \
  --output data/chen2020_supervised.npz \
  > results/logs/generate_supervised.log 2>&1 &
tail -f results/logs/generate_supervised.log
```

NPZ 包含 7 维 latent、10 个解码后设计量、4 个监督标签和 JSON metadata。标签是 1C 比能量、3C 比功率、1C/3C 平滑最低电压。失败样本不会进入训练集，其索引写入同名 JSON。

## 3. 训练并检验监督可学习性

```bash
python scripts/train_supervised_surrogate.py \
  --data data/chen2020_supervised.npz \
  --epochs 500 --batch-size 128 --patience 50 \
  --output-dir results/supervised_seed7
```

脚本使用 70%/15%/15% train/validation/test 划分，归一化只使用训练集统计量，并按验证集 early stopping 选模型。输出包括：

- `metrics.json`：每个物理标签的测试 MAE、RMSE、R2；
- `history.json`：训练及验证曲线原始数据；
- `best_model.pt`：模型、结构、归一化参数和标签名。

测试集 R2 明显为正且 train/validation 曲线同步下降，说明当前随机设计分布下存在监督可学习性。若 R2 很低，应先扩大数据量、检查失败样本分布和预测散点图，不能直接解释为物理映射不可学习。

## 推荐正式实验矩阵

```bash
for seed in 7 17 27; do
  python scripts/generate_supervised_data.py \
    --backend pybamm --samples 10000 --seed "$seed" \
    --output "data/chen2020_seed${seed}.npz"
  python scripts/train_supervised_surrogate.py \
    --data "data/chen2020_seed${seed}.npz" --seed "$seed" \
    --output-dir "results/supervised_seed${seed}"
done
```

最终汇总三个种子的 R2/MAE 均值与标准差。当前实验验证的是 Chen2020 SPMe 标签在指定设计分布上的函数逼近能力，不等价于真实电芯实验泛化能力。
