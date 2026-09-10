# paper探索：Qwen Embedding + MLP 首次训练指南

## 1. 当前训练方式

第一版采用“冻结Qwen、缓存embedding、单独训练MLP、完整链路测试”：

```text
requirement_text
  → frozen Qwen
  → attention-mask mean pooling
  → cached float32 embedding
  → multi-head MLP
  → five-dimensional latent
```

Qwen在GPU上只执行一次前向提取。MLP也默认在GPU训练，但不重复加载Qwen。PyBaMM不进入本轮反向传播。

## 2. 服务器准备

```bash
cd /path/to/grad_cell
git pull origin main
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[physics,language-gpu,dev]'
```

检查数据和GPU：

```bash
test -s data/paper_explore_2000/dataset.jsonl
wc -l data/paper_explore_2000/dataset.jsonl

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16:", torch.cuda.is_bf16_supported())
    free, total = torch.cuda.mem_get_info()
    print("free/total GiB:", free / 1024**3, total / 1024**3)
PY
```

## 3. 一键训练

A100 40GB等可直接加载Qwen3-8B BF16的GPU：

```bash
CUDA_DEVICE=0 \
QWEN_MODEL_NAME="$PWD/models/Qwen3-8B" \
bash scripts/run_paper_explore_training_server.sh
```

显存较小时使用4-bit：

```bash
CUDA_DEVICE=0 \
QWEN_MODEL_NAME="$PWD/models/Qwen3-8B" \
LOAD_IN_4BIT=1 \
EMBED_BATCH_SIZE=2 \
bash scripts/run_paper_explore_training_server.sh
```

脚本默认读取仓库中的 `models/Qwen3-8B`，因此通常可以直接运行：

```bash
CUDA_DEVICE=0 bash scripts/run_paper_explore_training_server.sh
```

如果模型位于服务器上的其他目录：

```bash
QWEN_MODEL_NAME=/absolute/path/to/Qwen3-8B \
bash scripts/run_paper_explore_training_server.sh
```

## 4. 分步执行

### 4.1 提取Qwen embedding

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/src"

python scripts/extract_paper_explore_embeddings.py \
  --data data/paper_explore_2000/dataset.jsonl \
  --model-name "$PWD/models/Qwen3-8B" \
  --output data/paper_explore_2000/qwen3_8b_embeddings.npz \
  --pooling mean --max-length 512 \
  --batch-size 8 --dtype bfloat16 \
  --local-files-only
```

每个batch写入`.partial`缓存。中断后重复相同命令会从已经完成的task ID前缀继续；配置或数据哈希改变时拒绝错误续跑。

### 4.2 第一次smoke训练

```bash
python scripts/train_paper_explore_mlp.py \
  --data data/paper_explore_2000/dataset.jsonl \
  --embeddings data/paper_explore_2000/qwen3_8b_embeddings.npz \
  --output-dir results/paper_explore_2000/qwen_mlp/smoke \
  --hidden-dim 128 --num-blocks 1 \
  --epochs 5 --batch-size 16 \
  --max-train-samples 100 --seed 7 --device cuda
```

必须生成：

- `best_model.pt`
- `history.json`
- `metrics.json`
- `test_predictions.npz`

### 4.3 正式训练

```bash
for seed in 7 17 27; do
  python scripts/train_paper_explore_mlp.py \
    --data data/paper_explore_2000/dataset.jsonl \
    --embeddings data/paper_explore_2000/qwen3_8b_embeddings.npz \
    --output-dir "results/paper_explore_2000/qwen_mlp/seed_${seed}" \
    --hidden-dim 512 --num-blocks 2 --dropout 0.1 \
    --batch-size 32 --epochs 300 \
    --learning-rate 1e-3 --weight-decay 1e-4 \
    --early-stopping-patience 30 --gradient-clip 1.0 \
    --latent-weight 1.0 --performance-weight 0.3 \
    --feasibility-weight 0.3 --unsupported-weight 0.1 \
    --seed "$seed" --device cuda
done
```

## 5. MLP输出和损失

MLP包含：

- 五维latent头，输出经`4*tanh`限制在`[-4,4]`；
- 需求可行性二分类头；
- E1C、R5、R6辅助性能头；
- 循环、安全、成本、低温和成品几何多标签头。

可行任务的latent监督采用Top-5 teacher集合最小Smooth-L1。不可行或超范围任务不计算latent和性能损失，只训练可行性及未支持标签。

## 6. 当前测试指标

训练脚本自动在固定测试集输出：

- Top-K teacher latent MAE；
- 可行性accuracy、precision、recall和F1；
- 可行样本E1C、R5、R6辅助预测MAE；
- 各损失分量；
- 每个测试任务的latent、可行性概率和性能预测。

这些是离线监督指标。要证明设计有效，下一步还必须把`test_predictions.npz`中的latent经过同一个`DesignSpace`解码，并执行hard-cutoff SPMe，计算需求满足率和oracle regret。

## 7. 常见问题

### CUDA不可用

脚本默认要求CUDA。如果确实只在CPU训练MLP，可传`--device cpu`；Qwen3-8B embedding不建议在CPU提取。

### BF16不支持

改用：

```bash
--dtype float16
```

### Qwen显存不足

先降低`--batch-size`，再启用`--load-in-4bit`。不要在embedding阶段启用训练梯度。

### 数据改变后缓存不匹配

embedding缓存记录数据SHA-256。DeepSeek重新改写数据后，需要删除旧embedding或使用：

```bash
--overwrite
```

重新提取。
