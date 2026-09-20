# DeepSeek 2158 条数据：严格划分与 Top-K 训练

## 1. 实验目标

本实验只使用已经通过脚本验收的 2158 条 `deepseek-v4-flash` 自然语言记录，训练从电池性能描述到多候选参数设计的逆模型。

这里的 K 指模型同时输出 K 个候选设计，不是 K-fold 交叉验证。当前每条数据最多包含：

```text
1 个主 teacher design + 5 个 alternative teacher designs = 6 个有效标签
```

因此默认使用 `K=6`，避免丢弃任何已验证的一对多标签。

实际有效标签数分布为：

| 每条记录的可行设计数 | 记录数 |
|---:|---:|
| 1 | 1924 |
| 2 | 30 |
| 3 | 33 |
| 4 | 39 |
| 5 | 33 |
| 6 | 99 |

其中 234 条至少含有两个经 DFN 独立仿真的可替代设计，是 Top-K 训练的核心子集。

## 2. 数据限制

这 2158 条数据全部来自：

```text
parameter set: Chen2020
generation mode: regular
```

因此实验只能回答“模型能否在 Chen2020 Regular Mode 内部从自然语言恢复一组或多组参数设计”。它不能证明：

- Extreme Mode 泛化；
- 跨 parameter set 泛化；
- 任意自然语言输入泛化；
- 所有输出候选都满足真实电池实验约束。

2158 条记录对应 720 个物理设计，其中 719 个设计有完整的三个语言版本，最后一个设计只有一个已成功生成的 DeepSeek 版本。为了严格保留用户指定的 2158 条，脚本保留这个不完整 family，但保证它只属于一个 split。

## 3. 严格数据划分

脚本：

```text
scripts/prepare_deepseek_topk_dataset.py
```

过滤条件：

1. `provenance.language_source == deepseek-v4-flash`；
2. 不存在 `provenance.language_error`；
3. 不存在 `quality_flags`；
4. 数据量必须严格等于 2158，数量变化时立即报错；
5. teacher、alternative count 和 equivalence group 必须完整。

划分单元不是单条文本，而是：

```text
performance-equivalence group
```

这会同时保证：

- 同一物理设计的语言改写不跨 split；
- 同一性能等价组不跨 split；
- 一对多替代设计附近的样本不会泄漏到测试集；
- 划分可由 seed 完全复现。

在当前 2158 条文件上，seed 7 的实际划分为：

| Split | 记录数 | 物理设计数 | 等价组数 |
|---|---:|---:|---:|
| Train | 1726 | 576 | 17 |
| Validation | 216 | 72 | 4 |
| Test | 216 | 72 | 4 |

物理设计泄漏和等价组泄漏均为 0。

单独执行划分：

```bash
python scripts/prepare_deepseek_topk_dataset.py \
  --input data/multiset_dfn_language/battery_description_modes_v3.jsonl \
  --output data/multiset_dfn_language/deepseek_2158_strict_topk.jsonl \
  --manifest data/multiset_dfn_language/deepseek_2158_strict_topk_manifest.json \
  --language-source deepseek-v4-flash \
  --expected-records 2158 \
  --train-ratio 0.80 \
  --validation-ratio 0.10 \
  --test-ratio 0.10 \
  --seed 7
```

## 4. 模型结构

```text
电池自然语言描述
→ 冻结的本地 Qwen3-8B
→ mean pooled embedding
→ Residual MLP
→ K × 7 个 log-multiplier
```

Qwen3-8B 只负责产生冻结的文本表示，不参与 MLP 反向传播。MLP 输出 K 个候选，每个候选包含七个参数倍率。

## 5. Top-K 集合损失

令模型候选为：

\[
\hat X=\{\hat x_1,\ldots,\hat x_K\}
\]

有效 teacher 集合为：

\[
Y=\{y_1,\ldots,y_M\},\quad 1\le M\le6
\]

候选和 teacher 之间使用逐参数 Smooth-L1 距离。覆盖损失为：

\[
L_{coverage}=\frac{1}{M}\sum_{m=1}^{M}
\min_{k}d(\hat x_k,y_m)
\]

它要求每个有效 teacher 至少被一个候选覆盖。

候选精度损失为：

\[
L_{precision}=\frac{1}{K}\sum_{k=1}^{K}
\min_m d(\hat x_k,y_m)
\]

它防止多余候选远离所有已知可行标签。

总损失：

\[
L=L_{coverage}+\lambda L_{precision}
\]

默认 `lambda=0.25`。对于只有一个 teacher 的记录，候选允许收敛到同一区域；只有在存在多个已验证标签时，coverage loss 才要求候选覆盖不同解。这比无依据地强制所有样本产生多样设计更可靠。

## 6. 评价指标

训练和测试报告：

- `primary_best_of_k_log_mae`：主标签与最佳候选的 log-multiplier MAE；
- `target_coverage_log_mae`：所有有效标签到最近候选的 MAE；
- `candidate_precision_log_mae`：每个候选到最近有效标签的 MAE；
- `ambiguous_target_coverage_log_mae`：只在一对多记录上计算的覆盖 MAE；
- `candidate_pairwise_log_rms`：候选之间的平均 log-RMS 距离。

正式判断 Top-K 是否有价值时，应将它与原单输出 MLP 比较：

```text
Best-of-K error
Set coverage error
DFN回放成功率
候选去重后的有效候选数
```

## 7. A100 40 GB 一键运行

本地 Qwen 模型默认路径：

```text
$PWD/models/Qwen3-8B
```

第 3 张物理 GPU 卡运行：

```bash
CUDA_DEVICE=2 \
QWEN_MODEL_NAME="$PWD/models/Qwen3-8B" \
bash scripts/run_deepseek_2158_topk_training_server.sh
```

如果服务器按 0 开始编号，则第 3 张卡对应 `CUDA_DEVICE=2`。脚本设置 `CUDA_VISIBLE_DEVICES=2` 后，训练程序内部仍使用 `cuda:0`，这是正常现象。

默认过程：

```text
过滤2158条成功数据
→ 严格group-aware 80/10/10划分
→ 本地Qwen3-8B提取embedding
→ K=6 Top-K MLP训练
→ seed 7/17/27重复实验
→ 保存每个seed的最佳模型和测试指标
```

输出文件：

```text
data/multiset_dfn_language/deepseek_2158_strict_topk.jsonl
data/multiset_dfn_language/deepseek_2158_strict_topk_manifest.json
data/multiset_dfn_language/deepseek_2158_qwen_embeddings.npz
results/deepseek_2158_topk/seed_7/
results/deepseek_2158_topk/seed_17/
results/deepseek_2158_topk/seed_27/
```

每个 seed 目录包括：

```text
best_model.pt
history.json
metrics.json
```

## 8. 可调整环境变量

```bash
export CUDA_DEVICE=2
export TOPK_CANDIDATE_COUNT=6
export EMBED_BATCH_SIZE=8
export MLP_BATCH_SIZE=64
export DEEPSEEK_EXPECTED_RECORDS=2158
```

若 embedding 阶段显存不足，将 `EMBED_BATCH_SIZE` 降为 4 或 2。MLP 本身很小，通常不会占满 A100。

如果严格数据文件发生变化，已有 embedding 的 SHA 校验会拒绝训练。此时应删除旧的：

```bash
rm data/multiset_dfn_language/deepseek_2158_qwen_embeddings.npz
```

然后重新运行一键脚本。

## 9. 验收条件

数据阶段：

```text
records = 2158
train/validation/test = 1726/216/216
physical_design_leakage = 0
group_leakage = 0
parameter_sets = [Chen2020]
modes = [regular]
```

训练阶段至少满足：

1. 三个 seed 均正常结束；
2. validation loss 在早停前下降；
3. 测试集 `primary_best_of_k_log_mae` 优于单输出基线；
4. 一对多子集的 `ambiguous_target_coverage_log_mae` 明显下降；
5. 最终候选还需要通过 DFN 回放，不能仅根据标签距离宣称物理有效。
