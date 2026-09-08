# GradCell-LM：Qwen3-8B结构化设计语言实验

_目标实验方案：Qwen 直接输出五个 K=0 latent token，并接受可微电池物理反馈。_

---

## 🎯 1. 研究问题

在固定 Chen2020 材料体系和现有五维结构空间中，Qwen3-8B 能否读取结构化的多目标
任务标签，生成接近参考 Pareto 前沿的量化设计 latent，并通过有限步可微 SPMe 物理细化
稳定改善设计？

第一阶段不增加温度、厚度或材料参数，避免语言表示变化与物理问题变化混杂。

## 🧾 2. 表示

输入采用固定语法：

```text
<TASK>
<OBJECTIVE_A>ENERGY_1C</OBJECTIVE_A>
<OBJECTIVE_B>MIN_RETENTION_5C_6C</OBJECTIVE_B>
<PREFERENCE><LEVEL_178></PREFERENCE>
</TASK>
<DESIGN>
```

输出为五个 256 档的 latent token：

```text
<U0><LEVEL_201></U0>
<U1><LEVEL_189></U1>
<U2><LEVEL_196></U2>
<U3><LEVEL_073></U3>
<U4><LEVEL_112></U4>
</DESIGN>
```

latent 截断在 `[-4, 4]` 后均匀量化。物理参数仍由现有 `DesignSpace` 解码，网络不能绕过
相体积分数、非活性相和 N/P 容量平衡约束。

## 🧠 3. 模型

- 基座：`Qwen/Qwen3-8B`；
- 第一阶段冻结基座，使用 LoRA 学习五个 level token 的结构化输出；
- 8GB显存采用NF4双重量化；高显存服务器可以使用BF16基座；
- 第二阶段通过 level logits 的连续期望值接收物理梯度，训练 LoRA，不再依赖外部连续 head；
- 全部阶段均冻结 Qwen3-8B 基座，不全量微调 8B 参数；
- 当前 GradCell refiner 接收同一个 task embedding，执行 K=1/K=3 物理细化。

## 🗂️ 4. 数据

从相同 reference front 和现有 K=0 checkpoint 生成训练标签。每个 preference 对应：

1. reference set 中 scalarized loss 最小的 oracle latent；
2. 现有 K=0 initializer latent，作为蒸馏教师；
3. top-k 与随机负例，供后续排序实验使用。

训练 preference 使用连续采样；测试必须同时包含标准 21 点、20 个离网格中点和至少一个
完整 held-out 区间。严禁随机拆分同一 preference 的重复序列来宣称语言泛化。

## 🔄 5. 训练阶段

### S1：结构化监督与蒸馏

不运行 PyBaMM。优化 level 交叉熵、连续 latent MSE，以及 token/continuous latent 一致性。

### S2：物理微调

冻结 Qwen 基座，训练阶段一 LoRA。Qwen 直接输出五个 K=0 level-token 分布，使用 softmax
期望值解码为连续 latent，再进入硬解码器和 SPMe。联合优化 level CE、K=0 物理 loss 与
小权重熵正则，使 PyBaMM 梯度真正进入 LLM 的 LoRA 参数。

### S3：K=3 物理细化

冻结 S2 的 Qwen K=0 生成器，训练 GradCell refiner 执行三次物理反馈修正，并从 K=0 至 K=3
候选中选择物理 loss 最低的设计。

## 📊 6. 基线和消融

必须比较：

1. 当前 Fourier-MLP K0；
2. Qwen 冻结 + continuous head K0；
3. Qwen 冻结 + level-token design K0；
4. Qwen continuous K1/K3；
5. 随机初始化 + exact-gradient K1/K3；
6. 去除结构标签、只输入普通数字文本；
7. 64/128/256/512 个量化档位；
8. 有无 token-continuous consistency loss。

## 📏 7. 指标

- hard-cutoff SPMe/DFN 成功率和约束满足率；
- scalarized regret、hypervolume 和 Pareto coverage；
- K=0 到 K=1/K=3 的 regret 改善；
- 设计标签格式正确率、每维 level accuracy 和 latent MAE；
- 相邻 preference 的性能单调性与最大 latent jump；
- 不同随机种子均值与标准差；
- 训练显存、wall-clock time 和每个设计的物理调用次数。

## ✅ 8. 第一阶段验收

Qwen K0 必须达到 100% 语法可解析和硬解码可行；其 held-out preference regret 应明显优于
随机设计。K1/K3 应在固定物理调用预算下稳定降低 Qwen K0 regret。只有达到这些条件后，
才扩展显式性能目标、温度和几何变量。
