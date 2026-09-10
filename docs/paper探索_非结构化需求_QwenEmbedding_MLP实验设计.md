# paper探索：非结构化电芯需求的 Qwen Embedding–MLP 设计实验

> 实验名称：`paper探索`；实验代号：`paper_explore_v1`；版本日期：2026-09-09。
> 当前定位：论文方向可行性探索，不宣称真实 70 Ah LFP 软包已经完成工程定型。

## 1. 研究问题

在当前 Chen2020 固定材料体系和五维可行结构空间内，冻结的 Qwen 是否能够把非结构化电芯需求编码为有用的语义表示，使一个小型多头 MLP 预测接近物理 oracle 的可行设计，并在未见过的需求表达和偏好组合上保持较低 regret？

本实验刻意不预测当前缺少可信标签的循环寿命、安全、成本和低温数值。这些内容可以出现在需求文本中，作为场景和偏好特征，但只用于考察语言表示能否识别它们，不把 DeepSeek 生成的数值当成监督真值。

## 2. 核心假设

### H1：非结构化语义可学习

Qwen embedding + MLP 在测试集上的五维 latent 误差、物理 regret 和需求满足率优于常数设计与随机设计。

### H2：Qwen embedding 提供额外价值

在同一数据划分下，Qwen embedding + MLP 在语言改写、参数顺序变化和单位变化测试上优于 TF-IDF/关键词特征 + MLP。

### H3：多任务辅助监督有益

在预测 latent 的基础上加入可行性和 SPMe 性能辅助头，可以降低 hard-cutoff SPMe regret 或提高需求满足率；若没有改善，应保留更简单的 latent-only 模型。

### H4：结构化字段仍是性能上界基线

标准化需求字段 + MLP 应作为输入信息无损时的上界参考。非结构化模型的目标不是必然超过它，而是尽量缩小差距并获得语言鲁棒性。

## 3. 实验边界

### 3.1 当前固定项

- 材料参数集：Chen2020。
- 温度：298.15 K。
- 正负极固相扩散系数乘子：1.0。
- 物理训练标签：PyBaMM SPMe。
- 最终复核：hard-cutoff SPMe；代表点可增加 DFN 复核。
- 设计空间：现有五维硬可行 latent/decoder。

### 3.2 当前不作为数值监督目标的字段

- 3000 次循环后的容量保持率。
- 热安全或滥用安全指标。
- 原材料和制造成本。
- 低温容量、低温充电能力和析锂风险。
- 70 Ah 软包的成品级层数、封边、极耳和 BOM 质量。

这些字段在 v1 中保存为 `context_tags`、`priority_tags` 或 `unverified_requirements`。模型可以识别其存在，但不输出未经验证的性能承诺。

## 4. 任务输入与输出

### 4.1 模型输入

输入为自然语言需求，例如：

> 希望设计一款偏储能用途的电芯，优先考虑能量密度，同时要求 5C 和 6C 的能量保持率分别不低于 50% 和 44%。材料体系先固定，工作温度为 25℃；如果循环寿命或低温能力目前无法可靠计算，请明确标记为待验证。

文本允许出现：

- 不同语序和表达风格；
- 百分数与小数混用；
- 摄氏度与开尔文转换；
- 明确偏好、模糊偏好和多目标权衡；
- 当前模型无法验证的工程要求；
- 缺失、冲突或超出当前设计域的要求。

### 4.2 标准化需求底稿

模型输入是非结构化文本，但每条样本必须保存标准化真值：

```json
task_id
requirement_family_id
requirement_text
preference_energy
preference_high_rate
target_energy_wh_kg
min_retention_5c
min_retention_6c
temperature_k
material_parameter_set
application_tag
priority_tags
unverified_requirements
requirement_feasible
infeasible_reasons
```

### 4.3 MLP 主要输出

主输出为五维连续 latent：

```text
u0: 正极孔隙率 latent
u1: 负极孔隙率 latent
u2: 隔膜孔隙率 latent
u3: 正极活性材料体积分数相关 latent
u4: N/P latent
```

预测 latent 经过现有 `DesignSpace` 硬解码器，确定性得到：

- 正极孔隙率；
- 负极孔隙率；
- 隔膜孔隙率；
- 正极活性材料体积分数；
- N/P；
- 负极活性材料体积分数；
- 标称容量 proxy；
- stack-level 质量 proxy。

结构参数、容量和质量以解码器输出为正式结果，不再由独立回归头决定最终数值。

### 4.4 辅助输出

多头 MLP 可以额外预测：

- `requirement_feasible`：当前设计域能否满足已验证的数值约束；
- `energy_1c_wh_kg`；
- `retention_5c`；
- `retention_6c`；
- `unsupported_requirement_flags`：是否包含循环、安全、成本、低温或成品几何等未建模要求。

辅助性能头只服务于表示学习和快速估计。正式评价一律对解码后的设计重新运行 hard-cutoff SPMe，不采用辅助头数值替代物理评价。

## 5. 可行性的定义

本实验区分两种可行性：

1. **结构可行性**：由硬解码器保证孔隙率、非活性相和 N/P 容量平衡合法。正常情况下该标签始终为真，不适合单独训练分类器。
2. **需求可行性**：在当前固定材料、设计域和 SPMe 定义下，是否存在设计满足用户提出的已建模目标。这是分类头真正预测的标签。

需求可行标签由 oracle 搜索结果确定：若参考 archive 或连续优化结果中存在同时满足目标的设计，则标记为可行；否则标记为不可行，并保存原因，例如：

- `retention_5c_above_observed_maximum`；
- `retention_6c_above_observed_maximum`；
- `energy_and_rate_jointly_infeasible`；
- `temperature_out_of_model_scope`；
- `material_system_out_of_scope`；
- `unsupported_lifetime_requirement`。

超出模型范围不等同于物理上绝对不可行，应输出“当前模型不可验证”，不能输出“该电芯不可能实现”。

## 6. 数据生成设计

### 6.1 数据规模

首轮生成 500 个独立需求族。每个需求族只保留一个主文本用于正式 500 条训练实验；额外语言改写单独形成鲁棒性测试集，不计入主训练样本。

| 数据类型 | 数量 | 标签方式 |
|---|---:|---|
| 常规可行需求 | 300 | oracle latent + SPMe 严格复算 |
| 边界可行需求 | 100 | 边界加密搜索 + SPMe 复算 |
| 已建模目标冲突 | 50 | oracle 搜索确认当前域内不可行 |
| 缺失/歧义/超范围需求 | 50 | 规则标签 + 人工审核 |

### 6.2 数值需求采样

首先从已有参考 archive 的真实性能分布反向采样任务，不直接让 DeepSeek自由编造阈值：

1. 从 Pareto 前沿或可行 archive 选择目标设计。
2. 根据该设计性能构造带不同松弛量的可行需求。
3. 在观测边界附近构造边界需求。
4. 在观测最大值之外或联合前沿之外构造不可行需求。
5. 用连续优化或有限 oracle 为每个可行任务选择最低 loss 的 teacher latent。
6. 对 teacher latent 运行 hard-cutoff SPMe，保存真实复算性能。

### 6.3 DeepSeek 的作用

DeepSeek只读取标准化需求底稿并生成自然语言表达，包括正式工程风格、口语表达、参数乱序、单位转换和冗余背景。DeepSeek不能修改数值真值，也不能生成 teacher latent、SPMe 性能或可行性结论。

每条生成文本经过程序重新解析和规则检查：文本中的关键数值必须与底稿一致；若发生遗漏或改写错误，则拒收或明确标注为缺失/冲突样本。

### 6.4 鲁棒性改写集

从测试需求族中为每族额外生成 3 种等价表达：

- 语序变化；
- 单位/百分数形式变化；
- 口语化或带冗余背景。

这些改写不能进入训练集。评价同一需求族不同文本的 latent 标准差、物理性能差异和预测一致率。

## 7. 数据划分

主数据按 `requirement_family_id` 分组划分：

| 集合 | 数量 | 用途 |
|---|---:|---|
| 训练集 | 350 | 模型拟合 |
| 验证集 | 75 | 早停和超参数选择 |
| 测试集 | 75 | 最终一次性评价 |

必须分层保持常规、边界、冲突和超范围样本比例。所有同源改写、相同 oracle 设计的近重复任务和同一参数模板必须位于同一集合。

由于样本量较小，主结论同时报告 5 折分组交叉验证，并至少运行模型种子 7、17、27。

## 8. 模型架构

### 8.1 文本编码

```text
requirement_text
    → Qwen tokenizer
    → 冻结 Qwen
    → attention-mask mean pooling
    → task embedding
```

第一版冻结全部 Qwen 权重，并离线缓存 embedding。mean pooling 只对非 padding token 求平均。另设最后有效 token pooling 作为消融，不默认使用它作为唯一方案。

### 8.2 多头 MLP

```text
Qwen embedding
    → LayerNorm
    → Linear + SiLU + Dropout
    → Residual MLP block × 2
    ├─ latent head: 5维
    ├─ feasibility head: 1维
    ├─ performance head: E1C、R5、R6
    └─ unsupported flags head: 多标签
```

latent head 使用有界映射：

\[
u = u_{limit}\tanh(z)
\]

`u_limit` 必须与 teacher archive 和现有实验保持一致，不允许训练集使用 `[-4,4]`、模型却按 `[-2,2]` 评价。

## 9. 损失函数

建议第一版使用：

\[
L = L_{latent}
+0.3L_{performance}
+0.3L_{feasibility}
+0.1L_{unsupported}
\]

其中：

- `L_latent`：五维标准化 Smooth L1；
- `L_performance`：标准化 Smooth L1，仅对具有有效 SPMe 标签的样本计算；
- `L_feasibility`：带类别权重的二元交叉熵；
- `L_unsupported`：多标签二元交叉熵。

不可行任务没有唯一正确 latent，因此不计算 `L_latent`；只训练可行性和原因标签。若一个需求存在多个近似等价 oracle，优先保存 Top-K 候选并使用集合最小损失，而不是把某一个随机候选强制作为唯一真值。

## 10. 基线与消融

### 10.1 必做基线

| 编号 | 方法 | 作用 |
|---|---|---|
| B0 | 标称常数设计 | 最低基线 |
| B1 | 随机可行设计 | 检查模型是否真正学习 |
| B2 | 结构化标准字段 + MLP | 信息无损上界基线 |
| B3 | TF-IDF + MLP/线性模型 | 非大模型文本基线 |
| B4 | 冻结 Qwen embedding + latent-only MLP | 主简单模型 |
| B5 | 冻结 Qwen embedding + 多头 MLP | 完整模型 |

### 10.2 消融实验

- mean pooling 与最后有效 token pooling；
- Qwen 不同层 embedding；
- latent-only 与多任务辅助头；
- 100、250、500 条训练规模；
- 是否加入冲突/超范围样本；
- 是否对 embedding 做 PCA 或低维投影；
- 随机划分与需求族分组划分，用于展示数据泄漏的影响。

第一版不进行 Qwen LoRA 微调。只有冻结 embedding 明显优于文本基线且数据规模扩大后，才考虑微调最后层或 LoRA。

### 10.3 提示词与外部模型对照

在同一测试需求、同一五维输出协议和同一 SPMe 评价器下增加：

| 编号 | 方法 | 输入上下文 |
|---|---|---|
| P0 | GPT/DeepSeek zero-shot | 仅需求和五维输出格式 |
| P1 | GPT/DeepSeek rule-prompt | 加入设计范围、容量平衡和不可验证边界 |
| P2 | GPT/DeepSeek few-shot | P1 加训练集内代表样例 |
| P3 | Battery-Sim-Agent 式迭代 | 给定固定次数的 PyBaMM 反馈后修改 latent |
| M1 | Qwen embedding + MLP | 离线训练，一次前向输出 latent |

比较必须记录模型版本、提示词、温度、token 用量、API 成本、延迟和物理调用次数。P3 与 M1 若物理调用预算不同，需要同时报告等预算结果。外部 LLM 的文本答案必须解析为同一五维 latent，再经同一个 `DesignSpace` 和 hard-cutoff SPMe 评价；不能用 LLM 自报的性能数字评分。

论文主要比较：需求满足率、regret、SPMe 成功率、语言改写稳定性、单样本延迟和成本。只有 M1 在这些指标中的至少一项显著改善，且没有以明显更差的物理可行性为代价，才能支持“训练 MLP 有效”的结论。

## 11. 评价指标

### 11.1 预测指标

- 每维 latent MAE、RMSE 和 R²；
- 解码后五个结构量的 MAE；
- 可行性 Accuracy、Precision、Recall、F1 和 AUROC；
- 未支持需求标签的 micro/macro F1；
- 辅助性能头的 E1C/R5/R6 MAE。

### 11.2 物理指标

对测试集预测 latent 统一执行 hard-cutoff SPMe：

- 求解成功率；
- 硬结构约束满足率；
- 用户已建模需求满足率；
- mean/median/max scalarized regret；
- 相对 oracle 的 1C 比能量差值；
- R5/R6 约束违反幅度；
- 优于标称设计比例。

### 11.3 语言鲁棒性指标

- 同义改写后的 feasibility 一致率；
- 同一需求族 latent 平均两两距离；
- 改写前后 SPMe 性能差异；
- 单位变化导致的严重错误率；
- 参数顺序变化导致的需求满足率下降。

### 11.4 统计报告

所有核心指标报告三个模型种子的均值、标准差和单次最差值。模型比较使用相同测试任务和配对差值；样本量允许时，对 regret 和需求满足率差异做 bootstrap 置信区间。

## 12. 验收标准

`paper_explore_v1` 通过的最低条件：

1. 测试集预测设计的硬解码可行率为 100%。
2. hard-cutoff SPMe 成功率不低于 95%。
3. 可行需求满足率显著高于随机可行设计，并优于常数标称设计。
4. mean scalarized regret 显著低于随机与标称基线。
5. 可行性分类 macro-F1 不低于 0.80；若类别样本不足，则以置信区间和错误分析替代单一门槛。
6. Qwen embedding 模型在鲁棒性改写集上优于 TF-IDF 基线。
7. 三个随机种子没有出现大面积输出坍缩或性能失效。
8. 所有训练和测试标签均具有 provenance，DeepSeek 没有直接提供物理真值。

论文探索是否继续进入第二版，依据物理 regret、语言鲁棒性和错误类型共同决定，而不是只看 latent MSE。

## 13. 失败分析

每个失败样本至少归入以下一种：

- 数值需求未被文本编码正确识别；
- 单位或百分数解析失败；
- 未支持要求被误当作已验证能力；
- MLP 输出接近 latent 边界；
- latent 接近 teacher 但 SPMe 性能差异较大；
- 多解任务被单标签监督误导；
- oracle 本身稀疏或不可信；
- SPMe 求解失败；
- 数据族泄漏或近重复。

对高 regret、边界和语言不一致样本形成定向补数清单，不直接无差别增加生成量。

## 14. 产物目录建议

```text
data/paper_explore_v1/
  canonical_tasks.jsonl
  generated_text_raw.jsonl
  dataset_clean.jsonl
  split_manifest.json
  embedding_cache/

results/paper_explore_v1/
  baselines/
  qwen_mlp/
    seed_7/
    seed_17/
    seed_27/
  robustness/
  physics_evaluation/
  figures/
  aggregate_metrics.json
  error_analysis.jsonl
  experiment_report.md
```

每次运行记录：代码 commit、配置、Qwen 模型标识、tokenizer、pooling 方法、数据哈希、随机种子、训练历史、checkpoint、逐样本预测和 SPMe 复算结果。

## 15. 执行顺序

1. 固定 `paper_explore_v1` 的任务字段、latent 范围和物理指标定义。
2. 从现有 archive 生成结构化任务和 oracle 标签。
3. 先人工审核 20–30 条黄金样本。
4. 让 DeepSeek 生成 50 条文本试样，做数值一致性与单位审计。
5. 通过审计后扩展为 500 个需求族。
6. 按需求族分组并冻结数据划分。
7. 缓存 Qwen embedding。
8. 训练 B0–B5 基线和三随机种子主模型。
9. 对测试预测统一运行 hard-cutoff SPMe。
10. 完成语言改写鲁棒性测试和失败分析。
11. 根据结果决定扩展数据、引入 LFP/软包几何模型，或再考虑 Qwen LoRA。

## 16. 论文可支持的结论边界

若实验通过，可以表述为：冻结的大语言模型表示能够在固定材料、受限结构域和仿真监督下，将自然语言多目标需求映射为具有物理意义的电芯结构候选；该方法相对浅层文本表示具有更好的语言改写鲁棒性。

不得表述为：模型已经验证真实 70 Ah LFP 软包的循环寿命、安全、低温和成本指标，或已经替代真实制造与测试。

## 17. Battery-Sim-Agent 的复用方式

Battery-Sim-Agent 原项目解决的是 LLM-agent 驱动 PyBaMM 做参数反演，并提供单参数、多参数和 SEI 退化模拟基准。`paper探索` 不直接复制它的标签空间，而复用以下实验范式：

- 参数采样后运行 PyBaMM；
- 明确保留求解失败并过滤无效标签；
- 使用统一模拟器反馈评价候选；
- 设置 BO/CMA-ES 或迭代 LLM agent 对照；
- 固定每种方法的物理调用预算；
- 聚合逐任务误差、成本与运行时间。

当前适配实现使用 GradCell 的五维硬可行 `DesignSpace` 和统一 SPMe evaluator 产生主数据。Battery-Sim-Agent 的粒径、厚度、Bruggeman 系数与 SEI 参数可以在 v2 扩展设计空间时加入，但不能在 v1 中与五维 latent 混为同一标签。

## 18. 第一批数据生成记录

2026-09-09 已使用 seed 101、五维 Sobol 采样、latent 范围 `[-4,4]` 生成 500 个 SPMe 候选。所有候选均完成容量校准及 1C/3C/5C/6C hard-cutoff 仿真，500/500 有效，失败率 0%。在此 archive 上已构造 500 条 canonical 需求，类型为 300/100/50/50，划分为 350/75/75，且物理来源索引不重复。

由于本机尚未配置 `DEEPSEEK_API_KEY`，当前文本由确定性模板生成；DeepSeek 改写代码已完成但尚未调用。API 改写后必须更新 manifest 哈希并重新执行文本数值一致性审计。
