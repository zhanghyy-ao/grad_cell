# paper探索：2000条数据服务器Workflow

## 1. 目标

在Linux服务器上构建约2000条“自然语言电芯需求—Top-K Pareto物理解”训练数据。物理标签来自GradCell/PyBaMM，DeepSeek只负责自然语言多样化，不生成设计真值。

## 2. 工作流

```text
4 × 2048 Sobol latent
        ↓
硬可行DesignSpace解码
        ↓
SPMe容量校准 + 1C/3C/5C/6C hard-cutoff
        ↓
合并8192候选并做三目标非支配筛选
        ↓
按随机偏好为每个任务选择Top-5 Pareto oracle
        ↓
构造2000条canonical需求
        ↓
DeepSeek生成多样化中文问题
        ↓
数值锚点校验、失败重试、断点保存
        ↓
1400/300/300训练验证测试划分
```

三个Pareto目标为最大化1C比能量、5C能量保持率和6C能量保持率。每条可行任务保存Top-5候选及其oracle loss，训练时可使用集合最小latent损失。

## 3. 服务器环境

```bash
cd /path/to/gradcell/grad_cell
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[physics,language,dev]'
python -c 'import torch, pybamm, numpy, yaml; print(torch.__version__, pybamm.__version__)'
```

如果仓库位于集群共享文件系统，建议把`data/paper_explore_2000`和`results/paper_explore_2000`放到本地高速盘，再在完成后同步回项目目录。

## 4. API安全配置

不要把API key写入YAML、Shell脚本、Git仓库或聊天记录。登录服务器后在当前Shell中设置：

```bash
export DEEPSEEK_API_KEY='你的密钥'
```

也可以由集群secret manager注入。运行日志不得打印该变量。

当前配置使用：

```yaml
base_url: https://api.deepseek.com
model: deepseek-v4-flash
api_key_env: DEEPSEEK_API_KEY
```

模型名称和接口可能变化，正式运行前应按DeepSeek官方文档确认并记录具体模型版本。

## 5. 一键运行

默认启动4个物理shard，每个2048个候选：

```bash
bash scripts/run_paper_explore_data_server.sh
```

这一步默认生成物理archive、Pareto前沿和2000条确定性文本，不调用DeepSeek。

检查产物后执行DeepSeek改写：

```bash
export DEEPSEEK_API_KEY='你的密钥'
WITH_DEEPSEEK=1 bash scripts/run_paper_explore_data_server.sh
```

已有物理shard会被跳过。DeepSeek每25条保存一次；相同模型已经成功改写的任务会被跳过，可在网络中断后重复运行。

## 6. 并发调整

物理并行度通过shard数量控制：

```bash
SHARD_COUNT=2 SAMPLES_PER_SHARD=4096 bash scripts/run_paper_explore_data_server.sh
```

建议起始设置：

| 服务器资源 | 推荐物理并发 |
|---|---:|
| 8–16 CPU核、16–32GB内存 | 2 |
| 16–32 CPU核、32–64GB内存 | 4 |
| 32核以上、64GB以上 | 4–8，先做小规模稳定性测试 |

PyBaMM/SPMe阶段主要使用CPU。GPU留给后续Qwen embedding提取和MLP训练。不要仅依据逻辑核数量把PyBaMM并发开满。

## 7. 分阶段运行与检查

### 7.1 物理archive

每个shard应满足：

- metadata状态为`completed`；
- `valid_samples`接近请求数量；
- 失败条目具有诊断信息；
- latent、design和targets行数一致。

### 7.2 Pareto前沿

检查：

- `candidate_count`约8192；
- `finite_count`和`feasible_count`；
- `pareto_count`不能过少；
- 三个目标范围；
- latent是否大量贴边。

如果Pareto点过少，不应直接生成2000条近重复标签；应扩大archive、加入局部加密或调整设计域。

### 7.3 canonical任务

配置为：

| 类型 | 数量 |
|---|---:|
| 常规可行 | 1200 |
| 边界可行 | 400 |
| 已建模冲突 | 200 |
| 超范围/歧义 | 200 |

每条记录包含需求底稿、teacher latent、teacher design、teacher performance、Top-5 Pareto候选、可行性和provenance。

### 7.4 DeepSeek改写

DeepSeek必须逐字保留下列数值锚点：

- 1C比能量目标；
- 5C/6C保持率；
- 能量与高倍率偏好权重；
- Chen2020和25℃。

改写失败时保留确定性模板并写入`quality_flags`。正式训练前应统计DeepSeek成功率、重试次数、失败类型和语言风格分布，并人工抽查至少100条。

## 8. 主要文件

- 配置：`configs/paper_explore_2000.yaml`
- 一键服务器脚本：`scripts/run_paper_explore_data_server.sh`
- 物理生成：`scripts/generate_supervised_data.py`
- Pareto筛选：`scripts/build_reference_front_3d.py`
- 任务与DeepSeek改写：`scripts/generate_paper_explore_dataset.py`
- 最终数据：`data/paper_explore_2000/dataset.jsonl`
- 审计清单：`data/paper_explore_2000/manifest.json`

## 9. 运行后验收

数据进入Qwen embedding训练前必须满足：

1. 2000条JSONL均可解析，task ID与需求族ID唯一。
2. 1400/300/300划分正确，同一需求族不跨集合。
3. 可行任务均有1–5个有效Pareto teacher。
4. teacher latent为五维有限数值且位于统一范围。
5. teacher性能来自物理archive，不来自DeepSeek输出。
6. DeepSeek改写数值锚点通过率达到100%，失败样本被显式标记。
7. 文本去重、近重复和语言风格分布完成审计。
8. manifest保存物理配置、模型名、随机种子、数据哈希和生成时间。

## 10. 后续训练接口

训练时只把`requirement_text`输入Qwen；监督标签使用`teacher_candidates`、`requirement_feasible`和未支持需求标签。`requirements_canonical`仅用于审计和结构化输入基线，不能混入主非结构化输入模型。
