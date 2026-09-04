# GradCell：基于可微电池仿真的目标条件化电芯设计

GradCell 是一个面向电芯逆向设计的研究原型。当前主实验接收 1C 比能量与 5C/6C 高倍率能量保持率之间的连续性能偏好，先预测满足制造约束的电芯设计，再利用 PyBaMM 一阶物理敏感度改善设计。第二目标定义为 `min(E_5C/E_1C, E_6C/E_1C)`；参考 Pareto 前沿只使用同时满足 5C 和 6C 最低能量保持率约束的样本。旧的 3C 容量保持率因大部分样本落在平台区，已不再作为条件化训练目标。

当前同时提供一条独立的电解液纯物理端到端路线：固定 Chen2020 电极与结构，网络由 CALiSol-23 配方特征直接预测有界 `log_conductivity_scale`，将其接入 PyBaMM DFN，并且只用 DFN 电压轨迹 loss 训练。Property Loss 已删除；实测电导率仅离线生成合成 DFN 目标并用于训练后诊断。DFN sensitivity 和自定义 `Jᵀv` 将电压梯度传回网络。详细流程见 `docs/第一版_固定材料电解液性质_DFN端到端实验流程.md`。

另外提供独立的 DFN 逆参数 benchmark 生成器。它参考 Battery-Sim-Agent 的
单参数/多参数扰动、失败过滤和 benchmark case 设计，但由 GradCell 独立实现：

```bash
PYTHONPATH=src python scripts/generate_dfn_parameter_benchmark.py \
  --family single --samples 100 --candidate-factor 5 \
  --c-rates 0.5,1,2 --seed 7 \
  --output data/dfn_parameter_benchmark_single_s7.npz
```

生成器要求 DFN 正常到达最低电压截止，过滤与 nominal Chen2020 在容量和曲线形状上
几乎不可区分的案例，并同时保存 `.npz`、可读的 `.yaml` case 配置及包含全部失败原因的
`.json` audit。电压曲线按实际放电过程归一化到 `[0, 1]`，实际截止时间和放电容量另存，
因此不会把不同样本错误地当成固定放电时长。`--family multi` 可生成 2–4 参数联合扰动。

服务器一键入口：`SEED=7 PHYSICS_SAMPLES=32 EPOCHS=100 bash scripts/run_electrolyte_v1_server.sh`。

纯物理流程的正式对照使用嵌套 physics 子集与多模型 seed 矩阵：

```bash
PYTHONPATH=src python scripts/run_electrolyte_v1_experiment_matrix.py \
  --split-seeds 7 --model-seeds 7 17 27 \
  --physics-samples 32 64 128 \
  --validation-physics-samples 32 --test-physics-samples 32 \
  --epochs 100 --output-dir results/electrolyte_v1_matrix
```

脚本为每个 DOI split 只生成一次最大物理数据集，使用保存的 `physics_rank` 构造
`32 ⊂ 64 ⊂ 128` 的公平嵌套训练子集；模型初始化 seed 与 split seed 分离。验证和
测试 physics 行不参与训练，只用于报告模型域 voltage RMSE、求解成功率和运行时间。
运行支持断点续跑，并汇总多 seed 均值与标准差到 `aggregate.json`/`aggregate.csv`。

正式准备数据前先执行可追溯清洗：

```bash
PYTHONPATH=src python scripts/clean_calisol23.py --overwrite
```

该命令生成全量规范化表 `data/calisol23_canonical.csv`、第一版训练表
`data/calisol23_model_v1.csv` 和审计记录 `data/calisol23_cleaning_report.json`。
`prepare_electrolyte_v1_data.py` 也会在内存中应用相同规则，不再把负浓度、缺失标签
或数值零直接送入 log-conductivity 回归。

当前电解液网络直接预测 `log_conductivity_scale`，通过 `tanh` 约束在
`[log(0.5), log(2.0)]`，然后直接进入 Chen2020 DFN。训练不再计算 Property Loss，
也不存在 physics weight sweep；验证电压 RMSE 是早停和 checkpoint 选择标准。

本仓库目前实现的是第一阶段 MVP，重点验证以下完整链路：

```text
性能偏好 λ
    ↓
目标编码器与设计初始化器
    ↓
5 维无约束结构 latent design u
    ↓
硬可行设计解码器 T(u)
    ↓
物理参数、额定容量、1C/5C/6C 电流和电芯质量
    ↓
PyBaMM SPMe + IDAKLU forward sensitivities
    ↓
电压轨迹和轨迹 Jacobian
    ↓
可微 1C 比能量、5C/6C 能量保持率与约束 Loss
    ↓
自定义 backward 执行 Jᵀv
    ↓
Initializer 参数训练或 K 步 Learned Refiner 修正
```

仓库中的 `AnalyticToyBackend` 只用于快速测试代码、梯度和训练闭环，不是科学电池模型。正式实验必须使用 `PyBaMMBackend`，并在后续使用 DFN 对候选设计进行统一复核。

GradCell 主任务的分阶段实验入口已经统一为：

```bash
python scripts/run_gradcell_exploration.py --stage checks
python scripts/run_gradcell_exploration.py --stage reference-data --reference-samples 5000
python scripts/run_gradcell_exploration.py --stage reference-front --reference-samples 5000 --retention-5c-min 0.50 --retention-6c-min 0.44 --constraint-weight 5.0
python scripts/run_gradcell_exploration.py --stage train-k0 --reference-samples 5000
python scripts/run_gradcell_exploration.py --stage evaluate-k0 --reference-samples 5000
python scripts/run_gradcell_exploration.py --stage train-refiner --reference-samples 5000
python scripts/run_gradcell_exploration.py --stage evaluate-refiner --reference-samples 5000
python scripts/run_gradcell_exploration.py --stage verify-dfn --reference-samples 5000
```

每一步的实验目的、判据和服务器命令见
[`docs/GradCell初探_分阶段实验执行.md`](docs/GradCell初探_分阶段实验执行.md)。
参考前沿同时保存由低倍率 PyBaMM 标定得到的容量乘子；GradCell 训练使用该乘子定义
1C/5C/6C 电流，避免解析标称容量与硬截止参考容量不一致而造成目标尺度错位。

## 1. 当前实现范围

当前版本包含：

- 目标偏好 Fourier 编码；
- 目标条件化电芯设计初始化器；
- 5 维硬可行结构设计解码器；
- 正负极容量平衡和 N/P 约束；
- 额定容量、C-rate 电流和 stack mass 计算；
- 解析 toy physics 后端；
- PyBaMM SPMe + IDAKLU sensitivity 后端；
- PyTorch 自定义 `autograd.Function`；
- 平滑截止电压、1C 比能量和 5C/6C 能量保持率指标；
- Smooth Tchebycheff 多目标损失；
- Learned diagonal physics refiner；
- 方向导数梯度检查；
- K=0 和 K>0 训练入口；
- 设计可行性、自定义反传和模型闭环测试。

当前主设计变量为：

| 索引 | 变量 | 含义 | 参数化方式 |
|---:|---|---|---|
| 0 | `eps_p` | 正极孔隙率 | bounded sigmoid |
| 1 | `eps_n` | 负极孔隙率 | bounded sigmoid |
| 2 | `eps_s` | 隔膜孔隙率 | bounded sigmoid |
| 3 | `phi_p` | 正极活性材料体积分数 | 耦合可行区间 |
| 4 | `np_ratio` | N/P 容量比 | bounded sigmoid |
`phi_n` 不由网络独立输出，而是根据 `phi_p`、N/P 比和材料容量常数解析计算，从而严格满足容量平衡。

正负极固相扩散率乘子在当前固定材料 MVP 中均固定为 `1.0`，不属于 5 维优化变量。颗粒半径、厚度和材料性质属于后续 normalized-coordinate 或经严格有限差分验证的 Track B。

## 2. 项目目录

```text
grad_cell/
├── .gitignore
├── README.md
├── pyproject.toml
├── setup.py
├── configs/
│   ├── design_space/
│   │   └── chen2020.yaml
│   ├── model/
│   │   └── gradcell.yaml
│   ├── physics/
│   │   ├── spme.yaml
│   │   └── toy.yaml
│   └── training/
│       └── mvp.yaml
├── data/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── scripts/
│   ├── run_baseline.py
│   ├── train_mvp.py
│   └── validate_gradients.py
├── src/
│   └── gradcell/
│       ├── __init__.py
│       ├── benchmark/
│       │   ├── __init__.py
│       │   └── regret.py
│       ├── design/
│       │   ├── __init__.py
│       │   ├── capacity_balance.py
│       │   ├── feasible_decoder.py
│       │   └── mass_model.py
│       ├── losses/
│       │   ├── __init__.py
│       │   └── scalarization.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── gradcell.py
│       │   ├── initializer.py
│       │   ├── refiner.py
│       │   └── task_encoder.py
│       ├── physics/
│       │   ├── __init__.py
│       │   ├── autograd_layer.py
│       │   ├── backend.py
│       │   ├── gradient_validation.py
│       │   └── soft_metrics.py
│       └── training/
│           ├── __init__.py
│           ├── task_sampler.py
│           └── trainer.py
└── tests/
    ├── test_autograd_layer.py
    ├── test_decoder.py
    └── test_model.py
```

目录按职责分为五层：

1. `configs/` 保存实验参数，避免把参数硬编码在运行脚本里；
2. `src/gradcell/design/` 将网络 latent 转成合法物理电芯；
3. `src/gradcell/physics/` 负责仿真、敏感度和 PyTorch 反传；
4. `src/gradcell/models/` 负责目标编码、初始化和物理梯度修正；
5. `scripts/` 和 `tests/` 分别负责实验入口和自动验证。

## 3. 根目录文件

### `.gitignore`

忽略虚拟环境、Python 缓存、构建产物、自动生成的数据、checkpoint 和实验结果。`data/.gitkeep` 与 `results/.gitkeep` 用于保留空目录结构。

### `pyproject.toml`

项目的主要打包和依赖配置文件。核心依赖包括 `numpy`、`torch` 和 `pyyaml`；`physics` 可选依赖组包含 PyBaMM、SciPy、pandas 和 matplotlib；`dev` 组包含 pytest、pytest-cov 和 Ruff。

`[tool.setuptools.packages.find]` 指定包代码位于 `src/`，其余部分设置 pytest 和 Ruff。

### `setup.py`

为旧版 pip 提供 editable install 兼容入口。具体项目信息仍由 `pyproject.toml` 管理。

## 4. 配置文件说明

### `configs/design_space/chen2020.yaml`

定义 Chen2020 MVP 的设计边界：三种孔隙率、正极活性材料比例下限、正负极最小非活性相比例和 N/P 比；两个扩散率乘子显式固定为 `1.0`。

`phi_p_min` 当前为 `0.20`。原设想中的 `0.35` 与最大负极孔隙率和最大 N/P 比组合时可能没有可行交集。如果提高该下限，必须同时收紧其他耦合边界并重新运行 decoder 可行性测试。

### `configs/model/gradcell.yaml`

保存模型结构超参数：Fourier frequency、task embedding 维度、initializer 宽度和 block 数、refiner hidden dimension、修正步数、最大更新步长和对角预条件器上下界。

当前脚本仍使用类的默认构造参数；后续可以增加统一配置加载器自动注入这些值。

### `configs/physics/spme.yaml`

保存正式物理实验设置：SPMe、Chen2020、温度、精度、固定仿真时域、输出变量、IDAKLU sensitivity 和 solver tolerance。

训练采用固定时域和 PyTorch 平滑电压 gate，不依赖 `pybamm.Experiment` 的硬截止事件，以降低事件时间不连续造成的梯度问题。

### `configs/physics/toy.yaml`

定义解析测试后端的时间点和时域，只用于软件测试。

### `configs/training/mvp.yaml`

保存随机种子、后端、训练步数、batch size、优化器参数、梯度裁剪、refinement steps 以及各类 loss 权重。

## 5. 设计空间代码

### `src/gradcell/design/capacity_balance.py`

实现电极容量平衡。

`CapacityConstants` 保存正负极厚度、最大固相浓度、化学计量区间和电极面积。模块常量 `FARADAY_C_PER_MOL` 是法拉第常数。

`negative_active_fraction(...)` 根据正极活性材料体积分数和目标 N/P 比推导负极活性材料体积分数：

```text
φn = rNP × (Lp cmax,p Δθp) / (Ln cmax,n Δθn) × φp
```

`nominal_capacity_ah(...)` 根据正极限制容量计算额定容量：

```text
Qnom = F × A × Lp × φp × cmax,p × Δθp / 3600
```

当前训练随后构造 1C、5C、6C 电流：`Qnom`、`5 × Qnom`、`6 × Qnom` A；参考数据还保存 3C 诊断。

### `src/gradcell/design/mass_model.py`

`MassConstants` 保存有效面积、各层厚度、材料密度和固定集流体质量。

`stack_mass_kg(...)` 计算正负极活性材料与非活性材料质量、电解液质量和固定质量，返回 stack-level mass。这个模型是工程 proxy，不等同于完整商业电芯质量模型。

### `src/gradcell/design/feasible_decoder.py`

这是设计层的核心文件。

`CellDesign` 保存解码后的孔隙率、活性材料比例、N/P、扩散率乘子、额定容量和质量。

`CellDesign.physics_tensor(c_rate)` 返回 PyBaMM 后端需要的 `[B, 8]` Tensor：

```text
[eps_p, eps_n, eps_s, phi_p, phi_n,
 diffusivity_p_multiplier, diffusivity_n_multiplier, current_a]
```

`DesignSpace` 将任意 `[..., 5]` latent Tensor 映射到硬可行设计。Chen2020
正负极固相扩散系数乘子固定为 `1.0`，不作为优化变量：

- `_bounded` 用 sigmoid 映射普通有界参数；
- `forward/decode` 执行完整设计解码；
- `nominal_latent` 返回全零 nominal latent。

正极活性相允许上界为：

```text
phi_p_max = min(
    1 - eps_p - inactive_p_min,
    (1 - eps_n - inactive_n_min) / kappa,
)
```

再将 `phi_p` 映射到 `[phi_p_min, phi_p_max]` 并解析计算 `phi_n`。因此相体积分数、最小非活性相、N/P 和参数边界都由结构严格保证，而不是依靠 penalty 近似满足。

## 6. 物理仿真和可微层

### `src/gradcell/physics/backend.py`

定义统一物理后端协议及两个实现。

`PhysicsBatch` 保存：

| 字段 | 形状 | 含义 |
|---|---|---|
| `trajectories` | `[B, O, T]` | 物理输出轨迹 |
| `jacobian` | `[B, O, T, P]` | 轨迹对输入参数的 Jacobian |
| `status` | `[B]` | 求解是否成功 |
| `runtime_s` | `[B]` | 每个样本求解时间 |

`PhysicsBackend` 是一个 `Protocol`，要求后端实现：

```python
solve_batch(inputs: np.ndarray) -> PhysicsBatch
```

`AnalyticToyBackend` 使用解析公式生成电压轨迹和精确 Jacobian。它用于测试项目结构、自定义 backward 和短训练，不可用于科学结论。

`PyBaMMBackend` 是正式 SPMe 后端。初始化时：

1. 构建 SPMe 和 Chen2020 参数集；
2. 将硬电压截止移动到训练区域之外；
3. 将孔隙率、活性材料比例和电流设为 `InputParameter`；
4. 用两个运行时乘子包装正负极固相扩散率；
5. 创建并缓存 IDAKLU solver 和 Simulation。

`_extract_sensitivity(...)` 获取 processed variable 的 sensitivity，并将 PyBaMM 自适应时间网格上的值插值到固定 `t_eval`。

`solve_batch(...)` 当前逐样本求解。成功时返回轨迹和 Jacobian；失败时返回 `status=0`、dummy trajectory 和零 Jacobian，上层 loss 再通过 nominal recovery barrier 提供恢复方向。失败不能从实验统计中静默删除。

### `src/gradcell/physics/autograd_layer.py`

将 NumPy/PyBaMM 后端接入 PyTorch autograd。

`_PhysicsFunction.forward(...)`：

1. 将 Tensor 转成 CPU `float64` NumPy；
2. 调用 `backend.solve_batch`；
3. 将结果转回 Tensor；
4. 保存 Jacobian；
5. 返回 trajectory、status 和 runtime。

这里的 `detach` 是有意的，因为 PyBaMM 不在 PyTorch 原生图内，梯度由自定义 backward 显式提供。

`_PhysicsFunction.backward(...)` 计算 VJP：

```text
∂L/∂p = Jᵀ ∂L/∂y
```

对应实现：

```python
torch.einsum("bot,botp->bp", grad_y, jacobian)
```

`DifferentiablePhysicsLayer` 是供 GradCell 使用的普通 `nn.Module` 包装器。

### `src/gradcell/physics/soft_metrics.py`

`PerformanceMetrics` 保存比能量、平滑可释放容量、诊断用比功率和平滑最低电压。

`voltage_gate(...)` 用 sigmoid 近似硬截止：

```text
a(t) = sigmoid((V(t) - Vcut) / τV)
```

`discharge_metrics(...)` 计算：

```text
usable energy = ∫ I V(t) a(t) dt
specific energy = usable energy / stack mass
delivered capacity = I × ∫ a(t) dt
R5 = delivered energy at 5C / delivered energy at 1C
R6 = delivered energy at 6C / delivered energy at 1C
high-rate objective = min(R5, R6)
```

最终评测仍需要运行真实硬 cutoff 仿真，以检查 smooth metric 是否产生系统性偏差。

### `src/gradcell/physics/gradient_validation.py`

`directional_derivative_check(...)` 比较：

```text
autodiff = gᵀv
finite difference = [f(x + εv) - f(x - εv)] / (2ε)
```

方向导数能同时覆盖 decoder、PyBaMM sensitivity、自定义 VJP、性能指标和 loss，比只检查单个坐标更适合验证整条计算链。

## 7. 神经网络代码

### `src/gradcell/models/task_encoder.py`

`FourierPreferenceEncoder` 输入连续偏好 `λ ∈ [0,1]`，构造：

```text
[λ, sin(2πλ), cos(2πλ), ..., sin(2πFλ), cos(2πFλ)]
```

再经过两层 MLP、SiLU 和 LayerNorm 输出 task embedding。完整版本可继续加入温度、工况和材料参数集物理指纹。

### `src/gradcell/models/initializer.py`

`ResidualBlock` 由两层线性层、SiLU 和 LayerNorm 组成，并使用 residual connection。

`DesignInitializer` 将 task embedding 映射到 5 维结构 latent：

```text
u0 = initial_scale × tanh(network(task_embedding))
```

输出层零初始化，使训练开始时所有偏好都从 nominal latent 出发，避免随机初始化产生大量不稳定电芯。

### `src/gradcell/models/refiner.py`

`DiagonalPhysicsRefiner` 接收 task embedding、当前 latent、物理指标、loss、当前梯度和前一步 GRU state。

梯度先做 RMS 归一化，随后 GRU 预测：

- 标量步长 `alpha`；
- 5 维正对角预条件器 `diagonal`。

更新规则：

```text
u(k+1) = u(k) - alpha(k) × diagonal(k) × stopgrad(g(k))
```

步长和对角元素都有边界，因此更新幅度受控且预条件器始终半正定。物理梯度被 detach，所以不要求 PyBaMM 提供二阶 sensitivity。

### `src/gradcell/models/gradcell.py`

这是完整模型的编排层。

`GradCellStep` 保存一次物理评估的 latent、设计、loss、1C 比能量、5C/6C 能量保持率和 solver status。

`GradCellOutput` 保存 K+1 次评估结果，`final` 属性返回最后一步。

`GradCell.__init__` 组合 decoder、task encoder、initializer、refiner、1C/5C/6C physics layer 和带高倍率约束的 Smooth Tchebycheff objective。

`GradCell.evaluate(...)`：

1. 解码硬可行设计；
2. 根据额定容量生成 1C、5C 和 6C 电流；
3. 分别调用两个物理层；
4. 计算 1C 比能量、5C/6C 能量保持率和约束 violation；
5. 计算偏好损失；
6. 对部分 solver failure 加入 recovery penalty；若整个 batch 失败则立即停止训练。

`GradCell.forward(...)` 执行：

```text
preference → embedding → initializer → u0 → evaluate
→ ∂loss/∂u → refiner update → u1 → ... → uK
```

`num_steps=0` 对应 direct initializer；`num_steps=1/3/5` 对应不同物理调用预算下的 GradCell。

## 8. Loss 与 benchmark

### `src/gradcell/losses/scalarization.py`

`SmoothTchebycheff` 首先将 1C 比能量和最差 5C/6C 能量保持率转成相对于 ideal/nadir 的无量纲距离：

```text
dE = (Eideal - E) / (Eideal - Enadir)
dR = (Rideal - min(R5, R6)) / (Rideal - Rnadir)
```

偏好权重为 `[λ, 1-λ]`，最终 loss 为：

```text
L_tch = τ logsumexp(wj dj / τ) + ρ Σ wj dj
L = L_tch + γ[ReLU(r5_min - R5) + ReLU(r6_min - R6)]
```

默认 `r5_min=0.50`、`r6_min=0.44`、`γ=5.0`。独立的硬截止 reference 数据自动计算
ideal/nadir；Pareto 前沿只从满足两个硬约束的样本中构建。

### `src/gradcell/benchmark/regret.py`

`normalized_regret(...)` 实现：

```text
R = (Lachieved - Loptimal) / (Lnominal - Loptimal + ε)
```

- `R=0` 表示达到参考最优；
- `R=1` 表示与 nominal cell 相当；
- `R>1` 表示比 nominal cell 更差。

正式实验应报告 Regret@0、@1、@3、@5，以及 solver calls、wall-clock、可行率和训练摊销成本。

## 9. 训练代码

### `src/gradcell/training/task_sampler.py`

`sample_preferences(...)` 使用混合采样：

- 50% `Uniform(0,1)`；
- 25% `Beta(0.5,2.0)`，加强 5C/6C 高倍率保持率端；
- 25% `Beta(2.0,0.5)`，加强能量端。

这可以减少模型只学习 Pareto front 中间区域的风险。

### `src/gradcell/training/trainer.py`

`TrainResult` 当前保存每一步 loss 历史。

`train(...)` 每一步执行：

1. 采样 preference；
2. 运行 K 步 GradCell；
3. 堆叠 intermediate loss；
4. 计算平均 loss；
5. K>0 时加入单调改进 penalty；
6. 加入 latent step penalty；
7. 反向传播；
8. 梯度裁剪；
9. AdamW 更新；
10. 记录并打印 loss。

当前 trainer 尚未包含验证集、早停、断点恢复和结构化日志。

## 10. 运行脚本

### `scripts/run_baseline.py`

运行 nominal Chen2020 SPMe 基准仿真并保存 solver status、运行时间、最低电压和最高电压。

```powershell
python scripts\run_baseline.py --c-rate 1
python scripts\run_baseline.py --c-rate 3 --output results\baseline_3c.json
```

### `scripts/validate_gradients.py`

验证 physics layer 的自定义 backward：

```powershell
python scripts\validate_gradients.py --backend toy
python scripts\validate_gradients.py --backend pybamm --eps 1e-4
```

`toy` 使用解析 Jacobian；`pybamm` 使用真实 SPMe/IDAKLU sensitivity。正式训练前应在多个设计点、多个 solver tolerance 和多个 `eps` 下重复验证。

### `scripts/qa_physics.py`

这是不训练网络的长时间物理层 QA runner。它按 0.5C、1C、2C、3C 和 nominal/扩散率扰动案例逐个执行 forward、有限差分 sensitivity 与自定义 autograd VJP 检查；每个案例立即追加到 `case_results.jsonl`，并更新 `summary.json`，中断后重新运行会自动跳过已完成案例。

```powershell
python scripts\qa_physics.py --model DFN --mode all --output-dir results\physics_qa
```

服务器上可用 `nohup` 或作业调度器运行：

```bash
mkdir -p results/physics_qa
nohup python scripts/qa_physics.py --model DFN --mode all \
  --output-dir results/physics_qa > results/physics_qa/stdout.log 2>&1 &
tail -f results/physics_qa/run.log
```

结果文件包括 `metadata.json`、`case_results.jsonl`、`summary.json` 和 `run.log`。若只先验证正向求解，可使用 `--mode forward --max-cases 3`；正式运行前应先用小规模案例检查 DFN 求解时间和失败率。

### `scripts/train_mvp.py`

创建两个 physics backend，构建 GradCell，训练并保存 checkpoint。主要参数为 `--backend`、`--steps`、`--batch-size`、`--refinement-steps` 和 `--checkpoint`。

```powershell
# K=0：只训练 initializer
python scripts\train_mvp.py --backend pybamm --steps 5000 --batch-size 2 --refinement-steps 0

# K=1：加入一次 learned refinement
python scripts\train_mvp.py --backend pybamm --steps 5000 --batch-size 2 --refinement-steps 1

# K=3：完整 MVP
python scripts\train_mvp.py --backend pybamm --steps 20000 --batch-size 2 --refinement-steps 3
```

checkpoint 保存模型 `state_dict` 和 loss 历史。

## 11. 自动测试

### `tests/test_decoder.py`

- `test_decoder_is_hard_feasible` 随机生成 1000 个 latent，检查相体积分数、最小非活性相和 N/P 边界；
- `test_np_ratio_is_exact` 独立重算两极容量，检查 `Qn/Qp` 与指定 N/P 一致。

### `tests/test_autograd_layer.py`

使用解析 toy backend 比较自定义 backward 和方向有限差分，定位参数顺序、Jacobian 维度或 VJP 实现问题。

### `tests/test_model.py`

- 验证 K=0 能完成反向传播且 initializer 收到非零梯度；
- 验证 K=1 能运行两次物理评估并产生有限 loss。

运行：

```powershell
pytest
ruff check src scripts tests
```

## 12. 安装与环境

推荐 Windows PowerShell、Python 3.10 或 3.11、CPU 和 `float64`。

```powershell
cd C:\Users\zhy\Desktop\gradcell\grad_cell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[physics,dev]"
```

检查版本：

```powershell
python -c "import torch, pybamm; print('torch:', torch.__version__); print('pybamm:', pybamm.__version__)"
```

正式实验应记录 Python、PyTorch、PyBaMM、操作系统、CPU、Git commit、随机种子、solver tolerance、mesh、参数集和设计范围。

## 13. 推荐实验顺序

### 第一步：纯软件测试

```powershell
pytest
python scripts\validate_gradients.py --backend toy
python scripts\train_mvp.py --backend toy --steps 100 --refinement-steps 0
python scripts\train_mvp.py --backend toy --steps 100 --refinement-steps 1
```

### 第二步：PyBaMM 基准

```powershell
python scripts\run_baseline.py --c-rate 1
python scripts\run_baseline.py --c-rate 3
```

检查 status、NaN、电压范围、重复性和运行时间。

### 第三步：真实梯度 QA

```powershell
python scripts\validate_gradients.py --backend pybamm --eps 1e-3
python scripts\validate_gradients.py --backend pybamm --eps 1e-4
python scripts\validate_gradients.py --backend pybamm --eps 1e-5
```

建议验收标准：median directional error `<1e-3`，95% error `<1e-2`，并且小步 exact-gradient 更新能使多数内部设计的 loss 下降。

### 第四步：课程训练

1. K=0，训练目标条件化 initializer；
2. K=1，学习一步 refiner；
3. K=3，联合训练 initializer 和 refiner；
4. 构建 reference front 并计算 Regret@K；
5. 最后使用 DFN 统一复核。

## 14. 训练中的数据形状

| 变量 | 形状 | 含义 |
|---|---|---|
| `preference` | `[B]` 或 `[B,1]` | 1C 比能量–5C/6C 高倍率能量保持率偏好 |
| `task_embedding` | `[B,128]` | 任务表示 |
| `latent` | `[B,5]` | 无约束结构设计 |
| `physics_inputs` | `[B,8]` | PyBaMM 输入和电流 |
| `trajectory` | `[B,O,T]` | 电压等轨迹 |
| `jacobian` | `[B,O,T,8]` | 轨迹 sensitivity |
| `grad_y` | `[B,O,T]` | loss 对轨迹的梯度 |
| `grad_inputs` | `[B,8]` | loss 对物理输入的梯度 |
| `grad_latent` | `[B,5]` | loss 对 latent 的梯度 |
| `loss` | `[B]` | 每个任务的标量损失 |

完整链式法则为：

```text
∂L/∂θ =
(∂L/∂y × ∂y/∂p × ∂p/∂u + ∂L/∂design × ∂design/∂u)
× ∂u/∂θ
```

PyBaMM 提供 `∂y/∂p`，自定义 backward 完成 `Jᵀv`，其他部分由 PyTorch 自动计算。

## 15. 当前限制

1. YAML 尚未通过统一 loader 自动注入所有类；
2. PyBaMM batch 当前逐样本循环；
3. 当前只输出电压，尚未加入电解液浓度和表面化学计量比；
4. reference front 仍是有限随机搜索近似，而非全局最优证明；
6. 尚未实现 NSGA-II、CMA-ES、BO 和 exact-gradient baseline；
7. 尚未实现 Preference-OOD split；
8. 已支持验证、早停、断点和结构化日志，但尚未完成正式多 seed 汇总；
9. 已提供 DFN 统一复核入口，但尚无正式复核实验结论；
10. 尚未接入真实电芯实验数据；
11. 质量模型仍是 stack-level proxy；
12. solver failure 的恢复梯度主要来自 latent-to-nominal barrier。

因此当前正确表述是“完成了可微 SPMe 训练闭环”，不能表述为已经完成真实电芯材料发现或真实实验验证。

## 16. 下一阶段建议增加的文件

```text
src/gradcell/
├── config.py
├── physics/
│   ├── model_factory.py
│   ├── input_registry.py
│   └── failure_policy.py
├── losses/
│   └── physical_constraints.py
├── benchmark/
│   ├── reference_front.py
│   ├── evaluator.py
│   └── baselines/
│       ├── exact_gradient.py
│       ├── random_search.py
│       ├── cma_es.py
│       ├── nsga2.py
│       └── bayesian_optimization.py
└── training/
    ├── checkpoint.py
    ├── logger.py
    └── curriculum.py

scripts/
├── generate_reference_front.py
├── evaluate_all.py
├── verify_dfn.py
└── sweep_design_space.py
```

优先顺序：多点梯度 QA 和 solver sweep → reference front → baseline → 正式课程训练 → 轨迹约束 → DFN 复核 → 多温度/参数集 → 真实数据。

## 17. 已完成的冒烟验证

- 现有单元测试全部通过；
- Ruff 静态检查通过；
- nominal Chen2020 SPMe 1C 求解成功；
- toy backend backward 与方向有限差分一致；
- PyBaMM sensitivity backward 与方向有限差分一致；
- toy backend K=0/K=1 短训练完成；
- PyBaMM backend K=0 单步训练完成。

这些结果只证明软件链路可运行，不能替代多点、多种子、完整预算和 DFN 复核实验。

## 18. 开发约定

- 物理与训练默认使用 `float64`；
- 第一阶段 CPU-first；
- 训练路径不使用不可控硬 cutoff event；
- 不静默删除 solver failure；
- 修改 decoder 后必须运行随机可行性测试；
- 修改 PyBaMM 后端后必须重新运行方向导数检查；
- 正式实验必须固定 seed、配置、版本和 Git commit；
- SPMe 不是 ground truth；
- DFN 只能称为 higher-model-fidelity verification；
- 真实结论必须由独立真实电芯数据支持。
