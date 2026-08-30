# 第一版：固定材料下的电解液性质预测与 DFN 端到端梯度实验

## 1. 研究问题

第一版回答两个彼此关联、但不能混为一谈的问题：

1. 在固定 Chen2020 电极材料、几何结构和电芯设置的情况下，常规监督模型能否由电解液配方、盐浓度和温度预测实验离子电导率？
2. 网络预测的电导率能否作为 Chen2020 DFN 的电解液电导率修正量进入 PyBaMM，并使 DFN 电压轨迹损失通过 IDAKLU sensitivity 和自定义 `Jᵀv` 反传到网络？

第一版不是电解液发现结论，也不声称 CALiSol-23 的每条性质记录对应一个真实 Chen2020 电芯。DFN 分支是模型域辅助约束和端到端梯度验证。

## 2. 固定项与学习项

### 固定项

- PyBaMM 参数集：Chen2020；
- 电化学模型：DFN；
- 正极、负极、隔膜、几何和初始状态；
- 训练探针工况：5 A 恒流，第一版不使用电流 ramp；
- 参数集内的最低/最高电压截止事件保留；
- IDAKLU：`rtol=1e-6`、`atol=1e-8`。

实测发现，在当前 PyBaMM 26.8.0、DFN、IDAKLU sensitivity 组合下，`1-exp(-t/1s)` ramp 会在 `t=0` 触发 IDAS 误差测试失败；直接恒流输入可以稳定建立一致初值。因此第一版锁定 `current_ramp_time_s=0`。若以后恢复 ramp，必须作为单独 solver 实验重新验证，不能只因直觉上“更平滑”就默认启用。

### 网络学习项

网络从 CALiSol-23 特征预测：

```text
log(k / [mS cm^-1])
```

输入包括温度、盐浓度、38 种溶剂比例，以及盐类型、浓度单位、溶剂比例单位的 one-hot 表示。

### DFN 接口量

预测的电导率通过参考值 10 mS/cm 转成对数修正：

```text
log_scale = predicted_log_k - log(10)
kappa_DFN(c_e, T) = kappa_Chen2020(c_e, T) * exp(log_scale)
```

这是第一版的低维代理。它没有把单点实验电导率误认为完整的 `kappa(c_e,T)` 函数，而是把它解释为 Chen2020 电导率函数的整体倍率。

## 3. 数据结构

### 主数据：CALiSol-23

- 目标：实验离子电导率，单位 mS/cm；
- 损失：对数空间 Huber loss；
- 划分：按来源 DOI 分组为 train/validation/test，禁止同一论文的相邻温度点跨集合泄漏；
- 归一化：只使用训练集统计量。

### 小比例 Chen2020–DFN 数据

从实验数据中随机选取一小部分处于稳定倍率范围内的样本：

1. 把其实验电导率换算成 Chen2020 电导率倍率；
2. 固定 Chen2020 材料和工况运行 DFN；
3. 保存 0–600 s 公共观测窗口内的电压轨迹；
4. 在线训练时，网络预测电导率后重新运行同一 DFN；
5. 预测轨迹与离线目标轨迹形成物理 loss。

这些电压是 **Chen2020 模型域标签**，不是 CALiSol-23 的真实电芯电压。

## 4. 为什么既保留电压截止，又使用 600 s 观测窗口

DFN 始终保留 Chen2020 的物理电压截止事件。600 s 只是梯度探针读取电压的公共观测窗口，不是把放电时间固定为 600 s，也不替代真实截止时间。

第一版不对事件终止时间求导。原因是当前 IDAKLU forward sensitivity 给出状态/输出对 `InputParameter` 的一阶敏感度，但不能直接把事件时间导数当作可靠监督。训练只使用预计发生截止之前的共同时间点；完整评测另行运行到最低电压截止，报告真实放电时间、容量和能量。

如果某个候选在 600 s 前已经触发截止，该样本的在线物理求解标记为失败，不会用零轨迹冒充有效标签。

## 5. 模型与损失

### 性质模型

```text
CALiSol features
→ standardized features
→ MLP + SiLU + LayerNorm
→ bounded log conductivity
```

输出边界对应 0.05–50 mS/cm，保证电导率为正并减少初期极端 DFN 输入。

### 联合损失

```text
L_total = L_property + w_physics * L_voltage
```

其中：

- `L_property`：实验 `log(k)` 的 Huber loss；
- `L_voltage`：DFN 预测电压和 Chen2020 辅助目标电压的 Huber loss；
- `w_physics`：默认 0.1；
- 电压先除以 0.05 V 再计算 loss，使数值尺度可解释。

DFN 的 forward sensitivity 组成：

```text
J = d voltage(t) / d [log_conductivity_scale, current]
```

自定义 PyTorch backward 执行：

```text
dL/dinput = J^T * dL/dvoltage
```

随后 PyTorch 自动把梯度从 `log_conductivity_scale` 传回 MLP 参数。

## 6. 完整执行流程

### E0：环境修复与安装

```powershell
cd C:\Users\zhy\Desktop\gradcell\grad_cell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[physics,dev]"
python -c "import torch, pybamm; print(torch.__version__, pybamm.__version__)"
```

验收：NumPy、PyTorch、PyBaMM 可正常导入，且 Python ABI 与二进制 wheel 一致。

### E1：软件测试

```powershell
pytest -q
ruff check src scripts tests
```

重点测试：性质模型输出为正；解析物理层方向导数与有限差分一致；只使用物理电压 loss 时网络仍收到非零梯度。

### E2：准备 CALiSol-23 与少量 DFN 标签

```powershell
python scripts\prepare_electrolyte_v1_data.py `
  --physics-backend dfn `
  --physics-samples 32 `
  --time-points 41 `
  --probe-horizon-s 600 `
  --current-a 5 `
  --seed 7 `
  --output data\electrolyte_v1_dfn_s7.npz
```

脚本会在缺少 CSV 时从 DTU Data/Figshare 官方下载 CALiSol-23。产物包含特征、`log(k)`、DOI group、物理掩码、DFN 电压目标和完整元数据。

验收：

- `physics_samples_valid > 0`；
- 电导率均为正且有限；
- DFN 电压轨迹有限；
- 失败样本及原因打印出来；
- 不把模型域电压标成实验电压。

### E3：梯度数值验证

先跑解析后端：

```powershell
python scripts\validate_electrolyte_v1_gradients.py --backend analytic --eps 1e-6
```

再跑真实 DFN：

```powershell
python scripts\validate_electrolyte_v1_gradients.py `
  --backend dfn `
  --time-points 21 `
  --probe-horizon-s 300 `
  --eps 1e-4
```

验收：解析后端相对方向误差 `<1e-5`；DFN 建议 `<2e-2`，并在 `eps=1e-3,1e-4,1e-5` 下检查稳定区间。必须同时检查梯度符号、有限性和非零性。

### E4：联合端到端训练

```powershell
python scripts\train_electrolyte_v1.py `
  --data data\electrolyte_v1_dfn_s7.npz `
  --physics-backend dfn `
  --epochs 100 `
  --batch-size 256 `
  --physics-batch-size 2 `
  --physics-weight 0.1 `
  --seed 7 `
  --output-dir results\electrolyte_v1_dfn_s7
```

每个更新同时取一个实验性质 batch 和一个小 DFN batch。因此所有训练轮次都有常规监督信号，且每个更新都有可选的在线 PyBaMM 物理梯度。

### E5：一键执行

```powershell
python scripts\run_electrolyte_v1_pipeline.py `
  --physics-backend dfn `
  --physics-samples 32 `
  --epochs 100 `
  --seed 7 `
  --output-dir results\electrolyte_v1_dfn_s7
```

Linux 服务器可直接使用：

```bash
SEED=7 PHYSICS_SAMPLES=32 EPOCHS=100 \
  bash scripts/run_electrolyte_v1_server.sh
```

该脚本依次执行测试、数据准备、解析梯度、DFN 多 epsilon 梯度扫描和联合训练。环境安装仍应由服务器的虚拟环境或作业镜像负责。

低成本流程验证可先使用：

```powershell
python scripts\run_electrolyte_v1_pipeline.py `
  --physics-backend analytic `
  --physics-samples 16 `
  --epochs 3 `
  --output-dir results\electrolyte_v1_analytic_smoke
```

## 7. 正式对照实验

至少比较：

1. 性质监督模型：`physics_weight=0`；
2. 性质监督 + 解析物理层：只验证软件链路；
3. 性质监督 + 在线 DFN loss：第一版完整模型；
4. 不同 DFN 数据比例：0、8、16、32、64；
5. 不同随机种子：7、17、27；
6. 随机行切分与 DOI group split：仅用于证明泄漏影响，正式结果只采用 group split。

## 8. 验收标准

- 性质测试集：模型优于训练集均值基线，并报告 MAE、RMSE、R²、相对误差；
- 梯度：DFN `Jᵀv` 与中心有限差分方向一致；
- 端到端：只保留 DFN 电压 loss 时，网络参数仍得到非零有限梯度；
- 训练：性质 loss 和 DFN loss 均不出现 NaN/Inf，DFN failure 不被静默删除；
- 物理解释：只声称在 Chen2020 模型域中完成端到端可微训练验证；
- 最终放电评价：使用真实最低电压截止，不把 600 s 探针窗口当作实际放电时间。

## 9. 当前边界与下一步

- 单一倍率修正不足以表达完整的 `kappa(c_e,T,composition)`；下一版可预测低维函数系数。
- CALiSol-23 同时存在质量比、体积比、摩尔比，以及 mol/L、mol/kg，正式化学泛化前需进行单位与组成标准化。
- DFN 辅助电压不是实验电芯标签，不能用来证明真实电芯准确性。
- 第一版只证明一阶端到端梯度与联合训练可行；事件时间梯度、容量 loss、多性质输出和真实整电芯性能属于后续阶段。
