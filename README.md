# ARPES 能带识别：物理模拟监督的 SRCNN

依据上传的论文 **Peng et al., “Super resolution convolutional neural network for feature extraction in spectroscopic data”, Rev. Sci. Instrum. 91, 033905 (2020)**，DOI: [10.1063/1.5132586](https://doi.org/10.1063/1.5132586)。这是独立实现，提供模拟、训练、验证、推理和 feature map 可视化。

输入为二维 ARPES 强度矩阵 `I[E, k]`，输出为每个像素存在能带的置信图，再从置信图提取能带位置。它不计算材料的 DFT 能带，也不直接拟合自能、有效质量或哈密顿量参数。

## 论文方法与本实现

| 项目 | 实现 |
| --- | --- |
| 网络 | Conv(1→64, 9×9) → ReLU → Conv(64→32, 1×1) → ReLU → Conv(32→1, 5×5) → Sigmoid |
| 参数量、感受野 | 8,129；13×13 |
| 边界 | 三层均无 padding，输出 `(H−12, W−12)`，对应输入 `[6:-6, 6:-6]` |
| 标签 | 随机紧束缚模型的色散曲线，4 倍超采样抗锯齿，标签为 [0,1] |
| 输入 | 标签 → 洛伦兹能量展宽 → 费米分布及平滑矩阵元 → 泊松噪声 → 二维高斯仪器展宽 → 高斯读出噪声 |
| 损失 | 无类别加权的逐像素二元交叉熵；代码用数值稳定的 `BCEWithLogits`，推理时 Sigmoid |
| 优化器 | Adam；随机抽取 32×32 patch，与中心 20×20 标签对齐 |
| 特征图 | 导出第一层 64 通道、第二层 32 通道及第一层卷积核；按响应方差排列绘图 |

这里的监督目标是最终的能带位置 feature map；中间 feature map 由反向传播自动学习，无需人工标注。训练与验证按不同随机哈密顿量划分，测试再使用不同的随机种子，不将同一模型的相邻切片分到不同集合。

论文没有公布完整的紧束缚参数、全部噪声分布或作者训练权重。本实现选择实对称的 1–4 轨道模型：

```text
H(kx,ky) = H0 + Cx*cos(kx) + Sx*sin(kx) + Cy*cos(ky) + Cxy*cos(kx+ky)
```

对角化得到能带，再沿不同 ky 切片。`|Im Σ|` 在 10–45 meV 之间随机取值，每张谱图中保持常量；`Re Σ=0`。温度为 10–80 K，仪器高斯 sigma 为每轴 0.3–1 像素，峰值计数为 100–1000，读出噪声标准差为归一化峰值的 0–3%。这些补充选择及学习率属于本实现的设置，不是论文公布的全部参数。模拟参数集中在 `arpes_bands/simulation.py` 的 `SimulationConfig`。

论文采用自然图像 SRCNN 权重初始化、约 3 千万步训练；本实现默认从随机权重开始，提供短程训练样例，未声称复现论文精度。`--init-checkpoint` 可载入本项目格式的兼容权重进行 warm start；它会重置优化器及步数，不是精确断点续训。

## 安装

当前云实例已准备好 `/workspace/arpes-venv`，可直接使用：

```bash
cd /workspace/ARPES-bandextraction-extrema
source /workspace/arpes-venv/bin/activate
arpes-bands --help
```

其他 CPU 机器可安装到独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

当前依赖使用 `pyproject.toml` 声明兼容范围，并非完整锁定的环境；已验证的具体版本记录在 `VALIDATION.md`。

## GPU 训练

代码支持 NVIDIA CUDA。GPU 机器需先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与驱动匹配的 **CUDA 版 PyTorch**，再运行 `python -m pip install -e .`。CPU 版 PyTorch 不会因添加参数而变成 CUDA 版。

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
arpes-bands train --device cuda --batch-size 64 --steps 50000 --output outputs/gpu
```

默认 `--device auto` 自动使用可用 CUDA，否则使用 CPU；指定 `--device cuda` 而无 CUDA 时会明确报错。根据显存调整 batch size；论文规模训练成本需在目标硬件上实测。当前云实例未检测到 NVIDIA 设备，已完成的是 CPU 验证，GPU 路径尚未实机测试。

## 训练、评估和可视化

以下为已运行的短程训练（192 张训练谱图、16 张独立验证谱图）：

```bash
arpes-bands train --device cpu --steps 1000 --models 12 --slices 16 \
  --val-models 4 --val-slices 4 --eval-every 100 --output outputs/smoke
arpes-bands evaluate --checkpoint outputs/smoke/best.pt
arpes-bands demo --checkpoint outputs/smoke/best.pt
python -m unittest discover -s tests -v
```

不指定数据规模时，默认生成 20 个随机模型、每个模型 50 张切片，即 1000 张训练谱图。增加训练步数的示例：

```bash
arpes-bands train --steps 50000 --output outputs/train
```

输出文件：

- `outputs/smoke/best.pt`：按验证集 BCE 选出的权重、优化器状态和参数元数据。
- `outputs/smoke/last.pt`：最近一次评估时的检查点。
- `outputs/smoke/history.json`：训练、验证曲线数据及常量先验 BCE 基线。
- `outputs/evaluation.json`：独立模拟测试集指标，默认使用训练和验证之外的种子。
- `outputs/demo/comparison.png`：模拟输入、标签、预测对比；青色点为阈值筛选的 EDC 极大值。
- `outputs/demo/filters.png`、`layer1.png`、`layer2.png`：卷积核和特征响应。
- `outputs/demo/prediction.npz`：完整数值结果；同名 CSV 保存能带候选坐标。

## 实验谱图推理

推荐输入 `.npz`，包含 `intensity`、`energy`、`k` 三个数组：

```python
import numpy as np

# intensity.shape == (len(energy), len(k))；energy 为相对 EF 的 eV。
np.savez("experiment.npz", intensity=intensity, energy=energy, k=k)
```

```bash
arpes-bands infer --checkpoint outputs/smoke/best.pt \
  --input experiment.npz --output outputs/experiment.npz --features
```

支持仅含二维数组的 `.npy`，此时输出轴为原始像素坐标，不赋予物理单位。输入应为原始强度数据，避免使用带坐标文字或伪彩色的截图。轴须均匀采样；降序轴会连同数据翻转到升序。能量正方向须为从费米能以下向费米能以上，结合能为正的数据请先转换。

推理使用逐图 min-max 归一化，不默认平滑。CNN 对像素宽度敏感；实验谱峰的展宽与模拟集在像素尺度上应相近。默认 128 点、−0.8 到 0.1 eV 的模拟范围，对应洛伦兹 HWHM 约 1.4–6.4 像素；必要时先对实验谱图重采样，保留对应物理轴。

输出 `confidence` 是 `(H−12, W−12)`，其轴 `energy`、`k` 已裁掉输入每侧 6 点。`--features` 还会保存 `layer1`、`layer2` 和 `filters`；两组中间响应的空间轴是 `feature_energy`、`feature_k`，对应输入每侧裁掉 4 点。边缘不通过补零伪造预测。

CSV 从每列的能量分布曲线寻找局部极大值，默认置信阈值 0.5、最小间隔 2 像素，可用 `--threshold` 和 `--min-distance` 调整。输出是候选位置，交叉处不强行分配能带身份，也未执行全局曲线连接。端点不计作局部极大值。

## 适用范围

模拟训练的置信值不等于经过实验标定的概率。强烈变化的自能、矩阵元抑制、异常背景、费米能以上缺乏信号和复杂交叉都可能导致漏检或伪特征；应结合原始谱图、传统 EDC/MDC 拟合及独立实验核对。尚未用真实 ARPES 数据验证。`outputs/` 为本地生成结果，默认不纳入 Git。
