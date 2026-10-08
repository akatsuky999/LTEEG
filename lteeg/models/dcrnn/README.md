# DCRNN

图扩散卷积循环网络。把每个导联当作图上的一个节点，用"扩散卷积 + GRU"在时间上逐步建模，每个时间步（默认 1 秒）输出一个发作 logit。

## 来源

- Li, Yu, Shahabi, Liu. *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* ICLR 2018（DCRNN 原始模型）。
- Tang, Dunnmon, Saab, Zhang, Huang, Dubost, Rubin, Lee-Messer. *Self-Supervised Graph Neural Networks for Improved Electroencephalographic Seizure Analysis.* ICLR 2022（DCRNN 在 EEG 癫痫检测上的设置）。
- 参考实现：[tsy935/eeg-gnn-ssl](https://github.com/tsy935/eeg-gnn-ssl)，commit `3e60b23`，`model/cell.py`、`model/model.py` 中的 `DCRNNModel_classification`，以及 `data/data_utils.py`、`data/dataloader_detection.py` 中的特征和建图函数。

## 计算流程

输入窗口 `x`：`(B, N, T)`，N 个导联即 N 个图节点。

1. **切时间步**：按 `step = step_sec × fs` 个采样点切成 `S = ceil(T / step)` 段，最后一段不足时补零。每个导联的每一段变成一个特征向量：默认是 FFT 对数幅度谱（`floor(step/2)` 个频点，含直流、不含 Nyquist），得到 `(B, S, N, D)`。
2. **建图**：
   - 默认 `graph: correlation`，每个窗口单独建图。两导联之间的权重 = 两者特征序列零延迟归一化互相关的绝对值。每个节点保留自环和最强的 `top_k` 个邻居，得到有向图，再转成"双向随机游走"支撑矩阵。
   - 也可以用 `graph: static` 加上 `adjacency` 给一张固定的图，例如按电极距离构造，默认转成缩放拉普拉斯矩阵。
3. **特征标准化**：每个导联一组统计量，见下文"与参考实现的差异"。
4. **DCGRU 编码器**：`num_rnn_layers` 层，每层在 S 个时间步上递推。GRU 的门和候选状态都由扩散卷积计算，扩散阶数为 `max_diffusion_step`。
5. **输出头**：对每个时间步、每个节点做 `Linear(ReLU(Dropout(h)))`，再在节点维上取最大值，得到 `(B, num_outputs, S)`。框架再把它线性插值回逐采样点。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `step_sec` | 1.0 | 时间步长（秒）；`step_sec × fs` 必须是整数个采样点 |
| `features` | `fft` | `fft`：FFT 对数幅度；`raw`：直接用原始采样点 |
| `graph` | `correlation` | `correlation`：每窗口互相关图；`static`：使用 `adjacency` |
| `top_k` | 3 | 互相关图中每个节点保留的邻居数 |
| `adjacency` | `null` | `graph: static` 时的 N×N 非负权重矩阵 |
| `filter_type` | 自动 | `dual_random_walk`（互相关图默认）、`random_walk`、`laplacian`（固定图默认） |
| `num_rnn_layers` | 2 | DCGRU 层数 |
| `rnn_units` | 64 | 每个节点的隐藏单元数 |
| `max_diffusion_step` | 2 | 扩散阶数 K |
| `activation` | `tanh` | 候选状态的激活函数：`tanh` 或 `relu` |
| `dropout` | 0.0 | 输出头之前的 dropout |
| `feature_norm` | `batch` | `batch`：按导联做无仿射的 BatchNorm；`none`：不做标准化 |

`fs`（采样率）由框架自动传入，不在 `model.params` 里设置。

## 逐点化的改动

参考实现做的是片段分类：60 秒片段，取顶层 GRU **最后一个时间步**的隐状态，经输出头得到一个 logit。这里把同一个输出头作用在**每一个时间步**上，得到每秒一个 logit。在最后一个时间步上，结果与参考实现的片段 logit 完全相同。

## 与参考实现的差异

1. **特征标准化**：参考实现用训练集上预先算好的每导联均值和标准差。这里用一个无仿射参数的 `BatchNorm1d(N)` 扮演同样的角色：统计量同样按导联计算；`momentum=None` 时运行均值是累积平均，会收敛到训练集统计量。设 `feature_norm: none` 可关闭。
2. **建图位置**：参考实现在数据加载器里用 numpy 建图；这里在模型内部、对整个 batch 一次性建图，不参与反向传播。建图同样使用标准化之前的特征，互相关在 float64 下计算，保证邻居排序不受设备和舍入影响。
3. **节点**：Tang 等人用 19 个单极导联（10-20 系统）；这里节点就是 `data.channels`，CHB-MIT 默认为 18 个双极导联。
4. **固定图**：参考实现的"距离图"从预先算好的文件读入；这里由 `adjacency` 参数直接给出矩阵。
5. **不包含自监督预训练**：Tang 等人的主要贡献是用"预测下一段信号"做预训练。这里只提供有监督的 DCRNN。
6. **长度不是步长整数倍时**：模型先补零到整数个时间步，在补齐后的长度上插值，再裁回 T，输出与输入逐点对齐。框架训练和推理用的窗口长度（60 s × 256 Hz）正好是整数倍，不会走到这条路径。

为了与参考实现保持一致，以下两处行为**有意保留**：

- 扩散卷积在多个支撑矩阵之间没有重置递推变量。这是原始 DCRNN 代码就有的行为：有两个支撑矩阵且 `max_diffusion_step ≥ 2` 时，第二个支撑矩阵从 `S₁x` 而不是 `x` 开始扩散。
- 门控偏置初始化为 0。参考实现在前向里传入的 `bias_start=1.0` 实际上没有生效。

使用时还应注意：GRU 是单向的，一个时间步的隐状态只看到窗口内它之前的时间步。但互相关图是用整个窗口算的，所以不能把这个模型当作严格因果的实时检测器。

## 核验

在同一组权重和输入上，与参考实现逐项对比（随机信号，B=3，18 导联，60 个 1 秒时间步，256 Hz）：

| 对比项 | 最大绝对误差 |
|---|---|
| FFT 对数幅度特征 | 1.6e-13 |
| 互相关邻接矩阵（邻居集合完全一致） | 3.0e-8 |
| 双向随机游走支撑矩阵 | 3.8e-8 |
| 缩放拉普拉斯矩阵 | 5.4e-8 |
| 全部 60 个时间步的 logits | 1.5e-7 |

参考实现的 `state_dict` 可以 `strict=True` 加载进本模型（`feature_norm: none` 时），说明参数名和形状完全一致。`tests/test_dcrnn.py` 用 numpy 复述了参考实现的各个定义，把这些行为固定下来。

## 使用

```bash
python -m lteeg check-model --config configs/experiments/dcrnn.yaml --batch-size 40
python -m lteeg train --config configs/experiments/dcrnn.yaml
```

`configs/experiments/dcrnn.yaml` 采用 Tang 等人的训练设置：Adam，学习率 3e-4，L2 系数 5e-4，余弦退火，梯度裁剪 5，batch 40。

默认规模约 0.31M 参数。在 CPU 上，batch 4、60 秒窗口的一次前向加反向约 0.24 s。

固定图：在实验配置里给出一个 N×N 的非负矩阵，行列顺序与 `data.channels` 相同（CHB-MIT 默认 18×18），例如按导联中点之间的距离用高斯核构造：

```yaml
model:
  name: dcrnn
  params:
    graph: static
    adjacency: [[1.0, 0.8, ...], ...]   # 18 x 18
```
