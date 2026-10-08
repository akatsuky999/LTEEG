# SeizureTransformer

U 形卷积网络，瓶颈处接一个 Transformer，每个采样点输出一个 logit。2025 年 SzCORE 癫痫检测挑战赛第一名，也是本框架的默认 baseline。

## 来源

- Wu, Zhao, Yener. *SeizureTransformer: Scaling U-Net with Transformer for Simultaneous Time-Step Level Seizure Detection from Long EEG Recordings.* arXiv:2504.00336, 2025（后续版本题为 *Large EEG-U-Transformer for Time-Step Level Detection Without Pre-Training*）。
- 参考实现：原项目 `time_step_level/model.py`。

## 结构

以 60 秒、256 Hz（15360 个采样点）的输入为例：

- `layers/unet.py`：**编码器** 5 级卷积，每级 MaxPool 把长度减半（15360 → 480），每级输出留作跳跃连接；**解码器** 5 级最近邻上采样，每级加上对应的跳跃连接，恢复到 15360。
- `layers/res_cnn.py`：瓶颈处的 7 个残差卷积块。
- `layers/positional.py`：正弦位置编码。
- `model.py`：把以上部件组装起来，在瓶颈处接 8 层 Transformer 编码器，最后用 1 维卷积输出 `num_outputs` 个通道。

## 输入与输出

- 输入 `(B, C, T)`，T ≥ 32 即可，不必等于训练窗长；
- 输出 `(B, num_outputs, T)`，每个采样点一个 logit。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `dim_feedforward` | 2048 | Transformer 前馈层宽度 |
| `num_layers` | 8 | Transformer 层数 |
| `num_heads` | 4 | 注意力头数 |
| `drop_rate` | 0.1 | 残差卷积块的空间 dropout |
| `transformer_dropout` | 0.1 | Transformer 与位置编码的 dropout |
| `filters` | (32, 64, 128, 256, 512) | 编码器各级通道数，最后一级即 Transformer 维度 |
| `kernel_sizes` | (11, 9, 7, 7, 5, 5, 3) | 编码器各级卷积核；解码器使用其反序的前若干个 |
| `res_cnn_kernels` | (3, 3, 3, 3, 2, 3, 2) | 7 个残差卷积块的卷积核 |
| `norm_first` | false | Transformer 是否用 pre-norm |

默认规模 37.85M 参数。

## 与原实现的差异

在原配置下，以下改动都不改变计算结果：

- 输出 logits 而不是 sigmoid 概率，以便用数值稳定的 `BCEWithLogits`，也才能开混合精度和多分类输出。
- 删掉了一层从未参与计算的冗余 Transformer 层。`nn.TransformerEncoder` 会深拷贝传入的层，原实现因此多带了约 315 万个闲置参数（41.0M → 37.85M）。
- 用 `MaxPool1d(ceil_mode=True)` 处理奇数长度（与原实现的 -1e10 右侧补齐等价），解码器按跳跃连接的长度裁剪，所以接受任意长度 ≥ 32 的输入。
- 位置编码表按输入长度确定大小，且不存入 state dict。
- Transformer 使用 `batch_first`：参数相同，计算更快。

## 核验

- 与原实现在相同权重和输入下逐点对比，最大误差约 1.2e-7。
- 重构为独立文件夹之后，参数名、相同随机种子下的初始化和前向输出都与重构前**逐位相同**，包括默认规模的模型。

## 加载原作者的权重

```bash
python -m lteeg train --set model.init_checkpoint=path/to/original.pth
```

`load_weights` 会自动调用 `SeizureTransformer.convert_state_dict`，去掉 `module.` 前缀、冗余层和位置编码表。

**注意**：比赛权重是在 19 个单极导联上训练的，第一层卷积的输入通道数与本框架 CHB-MIT 默认的 18 个双极导联不同，不能直接加载。需要先把数据换成相同的导联方案。

## 使用

```bash
python -m lteeg check-model --batch-size 86
python -m lteeg train
```
