# Dilated TCN

轻量的膨胀卷积网络：步幅卷积先把时间分辨率降低 `stem_stride` 倍，再叠加若干个膨胀残差块（膨胀率 1, 2, 4, …）扩大感受野，最后每 `stem_stride` 个采样点输出一个 logit。

## 来源

本框架自带的轻量 baseline，结构参考时间卷积网络（Bai, Kolter, Koltun. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* 2018），但卷积是非因果的（"same" 填充）。

## 用途

- 在真实数据上快速打通流程（`configs/experiments/tcn_debug.yaml`）；
- 作为参数量小、速度快的对照模型；
- 演示"输出比输入短"的情况：框架会把 logits 线性插值回逐采样点。

## 结构

- `model.py`：步幅卷积 stem（核长 `2 × stem_stride + 1`）+ BatchNorm + GELU → `levels` 个残差块 → 1×1 卷积输出头。
- `layers/residual.py`：残差块 `x + Conv1x1(Dropout(GELU(BN(DilatedConv(x)))))`。

## 输入与输出

- 输入 `(B, C, T)`；
- 输出 `(B, num_outputs, ceil(T / stem_stride))`。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `hidden` | 64 | 通道数 |
| `levels` | 8 | 残差块个数；第 i 块的膨胀率为 2^i |
| `kernel_size` | 3 | 膨胀卷积核长（必须是奇数） |
| `stem_stride` | 8 | 输出时间分辨率（每多少个采样点一个 logit） |
| `dropout` | 0.1 | 残差块内的 dropout |

默认规模约 0.15M 参数。

## 使用

```bash
python -m lteeg train --config configs/experiments/tcn_debug.yaml
```
