# <模型名>

> 复制本文件夹后，把这份 README 改成你的模型说明。下面的小节是建议保留的结构。

## 来源

论文（作者、会议、年份、链接）和参考实现（仓库地址、commit）。

## 输入与输出

- 输入：`(B, C, T)`，C = `len(data.channels)`，T = 训练窗口的采样点数。
- 输出：`(B, num_outputs, T')`。说明 T' 与 T 的关系（例如每 `stride` 个采样点一个 logit）。
- 是否需要框架额外提供 `fs` / `channel_names`。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `hidden` | 32 | 隐藏通道数 |
| `depth` | 2 | 残差块个数 |
| `stride` | 4 | 输出时间分辨率（每多少个采样点一个 logit） |

## 与参考实现的差异

逐条列出改动，并说明是否改变了计算结果。

## 核验

如何确认移植正确，例如与参考实现在相同权重和输入下的最大误差。

## 使用

```bash
python -m lteeg check-model --set model.name=<模型名>
python -m lteeg train --set model.name=<模型名> "model.params={hidden: 64}"
```
