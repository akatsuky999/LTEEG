# 在服务器（A100）上启动：CHB-MIT + SeizureTransformer

本文是一份照着做就能跑起来的启动清单，面向这样的环境：

- Linux 服务器，NVIDIA A100（40 GB 或 80 GB 均可），磁盘空间充足；
- CHB-MIT 已经转换成 LTEEG 要求的 h5 格式（见下文第 2 步）；
- 目标：用默认配置训练 SeizureTransformer baseline，在 dev 的完整长程记录上做事件级评估。

整个流程分 9 步。前 6 步都是检查和准备，每一步有问题都会当场报错，修好再往下走。

```
1 装环境 → 2 放数据 → 3 写机器配置 → 4 校验数据 → 5 测显存 → 6 建缓存
       → 7 空跑 + 冒烟 → 8 正式训练 → 9 训练后评估
```

> 说明：框架此前只在合成数据和 CPU 上验证过完整流程，还没有在真实 CHB-MIT 和 CUDA 上跑过。所以第 4 步和第 7 步不要跳过；第一次出现的任何报错，请保留完整日志。

---

## 1. 装环境

```bash
git clone <仓库地址> LTEEG
cd LTEEG            # 进入含 pyproject.toml 的框架目录（若仓库根目录下还有一层 LTEEG/，再 cd 一次）

conda create -n lteeg python=3.10 -y
conda activate lteeg

# 按 nvidia-smi 右上角显示的 CUDA 版本选择 torch，例如 CUDA 12.x：
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"

python -m pytest -q    # 应全部通过
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最后一行必须打印 `True` 和 `NVIDIA A100 ...`。如果是 `False`，说明装的是 CPU 版 torch 或驱动版本不匹配，先解决这个再继续。

## 2. 放数据

把 h5 数据拷到服务器上，比如 `/data/EEG/CHB-MIT`（路径里不要有空格）。目录结构必须是：

```
/data/EEG/CHB-MIT/
├── chb01/
│   ├── chb01_01.h5
│   ├── chb01_02.h5
│   ├── ...
│   └── chb01_annotations.txt
├── chb02/
...
└── chb24/
```

- 每个 h5 里有数据集 `signals`，形状 `(18, 采样点数)`，属性 `fs = 256`；
- 18 行的顺序必须是 `FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8, P8-O2, FZ-CZ, CZ-PZ`；
- 标注文件 `chbXX_annotations.txt` 的 `[seizures]` 段：表头一行以 `file` 开头，之后每行 `文件名<TAB>开始秒<TAB>结束秒`。

**强烈建议**在 EDF→h5 转换时把导联名也写进 h5 属性（例如 `channels`）。否则框架只能检查"是不是 18 行"，没法核对顺序。CHB-MIT 有少数记录中途换过导联方案或缺导联，转换脚本对它们怎么处理，要心里有数。

## 3. 写机器配置

和机器有关的设置（数据在哪、缓存放哪、结果放哪）不写进仓库里的 yaml，而是放在一个环境文件里，每次启动前 `source` 一下：

```bash
mkdir -p /data/lteeg
cat > ~/lteeg.env <<'EOF'
export LTEEG_SET="data.root=/data/EEG/CHB-MIT data.cache_dir=/data/lteeg/cache experiment.output_dir=/data/lteeg/runs"
EOF
source ~/lteeg.env
```

之后所有命令都写成 `--set $LTEEG_SET 其他覆盖项...`。两个注意点：

- `$LTEEG_SET` **不要加引号**，否则 shell 不会把它拆成多个参数；
- **一条命令里只能出现一次 `--set`**。写两次的话，只有最后一次生效。需要额外覆盖时，接在同一个 `--set` 后面：`--set $LTEEG_SET train.epochs=2`。

这样做的好处是，任何实验配置（比如 `configs/experiments/original_faithful.yaml`）都可以直接配上同一份机器设置，不需要再复制一份 yaml。

## 4. 校验数据

```bash
python -m lteeg inspect --set $LTEEG_SET --out /data/lteeg/inspect.json
```

它会逐个文件检查：采样率是否 256 Hz、是否 18 行、有没有存成转置、标注里的文件名能否对上 h5、发作有没有越界或重叠，最后核对发作总数。它只读文件头和标注，不读信号本身，所以很快；信号里的 NaN/Inf 会在第 6 步建缓存时检查。

通过时，输出末尾应满足：

- 每个病人一行统计，`train` 的 TOTAL 一栏 events = **159**，`dev` = **39**；
- 最后一行是 `All validation checks passed.`；
- 留意倒数第二行 `Channel names verified from file attributes in N recording(s)`：N 是用导联名核对过顺序的记录数。如果 N = 0，说明 h5 里没有导联名，顺序只能靠转换脚本保证。

没通过时，报错会指出具体文件和行号。修数据，或者把个别坏记录加进排除列表（例如 `--set $LTEEG_SET data.exclude_records=[chb12/chb12_27]`，被排除的文件会写进日志）。注意，排除含发作的记录会让发作总数对不上，需要同步修改 `configs/chbmit_split.json` 里的 `expected_seizures`，并在报告中写明。

## 5. 测显存，定 batch

```bash
CUDA_VISIBLE_DEVICES=0 python -m lteeg check-model --batch-size 86 --set $LTEEG_SET
```

看输出里的 `peak GPU memory (...)` 一行。按 CPU 上的实测推算，60 s 窗口每个样本约占 0.45 GB，batch 86 约需 40 GB，再加上优化器状态约 0.5 GB：

| 显卡 | 设置 |
|---|---|
| A100 80 GB | 用默认值（batch 86），什么都不用加 |
| A100 40 GB | batch 86 大概率放不下（`check-model` 报 CUDA out of memory）。在 `~/lteeg.env` 的 `LTEEG_SET` 末尾加上 `train.batch_size=43 train.accum_steps=2`，等效 batch 仍是 86 |

改完再跑一次 `check-model --batch-size 43` 确认放得下。

> A100 支持 bf16，`train.amp=bf16 inference.amp=bf16` 能把显存和耗时大约减半。但它改变了数值，而且没有在 CUDA 上测过。建议先用 fp32 跑出 baseline，再把 bf16 当作单独的实验。

## 6. 建缓存

```bash
python -m lteeg prepare --jobs 8 --set $LTEEG_SET
```

- 把每条记录做一次 z-score + 带通 + 陷波，存成可随机读取的 float32 文件，总共约 **65 GB**。磁盘充足，就保持 float32；
- 读信号时会检查 NaN/Inf，有问题的记录会直接报错并给出文件名；
- 每个进程同时处理一条记录，内存峰值约 2 GB，`--jobs 8` 需要约 16 GB 内存；
- 缓存目录名带有预处理参数的指纹。预处理配置不变就直接复用；配置变了会自动建一份新的（例如 `original_faithful` 实验会再占约 65 GB）；
- 这一步可以跳过，`train` 会自动补齐。但提前建好，正式训练时就不用等，多个实验同时启动时也不会抢着建同一份缓存。

## 7. 空跑 + 冒烟

先空跑，把数据、缓存、采样器和模型都准备一遍，然后退出：

```bash
CUDA_VISIBLE_DEVICES=0 python -m lteeg train --dry-run --set $LTEEG_SET
```

看日志里的这两处：

- `Data summary`：每个病人的记录数、时长、发作数；
- `Training windows available {...}, selected {...}`：各类窗口可用多少、每个 epoch 选了多少，以及发作采样点占比。

空跑会留下一个只有配置和统计的运行目录，可以删掉。

再跑 2 个 epoch，确认整条链路，同时测速度：

```bash
CUDA_VISIBLE_DEVICES=0 python -m lteeg train --set $LTEEG_SET experiment.name=smoke train.epochs=2
```

从日志里读三个数：

| 日志里的字样 | 含义 |
|---|---|
| `... win/s` | 训练吞吐，单位为每秒窗口数 |
| `epoch 1/2 train loss ... (XXs)` | 一个 epoch 的训练耗时 |
| `scored N recordings (... h of EEG) in XXs (...x real time)` | 一次 dev 整段推理的耗时（dev 约 220 小时 EEG） |

正式训练总耗时 ≈ 100 × (训练耗时 + dev 推理耗时)。如果 dev 推理占大头，可以加 `train.val_every=2`（每 2 个 epoch 验证一次，选模型的粒度也会相应变粗）。

2 个 epoch 的模型通常还学不到东西，在 0.8 阈值下 event F1 为 0 很正常。冒烟只是为了确认流程能走通。冒烟目录用完可以删掉，免得混进后面的 `compare`。

## 8. 正式训练

放在 tmux 里跑，断开 SSH 也不会中断：

```bash
tmux new -s lteeg
conda activate lteeg && source ~/lteeg.env && cd <框架目录>
CUDA_VISIBLE_DEVICES=0 python -m lteeg train --set $LTEEG_SET
# Ctrl-b 再按 d 退出 tmux；tmux attach -t lteeg 回来
```

默认配置和原论文一致：100 个 epoch、batch 86、RAdam lr 1e-4、不早停。每个 epoch 末在 dev 上整段推理，按 pooled event F1 保存 `best.pt`。

运行目录是 `/data/lteeg/runs/seizure_transformer_<时间戳>/`。训练过程中可以这样看：

```bash
tail -f /data/lteeg/runs/seizure_transformer_*/train.log   # 每个 epoch 的损失和验证指标
nvidia-smi                                                  # 显存、利用率
```

`train_log.csv` 每个 epoch 一行，可以直接画学习曲线。如果 `nvidia-smi` 里 GPU 利用率长时间低于 70%，说明数据读取跟不上，下次启动加 `train.num_workers=4`。

**中断了怎么办**（断电、被抢占、手动停止）：

```bash
python -m lteeg train --resume /data/lteeg/runs/seizure_transformer_<时间戳>
```

续训会读取运行目录里的 `config.yaml` 和 `checkpoints/last.pt`，从上一个完成的 epoch 接着跑，随机状态也会恢复。想多训几轮，可以加 `--set train.epochs=150`。

**多张 A100**：框架每次运行只用一张卡。多卡的用法是同时跑多个实验，每个实验用 `CUDA_VISIBLE_DEVICES` 指定一张卡，例如第二张卡跑"完全复刻原实现"的对照：

```bash
CUDA_VISIBLE_DEVICES=1 python -m lteeg train --config configs/experiments/original_faithful.yaml --set $LTEEG_SET
```

## 9. 训练后评估

训练结束时，框架会自动用 `best.pt` 在 dev 上做一次正式评估，结果在 `eval/dev_best/`。先看 `summary.md`，各文件含义见 README 的"怎么读结果"一节。

再评估最后一个 epoch：

```bash
python -m lteeg evaluate /data/lteeg/runs/seizure_transformer_<时间戳>/checkpoints/last.pt
```

结果写到 `eval/dev_last/`。**为什么要多跑这一次**：现在 split 里没有独立的 test 集，`best.pt` 是在 dev 上挑出来的，它在 dev 上的分数偏乐观。`last.pt` 没有经过挑选，它的分数可以作为对照。两者一起报告，读者才能看出挑选带来了多少偏差。

把所有实验并排比较：

```bash
python -m lteeg compare /data/lteeg/runs
```

换阈值、换最短事件长度，不用重新推理：

```bash
python -m lteeg sweep /data/lteeg/runs/<运行>/eval/dev_best --thresholds 0.5 0.6 0.7 0.8 0.9 --min-durations 2 5 10
```

在 dev 上扫出来的最优阈值只能作为分析，不能当作 dev 上的成绩来报告。主结果用固定的 0.8。

### 报告时至少给出

- 评分规则：SzCORE 默认参数（容差 −30 s / +60 s，相隔 90 s 内的事件合并，超过 300 s 拆分）；后处理：阈值 0.8、形态学核 5、最短事件 2 s；
- pooled 的 event F1、灵敏度、precision、FA/24h；
- macro 的 event F1（均值 ± 标准差）和**最差病人**（`event_f1_macro_min`）；
- `best.pt` 和 `last.pt` 两行；
- 运行目录里的 `config.yaml` 和 `env.json`（torch/CUDA/GPU 版本）随结果一起存档。

---

## 常见问题

| 现象 | 处理 |
|---|---|
| `inspect` 报错 | 报错会指出文件名和行号；修数据，或用 `data.exclude_records` 排除并写明原因 |
| `torch.cuda.is_available()` 是 False | torch 装成了 CPU 版，或 CUDA 版本与驱动不匹配；按 `nvidia-smi` 显示的版本重装 torch |
| CUDA out of memory | 见第 5 步：减小 `train.batch_size`，用 `train.accum_steps` 补回等效 batch |
| 日志出现 non-finite loss | 框架会跳过这一步，连续超过 20 步就停止训练。多半是数据问题（某条记录全零或有异常值），用 `inspect` 的统计排查 |
| 训练很慢、GPU 利用率低 | 加 `train.num_workers=4`；确认缓存放在本地盘，不是网络盘 |
| 每个 epoch 的 dev 验证太慢 | `train.val_every=2` |
| 两个 `--set` 只有一个生效 | 一条命令只能有一个 `--set`，把所有覆盖项接在同一个后面 |
| 想确认配置最终是什么 | 看运行目录里的 `config.yaml`，它是本次运行实际生效的完整配置 |

## 命令汇总

```bash
source ~/lteeg.env
python -m lteeg inspect --set $LTEEG_SET --out /data/lteeg/inspect.json
CUDA_VISIBLE_DEVICES=0 python -m lteeg check-model --batch-size 86 --set $LTEEG_SET
python -m lteeg prepare --jobs 8 --set $LTEEG_SET
CUDA_VISIBLE_DEVICES=0 python -m lteeg train --dry-run --set $LTEEG_SET
CUDA_VISIBLE_DEVICES=0 python -m lteeg train --set $LTEEG_SET experiment.name=smoke train.epochs=2
CUDA_VISIBLE_DEVICES=0 python -m lteeg train --set $LTEEG_SET                      # 在 tmux 里
python -m lteeg evaluate /data/lteeg/runs/<运行>/checkpoints/last.pt
python -m lteeg compare /data/lteeg/runs
```
