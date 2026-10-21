# DeCoSwin 代码

这是论文 **“Deformation-Aware Local Prototype Sampling for Oracle Bone Script Recognition”** 的 PyTorch 代码。代码不下载数据集或预训练权重。当前版本保留论文的主要计算路径，并提供易于接入数据的训练与推理接口。

## 实现了什么

| 论文模块 | 本仓库实现 |
| --- | --- |
| Swin-T 主干 | `torchvision.models.swin_t`，输入为 128×128；第 2/3/4 阶段分别输出 192/384/768 通道。 |
| AOPA，式 (1)–(5) | 在第 2 阶段的 16×16 特征上预测 8×8 偏移，双线性采样，和 256 个可学习共享原型做余弦匹配，每位置取 top-16，生成门控并乘回原始锚点特征；含正交损失。 |
| DMF，式 (6) | 第 3、4 阶段各一个模块：96 通道投影、膨胀率 1/3/5 的三条可变形深度卷积分支、融合及初值为 0.5 的残差系数。 |
| 分类头 | 汇合 384+768+768 通道的全局池化向量，投影到 768 通道再分类。 |
| 训练 | 类别平衡交叉熵、`1e-3` 倍原型正交损失、AdamW、余弦退火、梯度裁剪；可选简化 SimMIM 和原型 K-means 初始化。 |

代码入口：`decoswin/model.py`、`decoswin/train.py`、`decoswin/pretrain.py`、`decoswin/predict.py`。关键张量形状和近似处均有注释。

## 当前代码的实现范围

- 原文没有公开偏移预测器、可变形卷积算子等全部底层细节；这里用纯 PyTorch `grid_sample` 实现可学习的逐核点偏移。偏移幅度上限设为 1.5 个特征像素，这属于实现选择。
- 可选 SimMIM 只重建 4×4 图像块并迁移 patch embedding 与第 1 阶段。想接近论文训练条件，应自行提供合法获取的 ImageNet Swin-T 权重，以及论文协议的无标签预训练图像。
- 增广只包含旋转和亮度/对比度扰动；论文中的噪声、形态学变换和骨架损坏未在简化版实现。没有水平翻转。
- 数据划分、去重、缺席训练类处理、三随机种子实验及合成鲁棒性评估由数据使用者根据论文协议执行。本仓库不包含 HUST-OBC、EV-OBC 数据或论文权重。
- 若未使用论文中的数据、权重和训练条件，请勿将运行结果与论文表格数字直接比较。

## 安装

建议 Python 3.10+，GPU 训练建议 CUDA 版 PyTorch。先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装匹配本机 CUDA 的 `torch` 与 `torchvision`，然后：

```bash
pip install -r requirements.txt
```

Windows 上如果已有 Anaconda 环境混装多个 OpenMP 库，请先换干净的虚拟环境；不要把该环境问题当作模型错误。

## 接入数据

创建 UTF-8 CSV，表头为 `path,label,split`。`path` 可以是相对 CSV 所在目录的路径，也可以是绝对路径。`label` 是任意非空类别字符串，`split` 用 `train`、`val`、`test`。例如：

```csv
path,label,split
images/a/001.png,a,train
images/a/002.png,a,val
images/b/001.png,b,train
images/b/002.png,b,test
```

所有验证/测试类别必须在训练集出现。`val` 可省略：此时保存最后一轮；存在 `val` 时保存验证 Top-1 最优轮次。`test` 可省略。对 HUST-OBC 请优先使用官方划分并按论文处理泄漏与训练缺席类别；对 EV-OBC 请自行固定划分与随机种子。不要把测试集当作验证集。

预处理统一转 RGB、白底补方形、缩放到 128×128，并映射到 `[0,1]`。若原始图像的前景/背景定义不同，请在 `decoswin/data.py` 中调整预处理。

## 直接训练

```bash
python -m decoswin.train --manifest data/split.csv --output checkpoints/decoswin.pt \
  --epochs 50 --batch-size 64 --init-prototypes
```

若不传 `--init-prototypes`，原型随机初始化，能更快检查流程。第一次跑通建议 `--epochs 1 --batch-size 2 --num-prototypes 32 --topk 8`。完整默认值为 256 原型与 top-16；在类别很多时分类层会显著占显存。默认自动选 CUDA，CUDA 不可用时使用 CPU。Windows `cmd`/PowerShell 请将续行命令写在一行，或使用各自的续行符。

可选：下载权重是使用者自己的步骤。本程序只读取本地 `torchvision` Swin-T 格式权重：

```bash
python -m decoswin.train --manifest data/split.csv \
  --imagenet-weights path/to/swin_t_torchvision.pth --init-prototypes
```

## 可选 SimMIM 预训练

将允许使用的无标签图像放入任意目录（可有子目录）：

```bash
python -m decoswin.pretrain --images data/unlabeled \
  --imagenet-weights path/to/swin_t_torchvision.pth \
  --output checkpoints/simmim.pt --epochs 50
python -m decoswin.train --manifest data/split.csv \
  --imagenet-weights path/to/swin_t_torchvision.pth \
  --simmim-weights checkpoints/simmim.pt --init-prototypes
```

SimMIM 只覆盖主干前两部分；训练时先载入本地 ImageNet 权重，再以 SimMIM 的前层权重覆盖对应参数。若不提供 ImageNet 权重，其余阶段从随机权重开始。

## 推理与测试

```bash
python -m decoswin.predict --checkpoint checkpoints/decoswin.pt --image path/to/glyph.png --topk 5
```

训练程序每轮输出 loss 和验证 Top-1/Top-10；有 `test` 划分时，结束后用保存的检查点输出测试 Top-1/Top-10。输出的小数是比例，例如 `0.85` 表示 85%。

快速检验代码：

```bash
python -m pytest -q
```

## 实验记录建议

固定 Python、PyTorch、torchvision 版本、CSV 划分、随机种子与权重来源。论文 HUST-OBC 在训练缺席类别和训练/测试重复图像上有特别处理，EV-OBC 使用固定随机 9:1 划分；实验报告中请记录这些决定，以及是否使用了 SimMIM、ImageNet 初始化、原型 K-means 和完整数据增广。`checkpoints/`、`data/` 均已加入 `.gitignore`，避免误上传数据或大权重。

## 论文

Yanjie Wang, Dezhun Yu, Jiwei Song, Liang Chang. *Deformation-Aware Local Prototype Sampling for Oracle Bone Script Recognition*, ICASSP 2027 manuscript.
