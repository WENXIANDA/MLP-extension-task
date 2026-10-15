# MLP 全连接神经网络学习笔记

> 以「手写数字识别（MNIST）」为例，完整讲解多层感知机（MLP）的原理、实现思路和代码。
> 本文所有代码均可直接运行，最终模型验证准确率约 **98%**。

---

## 一、什么是 MLP？

**MLP（Multi-Layer Perceptron，多层感知机）** 是最基础的神经网络。它由一层层**全连接层（Linear）**堆叠而成，每个神经元与上一层的所有神经元相连，通过非线性激活函数引入表达能力。

一张 28×28 的手写数字图片，先"拉平"成 784 维向量，再送入网络，经过若干隐藏层变换，最后输出 10 个分数（对应数字 0~9）。

```
输入 784 维 ──► 隐藏层1 (256) ──► 隐藏层2 (128) ──► 输出 10 维
              ReLU               ReLU              (不接激活，交给损失函数)
```

### 全连接层的数学表达

每个线性层做的运算就是：**`y = W·x + b`**

- `x`：输入向量（上一层传来的）
- `W`：权重矩阵（可学习的参数）
- `b`：偏置（可学习的参数）

然后再过一个激活函数（这里用 ReLU）：`ReLU(z) = max(0, z)`，给网络引入非线性。如果没有激活函数，多层线性层叠起来仍然等价于一个线性层，无法拟合复杂函数。

---

## 二、整体流程

做一次完整的手写数字识别，分四步：

1. **准备数据**：读取 MNIST 数据，按类别打乱并切分训练集/验证集
2. **定义模型**：搭建 MLP 网络
3. **训练**：前向传播 → 算损失 → 反向传播 → 更新参数
4. **评估**：在验证集上算准确率

下面逐一讲解代码。

---

## 三、完整代码

### 3.1 数据加载 `data_loader.py`

MNIST 原始文件是 `.gz` 压缩的二进制格式（`idx3-ubyte` 存图片、`idx1-ubyte` 存标签），需要手动解析。

```python
# -*- coding: utf-8 -*-
"""
数据加载模块：解析 MNIST 原始 .gz 文件，并按类别随机打乱等分 5 份。
前 4 份用于训练，第 5 份用于测试（验证）。
"""
import gzip
import struct
import numpy as np


def load_mnist_images(path):
    """读取 MNIST idx3-ubyte 图片文件，返回 (N, 28, 28) 的 uint8 数组。"""
    with gzip.open(path, 'rb') as f:
        # 文件头 16 字节：magic(4) + 图片数(4) + 行数(4) + 列数(4)
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"非法 magic 值: {magic}"  # 2051 是图片文件的标识
        buf = f.read(num * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
    return data


def load_mnist_labels(path):
    """读取 MNIST idx1-ubyte 标签文件，返回 (N,) 的 int64 数组。"""
    with gzip.open(path, 'rb') as f:
        # 文件头 8 字节：magic(4) + 标签数(4)
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"非法 magic 值: {magic}"  # 2049 是标签文件的标识
        buf = f.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def split_by_class(images, labels, n_folds=5, seed=42):
    """
    将每个类别的样本随机打乱，并等分为 n_folds 份。
    前 n_folds-1 份用于训练，最后 1 份用于测试。
    按类别切分的好处：保证训练/测试集里每个数字的比例一致，评估更公平。
    """
    rng = np.random.default_rng(seed)   # 固定随机种子，保证结果可复现
    train_idx, valid_idx = [], []
    for c in np.unique(labels):         # 遍历 0~9 每个类别
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)                # 随机打乱该类别
        folds = np.array_split(idx, n_folds)   # 等分成 5 份
        train_idx.extend(np.concatenate(folds[:-1]))  # 前 4 份 → 训练
        valid_idx.extend(folds[-1])                  # 第 5 份 → 测试
    train_idx = np.array(train_idx)
    valid_idx = np.array(valid_idx)
    rng.shuffle(train_idx)              # 再打乱训练集整体顺序
    rng.shuffle(valid_idx)
    return train_idx, valid_idx


def load_data(root='.'):
    """加载训练集并按类别 5 等分切分，返回训练/验证的 X 和 y。"""
    images = load_mnist_images(f'{root}/train-images-idx3-ubyte.gz')
    labels = load_mnist_labels(f'{root}/train-labels-idx1-ubyte.gz')
    train_idx, valid_idx = split_by_class(images, labels, n_folds=5, seed=42)
    X_train = images[train_idx]   # (48004, 28, 28)
    y_train = labels[train_idx]
    X_valid = images[valid_idx]   # (11996, 28, 28)
    y_valid = labels[valid_idx]
    return X_train, y_train, X_valid, y_valid


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid = load_data('.')
    print(f'训练集: {X_train.shape}, 标签: {y_train.shape}')
    print(f'验证集: {X_valid.shape}, 标签: {y_valid.shape}')
    # 打印每个类别在训练/验证集中的数量，验证 4:1 比例
    for c in range(10):
        n_tr = (y_train == c).sum()
        n_va = (y_valid == c).sum()
        print(f'类别 {c}: 训练 {n_tr} / 验证 {n_va}')
```

**关键点**：
- `struct.unpack('>IIII', ...)` 里的 `>` 表示大端字节序，`I` 表示 4 字节无符号整数。
- 按类别切分（`split_by_class`）是重点：如果直接整体随机切，可能出现某些数字训练多、测试少，评估不公平。

---

### 3.2 模型定义 `model.py`

把网络写成**可配置**的形式，隐藏层数量和每层神经元数都能灵活指定。

```python
# -*- coding: utf-8 -*-
"""
模型定义模块：可配置隐藏层层数、每层神经元个数、Dropout 等。
输入为 28*28=784 维，输出为 10 类。
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_sizes=(256,), num_classes=10,
                 dropout=0.0):
        """
        hidden_sizes: 元组，表示各隐藏层的神经元个数。
                      例如 (256,)        为 1 个隐藏层；
                      (256, 128)        为 2 个隐藏层；
                      (256, 128, 64)    为 3 个隐藏层。
        dropout: 丢弃率，0 表示不使用 Dropout。
        """
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        # 循环构建隐藏层：Linear + ReLU (+ 可选 Dropout)
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hs
        # 最后接一个输出层（不加激活函数，交给 CrossEntropyLoss 处理）
        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)   # 把各层打包成一个顺序网络

    def forward(self, x):
        x = x.view(x.size(0), -1)   # 展平：把 (batch, 28, 28) 变成 (batch, 784)
        return self.net(x)
```

**关键点**：
- 继承 `nn.Module`，实现 `__init__`（定义层）和 `forward`（定义数据流向）。
- `nn.Sequential` 把层串起来，数据依次流过。
- 输出层**不加** Softmax，因为 `CrossEntropyLoss` 内部会自动做 Softmax + 对数运算。

---

### 3.3 训练与评估 `trainer.py`

封装训练循环和验证，返回每轮的损失和准确率。

```python
# -*- coding: utf-8 -*-
"""
训练与评估模块：封装单次实验的训练和测试流程，返回准确率等指标。
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def build_dataloaders(X_train, y_train, X_valid, y_valid, batch_size=64):
    """构建训练/验证 DataLoader。"""
    # 转成 tensor，图片像素 /255 归一化到 [0,1]
    X_train_t = torch.tensor(X_train, dtype=torch.float32) / 255.0
    y_train_t = torch.tensor(y_train, dtype=torch.long)   # 标签用 long 类型
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32) / 255.0
    y_valid_t = torch.tensor(y_valid, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    valid_ds = TensorDataset(X_valid_t, y_valid_t)

    # shuffle=True：训练时打乱顺序；验证时不用打乱
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def train_and_evaluate(model, train_loader, valid_loader,
                       optimizer, criterion, num_epochs=5, verbose=True):
    """训练并返回 (最终验证准确率, 每轮验证准确率列表, 每轮训练损失列表)。"""
    device = next(model.parameters()).device
    val_accs = []
    train_losses = []
    for epoch in range(num_epochs):
        # ---- 训练阶段 ----
        model.train()                     # 切换到训练模式（启用 Dropout 等）
        running_loss = 0.0
        total_batches = len(train_loader)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()         # 1. 梯度清零（否则会累加）
            outputs = model(inputs)       # 2. 前向传播
            loss = criterion(outputs, labels)  # 3. 计算损失
            loss.backward()               # 4. 反向传播，计算梯度
            optimizer.step()              # 5. 用梯度更新参数
            running_loss += loss.item()
        avg_loss = running_loss / total_batches
        train_losses.append(avg_loss)

        # ---- 验证阶段 ----
        model.eval()                      # 切换到评估模式（关闭 Dropout）
        correct = 0
        total = 0
        with torch.no_grad():             # 不计算梯度，省内存、加速
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)  # 取分数最大的类别
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        val_accs.append(acc)
        if verbose:
            print(f'  Epoch [{epoch + 1}/{num_epochs}] '
                  f'Loss: {avg_loss:.4f}, Val Acc: {acc:.2f}%')
    return val_accs[-1], val_accs, train_losses
```

**训练循环 5 步（务必背下来）**：

```python
optimizer.zero_grad()   # 1. 清空旧梯度
outputs = model(x)      # 2. 前向传播
loss = criterion(outputs, y)  # 3. 算损失
loss.backward()         # 4. 反向传播求梯度
optimizer.step()        # 5. 更新权重
```

**关键点**：
- 训练用 `model.train()`，评估用 `model.eval()`——二者会影响 Dropout 等层的表现。
- 评估时用 `torch.no_grad()` 关闭梯度计算，节省内存。

---

### 3.4 主程序：训练一个 MLP

把所有模块串起来，训练一个 2 隐藏层的网络：

```python
# -*- coding: utf-8 -*-
"""主程序：训练一个 2 隐藏层 MLP 识别手写数字。"""
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import MLP
from trainer import build_dataloaders, train_and_evaluate

torch.manual_seed(42)   # 固定随机种子，保证结果可复现

# 1. 准备数据
X_train, y_train, X_valid, y_valid = load_data('.')
train_loader, valid_loader = build_dataloaders(
    X_train, y_train, X_valid, y_valid, batch_size=64)

# 2. 定义模型：2 个隐藏层 (256, 128)
model = MLP(hidden_sizes=(256, 128))

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()          # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 4. 训练并评估
acc, val_accs, train_losses = train_and_evaluate(
    model, train_loader, valid_loader, optimizer, criterion, num_epochs=20)

print(f'\n最终验证准确率: {acc:.2f}%')

# 5. 保存模型
torch.save(model.state_dict(), 'best_model.pth')
print('模型已保存为 best_model.pth')
```

**运行结果**（CPU 训练约 1 分钟）：

```
Epoch [1/20] Loss: 0.3253, Val Acc: 95.06%
Epoch [2/20] Loss: 0.1250, Val Acc: 96.13%
...
Epoch [20/20] Loss: 0.0077, Val Acc: 97.78%
最终验证准确率: 97.78%
```

可以看到：**损失持续下降、准确率持续上升**，训练是正常的。

---

## 四、拓展实验：层数和神经元数的影响

用「控制变量法」——只改一个变量，其他条件完全一致——来研究网络结构的影响。

### 4.1 不同隐藏层层数

固定 SGD lr=0.01、10 epochs，只改层数：

| 层数 | 结构 | 准确率 |
|---|---|---|
| 1 层 | (256,) | 92.02% |
| 2 层 | (256, 128) | 92.76% |
| 3 层 | (256, 128, 64) | 92.89% |

**结论**：层数越多，网络表达能力越强，准确率小幅单调提升。但对 MNIST 这种简单任务，收益有限。

```python
# -*- coding: utf-8 -*-
"""对比 1/2/3 个隐藏层的准确率。"""
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import MLP
from trainer import build_dataloaders, train_and_evaluate

torch.manual_seed(42)
X_train, y_train, X_valid, y_valid = load_data('.')
train_loader, valid_loader = build_dataloaders(
    X_train, y_train, X_valid, y_valid, batch_size=64)
criterion = nn.CrossEntropyLoss()

for n_layers, hidden_sizes in [(1, (256,)), (2, (256, 128)), (3, (256, 128, 64))]:
    model = MLP(hidden_sizes=hidden_sizes)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    acc, _, _ = train_and_evaluate(
        model, train_loader, valid_loader, optimizer, criterion, num_epochs=10,
        verbose=False)
    print(f'{n_layers} 个隐藏层 {hidden_sizes}: 准确率 {acc:.2f}%')
```

### 4.2 2 隐藏层下不同神经元个数

固定 2 个隐藏层，改每层宽度：

| 结构 | 准确率 |
|---|---|
| (256, 128) | 92.60% |
| (128, 64) | 92.52% |
| (100, 50) | 92.54% |
| (512, 256) | 90.32% |

**结论**：神经元个数在 100~256 范围内对结果影响很小，说明 MNIST 不需要太大容量。(512,256) 反而最低，因为大网络在仅 10 轮内收敛更慢。

```python
# -*- coding: utf-8 -*-
"""对比 2 隐藏层下不同神经元个数的准确率。"""
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import MLP
from trainer import build_dataloaders, train_and_evaluate

torch.manual_seed(42)
X_train, y_train, X_valid, y_valid = load_data('.')
train_loader, valid_loader = build_dataloaders(
    X_train, y_train, X_valid, y_valid, batch_size=64)
criterion = nn.CrossEntropyLoss()

for hidden_sizes in [(256, 128), (128, 64), (512, 256), (100, 50)]:
    model = MLP(hidden_sizes=hidden_sizes)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    acc, _, _ = train_and_evaluate(
        model, train_loader, valid_loader, optimizer, criterion, num_epochs=10,
        verbose=False)
    print(f'隐藏层神经元 {hidden_sizes}: 准确率 {acc:.2f}%')
```

---

## 五、调参要点（结合实验总结）

我做了 24 组调参实验，总结出几条实用经验：

| 参数 | 作用 | 本次实验结论 |
|---|---|---|
| **Epoch** | 数据被训练的次数 | 5→30 持续提升（90.86%→96.04%），模型未收敛 |
| **学习率 lr** | 每次更新步长 | **影响最大**：0.001 仅 86.51%，0.05 达 97.46% |
| **优化器** | 参数更新算法 | Adam/AdamW/RMSprop 约 98%，优于小 lr 的 SGD |
| **Dropout** | 随机丢神经元防过拟合 | 反而略降——模型未过拟合 |
| **权重衰减** | L2 正则 | 增大反而降准确率——正则过强导致欠拟合 |

**最终最优配置**：2 隐藏层 (256,128) + Adam lr=0.001 + 20 epochs = **98.01%**。

### 最重要的调参心法

1. **每次只改一个参数**，否则无法归因。
2. **先定大方向，再微调**：先调学习率（影响最大）→ 再选优化器 → 最后调正则。
3. **固定随机种子**，否则两次跑同一配置结果都不一样，无法公平对比。

---

## 六、几个容易踩的坑

1. **忘记 `zero_grad()`**：梯度会累加，导致训练发散。
2. **评估时没设 `model.eval()`**：Dropout 在评估时仍会随机丢弃，结果偏随机。
3. **输出层加了 Softmax 又用 CrossEntropyLoss**：等于算了两次 Softmax，损失异常。
4. **标签类型错误**：`CrossEntropyLoss` 要求标签是 `torch.long`，不是 `float`。
5. **数据没归一化**：像素值 0~255 直接输入，训练慢且不稳定，应 `/255` 到 [0,1]。

---

## 七、小结

MLP 是深度学习入门的第一个模型。这篇笔记通过手写数字识别这个例子，把「**数据加载 → 模型定义 → 训练 → 评估 → 调参**」完整走了一遍。核心要记住三件事：

1. **网络结构**：`Linear + ReLU` 堆叠，最后一层不加激活函数。
2. **训练 5 步**：清梯度 → 前向 → 算损失 → 反向 → 更新。
3. **实验方法**：控制变量 + 固定种子，才能得出可信结论。

掌握了这套流程，以后换数据集、换模型（CNN、RNN）本质都一样——变的只是模型结构，训练套路完全通用。
