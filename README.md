# MLP 手写数字识别 - 拓展任务

基于 PyTorch 的全连接神经网络（MLP）实现 MNIST 手写数字识别，完成拓展任务与调参实验。

## 目录结构

| 文件 | 说明 |
| --- | --- |
| `data_loader.py` | 解析 MNIST `.gz` 数据，按类别随机打乱 5 等分（前 4 份训练 / 第 5 份验证） |
| `model.py` | 可配置隐藏层层数 / 神经元个数 / Dropout 的 MLP 模型 |
| `trainer.py` | 训练与评估流程封装 |
| `expand_task.py` | 拓展任务主实验：层数对比 + 神经元个数对比 |
| `tuning.py` | 系统调参实验（每次只调一个参数） |
| `final_model.py` | 用最优配置训练最终模型并可视化 |
| `gen_report.py` | 从 results.json 生成调参记录文档 |
| `调参记录.md` | 调参过程完整记录（参数变化 → 性能对照） |
| `results.json` | 所有实验的原始数据 |

## 拓展任务结论

**任务(3) 不同隐藏层层数对比**（10 epochs, SGD lr=0.01）：

| 层数 | 结构 | 准确率 |
| --- | --- | --- |
| 1 层 | (256,) | 92.02% |
| 2 层 | (256, 128) | 92.76% |
| 3 层 | (256, 128, 64) | 92.89% |

**任务(4) 2 个隐藏层不同神经元个数**（10 epochs）：

| 结构 | 准确率 |
| --- | --- |
| (256, 128) | 92.60% |
| (128, 64) | 92.52% |
| (512, 256) | 90.32% |
| (100, 50) | 92.54% |

## 调参结论

最优配置：**2 隐藏层 (256,128) + Adam lr=0.001 + 20 epochs**，验证准确率 **98.01%**。

详见 `调参记录.md`。

## 运行方式（PyCharm）

1. 用 PyCharm 打开本项目文件夹。
2. 配置 Python 解释器（建议用已有的 `C:\Users\HONOR\.workbuddy\binaries\python\envs\default\Scripts\python.exe`，已装好 torch，或自行 `pip install torch torchvision numpy matplotlib`）。
3. 依次运行：
   - `python data_loader.py` 验证数据切分
   - `python expand_task.py` 跑拓展任务
   - `python tuning.py` 跑完整调参（约 20 分钟）
   - `python final_model.py` 训练最终模型并输出可视化
