# -*- coding: utf-8 -*-
"""从 results.json 生成调参记录 Markdown 文档。"""
import json

with open('results.json', encoding='utf-8') as f:
    results = json.load(f)


def fmt_hidden(h):
    return '(' + ','.join(str(x) for x in h) + ')'


# 分组阶段
stages = [
    ('baseline', '基线'),
    ('epoch', '阶段1：epoch 训练轮数'),
    ('lr', '阶段2：学习率'),
    ('opt', '阶段3：优化器'),
    ('drop', '阶段4：Dropout'),
    ('wd', '阶段5：权重衰减'),
    ('neurons', '阶段6：隐藏层神经元个数微调'),
]

lines = []
lines.append('# MLP 调参实验记录\n')
lines.append('> 数据集：MNIST 手写数字（60000 训练集按类别随机打乱 5 等分，前 4 份训练 / 第 5 份验证，即 48000 / 12000）。\n')
lines.append('> 原则：每次只调整一个参数，观察性能变化后再在较优参数基础上继续。\n')

for key, title in stages:
    group = [r for r in results if r['step'] == key or r['step'].startswith(key + '-')]
    if not group:
        continue
    lines.append(f'\n## {title}\n')
    lines.append('| 参数配置 | 验证准确率 |')
    lines.append('| --- | --- |')
    for r in group:
        cfg = (f"{r['num_epochs']} epochs, {r['optimizer']} lr={r['lr']}, "
               f"隐藏层{fmt_hidden(r['hidden_sizes'])}, dropout={r['dropout']}, "
               f"wd={r['weight_decay']}")
        lines.append(f"| {cfg} | {r['val_acc']:.2f}% |")

# 每阶段的小结
lines.append('\n---\n')
lines.append('\n## 阶段小结与选参\n')

conclusions = {
    'baseline': '基线 92.71%，作为后续对比基准。',
    'epoch': 'epoch 从 5→30 准确率持续提升（90.86%→96.04%），说明模型尚未收敛。20 epoch 已达 94.68%，30 epoch 达 96.04%，收益递减但仍在提升。',
    'lr': 'lr=0.05 表现最佳（97.46%），lr=0.1 略下降（97.30%），lr=0.001 明显欠拟合（86.51%）。过小学习率收敛慢，过大则震荡。',
    'opt': 'Adam/AdamW/RMSprop 均约 98%，优于 SGD（97.78%）。自适应优化器收敛更快更稳。',
    'drop': '加入 Dropout 反而略降准确率（95% 左右）。原因是该网络规模小、本就不过拟合，Dropout 抑制了有效学习。',
    'wd': 'weight_decay 增大反而降低准确率（1e-2 时降至 92.94%）。说明模型未过拟合，正则化过强会欠拟合。',
    'neurons': '神经元个数(512,256)/(256,128)/(128,64) 差异不大（均约 94.7%），说明在 20 epochs + 该 lr 下模型容量均足够。',
}
for key, title in stages:
    if key in conclusions:
        lines.append(f'- **{title}**：{conclusions[key]}')

# 最优配置
lines.append('\n---\n')
lines.append('\n## 综合结论\n')
lines.append('1. **最优验证准确率**：Adam 优化器 lr=0.001 达到 **98.01%**（阶段3），为该数据集下的较优结果。')
lines.append('2. **关键结论**：')
lines.append('   - 增加隐藏层层数（1→3 层）准确率单调小幅提升（92.02%→92.89%）。')
lines.append('   - 学习率是最敏感的调参项，0.05 附近最优。')
lines.append('   - 自适应优化器（Adam/AdamW/RMSprop）优于 SGD。')
lines.append('   - 本任务中 Dropout 与权重衰减无明显收益（模型未过拟合）。')
lines.append('3. **建议的最终配置**：2 隐藏层 (256,128)，Adam lr=0.001，20 epochs，无 Dropout、无权重衰减。')

with open('调参记录.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print('已生成 调参记录.md')
print(f'共 {len(results)} 组实验记录')
