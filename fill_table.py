import json
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill

XLSX = 'C:/Users/HONOR/Desktop/MLP调参实验记录表.xlsx'

with open('D:/Code/python/MLP_拓展任务/results.json', encoding='utf-8') as f:
    results = json.load(f)

wb = openpyxl.load_workbook(XLSX)

ws = wb['基准配置']
base = {
    '数据集': 'MNIST 手写数字（train-images 60000 张，28×28 灰度）',
    '任务类型（分类/回归）': '分类（10 类）',
    '输入维度': '784（28×28 展平）',
    '类别数/输出维度': '10',
    '模型结构（层数/每层神经元）': '2 个隐藏层 (256, 128)，输出 10',
    '激活函数': 'ReLU',
    '优化器': 'SGD（调参阶段也对比了 Adam/AdamW/RMSprop）',
    '初始学习率': '0.01（调参阶段从 0.001~0.1 扫描）',
    '权重衰减 weight_decay': '0（调参阶段测试了 1e-4 / 1e-3 / 1e-2）',
    'Dropout 比率': '0（调参阶段测试了 0.1 / 0.2 / 0.3 / 0.5）',
    'Batch Size': '64',
    'Epoch 数': '10（调参阶段测试了 5 / 10 / 20 / 30）',
    '随机种子 seed': '42',
    '损失函数': 'CrossEntropyLoss',
    '是否使用 BatchNorm': '否',
    '是否使用早停': '否',
    '训练设备（CPU/GPU）': 'CPU',
    '数据预处理方式': '像素值 /255 归一化到 [0,1]，展平为一维',
    '验证集划分比例': '按类别随机打乱 5 等分，前 4 份训练(48004)/第 5 份验证(11996)，即 8:2',
    '备注': '调参遵循"每次只改一个参数"原则；最终最优：2 隐藏层(256,128) + Adam lr=0.001 + 20 epochs = 98.01%',
}
for row in ws.iter_rows(min_row=2, max_row=21):
    key = row[0].value
    if key and key in base:
        row[1].value = base[key]
        row[1].alignment = Alignment(wrap_text=True, vertical='center')

ws2 = wb['调参实验记录']
headers = [c.value for c in ws2[2]]
records = [
    (1, '基线', '-', 'baseline', '基准配置，作为后续对比起点'),
    (2, 'Epoch', '10 → 5', 'epoch-5', '训练轮数减少，准确率下降'),
    (3, 'Epoch', '5 → 10', 'epoch-10', ''),
    (4, 'Epoch', '10 → 20', 'epoch-20', '准确率明显提升'),
    (5, 'Epoch', '20 → 30', 'epoch-30', '继续提升，但收益递减'),
    (6, '学习率', '0.01 → 0.001', 'lr-0.001', '学习率过小，欠拟合明显'),
    (7, '学习率', '0.001 → 0.005', 'lr-0.005', ''),
    (8, '学习率', '0.005 → 0.01', 'lr-0.01', ''),
    (9, '学习率', '0.01 → 0.05', 'lr-0.05', '学习率最优，收敛快且稳定'),
    (10, '学习率', '0.05 → 0.1', 'lr-0.1', '略下降，出现轻微震荡'),
    (11, '优化器', 'SGD → Adam', 'opt-Adam', '自适应优化器，收敛更快'),
    (12, '优化器', 'Adam → AdamW', 'opt-AdamW', ''),
    (13, '优化器', 'AdamW → RMSprop', 'opt-RMSprop', ''),
    (14, '优化器', 'RMSprop → SGD(lr=0.05)', 'opt-SGD', '高学习率 SGD 也可达 97.78%'),
    (15, 'Dropout', '0 → 0.1', 'drop-0.1', '加 Dropout 反而略降，模型未过拟合'),
    (16, 'Dropout', '0.1 → 0.2', 'drop-0.2', ''),
    (17, 'Dropout', '0.2 → 0.3', 'drop-0.3', ''),
    (18, 'Dropout', '0.3 → 0.5', 'drop-0.5', ''),
    (19, '权重衰减', '0 → 1e-4', 'wd-1e-4', ''),
    (20, '权重衰减', '1e-4 → 1e-3', 'wd-1e-3', ''),
    (21, '权重衰减', '1e-3 → 1e-2', 'wd-1e-2', '正则过强，准确率下降'),
    (22, '隐藏层神经元数', '(256,128) → (512,256)', 'neurons-512-256', ''),
    (23, '隐藏层神经元数', '(512,256) → (256,128)', 'neurons-256-128', ''),
    (24, '隐藏层神经元数', '(256,128) → (128,64)', 'neurons-128-64', '神经元数影响不大'),
]

by_step = {r['step']: r for r in results}

def fmt_hidden(h):
    return ','.join(str(x) for x in h)

row_idx = 3
for (num, param_name, change, step, note) in records:
    r = by_step[step]
    layer_count = len(r['hidden_sizes'])
    hidden_str = fmt_hidden(r['hidden_sizes'])
    train_loss = r['train_loss_curve'][-1]
    val_acc = r['val_acc']
    overfit = ''
    if train_loss < 0.05 and val_acc < 95:
        overfit = '疑似'
    values = [
        num, param_name, change,
        r['num_epochs'],
        layer_count,
        hidden_str,
        r['optimizer'],
        r['lr'],
        r['dropout'],
        r['weight_decay'],
        64,
        'ReLU',
        round(train_loss, 4),
        '',
        round(val_acc, 2),
        '',
        overfit,
        note,
    ]
    for col, v in enumerate(values, start=1):
        ws2.cell(row=row_idx, column=col, value=v)
    row_idx += 1

best_row = 34
if 'A34:R34' in [str(r) for r in ws2.merged_cells.ranges]:
    ws2.unmerge_cells('A34:R34')
best_cfg = [
    '★ 当前最优配置（找到最佳参数后填写）',
    'Adam', '', 20, 2, '256,128', 'Adam', 0.001, 0, 0, 64, 'ReLU',
    round(by_step['opt-Adam']['train_loss_curve'][-1], 4), '',
    round(by_step['opt-Adam']['val_acc'], 2), '', '',
    '2 隐藏层(256,128) + Adam lr=0.001 + 20 epochs，验证准确率 98.01%',
]
for col, v in enumerate(best_cfg, start=1):
    ws2.cell(row=best_row, column=col, value=v)

ws3 = wb['训练曲线数据']
best = by_step['opt-Adam']
row3 = 3
for ep, (loss, acc) in enumerate(zip(best['train_loss_curve'], best['val_acc_curve']), start=1):
    ws3.cell(row=row3, column=1, value='实验11(Adam)')
    ws3.cell(row=row3, column=2, value=ep)
    ws3.cell(row=row3, column=3, value=round(loss, 4))
    ws3.cell(row=row3, column=4, value='')
    ws3.cell(row=row3, column=5, value=round(acc, 2))
    row3 += 1

ws5 = wb['实验结论']
conclusions = {
    '拓展任务做了什么': '基于 PyTorch 建立多隐藏层 MLP 做 MNIST 手写数字识别：(1)构建≥2隐藏层网络；(2)按类别随机打乱5等分，前4份训练/第5份验证；(3)对比1/2/3个隐藏层准确率；(4)对比2隐藏层不同神经元个数。',
    '尝试调了哪些参数': 'Epoch(5/10/20/30)、学习率(0.001~0.1)、优化器(SGD/Adam/AdamW/RMSprop)、Dropout(0.1~0.5)、权重衰减(1e-4~1e-2)、隐藏层神经元个数((512,256)/(256,128)/(128,64))。',
    '最佳参数组合': '2 个隐藏层(256,128)，Adam 优化器 lr=0.001，20 epochs，无 Dropout、无权重衰减，Batch Size=64。',
    '最佳验证准确率': '98.01%（Adam lr=0.001）；其次 SGD lr=0.05 达 97.46%。',
    '哪个参数影响最大': '学习率影响最大：lr=0.001 时仅 86.51%，lr=0.05 时达 97.46%，相差超 10 个百分点。其次优化器（自适应优化器显著优于小 lr SGD）。',
    '是否出现过拟合？如何处理': '未明显过拟合。训练 loss 持续下降、验证准确率同步上升，二者无明显背离。故 Dropout 和权重衰减均无收益，反而略降性能。',
    '是否出现欠拟合？如何处理': '出现过欠拟合：lr=0.001 时验证准确率仅 86.51%、训练 loss 高达 0.52，属学习率过小导致欠拟合。通过增大学习率解决。',
    '反直觉的现象': '1) Dropout 和权重衰减反而降低准确率——因为该网络对 MNIST 容量偏小、本就不易过拟合；2) 大网络(512,256)在 10 epochs 下准确率反而低于小网络，大模型收敛更慢。',
    '待解决的问题/下一步计划': '可在更深更宽的网络(如3~4层、每层512+)上重新对比 Dropout/权重衰减，观察正则化的真实收益；也可加入早停、BatchNorm、学习率衰减等进一步优化。',
    '其他疑问（可拿来讨论）': '为什么小网络+MNIST 下正则化无效？是否应该在过拟合明显的数据集(如复杂图像)上才有正则化收益？',
}
for row in ws5.iter_rows(min_row=2, max_row=11):
    key = row[0].value
    if key and key in conclusions:
        row[1].value = conclusions[key]
        row[1].alignment = Alignment(wrap_text=True, vertical='top')

wb.save(XLSX)
print('表格已填充完成')
