# -*- coding: utf-8 -*-
"""
调参实验脚本：按"每次只调一个参数"的原则，系统性地调整超参数并记录结果。

基线配置（Baseline）：
  隐藏层: (256, 128)，2 个隐藏层
  优化器: SGD，学习率 lr=0.01
  Dropout: 0（不使用）
  权重衰减 weight_decay: 0
  epoch: 10

调参顺序（每次只改一个参数，基于上一步最优继续）：
  1. epoch 训练轮数
  2. 学习率 lr
  3. 优化器 (SGD / Adam)
  4. Dropout
  5. 权重衰减 weight_decay
  6. 隐藏层神经元个数（在最优配置基础上微调）

所有结果会写入 results.json 和 调参记录.md。
"""
import json
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
CRITERION = nn.CrossEntropyLoss()

results = []  # 保存所有实验记录


def make_optimizer(model, name, lr, weight_decay):
    if name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'未知优化器: {name}')


def run_exp(step, desc, hidden_sizes=(256, 128), lr=0.01, num_epochs=10,
            optimizer_name='SGD', dropout=0.0, weight_decay=0.0):
    model = MLP(hidden_sizes=hidden_sizes, dropout=dropout)
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
    acc, val_accs, train_losses = train_and_evaluate(
        model, train_loader, valid_loader, optimizer, CRITERION,
        num_epochs=num_epochs, verbose=True)
    record = {
        'step': step,
        'desc': desc,
        'hidden_sizes': list(hidden_sizes),
        'lr': lr,
        'num_epochs': num_epochs,
        'optimizer': optimizer_name,
        'dropout': dropout,
        'weight_decay': weight_decay,
        'val_acc': round(acc, 4),
        'val_acc_curve': [round(a, 4) for a in val_accs],
        'train_loss_curve': [round(l, 4) for l in train_losses],
    }
    results.append(record)
    print(f'>>> [{step}] {desc} -> 最终准确率 {acc:.2f}%')
    print()
    return acc


if __name__ == '__main__':
    print('#' * 70)
    print('# 调参实验开始')
    print('#' * 70)

    # 基线
    run_exp('baseline', '基线: 2隐藏层(256,128), SGD lr=0.01, 10 epochs')

    # ---------- 1. epoch 轮数 ----------
    print('\n### 阶段1: 调整 epoch 轮数（其他不变）')
    run_exp('epoch-5', 'epoch=5', num_epochs=5)
    run_exp('epoch-10', 'epoch=10', num_epochs=10)  # 基线已含
    run_exp('epoch-20', 'epoch=20', num_epochs=20)
    run_exp('epoch-30', 'epoch=30', num_epochs=30)

    # ---------- 2. 学习率 ----------
    print('\n### 阶段2: 调整学习率（在 epoch 最优基础上）')
    run_exp('lr-0.001', 'lr=0.001', num_epochs=20, lr=0.001)
    run_exp('lr-0.005', 'lr=0.005', num_epochs=20, lr=0.005)
    run_exp('lr-0.01', 'lr=0.01', num_epochs=20, lr=0.01)
    run_exp('lr-0.05', 'lr=0.05', num_epochs=20, lr=0.05)
    run_exp('lr-0.1', 'lr=0.1', num_epochs=20, lr=0.1)

    # ---------- 3. 优化器 ----------
    print('\n### 阶段3: 调整优化器（在 lr 最优基础上）')
    run_exp('opt-Adam', 'Adam lr=0.001', num_epochs=20,
            optimizer_name='Adam', lr=0.001)
    run_exp('opt-AdamW', 'AdamW lr=0.001', num_epochs=20,
            optimizer_name='AdamW', lr=0.001)
    run_exp('opt-RMSprop', 'RMSprop lr=0.001', num_epochs=20,
            optimizer_name='RMSprop', lr=0.001)
    run_exp('opt-SGD', 'SGD lr=0.05', num_epochs=20,
            optimizer_name='SGD', lr=0.05)

    # ---------- 4. Dropout ----------
    print('\n### 阶段4: 调整 Dropout（在最优优化器基础上）')
    run_exp('drop-0.1', 'Dropout=0.1', num_epochs=20, dropout=0.1)
    run_exp('drop-0.2', 'Dropout=0.2', num_epochs=20, dropout=0.2)
    run_exp('drop-0.3', 'Dropout=0.3', num_epochs=20, dropout=0.3)
    run_exp('drop-0.5', 'Dropout=0.5', num_epochs=20, dropout=0.5)

    # ---------- 5. 权重衰减 ----------
    print('\n### 阶段5: 调整权重衰减（在最优 Dropout 基础上）')
    run_exp('wd-1e-4', 'weight_decay=1e-4', num_epochs=20, weight_decay=1e-4)
    run_exp('wd-1e-3', 'weight_decay=1e-3', num_epochs=20, weight_decay=1e-3)
    run_exp('wd-1e-2', 'weight_decay=1e-2', num_epochs=20, weight_decay=1e-2)

    # ---------- 6. 神经元个数微调 ----------
    print('\n### 阶段6: 在最优配置基础上微调神经元个数')
    run_exp('neurons-512-256', '神经元(512,256)', num_epochs=20,
            hidden_sizes=(512, 256))
    run_exp('neurons-256-128', '神经元(256,128)', num_epochs=20,
            hidden_sizes=(256, 128))
    run_exp('neurons-128-64', '神经元(128,64)', num_epochs=20,
            hidden_sizes=(128, 64))

    # 保存结果
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\n所有结果已保存到 results.json')
