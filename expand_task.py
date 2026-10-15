# -*- coding: utf-8 -*-
"""
拓展任务主实验：
  (1) 建立至少 2 个隐藏层的全连接神经网络；
  (3) 比较 1 / 2 / 3 个隐藏层的预测准确率；
  (4) 比较 2 个隐藏层下不同神经元个数的准确率。
"""
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import MLP
from trainer import build_dataloaders, train_and_evaluate

torch.manual_seed(42)

# ---------- 数据准备 ----------
X_train, y_train, X_valid, y_valid = load_data('.')
train_loader, valid_loader = build_dataloaders(
    X_train, y_train, X_valid, y_valid, batch_size=64)

CRITERION = nn.CrossEntropyLoss()


def run_exp(hidden_sizes, dropout=0.0, lr=0.01, num_epochs=10,
            optimizer_name='SGD', weight_decay=0.0):
    """运行一次实验，返回最终验证准确率。"""
    model = MLP(hidden_sizes=hidden_sizes, dropout=dropout)
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'未知优化器: {optimizer_name}')
    acc, _, _ = train_and_evaluate(
        model, train_loader, valid_loader, optimizer, CRITERION,
        num_epochs=num_epochs, verbose=False)
    return acc


if __name__ == '__main__':
    print('=' * 60)
    print('任务(3)：不同隐藏层层数的准确率比较')
    print('=' * 60)
    results_layers = []
    for n_layers in [1, 2, 3]:
        if n_layers == 1:
            hidden_sizes = (256,)
        elif n_layers == 2:
            hidden_sizes = (256, 128)
        else:
            hidden_sizes = (256, 128, 64)
        acc = run_exp(hidden_sizes, num_epochs=10)
        results_layers.append((n_layers, hidden_sizes, acc))
        print(f'{n_layers} 个隐藏层 {hidden_sizes}: 准确率 {acc:.2f}%')

    print()
    print('=' * 60)
    print('任务(4)：2 个隐藏层下不同神经元个数的比较')
    print('=' * 60)
    configs = [
        (256, 128),
        (128, 64),
        (512, 256),
        (100, 50),
    ]
    results_neurons = []
    for hs in configs:
        acc = run_exp(hs, num_epochs=10)
        results_neurons.append((hs, acc))
        print(f'隐藏层神经元 {hs}: 准确率 {acc:.2f}%')

    print()
    print('=' * 60)
    print('总结')
    print('=' * 60)
    print('【层数对比】')
    for n_layers, hs, acc in results_layers:
        print(f'  {n_layers} 层 {hs} -> {acc:.2f}%')
    print('【神经元个数对比】(2 隐藏层)')
    for hs, acc in results_neurons:
        print(f'  {hs} -> {acc:.2f}%')
