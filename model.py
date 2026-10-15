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
                      例如 (256,) 为 1 个隐藏层；
                      (256, 128) 为 2 个隐藏层；
                      (256, 128, 64) 为 3 个隐藏层。
        dropout: 丢弃率，0 表示不使用 Dropout。
        """
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hs
        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平为一维向量
        return self.net(x)
