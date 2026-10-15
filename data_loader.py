# -*- coding: utf-8 -*-
"""
数据加载模块：解析 MNIST 原始 .gz 文件，并按类别随机打乱等分 5 份。
前 4 份用于训练，第 5 份用于测试（验证）。
"""
import gzip
import struct
import numpy as np
from collections import defaultdict


def load_mnist_images(path):
    """读取 MNIST idx3-ubyte 图片文件，返回 (N, 28, 28) 的 uint8 数组。"""
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"非法 magic 值: {magic}"
        buf = f.read(num * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
    return data


def load_mnist_labels(path):
    """读取 MNIST idx1-ubyte 标签文件，返回 (N,) 的 int64 数组。"""
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"非法 magic 值: {magic}"
        buf = f.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def split_by_class(images, labels, n_folds=5, seed=42):
    """
    将每个类别的样本随机打乱，并等分为 n_folds 份。
    返回 train_idx 和 valid_idx 的数组。
    这里取前 4 份为训练，第 5 份为测试。
    """
    rng = np.random.default_rng(seed)
    train_idx, valid_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)  # 随机打乱该类别
        # 等分为 n_folds 份
        folds = np.array_split(idx, n_folds)
        # 前 n_folds-1 份用于训练，最后 1 份用于测试
        train_idx.extend(np.concatenate(folds[:-1]))
        valid_idx.extend(folds[-1])
    train_idx = np.array(train_idx)
    valid_idx = np.array(valid_idx)
    # 整体再打乱一次训练集顺序
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    return train_idx, valid_idx


def load_data(root='.'):
    """
    加载训练集并按类别 5 等分切分。
    返回：
      X_train, y_train  (48000 张，前 4 份)
      X_valid, y_valid  (12000 张，第 5 份)
    """
    images = load_mnist_images(f'{root}/train-images-idx3-ubyte.gz')
    labels = load_mnist_labels(f'{root}/train-labels-idx1-ubyte.gz')
    train_idx, valid_idx = split_by_class(images, labels, n_folds=5, seed=42)
    X_train = images[train_idx]
    y_train = labels[train_idx]
    X_valid = images[valid_idx]
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
