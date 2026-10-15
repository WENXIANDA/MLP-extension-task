import gzip
import struct
import numpy as np
from collections import defaultdict


def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"非法 magic 值: {magic}"
        buf = f.read(num * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
    return data


def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"非法 magic 值: {magic}"
        buf = f.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def split_by_class(images, labels, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, valid_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        folds = np.array_split(idx, n_folds)
        train_idx.extend(np.concatenate(folds[:-1]))
        valid_idx.extend(folds[-1])
    train_idx = np.array(train_idx)
    valid_idx = np.array(valid_idx)
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    return train_idx, valid_idx


def load_data(root='.'):
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
    for c in range(10):
        n_tr = (y_train == c).sum()
        n_va = (y_valid == c).sum()
        print(f'类别 {c}: 训练 {n_tr} / 验证 {n_va}')
