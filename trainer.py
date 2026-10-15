# -*- coding: utf-8 -*-
"""
训练与评估模块：封装单次实验的训练和测试流程，返回准确率等指标。
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def build_dataloaders(X_train, y_train, X_valid, y_valid, batch_size=64):
    """构建训练/验证 DataLoader。"""
    X_train_t = torch.tensor(X_train, dtype=torch.float32) / 255.0
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32) / 255.0
    y_valid_t = torch.tensor(y_valid, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    valid_ds = TensorDataset(X_valid_t, y_valid_t)

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
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / total_batches
        train_losses.append(avg_loss)

        # ---- 验证阶段 ----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        val_accs.append(acc)
        if verbose:
            print(f'  Epoch [{epoch + 1}/{num_epochs}] '
                  f'Loss: {avg_loss:.4f}, Val Acc: {acc:.2f}%')
    return val_accs[-1], val_accs, train_losses
