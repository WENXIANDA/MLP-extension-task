import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from data_loader import load_data
from model import MLP
from trainer import build_dataloaders, train_and_evaluate

torch.manual_seed(42)

X_train, y_train, X_valid, y_valid = load_data('.')
train_loader, valid_loader = build_dataloaders(
    X_train, y_train, X_valid, y_valid, batch_size=64)

model = MLP(hidden_sizes=(256, 128), dropout=0.0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('开始训练最终模型（Adam lr=0.001, 2隐藏层(256,128), 20 epochs）...')
acc, val_accs, train_losses = train_and_evaluate(
    model, train_loader, valid_loader, optimizer, criterion,
    num_epochs=20, verbose=True)
print(f'\n最终验证准确率: {acc:.2f}%')

torch.save(model.state_dict(), 'best_model.pth')
print('模型已保存为 best_model.pth')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 21), train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(1, 21), val_accs, marker='o', color='orange', label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('验证准确率曲线')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png', dpi=100)
print('训练曲线已保存为 training_curves.png')


def get_wrong_predictions(model, valid_loader):
    wrong_images, wrong_labels, wrong_preds = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            wrong_idx = (predicted != labels).nonzero(as_tuple=True)[0]
            wrong_images.append(images[wrong_idx])
            wrong_labels.append(labels[wrong_idx])
            wrong_preds.append(predicted[wrong_idx])
    wrong_images = torch.cat(wrong_images)
    wrong_labels = torch.cat(wrong_labels)
    wrong_preds = torch.cat(wrong_preds)
    return wrong_images, wrong_labels, wrong_preds


def show_wrong_predictions(wrong_images, wrong_labels, wrong_preds, num_images=6):
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i in range(num_images):
        ax = axes[i]
        img = wrong_images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {wrong_labels[i].item()} Pred: {wrong_preds[i].item()}',
                     loc='left')
        ax.axis('off')
    plt.savefig('wrong_predictions.png', dpi=100, bbox_inches='tight')
    print('错误预测样本已保存为 wrong_predictions.png')


wrong_images, wrong_labels, wrong_preds = get_wrong_predictions(model, valid_loader)
show_wrong_predictions(wrong_images, wrong_labels, wrong_preds, num_images=6)
