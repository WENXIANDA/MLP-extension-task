import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_sizes=(256,), num_classes=10,
                 dropout=0.0):
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
        x = x.view(x.size(0), -1)
        return self.net(x)
