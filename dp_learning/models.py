from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden: int = 256, num_classes: int = 47):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
