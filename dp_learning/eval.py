from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_TEST_LOADER_CACHE: Dict[Tuple[str, int], DataLoader] = {}


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: str = "cpu"):
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def _build_test_loader(dataset_name: str, batch_size: int) -> DataLoader:
    if dataset_name == "emnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),
                transforms.Lambda(lambda x: torch.flip(x, [2])),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        test_data = datasets.EMNIST(
            root="./data",
            split="balanced",
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )
        test_data = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError("Unsupported dataset")

    return DataLoader(test_data, batch_size=batch_size, shuffle=False)


def test_model(model, dataset_name: str = "emnist", batch_size: int = 512, device: str = "cpu"):
    key = (dataset_name, batch_size)

    if key not in _TEST_LOADER_CACHE:
        _TEST_LOADER_CACHE[key] = _build_test_loader(dataset_name, batch_size)

    model.eval()
    with torch.no_grad():
        test_loss, test_acc = evaluate_model(
            model,
            _TEST_LOADER_CACHE[key],
            device=device,
        )

    print(
        f"[{dataset_name.upper()}] "
        f"Test loss: {test_loss:.4f}, "
        f"Test accuracy: {test_acc * 100:.2f}%"
    )

    return test_loss, test_acc
