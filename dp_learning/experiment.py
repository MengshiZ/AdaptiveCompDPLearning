from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torchvision import datasets, transforms

from .configs import ExperimentLog, LearningConfig, METHOD_CONFIGS
from .data import make_train_loader
from .eval import test_model
from .models import SimpleCNN
from .trainers import DPSGDTrainer, NormalTrainer, StandardSGD
from .utils import set_seed


@dataclass
class VisionExperimentConfig:
    dataset_name: str
    method: str
    batch_size: int = 256
    epochs: int = 5
    lr: float = 0.1
    agg_epsilon: float | None = None
    agg_delta: float | None = None
    bin_epsilon: float | None = None
    bin_delta: float | None = None
    seed: int = 0
    max_grad_norm: float = 1.0
    dp_mechanism: str = "Naive"
    log_every: int | None = 50
    device: str | None = None


def _build_vision_dataset(dataset_name: str):
    if dataset_name == "emnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),
                transforms.Lambda(lambda x: torch.flip(x, [2])),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        train_dataset = datasets.EMNIST(
            root="./data",
            split="balanced",
            train=True,
            download=True,
            transform=transform,
        )

        model = SimpleCNN(num_classes=47, in_channels=1)
    elif dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        model = SimpleCNN(num_classes=10, in_channels=1)
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

        train_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        model = SimpleCNN(num_classes=10, in_channels=3)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, model


def run_experiment(
    model: torch.nn.Module,
    config: LearningConfig,
    log: ExperimentLog,
) -> Tuple[torch.nn.Module, ExperimentLog]:
    if log.method not in METHOD_CONFIGS:
        raise ValueError(
            f"Unknown method '{log.method}'. Available: {list(METHOD_CONFIGS.keys())}"
        )

    cfg = METHOD_CONFIGS[log.method]
    use_dp = cfg.use_dp
    random_batch = cfg.random_batch

    if log.seed is not None:
        set_seed(log.seed)

    train_loader = make_train_loader(
        dataset=config.dataset,
        batch_size=config.batch_size,
        random_batch=random_batch,
        epsilon=log.bin_epsilon if random_batch else None,
        delta=log.bin_delta if random_batch else None,
    )

    optimizer_kwargs = config.optimizer_kwargs or {}
    optimizer = config.optimizer_class(
        model.parameters(),
        lr=config.lr,
        **optimizer_kwargs,
    )

    if use_dp:
        if log.agg_epsilon is None or log.agg_delta is None:
            raise ValueError("agg_epsilon and agg_delta must be provided for DP runs")

        trainer = DPSGDTrainer(
            model=model,
            optimizer=optimizer,
            epsilon=log.agg_epsilon,
            delta=log.agg_delta,
            device=config.device,
            max_grad_norm=config.max_grad_norm,
            dp_mechanism=log.dp_mechanism,
        )
        print(
            f"Running {log.method} | "
            f"random_batch={random_batch}, agg_epsilon={log.agg_epsilon}, agg_delta={log.agg_delta}"
        )
    else:
        if log.method == "StandardSGD":
            trainer = StandardSGD(
                model=model,
                optimizer=optimizer,
                device=config.device,
            )
            print("Running StandardSGD (non-DP)")
        else:
            trainer = NormalTrainer(
                model=model,
                optimizer=optimizer,
                device=config.device,
                max_grad_norm=config.max_grad_norm,
            )
            print("Running baseline (non-DP)")

    model.to(config.device)

    for epoch in range(1, config.epochs + 1):
        model.train()

        if use_dp:
            trainer.reset_presum()

        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            loss = trainer.train_batch(x, y, config.loss_fn)
            total_loss += loss
            if config.log_every and batch_idx % config.log_every == 0:
                avg_so_far = total_loss / batch_idx
                print(
                    f"Epoch {epoch}/{config.epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Avg loss: {avg_so_far:.4f}"
                )

        avg_loss = total_loss / len(train_loader)

        log.epochs.append(epoch)
        log.train_loss.append(avg_loss)

        model.eval()
        with torch.no_grad():
            test_loss, acc = test_model(
                model=model,
                dataset_name=log.dataset,
                batch_size=config.batch_size,
                device=config.device,
            )

        log.test_loss.append(test_loss)
        log.test_acc.append(acc)

        print(f"Epoch {epoch}/{config.epochs} | Train loss: {avg_loss:.4f}")

    return model, log


def run_vision_experiment(config: VisionExperimentConfig):
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, model = _build_vision_dataset(config.dataset_name)

    learning_config = LearningConfig(
        dataset=train_dataset,
        batch_size=config.batch_size,
        epochs=config.epochs,
        lr=config.lr,
        device=device,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.SGD,
        optimizer_kwargs={},
        max_grad_norm=config.max_grad_norm,
        log_every=config.log_every,
    )

    log = ExperimentLog(
        method=config.method,
        dataset=config.dataset_name,
        batch_size=config.batch_size,
        seed=config.seed,
        agg_epsilon=config.agg_epsilon,
        agg_delta=config.agg_delta,
        bin_epsilon=config.bin_epsilon,
        bin_delta=config.bin_delta,
        dp_mechanism=config.dp_mechanism,
    )

    return run_experiment(
        model=model,
        config=learning_config,
        log=log,
    )
