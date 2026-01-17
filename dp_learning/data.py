from __future__ import annotations

import math
import random
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader, Sampler


def compute_epsilon_for_range(mean_batch_size: int, delta: float) -> float:
    if mean_batch_size <= 0:
        raise ValueError("mean_batch_size must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    u_bound = mean_batch_size / 3
    return math.log(2 / delta) / u_bound


def sample_truncated_geometric(epsilon: float, delta: float) -> int:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    alpha = math.exp(epsilon)
    u_bound = math.log(2 / delta) / epsilon

    p = 1 - 1 / alpha
    magnitude = torch.distributions.Geometric(p).sample().item()
    sign = -1 if torch.rand(1).item() < 0.5 else 1
    r = sign * magnitude

    r = max(-u_bound, min(u_bound, r))
    return int(r)


class TruncatedGeometricBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset_size: int,
        mean_batch_size: int,
        delta: float,
        shuffle: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be > 0")
        if mean_batch_size <= 0:
            raise ValueError("mean_batch_size must be > 0")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")

        self.dataset_size = dataset_size
        self.mean_batch_size = mean_batch_size
        self.epsilon = compute_epsilon_for_range(mean_batch_size, delta)
        self.delta = delta
        self.shuffle = shuffle

    def __iter__(self) -> Iterable[List[int]]:
        indices = list(range(self.dataset_size))
        if self.shuffle:
            random.shuffle(indices)

        i = 0
        while i < self.dataset_size:
            noise = sample_truncated_geometric(self.epsilon / 2, self.delta / 2)
            batch_size = max(1, self.mean_batch_size + noise)

            batch = indices[i : i + batch_size]
            yield batch
            i += batch_size

    def __len__(self) -> int:
        return max(1, self.dataset_size // self.mean_batch_size)


def make_train_loader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    random_batch: bool = False,
    delta: float | None = None,
) -> DataLoader:
    if random_batch:
        if delta is None:
            raise ValueError("delta must be provided for random_batch")

        batch_sampler = TruncatedGeometricBatchSampler(
            dataset_size=len(dataset),
            mean_batch_size=batch_size,
            delta=delta,
            shuffle=shuffle,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
