from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def flatten_grads(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)


def unflatten_grads(params: Iterable[torch.nn.Parameter], flat_grad: torch.Tensor) -> None:
    idx = 0
    for p in params:
        numel = p.numel()
        grad_view = flat_grad[idx : idx + numel].view_as(p)

        if p.grad is None:
            p.grad = torch.empty_like(p)
        p.grad.copy_(grad_view)
        idx += numel
