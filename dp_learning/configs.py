from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MethodConfig:
    name: str
    use_dp: bool
    random_batch: bool


METHOD_CONFIGS: Dict[str, MethodConfig] = {
    "baseline": MethodConfig(name="baseline", use_dp=False, random_batch=False),
    "StandardSGD": MethodConfig(name="StandardSGD", use_dp=False, random_batch=False),
    "Hamming-Style DP": MethodConfig(
        name="Hamming-Style DP",
        use_dp=True,
        random_batch=False,
    ),
    "Edit-Style DP": MethodConfig(name="Edit-Style DP", use_dp=True, random_batch=True),
}


@dataclass
class ExperimentLog:
    method: str
    dataset: str
    batch_size: Optional[int] = None
    seed: Optional[int] = None

    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_acc: Optional[float] = None

    agg_epsilon: Optional[float] = None
    agg_delta: Optional[float] = None
    bin_epsilon: Optional[float] = None
    bin_delta: Optional[float] = None
    dp_mechanism: str = "Naive"


@dataclass
class LearningConfig:
    dataset: Any
    batch_size: int
    epochs: int
    lr: float
    max_grad_norm: float

    device: str
    loss_fn: Any
    optimizer_class: Any
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    log_every: Optional[int] = None
