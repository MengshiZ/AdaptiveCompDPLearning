from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .configs import ExperimentLog


def logs_to_df(logs: Iterable[ExperimentLog]) -> pd.DataFrame:
    records = []
    for log in logs:
        if len(log.test_acc) == 0:
            continue
        records.append(
            {
                "dataset": log.dataset,
                "method": log.method,
                "agg_epsilon": log.agg_epsilon,
                "agg_delta": log.agg_delta,
                "seed": log.seed,
                "final_acc": log.test_acc[-1],
            }
        )
    return pd.DataFrame(records)


def plot_acc_vs_epsilon(df: pd.DataFrame, dataset: str, epsilon_column: str = "agg_epsilon"):
    if epsilon_column not in df.columns:
        raise ValueError(f"Column '{epsilon_column}' not found in dataframe")

    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=df[df["dataset"] == dataset],
        x=epsilon_column,
        y="final_acc",
        hue="method",
        marker="o",
        errorbar="sd",
    )
    plt.xscale("log")
    plt.xlabel("Privacy budget ε")
    plt.ylabel("Test accuracy")
    plt.title(f"{dataset}: Accuracy vs Privacy Budget")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(
    logs: List[ExperimentLog],
    dataset: str,
    method: str,
    agg_epsilon: float,
):
    curves = [
        log
        for log in logs
        if log.dataset == dataset and log.method == method and log.agg_epsilon == agg_epsilon
    ]

    if not curves:
        raise ValueError("No matching logs found for requested curve.")

    accs = np.array([log.test_acc for log in curves])
    mean = accs.mean(axis=0)
    std = accs.std(axis=0)

    epochs = curves[0].epochs

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, mean, label=method)
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title(f"{dataset} Learning Curve (ε={agg_epsilon})")
    plt.legend()
    plt.tight_layout()
    plt.show()
