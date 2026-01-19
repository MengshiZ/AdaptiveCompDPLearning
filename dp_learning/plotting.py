from __future__ import annotations

from pathlib import Path
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
        best_acc = (
            log.best_acc
            if log.best_acc is not None
            else log.test_acc[-1]
        )
        records.append(
            {
                "dataset": log.dataset,
                "method": log.method,
                "batch_size": log.batch_size,
                "agg_epsilon": log.agg_epsilon,
                "agg_delta": log.agg_delta,
                "dp_mechanism": log.dp_mechanism,
                "seed": log.seed,
                "best_acc": best_acc,
            }
        )
    return pd.DataFrame(records)


def plot_acc_vs_epsilon(
    df: pd.DataFrame,
    dataset: str,
    epsilon_column: str = "agg_epsilon",
    batch_size: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
):
    if epsilon_column not in df.columns:
        raise ValueError(f"Column '{epsilon_column}' not found in dataframe")

    plot_df = df[df["dataset"] == dataset]
    if batch_size is not None and "batch_size" in plot_df.columns:
        plot_df = plot_df[plot_df["batch_size"] == batch_size]

    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=plot_df,
        x=epsilon_column,
        y="best_acc",
        hue="method",
        marker="o",
        errorbar="sd",
    )
    plt.xscale("log")
    plt.xlabel("Privacy budget ε")
    plt.ylabel("Best test accuracy")
    plt.title(f"{dataset}: Accuracy vs Privacy Budget")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curve(
    logs: List[ExperimentLog],
    dataset: str,
    method: str,
    agg_epsilon: float,
    save_path: str | Path | None = None,
    show: bool = True,
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
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title(f"{dataset} Learning Curve (ε={agg_epsilon})")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_accuracy_summary(
    logs: Iterable[ExperimentLog],
    dataset: str,
    output_dir: str | Path | None = None,
    show: bool = True,
):
    df = logs_to_df(logs)
    plot_df = df[df["dataset"] == dataset]
    if plot_df.empty:
        print(f"No results available for dataset '{dataset}'.")
        return None

    plot_dir = Path(output_dir) if output_dir is not None else None
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(plot_df["batch_size"].dropna().unique())
    if not batch_sizes:
        print(f"No batch sizes available for dataset '{dataset}'.")
        return None

    axes = []
    for batch_size in batch_sizes:
        batch_df = plot_df[plot_df["batch_size"] == batch_size]
        if batch_df.empty:
            continue

        plt.figure(figsize=(7, 4))

        dp_df = batch_df[batch_df["agg_epsilon"].notna()]
        if not dp_df.empty:
            sns.lineplot(
                data=dp_df,
                x="agg_epsilon",
                y="best_acc",
                hue="method",
                style="dp_mechanism",
                markers=True,
                dashes=False,
                errorbar="sd",
            )
            plt.xscale("log")

        baseline_df = batch_df[batch_df["agg_epsilon"].isna()]
        if not baseline_df.empty:
            grouped = baseline_df.groupby(["method"], dropna=False)["best_acc"]
            for method, values in grouped:
                label = f"{method} (baseline)"
                plt.axhline(values.mean(), linestyle="--", alpha=0.7, label=label)

        plt.xlabel("Privacy budget ε")
        plt.ylabel("Best test accuracy")
        plt.title(f"{dataset}: Accuracy vs Privacy Budget (bs={int(batch_size)})")
        plt.tight_layout()

        if plot_dir is not None:
            filename = f"acc_summary_{dataset}_bs{int(batch_size)}.png"
            plt.savefig(plot_dir / filename)
        if show:
            plt.show()
        else:
            plt.close()

        axes.append(plt.gca())

    return axes
