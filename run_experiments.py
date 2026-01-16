from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dp_learning.experiment import VisionExperimentConfig, run_vision_experiment
from dp_learning.plotting import logs_to_df, plot_acc_vs_epsilon, plot_learning_curve
from dp_learning.configs import ExperimentLog


def _parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _timestamped_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / timestamp
    _ensure_dir(run_dir)
    return run_dir


def _find_latest_timestamp_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {base_dir}")
    candidates = sorted([path for path in base_dir.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No timestamped runs found in {base_dir}")
    return candidates[-1]


def run_experiments(
    datasets: Iterable[str],
    methods: Iterable[str],
    agg_epsilons: Iterable[float],
    agg_delta: float,
    bin_epsilon: float | None,
    bin_delta: float | None,
    batch_sizes: Iterable[int],
    epochs: int,
    lr: float,
    max_grad_norm: float,
    dp_mechanisms: Iterable[str],
    log_every: int | None,
    seeds: Iterable[int],
    output_dir: Path,
) -> None:
    run_output_dir = _timestamped_output_dir(output_dir)

    logs = []

    for dataset_name in datasets:
        for batch_size in batch_sizes:
            for seed in seeds:
                for method in methods:
                    if method in {"baseline", "StandardSGD"}:
                        config = VisionExperimentConfig(
                            dataset_name=dataset_name,
                            method=method,
                            batch_size=batch_size,
                            epochs=epochs,
                            lr=lr,
                            max_grad_norm=max_grad_norm,
                            dp_mechanism="Naive",
                            seed=seed,
                            log_every=log_every,
                        )
                        _, log = run_vision_experiment(config)
                        logs.append(log)
                        _save_run(
                            output_dir=run_output_dir,
                            dataset=dataset_name,
                            log=log,
                            params=asdict(config),
                        )
                        continue

                    for dp_mechanism in dp_mechanisms:
                        for epsilon in agg_epsilons:
                            config = VisionExperimentConfig(
                                dataset_name=dataset_name,
                                method=method,
                                agg_epsilon=epsilon,
                                agg_delta=agg_delta,
                                bin_epsilon=bin_epsilon if method == "Edit-Style DP" else None,
                                bin_delta=bin_delta if method == "Edit-Style DP" else None,
                                batch_size=batch_size,
                                epochs=epochs,
                                lr=lr,
                                max_grad_norm=max_grad_norm,
                                dp_mechanism=dp_mechanism,
                                seed=seed,
                                log_every=log_every,
                            )
                            _, log = run_vision_experiment(config)
                            logs.append(log)
                            _save_run(
                                output_dir=run_output_dir,
                                dataset=dataset_name,
                                log=log,
                                params=asdict(config),
                            )

    raw_path = run_output_dir / "raw_logs.json"
    raw_path.write_text(json.dumps([asdict(log) for log in logs], indent=2))

    df = logs_to_df(logs)
    df.to_csv(run_output_dir / "summary.csv", index=False)

    for dataset_name in datasets:
        for batch_size in batch_sizes:
            plot_acc_vs_epsilon(
                df,
                dataset=dataset_name,
                batch_size=batch_size,
                save_path=run_output_dir
                / f"acc_vs_epsilon_{dataset_name}_bs{batch_size}.png",
                show=False,
            )

    dp_methods = [method for method in methods if method not in {"baseline", "StandardSGD"}]
    for dataset_name in datasets:
        for batch_size in batch_sizes:
            filtered_logs = [
                log
                for log in logs
                if log.dataset == dataset_name and log.batch_size == batch_size
            ]
            if not filtered_logs:
                continue
            for method in dp_methods:
                for epsilon in agg_epsilons:
                    safe_method = method.replace(" ", "_")
                    plot_learning_curve(
                        logs=filtered_logs,
                        dataset=dataset_name,
                        method=method,
                        agg_epsilon=epsilon,
                        save_path=run_output_dir
                        / f"learning_curve_{dataset_name}_bs{batch_size}_{safe_method}_{epsilon}.png",
                        show=False,
                    )


def visualize_results(
    output_dir: Path,
    timestamp: str | None = None,
) -> Path:
    run_dir = output_dir / timestamp if timestamp else _find_latest_timestamp_dir(output_dir)

    raw_logs_path = run_dir / "raw_logs.json"
    if not raw_logs_path.exists():
        raise FileNotFoundError(f"Missing raw logs: {raw_logs_path}")

    raw_logs = json.loads(raw_logs_path.read_text())
    logs = [ExperimentLog(**log) for log in raw_logs]

    if not logs:
        raise ValueError("No logs found to visualize.")

    df = logs_to_df(logs)

    datasets = sorted({log.dataset for log in logs})
    batch_sizes = sorted({log.batch_size for log in logs if log.batch_size is not None})
    methods = sorted({log.method for log in logs})
    dp_methods = [method for method in methods if method not in {"baseline", "StandardSGD"}]
    baseline_methods = [method for method in methods if method in {"baseline", "StandardSGD"}]

    for dataset_name in datasets:
        for batch_size in batch_sizes:
            plot_df = df[
                (df["dataset"] == dataset_name) & (df["batch_size"] == batch_size)
            ]
            if not plot_df.empty:
                plt.figure(figsize=(6, 4))
                dp_df = plot_df[plot_df["agg_epsilon"].notna()]
                if not dp_df.empty:
                    sns.lineplot(
                        data=dp_df,
                        x="agg_epsilon",
                        y="final_acc",
                        hue="method",
                        marker="o",
                        errorbar="sd",
                    )
                    plt.xscale("log")

                baseline_df = plot_df[plot_df["method"].isin(baseline_methods)]
                for method in baseline_methods:
                    method_df = baseline_df[baseline_df["method"] == method]
                    if method_df.empty:
                        continue
                    value = method_df["final_acc"].mean()
                    plt.axhline(value, label=f"{method} (baseline)", linestyle="--")

                plt.xlabel("Privacy budget ε")
                plt.ylabel("Test accuracy")
                plt.title(f"{dataset_name}: Accuracy vs Privacy Budget (bs={batch_size})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    run_dir / f"acc_vs_epsilon_{dataset_name}_bs{batch_size}.png"
                )
                plt.close()

            filtered_logs = [
                log
                for log in logs
                if log.dataset == dataset_name and log.batch_size == batch_size
            ]
            if not filtered_logs:
                continue

            for method in dp_methods:
                epsilons = sorted(
                    {
                        log.agg_epsilon
                        for log in filtered_logs
                        if log.method == method and log.agg_epsilon is not None
                    }
                )
                for epsilon in epsilons:
                    safe_method = method.replace(" ", "_")
                    plot_learning_curve(
                        logs=filtered_logs,
                        dataset=dataset_name,
                        method=method,
                        agg_epsilon=epsilon,
                        save_path=run_dir
                        / f"learning_curve_{dataset_name}_bs{batch_size}_{safe_method}_{epsilon}.png",
                        show=False,
                    )

            if baseline_methods:
                for method in baseline_methods:
                    method_logs = [
                        log for log in filtered_logs if log.method == method
                    ]
                    if not method_logs:
                        continue
                    epochs = method_logs[0].epochs
                    finals = [log.test_acc[-1] for log in method_logs if log.test_acc]
                    if not finals or not epochs:
                        continue
                    baseline_value = float(np.mean(finals))
                    plt.figure(figsize=(6, 4))
                    plt.plot(
                        epochs,
                        [baseline_value] * len(epochs),
                        label=f"{method} (baseline)",
                    )
                    plt.xlabel("Epoch")
                    plt.ylabel("Test accuracy")
                    plt.title(
                        f"{dataset_name} Baseline Accuracy Curve (bs={batch_size})"
                    )
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(
                        run_dir
                        / f"learning_curve_{dataset_name}_bs{batch_size}_{method}_baseline.png"
                    )
                    plt.close()

    return run_dir


def _save_run(
    output_dir: Path,
    dataset: str,
    log,
    params: dict,
) -> None:
    safe_method = log.method.replace(" ", "_")
    safe_dp_mechanism = str(log.dp_mechanism).replace(" ", "_")
    parts = [
        f"dataset={dataset}",
        f"batch_size={log.batch_size}",
        f"method={safe_method}",
        f"dp_mechanism={safe_dp_mechanism}",
        f"agg_epsilon={log.agg_epsilon}",
        f"agg_delta={log.agg_delta}",
        f"bin_epsilon={log.bin_epsilon}",
        f"bin_delta={log.bin_delta}",
        f"max_grad_norm={params.get('max_grad_norm')}",
        f"epochs={params.get('epochs')}",
        f"lr={params.get('lr')}",
        f"seed={log.seed}",
    ]
    run_dir = output_dir / "_".join(str(part) for part in parts)
    _ensure_dir(run_dir)

    (run_dir / "params.json").write_text(json.dumps(params, indent=2))
    (run_dir / "log.json").write_text(json.dumps(asdict(log), indent=2))
    logs_to_df([log]).to_csv(run_dir / "summary.csv", index=False)

    if log.agg_epsilon is not None:
        plot_learning_curve(
            logs=[log],
            dataset=dataset,
            method=log.method,
            agg_epsilon=log.agg_epsilon,
            save_path=run_dir / "learning_curve.png",
            show=False,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DP learning experiments and save raw logs + plots."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Only visualize results from an existing run.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Timestamp folder to visualize (defaults to latest).",
    )
    parser.add_argument(
        "--datasets",
        default="emnist,mnist,cifar10",
        help="Comma-separated datasets to run (emnist,mnist,cifar10).",
    )
    parser.add_argument(
        "--methods",
        default="baseline,StandardSGD,Hamming-Style DP,Edit-Style DP",
        help="Comma-separated list of methods to run.",
    )
    parser.add_argument(
        "--agg-epsilons",
        default="0.5,1.0,2.0,4.0,8.0",
        help="Comma-separated agg epsilons for DP runs.",
    )
    parser.add_argument("--agg-delta", type=float, default=1e-5)
    parser.add_argument("--bin-epsilon", type=float, default=0.5)
    parser.add_argument("--bin-delta", type=float, default=1e-5)
    parser.add_argument(
        "--batch-sizes",
        default="128,256,512",
        help="Comma-separated list of batch sizes.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--dp-mechanisms",
        default="Naive,DPPreSum",
        help="Comma-separated list of DP mechanisms to use for DP methods.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated list of seeds.")
    parser.add_argument("--output-dir", default="runs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.visualize:
        visualize_results(output_dir=output_dir, timestamp=args.timestamp)
        return

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    agg_epsilons = _parse_float_list(args.agg_epsilons)
    dp_mechanisms = [item.strip() for item in args.dp_mechanisms.split(",") if item.strip()]
    batch_sizes = _parse_int_list(args.batch_sizes)
    seeds = _parse_int_list(args.seeds)

    run_experiments(
        datasets=datasets,
        methods=methods,
        agg_epsilons=agg_epsilons,
        agg_delta=args.agg_delta,
        bin_epsilon=args.bin_epsilon,
        bin_delta=args.bin_delta,
        batch_sizes=batch_sizes,
        epochs=args.epochs,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        dp_mechanisms=dp_mechanisms,
        log_every=args.log_every,
        seeds=seeds,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
