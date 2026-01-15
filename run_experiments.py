from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from dp_learning.experiment import VisionExperimentConfig, run_vision_experiment
from dp_learning.plotting import logs_to_df, plot_acc_vs_epsilon, plot_learning_curve


def _parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_experiments(
    dataset_name: str,
    methods: Iterable[str],
    agg_epsilons: Iterable[float],
    agg_delta: float,
    bin_epsilon: float | None,
    bin_delta: float | None,
    batch_size: int,
    epochs: int,
    lr: float,
    max_grad_norm: float,
    dp_mechanisms: Iterable[str],
    log_every: int | None,
    seeds: Iterable[int],
    output_dir: Path,
) -> None:
    _ensure_dir(output_dir)

    logs = []

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
                    output_dir=output_dir,
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
                        output_dir=output_dir,
                        dataset=dataset_name,
                        log=log,
                        params=asdict(config),
                    )

    raw_path = output_dir / "raw_logs.json"
    raw_path.write_text(json.dumps([asdict(log) for log in logs], indent=2))

    df = logs_to_df(logs)
    df.to_csv(output_dir / "summary.csv", index=False)

    plot_acc_vs_epsilon(
        df,
        dataset=dataset_name,
        save_path=output_dir / "acc_vs_epsilon.png",
        show=False,
    )

    dp_methods = [method for method in methods if method not in {"baseline", "StandardSGD"}]
    for method in dp_methods:
        for epsilon in agg_epsilons:
            safe_method = method.replace(" ", "_")
            plot_learning_curve(
                logs=logs,
                dataset=dataset_name,
                method=method,
                agg_epsilon=epsilon,
                save_path=output_dir / f"learning_curve_{safe_method}_{epsilon}.png",
                show=False,
            )


def _save_run(
    output_dir: Path,
    dataset: str,
    log,
    params: dict,
) -> None:
    safe_method = log.method.replace(" ", "_")
    parts = [
        f"dataset={dataset}",
        f"method={safe_method}",
        f"dp_mechanism={log.dp_mechanism}",
        f"agg_epsilon={log.agg_epsilon}",
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
    parser.add_argument("--dataset", default="emnist", choices=["emnist", "mnist", "cifar10"])
    parser.add_argument(
        "--methods",
        default="baseline,StandardSGD,Hamming-Style DP,Edit-Style DP",
        help="Comma-separated list of methods to run.",
    )
    parser.add_argument(
        "--agg-epsilons",
        default="0.5,1.0,2.0,4.0",
        help="Comma-separated agg epsilons for DP runs.",
    )
    parser.add_argument("--agg-delta", type=float, default=1e-5)
    parser.add_argument("--bin-epsilon", type=float, default=0.5)
    parser.add_argument("--bin-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--dp-mechanisms",
        default="Naive,DPPreSum",
        help="Comma-separated list of DP mechanisms to use for DP methods.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seeds", default="0", help="Comma-separated list of seeds.")
    parser.add_argument("--output-dir", default="runs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    agg_epsilons = _parse_float_list(args.agg_epsilons)
    dp_mechanisms = [item.strip() for item in args.dp_mechanisms.split(",") if item.strip()]
    seeds = _parse_int_list(args.seeds)

    run_experiments(
        dataset_name=args.dataset,
        methods=methods,
        agg_epsilons=agg_epsilons,
        agg_delta=args.agg_delta,
        bin_epsilon=args.bin_epsilon,
        bin_delta=args.bin_delta,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        dp_mechanisms=dp_mechanisms,
        log_every=args.log_every,
        seeds=seeds,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
