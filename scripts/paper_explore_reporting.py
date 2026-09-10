from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
LATENT_NAMES = ("eps_p", "eps_n", "eps_s", "phi_p", "N/P")
PERFORMANCE_NAMES = ("1C energy", "5C retention", "6C retention")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )
    return plt


def _save(fig: Any, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.clf()


def plot_training_history(history: list[dict[str, float]], output_dir: Path) -> None:
    if not history:
        return
    plt = _pyplot()
    epochs = np.asarray([row["epoch"] for row in history])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train", color=OKABE_ITO[0])
    axes[0].plot(
        epochs, [row["validation_loss"] for row in history], label="Validation", color=OKABE_ITO[1]
    )
    axes[0].set(xlabel="Epoch", ylabel="Weighted loss", title="A  Total objective")
    axes[0].legend(frameon=False)
    component_names = ("latent_loss", "performance_loss", "feasibility_loss", "unsupported_loss")
    for index, name in enumerate(component_names):
        key = f"validation_{name}"
        if key in history[0]:
            axes[1].plot(
                epochs,
                [row[key] for row in history],
                label=name.removesuffix("_loss").replace("_", " "),
                color=OKABE_ITO[index],
                linestyle=("-", "--", "-.", ":")[index],
            )
    axes[1].set(xlabel="Epoch", ylabel="Validation loss", title="B  Objective components")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, output_dir / "figures" / "training_curves")
    plt.close(fig)


def write_metrics_csv(metrics: dict[str, Any], path: Path) -> None:
    rows: list[tuple[str, Any]] = []

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                flatten(f"{prefix}.{key}" if prefix else str(key), nested)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            rows.append((prefix, value))

    flatten("", metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def plot_supervised_benchmark(
    *,
    predicted_latent: np.ndarray,
    candidates: np.ndarray,
    candidate_mask: np.ndarray,
    feasible: np.ndarray,
    feasibility_probability: np.ndarray,
    predicted_performance: np.ndarray,
    target_performance: np.ndarray,
    output_dir: Path,
) -> None:
    plt = _pyplot()
    valid_rows = np.flatnonzero(feasible)
    nearest = np.zeros_like(predicted_latent)
    for row in valid_rows:
        error = np.abs(candidates[row] - predicted_latent[row]).mean(axis=1)
        error[~candidate_mask[row]] = np.inf
        nearest[row] = candidates[row, int(error.argmin())]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    if len(valid_rows):
        axes[0, 0].boxplot(
            np.abs(predicted_latent[valid_rows] - nearest[valid_rows]),
            tick_labels=LATENT_NAMES,
            showfliers=False,
        )
    axes[0, 0].set(ylabel="Absolute latent error", title="A  Nearest Top-K latent error")
    labels = feasible.astype(int)
    axes[0, 1].hist(
        [feasibility_probability[labels == 0], feasibility_probability[labels == 1]],
        bins=np.linspace(0, 1, 16),
        label=("Infeasible", "Feasible"),
        color=(OKABE_ITO[1], OKABE_ITO[0]),
        alpha=0.75,
    )
    axes[0, 1].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(xlabel="Predicted probability", ylabel="Samples", title="B  Feasibility scores")
    axes[0, 1].legend(frameon=False)
    if len(valid_rows):
        target_valid = target_performance[valid_rows]
        predicted_valid = predicted_performance[valid_rows]
        scale = np.maximum(target_valid.std(axis=0), 1e-8)
        center = target_valid.mean(axis=0)
        target_standardized = (target_valid - center) / scale
        predicted_standardized = (predicted_valid - center) / scale
        for index, name in enumerate(PERFORMANCE_NAMES):
            axes[1, 0].scatter(
                target_standardized[:, index],
                predicted_standardized[:, index],
                s=10,
                alpha=0.55,
                color=OKABE_ITO[index],
                label=name,
            )
        lower = min(target_standardized.min(), predicted_standardized.min())
        upper = max(target_standardized.max(), predicted_standardized.max())
        axes[1, 0].plot((lower, upper), (lower, upper), color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(
        xlabel="Standardized target", ylabel="Standardized prediction", title="C  Performance proxy"
    )
    axes[1, 0].legend(frameon=False)
    if len(valid_rows):
        mae = np.abs(predicted_standardized - target_standardized).mean(axis=0)
        axes[1, 1].bar(PERFORMANCE_NAMES, mae, color=OKABE_ITO[:3])
        axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].set(ylabel="Standardized mean absolute error", title="D  Benchmark errors")
    fig.tight_layout()
    _save(fig, output_dir / "figures" / "supervised_benchmark")
    plt.close(fig)


def plot_joint_physics(
    predicted_performance: np.ndarray, physics: dict[str, np.ndarray], output_dir: Path
) -> None:
    plt = _pyplot()
    physical = np.column_stack(
        [physics["energy_wh_kg"], physics["energy_retention_5c"], physics["energy_retention_6c"]]
    )
    valid = physics["status"] == 1
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for index, (ax, name) in enumerate(zip(axes, PERFORMANCE_NAMES, strict=True)):
        ax.scatter(
            physical[valid, index], predicted_performance[valid, index], s=11, alpha=0.55, color=OKABE_ITO[index]
        )
        if valid.any():
            lower = min(physical[valid, index].min(), predicted_performance[valid, index].min())
            upper = max(physical[valid, index].max(), predicted_performance[valid, index].max())
            ax.plot((lower, upper), (lower, upper), color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="SPMe result", ylabel="MLP proxy", title=f"{chr(65 + index)}  {name}")
    fig.tight_layout()
    _save(fig, output_dir / "figures" / "mlp_proxy_vs_physics")
    plt.close(fig)
