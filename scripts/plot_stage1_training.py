from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def series(history: list[dict[str, Any]], current: str, legacy: str | None = None) -> np.ndarray:
    values = []
    for row in history:
        if current in row:
            value = row[current]
        elif legacy is not None and legacy in row:
            value = row[legacy]
        else:
            raise KeyError(f"History does not contain {current!r}")
        values.append(float(value))
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"History field {current!r} contains non-finite values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stage-1 supervised MLP training metrics.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--title", default="Stage 1: supervised language-to-design training")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    history_path = args.run_dir / "history.json"
    metrics_path = args.run_dir / "metrics.json"
    history = read_json(history_path)
    if not isinstance(history, list) or not history:
        raise ValueError("history.json must contain a non-empty list")
    metrics = read_json(metrics_path) if metrics_path.exists() else None
    epochs = series(history, "epoch").astype(int)
    train_loss = series(history, "train_design_smooth_l1", "train_design_loss")
    validation_loss = series(
        history, "validation_design_smooth_l1", "normalized_design_smooth_l1"
    )
    validation_mse = series(history, "validation_design_mse", "normalized_design_mse")
    generalization_gap = validation_loss - train_loss
    best_position = int(np.argmin(validation_loss))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    fig.suptitle(args.title, fontsize=14, fontweight="bold")

    axes[0, 0].plot(
        epochs, train_loss, color="#0072B2", marker="o", linewidth=2, label="Train Smooth-L1"
    )
    axes[0, 0].plot(
        epochs,
        validation_loss,
        color="#D55E00",
        marker="s",
        linewidth=2,
        label="Validation Smooth-L1",
    )
    axes[0, 0].scatter(
        [epochs[best_position]],
        [validation_loss[best_position]],
        color="#009E73",
        marker="*",
        s=130,
        zorder=5,
        label=f"Best epoch {epochs[best_position]}",
    )
    axes[0, 0].set(title="Supervised design loss", xlabel="Epoch", ylabel="Normalized loss")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        epochs, validation_mse, color="#CC79A7", marker="o", linewidth=2
    )
    axes[0, 1].set(
        title="Validation parameter error",
        xlabel="Epoch",
        ylabel="Normalized design MSE",
    )

    axes[1, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1, 0].plot(
        epochs, generalization_gap, color="#E69F00", marker="o", linewidth=2
    )
    axes[1, 0].set(
        title="Generalization gap",
        xlabel="Epoch",
        ylabel="Validation Smooth-L1 - train Smooth-L1",
    )

    if metrics is not None:
        split_names = ("validation", "test")
        x = np.arange(len(split_names), dtype=np.float64)
        width = 0.34
        mse = [float(metrics[name]["normalized_design_mse"]) for name in split_names]
        smooth = [
            float(metrics[name]["normalized_design_smooth_l1"]) for name in split_names
        ]
        axes[1, 1].bar(x - width / 2, mse, width, color="#56B4E9", label="MSE")
        axes[1, 1].bar(
            x + width / 2, smooth, width, color="#009E73", label="Smooth-L1"
        )
        axes[1, 1].set_xticks(x, [name.capitalize() for name in split_names])
        axes[1, 1].set(title="Best-checkpoint evaluation", ylabel="Normalized error")
        axes[1, 1].legend(frameon=False)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "metrics.json not found\nFinal split metrics unavailable",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    for axis in axes.flat:
        if axis.axison:
            axis.grid(True, alpha=0.22, linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)

    prefix = args.output_prefix or (args.run_dir / "stage1_training_curves")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = prefix.with_suffix(".png")
    pdf_path = prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "schema": "gradcell.stage1_training_plot_summary.v1",
        "history": str(history_path),
        "epochs_completed": int(len(epochs)),
        "best_epoch": int(epochs[best_position]),
        "best_validation_design_smooth_l1": float(validation_loss[best_position]),
        "final_train_design_smooth_l1": float(train_loss[-1]),
        "final_validation_design_smooth_l1": float(validation_loss[-1]),
        "final_validation_design_mse": float(validation_mse[-1]),
        "final_generalization_gap": float(generalization_gap[-1]),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    summary_path = prefix.with_name(prefix.name + "_summary").with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
