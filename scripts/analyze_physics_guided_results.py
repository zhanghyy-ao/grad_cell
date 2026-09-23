from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARAMETER_SHORT_NAMES = (
    "Pos. porosity",
    "Neg. porosity",
    "Sep. porosity",
    "Pos. active frac.",
    "Neg. active frac.",
    "Pos. diffusivity",
    "Neg. diffusivity",
)
PERFORMANCE_SHORT_NAMES = (
    "Capacity 1C",
    "Energy 1C",
    "Capacity 5C",
    "Energy 5C",
    "Retention 5C",
    "Capacity 6C",
    "Energy 6C",
    "Retention 6C",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def save_figure(fig: plt.Figure, prefix: Path, dpi: int) -> None:
    fig.savefig(prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def relative_error(predicted: float, target: float) -> float:
    return abs(predicted - target) / max(abs(target), 1e-12)


def grouped_prediction_consistency(
    rows: list[dict[str, Any]], group_field: str, value_field: str, names: list[str]
) -> dict[str, Any]:
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        values = np.asarray([float(row[value_field][name]) for name in names], dtype=np.float64)
        groups[str(row[group_field])].append(np.log(values))
    per_group = []
    for group, vectors in groups.items():
        matrix = np.stack(vectors)
        if len(matrix) < 2:
            continue
        rms = float(np.sqrt(np.mean(np.var(matrix, axis=0, ddof=0))))
        maximum_range = float(np.ptp(matrix, axis=0).max())
        per_group.append((group, len(matrix), rms, maximum_range))
    rms_values = np.asarray([item[2] for item in per_group], dtype=np.float64)
    range_values = np.asarray([item[3] for item in per_group], dtype=np.float64)
    return {
        "groups": len(per_group),
        "mean_log_rms": float(rms_values.mean()) if len(rms_values) else None,
        "p90_log_rms": float(np.quantile(rms_values, 0.9)) if len(rms_values) else None,
        "mean_maximum_log_range": float(range_values.mean()) if len(range_values) else None,
        "p90_maximum_log_range": float(np.quantile(range_values, 0.9)) if len(range_values) else None,
        "per_group": per_group,
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def decorate(axes: np.ndarray) -> None:
    for axis in axes.flat:
        if axis.axison:
            axis.grid(True, alpha=0.22, linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)


def plot_training(
    surrogate_history: list[dict[str, Any]],
    seed_histories: dict[int, list[dict[str, Any]]],
    output: Path,
    dpi: int,
) -> None:
    colors = {7: "#0072B2", 17: "#D55E00", 27: "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    fig.suptitle("Physics-guided training convergence")
    surrogate_epochs = [row["epoch"] for row in surrogate_history]
    axes[0, 0].plot(
        surrogate_epochs,
        [row["train_standardized_mse"] for row in surrogate_history],
        label="Train",
        color="#0072B2",
    )
    axes[0, 0].plot(
        surrogate_epochs,
        [row["validation_standardized_mse"] for row in surrogate_history],
        label="Validation",
        color="#D55E00",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(title="DFN surrogate convergence", xlabel="Epoch", ylabel="Standardized MSE")
    axes[0, 0].legend(frameon=False)

    for seed, history in seed_histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0, 1].plot(
            epochs,
            [row["train_loss"] for row in history],
            color=colors[seed],
            alpha=0.45,
            linewidth=1,
        )
        axes[0, 1].plot(
            epochs,
            [row["validation_loss"] for row in history],
            color=colors[seed],
            linewidth=1.8,
            label=f"Seed {seed} validation",
        )
        axes[1, 0].plot(
            epochs,
            [100.0 * row["validation_surrogate_performance_mean_absolute_percentage_error"] for row in history],
            color=colors[seed],
            linewidth=1.6,
            label=f"Seed {seed}",
        )
        axes[1, 1].plot(
            epochs,
            [100.0 * row["validation_surrogate_all_metrics_within_5pct_rate"] for row in history],
            color=colors[seed],
            linewidth=1.6,
            label=f"Seed {seed}",
        )
    axes[0, 1].set(title="Language-to-design objective", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].set(title="Validation surrogate performance error", xlabel="Epoch", ylabel="MAPE (%)")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].set(
        title="Validation samples passing all 5% metrics",
        xlabel="Epoch",
        ylabel="Pass rate (%)",
        ylim=(0, 100),
    )
    axes[1, 1].legend(frameon=False)
    decorate(axes)
    save_figure(fig, output / "figure_1_training_convergence", dpi)


def plot_seed_summary(
    seed_metrics: dict[int, dict[str, Any]], output: Path, dpi: int
) -> None:
    seeds = sorted(seed_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Three-seed generalization and error structure")
    x = np.arange(len(seeds))
    width = 0.36
    design = [100 * seed_metrics[s]["test"]["design_mean_absolute_percentage_error"] for s in seeds]
    performance = [
        100 * seed_metrics[s]["test"]["surrogate_performance_mean_absolute_percentage_error"]
        for s in seeds
    ]
    axes[0, 0].bar(x - width / 2, design, width, label="Design MAPE", color="#56B4E9")
    axes[0, 0].bar(x + width / 2, performance, width, label="Performance MAPE", color="#E69F00")
    axes[0, 0].set_xticks(x, [str(seed) for seed in seeds])
    axes[0, 0].set(title="Independent test error", xlabel="Seed", ylabel="MAPE (%)")
    axes[0, 0].legend(frameon=False)

    validation_pass = [
        100 * seed_metrics[s]["validation"]["surrogate_all_metrics_within_5pct_rate"] for s in seeds
    ]
    test_pass = [100 * seed_metrics[s]["test"]["surrogate_all_metrics_within_5pct_rate"] for s in seeds]
    axes[0, 1].bar(x - width / 2, validation_pass, width, label="Validation", color="#009E73")
    axes[0, 1].bar(x + width / 2, test_pass, width, label="Test", color="#CC79A7")
    axes[0, 1].set_xticks(x, [str(seed) for seed in seeds])
    axes[0, 1].set(
        title="All eight metrics within 5%", xlabel="Seed", ylabel="Pass rate (%)", ylim=(0, 100)
    )
    axes[0, 1].legend(frameon=False)

    parameter_names = list(next(iter(seed_metrics.values()))["parameter_names"])
    parameter_values = np.asarray(
        [
            [
                100 * seed_metrics[seed]["test"]["per_parameter"][name]["mean_absolute_percentage_error"]
                for name in parameter_names
            ]
            for seed in seeds
        ]
    )
    parameter_mean = parameter_values.mean(axis=0)
    parameter_std = parameter_values.std(axis=0)
    y = np.arange(len(parameter_names))
    axes[1, 0].barh(y, parameter_mean, xerr=parameter_std, color="#0072B2", alpha=0.85, capsize=3)
    axes[1, 0].set_yticks(y, PARAMETER_SHORT_NAMES)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(title="Test design error across seeds", xlabel="MAPE mean ± population SD (%)")

    performance_names = list(next(iter(seed_metrics.values()))["performance_fields"])
    performance_values = np.asarray(
        [
            [
                100
                * seed_metrics[seed]["test"]["per_performance_field"][name][
                    "mean_absolute_percentage_error"
                ]
                for name in performance_names
            ]
            for seed in seeds
        ]
    )
    performance_mean = performance_values.mean(axis=0)
    performance_std = performance_values.std(axis=0)
    y = np.arange(len(performance_names))
    axes[1, 1].barh(
        y, performance_mean, xerr=performance_std, color="#D55E00", alpha=0.85, capsize=3
    )
    axes[1, 1].set_yticks(y, PERFORMANCE_SHORT_NAMES)
    axes[1, 1].invert_yaxis()
    axes[1, 1].axvline(5.0, color="black", linestyle="--", linewidth=1, label="5%")
    axes[1, 1].set(title="Test surrogate-performance error", xlabel="MAPE mean ± population SD (%)")
    axes[1, 1].legend(frameon=False)
    decorate(axes)
    save_figure(fig, output / "figure_2_three_seed_summary", dpi)


def plot_dfn_replay(
    replay_metrics: dict[str, Any], replay_rows: list[dict[str, Any]], output: Path, dpi: int
) -> dict[str, Any]:
    fields = list(replay_metrics["per_field"])
    dfn_errors = np.asarray(
        [[float(row["dfn_replay"]["relative_errors"][field]) for field in fields] for row in replay_rows]
    )
    surrogate_dfn_errors = np.asarray(
        [
            [
                relative_error(
                    float(row["surrogate_predicted_performance"][field]),
                    float(row["dfn_replay"]["performance"][field]),
                )
                for field in fields
            ]
            for row in replay_rows
        ]
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Strict DFN replay: seed 7")
    x = np.arange(len(fields))
    means = [100 * replay_metrics["per_field"][field]["mean_absolute_percentage_error"] for field in fields]
    medians = [
        100 * replay_metrics["per_field"][field]["median_absolute_percentage_error"] for field in fields
    ]
    p90 = [100 * replay_metrics["per_field"][field]["p90_absolute_percentage_error"] for field in fields]
    axes[0, 0].plot(x, means, marker="o", color="#D55E00", label="Mean")
    axes[0, 0].plot(x, medians, marker="s", color="#0072B2", label="Median")
    axes[0, 0].plot(x, p90, marker="^", color="#CC79A7", label="P90")
    axes[0, 0].axhline(5.0, color="black", linestyle="--", linewidth=1, label="5% tolerance")
    axes[0, 0].set_xticks(x, PERFORMANCE_SHORT_NAMES, rotation=35, ha="right")
    axes[0, 0].set(title="DFN-to-target relative error", ylabel="Absolute percentage error (%)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(
        x, 100 * surrogate_dfn_errors.mean(axis=0), color="#009E73", alpha=0.85
    )
    axes[0, 1].set_xticks(x, PERFORMANCE_SHORT_NAMES, rotation=35, ha="right")
    axes[0, 1].set(
        title="Frozen surrogate vs direct DFN",
        ylabel="Mean absolute percentage difference (%)",
    )

    flattened = 100 * dfn_errors.reshape(-1)
    axes[1, 0].hist(flattened, bins=30, color="#56B4E9", edgecolor="white")
    axes[1, 0].axvline(5.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(title="All sample-metric DFN errors", xlabel="Absolute percentage error (%)", ylabel="Count")

    target = np.asarray(
        [[float(row["target_verified_performance"][field]) for field in fields] for row in replay_rows]
    )
    dfn = np.asarray(
        [[float(row["dfn_replay"]["performance"][field]) for field in fields] for row in replay_rows]
    )
    normalized_target = target / np.median(target, axis=0, keepdims=True)
    normalized_dfn = dfn / np.median(target, axis=0, keepdims=True)
    axes[1, 1].scatter(
        normalized_target.reshape(-1),
        normalized_dfn.reshape(-1),
        s=9,
        alpha=0.45,
        color="#0072B2",
        rasterized=True,
    )
    lower = min(float(normalized_target.min()), float(normalized_dfn.min()))
    upper = max(float(normalized_target.max()), float(normalized_dfn.max()))
    axes[1, 1].plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(
        title="DFN prediction vs requested performance",
        xlabel="Target / field median",
        ylabel="DFN / field median",
    )
    decorate(axes)
    save_figure(fig, output / "figure_3_strict_dfn_replay", dpi)
    return {
        "surrogate_vs_dfn_mean_absolute_percentage_difference": float(
            surrogate_dfn_errors.mean()
        ),
        "surrogate_vs_dfn_per_field": {
            field: float(surrogate_dfn_errors[:, index].mean())
            for index, field in enumerate(fields)
        },
        "dfn_error_mean_over_all_sample_metrics": float(dfn_errors.mean()),
        "dfn_error_median_over_all_sample_metrics": float(np.median(dfn_errors)),
        "dfn_error_p90_over_all_sample_metrics": float(np.quantile(dfn_errors, 0.9)),
    }


def plot_consistency(
    prompt_consistency: dict[int, dict[str, Any]],
    cross_seed_consistency: dict[str, Any],
    output: Path,
    dpi: int,
) -> None:
    seeds = sorted(prompt_consistency)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    fig.suptitle("Prediction stability diagnostics")
    data = [
        [100 * item[2] for item in prompt_consistency[seed]["per_group"]] for seed in seeds
    ]
    axes[0].boxplot(data, tick_labels=[str(seed) for seed in seeds], showfliers=False)
    axes[0].set(
        title="Equivalent descriptions of one design",
        xlabel="Seed",
        ylabel="Within-family log RMS (%)",
    )
    cross_data = [100 * item[2] for item in cross_seed_consistency["per_group"]]
    axes[1].hist(cross_data, bins=24, color="#CC79A7", edgecolor="white")
    axes[1].set(
        title="Same prompt across three seeds",
        xlabel="Across-seed log RMS (%)",
        ylabel="Task count",
    )
    for axis in axes:
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output / "figure_4_prediction_stability", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GradCell physics-guided experiment results.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    output = (args.output_dir or root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()

    surrogate_dir = root / "dfn_surrogate"
    surrogate_metrics = read_json(surrogate_dir / "metrics.json")
    surrogate_history = read_json(surrogate_dir / "history.json")
    seed_dirs = sorted(
        [path for path in root.glob("seed_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not seed_dirs:
        raise ValueError("No seed_* result directories were found")
    seed_metrics: dict[int, dict[str, Any]] = {}
    seed_histories: dict[int, list[dict[str, Any]]] = {}
    seed_predictions: dict[int, list[dict[str, Any]]] = {}
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])
        seed_metrics[seed] = read_json(seed_dir / "metrics.json")
        seed_histories[seed] = read_json(seed_dir / "history.json")
        seed_predictions[seed] = read_jsonl(seed_dir / "test_predictions.jsonl")

    plot_training(surrogate_history, seed_histories, output, args.dpi)
    plot_seed_summary(seed_metrics, output, args.dpi)
    parameter_names = list(next(iter(seed_metrics.values()))["parameter_names"])
    prompt_consistency = {
        seed: grouped_prediction_consistency(
            rows,
            "physical_design_id",
            "predicted_parameter_multipliers",
            parameter_names,
        )
        for seed, rows in seed_predictions.items()
    }
    cross_seed_rows = []
    for seed, rows in seed_predictions.items():
        for row in rows:
            cross_seed_rows.append({**row, "seed_task": row["task_id"], "seed": seed})
    cross_seed_consistency = grouped_prediction_consistency(
        cross_seed_rows, "seed_task", "predicted_parameter_multipliers", parameter_names
    )
    plot_consistency(prompt_consistency, cross_seed_consistency, output, args.dpi)

    replay_path = root / "seed_7" / "dfn_replay_metrics.json"
    replay_rows_path = root / "seed_7" / "test_predictions_dfn.jsonl"
    replay_metrics = read_json(replay_path) if replay_path.exists() else None
    replay_rows = read_jsonl(replay_rows_path) if replay_rows_path.exists() else []
    replay_additional = (
        plot_dfn_replay(replay_metrics, replay_rows, output, args.dpi)
        if replay_metrics is not None and replay_rows
        else None
    )

    seeds = sorted(seed_metrics)
    aggregate = {
        "test_design_mape": mean_std(
            [seed_metrics[seed]["test"]["design_mean_absolute_percentage_error"] for seed in seeds]
        ),
        "test_surrogate_performance_mape": mean_std(
            [
                seed_metrics[seed]["test"]["surrogate_performance_mean_absolute_percentage_error"]
                for seed in seeds
            ]
        ),
        "test_all_metrics_within_5pct_rate": mean_std(
            [
                seed_metrics[seed]["test"]["surrogate_all_metrics_within_5pct_rate"]
                for seed in seeds
            ]
        ),
        "validation_all_metrics_within_5pct_rate": mean_std(
            [
                seed_metrics[seed]["validation"]["surrogate_all_metrics_within_5pct_rate"]
                for seed in seeds
            ]
        ),
    }
    best_epochs = {
        str(seed): {
            "epochs_recorded": len(seed_histories[seed]),
            "minimum_validation_loss_epoch": int(
                min(seed_histories[seed], key=lambda row: row["validation_loss"])["epoch"]
            ),
            "minimum_validation_loss": float(
                min(row["validation_loss"] for row in seed_histories[seed])
            ),
        }
        for seed in seeds
    }
    summary = {
        "schema": "gradcell.physics_guided_experiment_analysis.v1",
        "results_dir": str(root),
        "file_inventory": {
            "seed_count": len(seeds),
            "seeds": seeds,
            "prediction_records_per_seed": {
                str(seed): len(seed_predictions[seed]) for seed in seeds
            },
            "strict_dfn_replay_seeds": [7] if replay_metrics is not None else [],
        },
        "surrogate": {
            "physical_designs": surrogate_metrics["physical_designs"],
            "validation_mape": surrogate_metrics["validation"]["mean_absolute_percentage_error"],
            "test_mape": surrogate_metrics["test"]["mean_absolute_percentage_error"],
            "validation_to_test_mape_ratio": (
                surrogate_metrics["test"]["mean_absolute_percentage_error"]
                / surrogate_metrics["validation"]["mean_absolute_percentage_error"]
            ),
            "validation_quality_gate_passed": surrogate_metrics["quality_gate"]["passed"],
            "test_negative_r2_fields": [
                field
                for field, values in surrogate_metrics["test"]["per_field"].items()
                if values["r2_log_space"] < 0.0
            ],
        },
        "aggregate_across_seeds": aggregate,
        "per_seed_best_epoch": best_epochs,
        "prompt_variant_consistency": {
            str(seed): {key: value for key, value in result.items() if key != "per_group"}
            for seed, result in prompt_consistency.items()
        },
        "cross_seed_consistency": {
            key: value for key, value in cross_seed_consistency.items() if key != "per_group"
        },
        "strict_dfn_replay": {
            "report": replay_metrics,
            "additional_analysis": replay_additional,
        }
        if replay_metrics is not None
        else None,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, seed_metrics, surrogate_metrics, output / "analysis_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_report(
    summary: dict[str, Any],
    seed_metrics: dict[int, dict[str, Any]],
    surrogate_metrics: dict[str, Any],
    path: Path,
) -> None:
    aggregate = summary["aggregate_across_seeds"]
    replay = summary["strict_dfn_replay"]
    test_parameter_rows = []
    for index, name in enumerate(next(iter(seed_metrics.values()))["parameter_names"]):
        values = [
            100
            * metric["test"]["per_parameter"][name]["mean_absolute_percentage_error"]
            for metric in seed_metrics.values()
        ]
        stats = mean_std(values)
        test_parameter_rows.append(
            f"| {PARAMETER_SHORT_NAMES[index]} | {stats['mean']:.2f} | {stats['std_population']:.2f} |"
        )
    test_performance_rows = []
    for index, name in enumerate(next(iter(seed_metrics.values()))["performance_fields"]):
        values = [
            100
            * metric["test"]["per_performance_field"][name]["mean_absolute_percentage_error"]
            for metric in seed_metrics.values()
        ]
        stats = mean_std(values)
        test_performance_rows.append(
            f"| {PERFORMANCE_SHORT_NAMES[index]} | {stats['mean']:.2f} | {stats['std_population']:.2f} |"
        )
    negative_fields = ", ".join(summary["surrogate"]["test_negative_r2_fields"])
    report = f"""# DeepSeek-2158 physics-guided 实验分析

## 数据与文件完整性

- 语言设计数据：2158 条，严格划分为 1726/216/216。
- 独立物理设计：720 个，代理模型按物理设计划分为 576/72/72。
- 语言设计随机种子：{', '.join(map(str, summary['file_inventory']['seeds']))}。
- 每个种子测试预测：216 条。
- 严格 DFN replay：目前仅 seed 7，共 {replay['report']['records'] if replay else 0} 条。

## 核心结论

1. 三个种子的结构可行性均为 100%，说明有界 MLP 和结构约束有效。
2. 三 seed 测试集设计参数 MAPE 为 {100 * aggregate['test_design_mape']['mean']:.2f}% ± {100 * aggregate['test_design_mape']['std_population']:.2f}%。
3. 代理模型预测的测试性能 MAPE 为 {100 * aggregate['test_surrogate_performance_mape']['mean']:.2f}% ± {100 * aggregate['test_surrogate_performance_mape']['std_population']:.2f}%。
4. 八项性能同时在 5% 内的测试通过率为 {100 * aggregate['test_all_metrics_within_5pct_rate']['mean']:.2f}% ± {100 * aggregate['test_all_metrics_within_5pct_rate']['std_population']:.2f}%，明显低于验证集的 {100 * aggregate['validation_all_metrics_within_5pct_rate']['mean']:.2f}%。
5. seed 7 严格 DFN replay 成功率为 {100 * replay['report']['dfn_success_rate']:.1f}%，但八项指标同时在 5% 内的比例只有 {100 * replay['report']['all_metrics_within_tolerance_rate']:.2f}%。因此模型已经能够生成可仿真的设计，但尚不能稳定满足全部目标性能。

## DFN 性能代理模型

代理模型验证集 MAPE 为 {100 * surrogate_metrics['validation']['mean_absolute_percentage_error']:.3f}%，独立测试集为 {100 * surrogate_metrics['test']['mean_absolute_percentage_error']:.2f}%，测试误差是验证误差的 {summary['surrogate']['validation_to_test_mape_ratio']:.1f} 倍。当前 quality gate 使用验证集，因此虽然报告为 passed，也不能证明代理模型具有同等测试泛化能力。

测试集 R² 为负的指标包括：{negative_fields}。这些负 R² 表明在相应未见物理设计上，代理模型甚至弱于使用测试集均值作为预测，尤其需要警惕高倍率指标。

另一方面，在 seed 7 模型实际生成的设计附近，代理预测与直接 DFN 的平均差异为 {100 * replay['additional_analysis']['surrogate_vs_dfn_mean_absolute_percentage_difference']:.3f}%。这说明 seed 7 的主要误差来自语言到设计映射没有完全命中目标，而不是 replay 点附近代理模型与 DFN 的数值不一致。

## 七个设计参数的测试误差

| 参数 | 三 seed 平均 MAPE (%) | seed 间标准差 (%) |
|---|---:|---:|
{chr(10).join(test_parameter_rows)}

孔隙率和活性材料比例整体较准，误差主要集中在正负极扩散率倍率。扩散率对有限的 1C/5C/6C 汇总性能并非总是唯一可辨识，因此较大的参数误差不必然意味着性能同样差，但反映出逆问题的一对多性与跨 seed 不稳定性。

## 八项性能的测试误差

| 性能指标 | 三 seed 平均 MAPE (%) | seed 间标准差 (%) |
|---|---:|---:|
{chr(10).join(test_performance_rows)}

1C 容量和能量相对稳定；5C/6C 能量保持率最难预测。seed 7 的真实 DFN replay 中，5C 保持率 MAPE 为 {100 * replay['report']['per_field']['energy_retention_5c']['mean_absolute_percentage_error']:.2f}%，6C 保持率为 {100 * replay['report']['per_field']['energy_retention_6c']['mean_absolute_percentage_error']:.2f}%。这两项是当前模型不能稳定通过 5% 联合阈值的主要原因。

## 稳定性与可信度限制

- 三个 seed 的结果差异明显，seed 27 整体最好，但扩散率误差的优劣方向会随 seed 改变。
- 目前只有 seed 7 做了严格 DFN replay，不能用 seed 7 的 24.07% 通过率代表三 seed 平均物理性能。
- 测试集包含同一物理设计的多个自然语言变体；报告中的一致性图检验了模型是否对等价描述保持稳定。
- 当前自然语言直接给出了容量、能量和倍率性能数值，所以实验验证的是“性能描述到参数逆映射”，还没有验证更开放的工程描述、循环寿命、安全和成本语义。
- 代理模型的 validation/test 差距表明后续 quality gate 必须同时检查独立 test split，或采用按物理设计的交叉验证。

## 建议

1. 对 seed 17 和 seed 27 也执行相同的 216 条严格 DFN replay，再报告三 seed 的 DFN 均值和标准差。
2. 将代理模型 quality gate 改为独立测试集或 5 折 physical-design group cross-validation，不能只使用 validation。
3. 对扩散率使用多解标签、Top-k 输出或性能等价损失，避免强迫模型恢复不可唯一辨识的单一参数答案。
4. 提高 5C/6C retention 的损失权重，或对高倍率区域进行分层采样和难例再训练。
5. 最终论文应同时报告结构可行率、DFN 成功率、每项 MAPE、八项联合通过率和跨 seed 方差。

## 图表

- `figure_1_training_convergence.png/pdf`：代理模型和三 seed 收敛曲线。
- `figure_2_three_seed_summary.png/pdf`：跨 seed 测试误差及参数/性能分解。
- `figure_3_strict_dfn_replay.png/pdf`：seed 7 严格 DFN replay。
- `figure_4_prediction_stability.png/pdf`：自然语言变体一致性与跨 seed 稳定性。
"""
    path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
