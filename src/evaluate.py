import argparse
import json
import os
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

matplotlib.use("Agg")


def parse_kv_args(argv: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key.lstrip("-")] = value
    return parsed


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def pick_x_axis(history: pd.DataFrame) -> pd.Series:
    if "iteration" in history.columns:
        return history["iteration"]
    if "_step" in history.columns:
        return history["_step"]
    return pd.Series(range(len(history)))


def generate_learning_curve(run_id: str, history: pd.DataFrame, out_dir: str) -> List[str]:
    paths = []
    if "eval_accuracy" not in history.columns:
        return paths
    x = pick_x_axis(history)
    y = history["eval_accuracy"].fillna(method="ffill")
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", label="Eval Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve: {run_id}")
    if len(y) > 0:
        plt.annotate(f"{y.iloc[-1]:.3f}", (x.iloc[-1], y.iloc[-1]))
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_learning_curve.pdf")
    plt.savefig(path)
    plt.close()
    paths.append(path)
    return paths


def generate_metric_curve(
    run_id: str, history: pd.DataFrame, metric: str, out_dir: str
) -> List[str]:
    paths = []
    if metric not in history.columns:
        return paths
    x = pick_x_axis(history)
    y = history[metric].fillna(method="ffill")
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", label=metric)
    plt.xlabel("Iteration")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()}: {run_id}")
    if len(y) > 0:
        plt.annotate(f"{y.iloc[-1]:.3f}", (x.iloc[-1], y.iloc[-1]))
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_{metric}_curve.pdf")
    plt.savefig(path)
    plt.close()
    paths.append(path)
    return paths


def generate_confusion_matrix(run_id: str, summary: Dict, out_dir: str) -> List[str]:
    paths = []
    cm = summary.get("confusion_matrix")
    if cm is None:
        return paths
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    plt.figure(figsize=(4, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {run_id}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_confusion_matrix.pdf")
    plt.savefig(path)
    plt.close()
    paths.append(path)
    return paths


def primary_metric_value(summary: Dict, history: pd.DataFrame) -> float:
    if "final_accuracy" in summary:
        return float(summary["final_accuracy"])
    if "best_accuracy" in summary:
        return float(summary["best_accuracy"])
    if "eval_accuracy" in history.columns:
        series = history["eval_accuracy"].dropna()
        if len(series) > 0:
            return float(series.iloc[-1])
    return float("nan")


def metric_direction(metric_name: str) -> str:
    tokens = ["loss", "error", "perplexity", "distance"]
    if any(token in metric_name for token in tokens):
        return "min"
    return "max"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--run_ids", type=str, default=None)
    known_args, _ = parser.parse_known_args()
    kv_args = parse_kv_args([arg for arg in os.sys.argv[1:] if "=" in arg])

    results_dir = kv_args.get("results_dir") or known_args.results_dir
    run_ids_arg = kv_args.get("run_ids") or known_args.run_ids
    if results_dir is None or run_ids_arg is None:
        raise ValueError("results_dir and run_ids must be provided.")

    run_ids = json.loads(run_ids_arg)
    cfg = OmegaConf.load(os.path.join(os.getcwd(), "config", "config.yaml"))
    entity = cfg.wandb.entity
    project = cfg.wandb.project

    api = wandb.Api()
    generated_paths: List[str] = []
    comparison_metrics: Dict[str, Dict[str, float]] = {}
    history_map: Dict[str, pd.DataFrame] = {}
    summary_map: Dict[str, Dict] = {}
    config_map: Dict[str, Dict] = {}

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history(samples=100000)
        summary = run.summary._json_dict
        config = dict(run.config)
        history = history.replace({np.nan: None})
        history_map[run_id] = history
        summary_map[run_id] = summary
        config_map[run_id] = config

        run_dir = os.path.join(results_dir, run_id)
        ensure_dir(run_dir)
        metrics_path = os.path.join(run_dir, "metrics.json")
        save_json(
            metrics_path,
            {
                "history": history.to_dict(orient="records"),
                "summary": summary,
                "config": config,
            },
        )
        generated_paths.append(metrics_path)

        generated_paths += generate_learning_curve(run_id, history, run_dir)
        generated_paths += generate_metric_curve(run_id, history, "prompt_length", run_dir)
        generated_paths += generate_metric_curve(run_id, history, "worst_corpus_accuracy", run_dir)
        generated_paths += generate_metric_curve(run_id, history, "accepted_update_count", run_dir)
        generated_paths += generate_metric_curve(run_id, history, "edit_distance_words", run_dir)
        generated_paths += generate_confusion_matrix(run_id, summary, run_dir)

        for metric, value in summary.items():
            if isinstance(value, (int, float)):
                comparison_metrics.setdefault(metric, {})[run_id] = float(value)

    comparison_dir = os.path.join(results_dir, "comparison")
    ensure_dir(comparison_dir)

    primary_metric = "accuracy"
    primary_values = {
        run_id: primary_metric_value(summary_map[run_id], history_map[run_id])
        for run_id in run_ids
    }
    comparison_metrics.setdefault(primary_metric, primary_values)

    proposed_runs = []
    baseline_runs = []
    for run_id in run_ids:
        method_name = str(config_map[run_id].get("method", "")).lower()
        if "proposed" in run_id or "safe-refine" in method_name:
            proposed_runs.append(run_id)
        elif (
            "comparative" in run_id
            or "baseline" in run_id
            or "roppo" in method_name
            or "greedy" in method_name
            or "fixed" in method_name
        ):
            baseline_runs.append(run_id)

    best_proposed = max(
        proposed_runs, key=lambda rid: primary_values.get(rid, float("-inf")), default=None
    )
    best_baseline = max(
        baseline_runs, key=lambda rid: primary_values.get(rid, float("-inf")), default=None
    )

    best_proposed_val = (
        primary_values.get(best_proposed, float("nan")) if best_proposed else float("nan")
    )
    best_baseline_val = (
        primary_values.get(best_baseline, float("nan")) if best_baseline else float("nan")
    )

    gap = float("nan")
    if best_baseline and not np.isnan(best_baseline_val) and best_baseline_val != 0:
        improvement = (best_proposed_val - best_baseline_val) / best_baseline_val * 100
        gap = improvement if metric_direction(primary_metric) == "max" else -improvement

    stat_tests = {}
    if len(proposed_runs) >= 2 and len(baseline_runs) >= 2:
        proposed_vals = [primary_values[rid] for rid in proposed_runs]
        baseline_vals = [primary_values[rid] for rid in baseline_runs]
        ttest = stats.ttest_ind(proposed_vals, baseline_vals, equal_var=False)
        stat_tests["proposed_vs_baseline"] = {
            "t_stat": float(ttest.statistic),
            "p_value": float(ttest.pvalue),
        }

    aggregated = {
        "primary_metric": primary_metric,
        "metrics": comparison_metrics,
        "best_proposed": {"run_id": best_proposed, "value": best_proposed_val},
        "best_baseline": {"run_id": best_baseline, "value": best_baseline_val},
        "gap": gap,
        "stat_tests": stat_tests,
    }

    aggregated_path = os.path.join(comparison_dir, "aggregated_metrics.json")
    save_json(aggregated_path, aggregated)
    generated_paths.append(aggregated_path)

    plt.figure(figsize=(7, 4))
    run_labels = list(primary_values.keys())
    run_scores = [primary_values[rid] for rid in run_labels]
    sns.barplot(x=run_labels, y=run_scores)
    plt.ylabel("Accuracy")
    plt.title("Primary Metric Comparison")
    for idx, val in enumerate(run_scores):
        if not np.isnan(val):
            plt.text(idx, val, f"{val:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    bar_path = os.path.join(comparison_dir, "comparison_accuracy_bar_chart.pdf")
    plt.savefig(bar_path)
    plt.close()
    generated_paths.append(bar_path)

    if len(run_ids) > 1:
        method_labels = [str(config_map[rid].get("method", rid)) for rid in run_labels]
        df = pd.DataFrame({"run_id": run_labels, "method": method_labels, "accuracy": run_scores})
        plt.figure(figsize=(7, 4))
        sns.boxplot(x="method", y="accuracy", data=df)
        sns.stripplot(x="method", y="accuracy", data=df, color="black", size=4)
        plt.ylabel("Accuracy")
        plt.title("Accuracy Distribution by Method")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        box_path = os.path.join(comparison_dir, "comparison_accuracy_boxplot.pdf")
        plt.savefig(box_path)
        plt.close()
        generated_paths.append(box_path)

    if "final_acc_in" in comparison_metrics and "final_acc_ood" in comparison_metrics:
        plt.figure(figsize=(5, 5))
        for run_id in run_ids:
            x = comparison_metrics["final_acc_in"].get(run_id)
            y = comparison_metrics["final_acc_ood"].get(run_id)
            if x is None or y is None:
                continue
            plt.scatter(x, y, label=run_id)
            plt.annotate(run_id, (x, y))
        plt.xlabel("In-domain Accuracy")
        plt.ylabel("OOD Accuracy")
        plt.title("In-domain vs OOD Accuracy")
        plt.legend(fontsize=8)
        plt.tight_layout()
        scatter_path = os.path.join(comparison_dir, "comparison_in_ood_scatter.pdf")
        plt.savefig(scatter_path)
        plt.close()
        generated_paths.append(scatter_path)

    table_metrics = ["final_accuracy", "final_worst_corpus_accuracy", "accepted_update_count"]
    table_data = {
        metric: [comparison_metrics.get(metric, {}).get(rid, np.nan) for rid in run_labels]
        for metric in table_metrics
    }
    table_df = pd.DataFrame(table_data, index=run_labels)
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.4 * len(run_labels)))
    ax.axis("off")
    tbl = ax.table(
        cellText=np.round(table_df.values, 3),
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    plt.tight_layout()
    table_path = os.path.join(comparison_dir, "comparison_metrics_table.pdf")
    plt.savefig(table_path)
    plt.close()
    generated_paths.append(table_path)

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
