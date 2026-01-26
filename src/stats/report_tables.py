"""Generate tables, figures, and summaries for evaluation runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _require_packages() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies for stats reporting. Install numpy, pandas, scipy, matplotlib."
        ) from exc
    return np, pd, stats


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _iter_attempt_paths(results_root: Path, experiment: str, runs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for run_id in runs:
        run_dir = results_root / f"arm={experiment}" / f"run={run_id}"
        paths.extend(sorted(run_dir.glob("model=*.jsonl")))
    return paths


def _validate_schema(records: list[dict[str, Any]]) -> None:
    for record in records:
        if record.get("schema_version") != "eval.v1":
            raise ValueError(f"Unsupported schema_version: {record.get('schema_version')}")


def _classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    import numpy as np
    import pandas as pd

    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return {
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
            "f1_macro": float("nan"),
            "precision_micro": float("nan"),
            "recall_micro": float("nan"),
            "f1_micro": float("nan"),
        }
    table = pd.crosstab(
        pd.Categorical(y_true, categories=labels),
        pd.Categorical(y_pred, categories=labels),
    )
    table = table.reindex(index=labels, columns=labels, fill_value=0)
    cm = table.to_numpy()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    macro_precision = float(np.mean(precision)) if precision.size else float("nan")
    macro_recall = float(np.mean(recall)) if recall.size else float("nan")
    macro_f1 = float(np.mean(f1)) if f1.size else float("nan")
    total_tp = float(tp.sum())
    total = float(cm.sum()) or 1.0
    micro_precision = total_tp / total
    micro_recall = total_tp / total
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0
    return {
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
        "f1_micro": micro_f1,
    }


def _rank_biserial_from_wilcoxon(statistic: float, n: int) -> float:
    if n == 0:
        return float("nan")
    total_rank = n * (n + 1) / 2
    w_pos = statistic
    w_neg = total_rank - w_pos
    return float((w_pos - w_neg) / total_rank)


def _holm_correction(pvals: list[float]) -> list[float]:
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    m = len(pvals)
    adjusted = [0.0] * m
    for rank, (idx, pval) in enumerate(indexed, start=1):
        adjusted[idx] = min(pval * (m - rank + 1), 1.0)
    return adjusted


def _bootstrap_ci(values: list[float], n_boot: int, seed: int = 7) -> tuple[float, float, float]:
    import numpy as np

    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    means = samples.mean(axis=1)
    lower = float(np.percentile(means, 2.5))
    upper = float(np.percentile(means, 97.5))
    return float(arr.mean()), lower, upper


def _bootstrap_macro_f1(df, n_boot: int, seed: int = 7) -> tuple[float, float, float]:
    import numpy as np

    df = df.dropna(subset=["gold.label", "scoring.pred_label"])
    if df.empty:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    f1s = []
    for _ in range(n_boot):
        sample = df.iloc[rng.choice(idx, size=len(idx), replace=True)]
        metrics = _classification_metrics(
            sample["gold.label"].astype(str).tolist(),
            sample["scoring.pred_label"].astype(str).tolist(),
        )
        f1s.append(metrics["f1_macro"])
    f1s_arr = np.array(f1s, dtype=float)
    return float(f1s_arr.mean()), float(np.percentile(f1s_arr, 2.5)), float(np.percentile(f1s_arr, 97.5))


def _plot_accuracy_ci(df, output_path: Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    df_sorted = df.sort_values(metric)
    ax.errorbar(
        df_sorted["model_key"],
        df_sorted[metric],
        yerr=[
            df_sorted[metric] - df_sorted[f"{metric}_ci_lower"],
            df_sorted[f"{metric}_ci_upper"] - df_sorted[metric],
        ],
        fmt="o",
        capsize=4,
    )
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Model")
    ax.set_xticklabels(df_sorted["model_key"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_latency_ecdf(df, output_path: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_key, group in df.dropna(subset=["latency_ms"]).groupby("model_key"):
        values = np.sort(group["latency_ms"].astype(float).to_numpy())
        if values.size == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", label=model_key)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("ECDF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_confusion_matrix(df, output_path: Path, model_key: str) -> None:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = df[(df["model_key"] == model_key)].dropna(subset=["gold.label", "scoring.pred_label"])
    labels = sorted(set(df["gold.label"].astype(str)) | set(df["scoring.pred_label"].astype(str)))
    if not labels:
        return
    table = pd.crosstab(
        pd.Categorical(df["gold.label"].astype(str), categories=labels),
        pd.Categorical(df["scoring.pred_label"].astype(str), categories=labels),
    )
    table = table.reindex(index=labels, columns=labels, fill_value=0)
    cm = table.to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(f"Confusion Matrix: {model_key}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_cd_diagram(rank_df, output_path: Path, n_units: int, alpha: float = 0.05) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import studentized_range

    k = len(rank_df)
    if k < 2:
        return
    q_alpha = studentized_range.ppf(1 - alpha, k, math.inf)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n_units))

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.set_title("Critical Difference Diagram")
    ax.set_xlim(1, k)
    ax.set_ylim(0, 1)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Average Rank (lower is better)")
    for _, row in rank_df.iterrows():
        ax.plot(row["avg_rank"], 0.5, "o", color="black")
        ax.text(row["avg_rank"], 0.6, row["model_key"], ha="center", fontsize=8)
    ax.hlines(0.2, 1, 1 + cd, colors="black", linewidth=2)
    ax.text(1 + cd / 2, 0.1, f"CD={cd:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation tables and figures.")
    parser.add_argument("--experiment", required=True, help="Experiment ID (e.g., A1)")
    parser.add_argument("--runs", required=True, help="Comma-separated run IDs")
    parser.add_argument("--results-root", default="results", help="Root directory for attempt logs")
    parser.add_argument("--output-dir", default=None, help="Output directory for reports/figures")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap samples")
    parser.add_argument("--topk", type=int, default=3, help="Top-K models for confusion plots")
    args = parser.parse_args()

    np, pd, stats = _require_packages()

    experiment = args.experiment
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir or f"reports/{experiment.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = _iter_attempt_paths(results_root, experiment, runs)
    if not paths:
        raise SystemExit(f"No attempt logs found under {results_root} for {experiment} and runs {runs}")

    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_jsonl(path))
    _validate_schema(records)

    df = pd.json_normalize(records)
    if "scoring.is_correct" not in df:
        df["scoring.is_correct"] = np.nan
    if "scoring.is_partial" not in df:
        df["scoring.is_partial"] = np.nan
    if "latency_ms" not in df:
        df["latency_ms"] = np.nan
    if "cost_usd" not in df:
        df["cost_usd"] = np.nan
    df["scoring.is_correct"] = pd.to_numeric(df["scoring.is_correct"], errors="coerce")
    df["scoring.is_partial"] = pd.to_numeric(df["scoring.is_partial"], errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce")

    if "errors.provider_error" not in df:
        df["errors.provider_error"] = np.nan
    if "errors.timeout" not in df:
        df["errors.timeout"] = False
    if "errors.parse_error" not in df:
        df["errors.parse_error"] = False
    df["has_error"] = (
        df["errors.provider_error"].notna()
        | df["errors.timeout"].fillna(False)
        | df["errors.parse_error"].fillna(False)
    )

    metric_rows = []
    for (run_id, model_key), group in df.groupby(["run_id", "model_key"]):
        model_id = group["model_id"].iloc[0]
        model_class = group["model_class"].iloc[0] if "model_class" in group else None
        accuracy = group["scoring.is_correct"].mean()
        partial = group["scoring.is_partial"].mean()
        coverage = 1.0 - group["has_error"].mean()
        latency_median = group["latency_ms"].median()
        latency_p90 = group["latency_ms"].quantile(0.9)
        cost_total = group["cost_usd"].sum()
        cost_per_q = cost_total / len(group) if len(group) else float("nan")
        if "gold.label" in group and "scoring.pred_label" in group:
            filtered = group.dropna(subset=["gold.label", "scoring.pred_label"])
            classification = _classification_metrics(
                filtered["gold.label"].astype(str).tolist(),
                filtered["scoring.pred_label"].astype(str).tolist(),
            )
        else:
            classification = _classification_metrics([], [])
        metric_rows.append(
            {
                "experiment_id": experiment,
                "run_id": run_id,
                "model_key": model_key,
                "model_id": model_id,
                "model_class": model_class,
                "n_items": len(group),
                "accuracy": accuracy,
                "partial_accuracy": partial,
                "coverage": coverage,
                "latency_median_ms": latency_median,
                "latency_p90_ms": latency_p90,
                "cost_total_usd": cost_total,
                "cost_per_question_usd": cost_per_q,
                **classification,
            }
        )

    by_run_df = pd.DataFrame(metric_rows)
    by_run_path = output_dir / f"{experiment.lower()}_model_metrics_by_run.csv"
    by_run_df.to_csv(by_run_path, index=False)

    agg_rows = []
    for model_key, group in df.groupby("model_key"):
        accuracy_values = group["scoring.is_correct"].dropna().astype(float).tolist()
        partial_values = group["scoring.is_partial"].dropna().astype(float).tolist()
        acc_mean, acc_low, acc_high = _bootstrap_ci(accuracy_values, args.bootstrap)
        part_mean, part_low, part_high = _bootstrap_ci(partial_values, args.bootstrap)
        f1_mean, f1_low, f1_high = _bootstrap_macro_f1(group, args.bootstrap)
        agg_rows.append(
            {
                "experiment_id": experiment,
                "model_key": model_key,
                "model_id": group["model_id"].iloc[0],
                "model_class": group["model_class"].iloc[0] if "model_class" in group else None,
                "accuracy": acc_mean,
                "accuracy_ci_lower": acc_low,
                "accuracy_ci_upper": acc_high,
                "partial_accuracy": part_mean,
                "partial_accuracy_ci_lower": part_low,
                "partial_accuracy_ci_upper": part_high,
                "f1_macro": f1_mean,
                "f1_macro_ci_lower": f1_low,
                "f1_macro_ci_upper": f1_high,
            }
        )

    agg_df = pd.DataFrame(agg_rows)
    agg_path = output_dir / f"{experiment.lower()}_model_metrics_agg_ci.csv"
    agg_df.to_csv(agg_path, index=False)

    # Friedman test (paired per run/question)
    pivot = df.pivot_table(
        index=["run_id", "question_id"],
        columns="model_key",
        values="scoring.is_correct",
    ).dropna()
    friedman_path = output_dir / f"{experiment.lower()}_friedman_global.csv"
    if not pivot.empty:
        stat, pval = stats.friedmanchisquare(*[pivot[col].values for col in pivot.columns])
        friedman_df = pd.DataFrame(
            [{
                "metric": "is_correct",
                "statistic": stat,
                "p_value": pval,
                "n_units": len(pivot),
                "n_models": len(pivot.columns),
            }]
        )
        friedman_df.to_csv(friedman_path, index=False)
    else:
        pd.DataFrame([]).to_csv(friedman_path, index=False)

    # Post-hoc Wilcoxon with Holm correction
    posthoc_rows = []
    if not pivot.empty:
        model_keys = list(pivot.columns)
        pvals = []
        pairs = []
        stats_vals = []
        ns = []
        for i, mk_a in enumerate(model_keys):
            for mk_b in model_keys[i + 1:]:
                paired = pivot[[mk_a, mk_b]].dropna()
                diffs = paired[mk_a] - paired[mk_b]
                diffs = diffs[diffs != 0]
                if diffs.empty:
                    statistic = float("nan")
                    pval = 1.0
                    n = 0
                else:
                    statistic, pval = stats.wilcoxon(paired[mk_a], paired[mk_b])
                    n = len(diffs)
                pvals.append(pval)
                pairs.append((mk_a, mk_b))
                stats_vals.append(statistic)
                ns.append(n)
        corrected = _holm_correction(pvals)
        for (mk_a, mk_b), pval, pval_corr, stat_val, n in zip(pairs, pvals, corrected, stats_vals, ns):
            posthoc_rows.append(
                {
                    "metric": "is_correct",
                    "model_a": mk_a,
                    "model_b": mk_b,
                    "p_value": pval,
                    "p_value_holm": pval_corr,
                    "effect_size_r": _rank_biserial_from_wilcoxon(stat_val, n),
                    "n_units": n,
                }
            )
    posthoc_df = pd.DataFrame(posthoc_rows)
    posthoc_path = output_dir / f"{experiment.lower()}_posthoc_pairwise_holm.csv"
    posthoc_df.to_csv(posthoc_path, index=False)

    # Large vs small comparison
    group_path = output_dir / f"{experiment.lower()}_group_large_vs_small.csv"
    group_rows = []
    if "model_class" in df:
        grouped = df.pivot_table(
            index=["run_id", "question_id"],
            columns="model_class",
            values="scoring.is_correct",
        )
        if {"large", "small"}.issubset(grouped.columns):
            paired = grouped.dropna(subset=["large", "small"])
            statistic, pval = stats.wilcoxon(paired["large"], paired["small"])
            diffs = paired["large"] - paired["small"]
            diffs = diffs[diffs != 0]
            n = len(diffs)
            group_rows.append(
                {
                    "metric": "is_correct",
                    "group_a": "large",
                    "group_b": "small",
                    "p_value": pval,
                    "effect_size_r": _rank_biserial_from_wilcoxon(statistic, n),
                    "n_units": n,
                    "mean_large": paired["large"].mean(),
                    "mean_small": paired["small"].mean(),
                }
            )
    pd.DataFrame(group_rows).to_csv(group_path, index=False)

    # Latency + cost summaries
    latency_summary_path = output_dir / f"{experiment.lower()}_latency_summary.csv"
    df.groupby("model_key")["latency_ms"].agg(
        latency_median_ms="median",
        latency_p90_ms=lambda x: x.quantile(0.9),
    ).reset_index().to_csv(latency_summary_path, index=False)

    cost_path = output_dir / f"{experiment.lower()}_cost_summary.csv"
    df.groupby("model_key")["cost_usd"].agg(
        cost_total_usd="sum",
        cost_per_question_usd="mean",
    ).reset_index().to_csv(cost_path, index=False)

    error_path = output_dir / f"{experiment.lower()}_error_coverage.csv"
    df.groupby("model_key")["has_error"].agg(
        error_rate="mean",
        coverage=lambda x: 1.0 - x.mean(),
    ).reset_index().to_csv(error_path, index=False)

    # Figures
    acc_fig = figures_dir / f"{experiment.lower()}_accuracy_ci.png"
    _plot_accuracy_ci(agg_df, acc_fig, "accuracy")
    f1_fig = figures_dir / f"{experiment.lower()}_f1_ci.png"
    if not agg_df["f1_macro"].isna().all():
        _plot_accuracy_ci(agg_df.dropna(subset=["f1_macro"]), f1_fig, "f1_macro")
    latency_fig = figures_dir / f"{experiment.lower()}_latency_ecdf.png"
    _plot_latency_ecdf(df, latency_fig)

    # Confusion matrices for top models
    top_models = (
        agg_df.sort_values("accuracy", ascending=False)["model_key"].head(args.topk).tolist()
    )
    for model_key in top_models:
        _plot_confusion_matrix(
            df,
            figures_dir / f"{experiment.lower()}_confusion_{model_key}.png",
            model_key,
        )

    # Critical difference diagram
    if not pivot.empty:
        rank_rows = []
        ranks = []
        for _, row in pivot.iterrows():
            scores = row.values
            rank = stats.rankdata(-scores, method="average")
            ranks.append(rank)
        ranks_arr = np.vstack(ranks)
        avg_ranks = ranks_arr.mean(axis=0)
        rank_df = pd.DataFrame(
            {"model_key": list(pivot.columns), "avg_rank": avg_ranks}
        ).sort_values("avg_rank")
        _plot_cd_diagram(rank_df, figures_dir / f"{experiment.lower()}_cd_diagram.png", n_units=len(pivot))

    # Summary markdown
    summary_path = output_dir / "summary.md"
    summary_lines = [
        f"# {experiment} summary",
        "",
        "## Setting",
        "- no-RAG, oneshot" if experiment.upper() == "A1" else f"- arm: {experiment}",
        f"- runs: {', '.join(runs)}",
        f"- total items: {df['question_id'].nunique()}",
        "",
        "## Metrics",
        f"- models: {df['model_key'].nunique()}",
        f"- accuracy (mean): {agg_df['accuracy'].mean():.3f}",
        f"- partial accuracy (mean): {agg_df['partial_accuracy'].mean():.3f}",
        "",
        "## Statistics",
        "- global test: Friedman over is_correct",
        "- post-hoc: pairwise Wilcoxon signed-rank with Holm correction",
        "- large vs small: paired Wilcoxon on per-question group means",
        "",
        "## Outputs",
        f"- tables: {by_run_path}, {agg_path}, {friedman_path}, {posthoc_path}, {group_path}",
        f"- figures: {acc_fig}, {latency_fig}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
