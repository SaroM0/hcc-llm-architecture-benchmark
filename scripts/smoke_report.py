"""Generate a smoke-test report for SCT runs with lightweight SVG charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from oncology_rag.eval.sct.loader import load_sct_as_qa_items
from oncology_rag.eval.sct.metrics import calculate_sct_metrics


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _latest_runs(runs_dir: Path) -> dict[str, Path]:
    run_map: dict[str, tuple[float, Path]] = {}
    for manifest_path in runs_dir.glob("experiments/*/manifest.json"):
        run_dir = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        experiment = manifest.get("experiment", {}) or {}
        exp_id = experiment.get("id") or run_dir.name
        mtime = run_dir.stat().st_mtime
        current = run_map.get(exp_id)
        if current is None or mtime > current[0]:
            run_map[exp_id] = (mtime, run_dir)
    return {key: value[1] for key, value in run_map.items()}


def _build_metrics(
    run_dir: Path,
    expected_map: dict[str, str | None],
) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    predictions = _load_predictions(predictions_path)
    for pred in predictions:
        expected = expected_map.get(pred.get("question_id", ""))
        if expected is not None:
            pred["expected_answer"] = expected
    metrics = calculate_sct_metrics(predictions, expected_key="expected_answer")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = manifest.get("experiment", {}) or {}
    arm_id = experiment.get("arm", "unknown")
    return {
        "run_dir": str(run_dir),
        "arm": arm_id,
        "experiment_id": experiment.get("id", "unknown"),
        "total_items": metrics.total_items,
        "accuracy": metrics.accuracy,
        "partial_accuracy": metrics.partial_accuracy,
        "parse_rate": metrics.parse_rate,
        "mae": metrics.mean_absolute_error,
    }


def _svg_bar_chart(
    rows: list[dict[str, Any]],
    *,
    title: str,
    metrics: list[str],
    colors: list[str],
    width: int = 900,
    height: int = 420,
) -> str:
    margin = 60
    chart_w = width - margin * 2
    chart_h = height - margin * 2
    max_value = 1.0

    arms = [row["arm"] for row in rows]
    group_count = len(arms)
    bars_per_group = len(metrics)
    group_width = chart_w / max(1, group_count)
    bar_width = group_width / (bars_per_group + 1)

    def y_pos(value: float) -> float:
        return margin + chart_h * (1 - value / max_value)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-size="18" '
        f'font-family="Arial, sans-serif">{title}</text>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + chart_h}" '
        f'stroke="#111" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{margin + chart_h}" x2="{margin + chart_w}" '
        f'y2="{margin + chart_h}" stroke="#111" stroke-width="1"/>',
    ]

    for i, row in enumerate(rows):
        x0 = margin + i * group_width
        for j, metric in enumerate(metrics):
            value = float(row.get(metric, 0.0) or 0.0)
            bar_h = chart_h * (value / max_value)
            x = x0 + (j + 0.5) * bar_width
            y = margin + chart_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.8:.1f}" '
                f'height="{bar_h:.1f}" fill="{colors[j]}" />'
            )
        label_x = x0 + group_width / 2
        parts.append(
            f'<text x="{label_x:.1f}" y="{margin + chart_h + 24}" text-anchor="middle" '
            f'font-size="12" font-family="Arial, sans-serif">{row["arm"]}</text>'
        )

    for idx, metric in enumerate(metrics):
        lx = margin + idx * 160
        ly = height - 16
        parts.append(
            f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" '
            f'fill="{colors[idx]}" />'
        )
        parts.append(
            f'<text x="{lx + 18}" y="{ly}" font-size="12" '
            f'font-family="Arial, sans-serif">{metric}</text>'
        )

    for tick in [0.0, 0.5, 1.0]:
        y = y_pos(tick)
        parts.append(
            f'<line x1="{margin - 4}" y1="{y:.1f}" x2="{margin}" y2="{y:.1f}" '
            f'stroke="#111" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" font-family="Arial, sans-serif">{tick:.1f}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate smoke test report charts.")
    parser.add_argument("--runs-dir", default="runs", help="Runs directory.")
    parser.add_argument("--dataset", required=True, help="SCT dataset JSON path.")
    parser.add_argument(
        "--output-dir",
        default="runs/reports",
        help="Output directory for report artifacts.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment IDs to include (default: all latest).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_items = load_sct_as_qa_items(Path(args.dataset))
    expected_map = {item.question_id: item.metadata.get("expected_answer") for item in qa_items}

    latest = _latest_runs(runs_dir)
    selected = args.experiments or sorted(latest.keys())

    rows: list[dict[str, Any]] = []
    for exp_id in selected:
        run_dir = latest.get(exp_id)
        if not run_dir:
            continue
        rows.append(_build_metrics(run_dir, expected_map))

    rows.sort(key=lambda row: row["arm"])
    output_dir.joinpath("smoke_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    csv_lines = [
        "arm,experiment_id,total_items,accuracy,partial_accuracy,parse_rate,mae,run_dir"
    ]
    for row in rows:
        csv_lines.append(
            ",".join(
                [
                    row["arm"],
                    row["experiment_id"],
                    str(row["total_items"]),
                    f"{row['accuracy']:.3f}",
                    f"{row['partial_accuracy']:.3f}",
                    f"{row['parse_rate']:.3f}",
                    "" if row["mae"] is None else f"{row['mae']:.3f}",
                    row["run_dir"],
                ]
            )
        )
    output_dir.joinpath("smoke_metrics.csv").write_text(
        "\n".join(csv_lines), encoding="utf-8"
    )

    svg = _svg_bar_chart(
        rows,
        title="Smoke Test Metrics (Small Models)",
        metrics=["accuracy", "partial_accuracy", "parse_rate"],
        colors=["#3b82f6", "#f59e0b", "#10b981"],
    )
    output_dir.joinpath("smoke_metrics.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
