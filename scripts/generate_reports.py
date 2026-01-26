#!/usr/bin/env python3
"""Generate comprehensive reports for A1, A3, and A4 experiments."""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List


def load_predictions(pred_file: Path) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(pred_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def generate_a1_report():
    """Generate comprehensive A1 report."""
    print("Generating A1 Report...")

    # Load A1 summary
    a1_summary = json.load(open("runs/matrix/summary_20260124_193025.json"))

    # Sort by accuracy
    results = sorted(a1_summary["results"], key=lambda x: x["accuracy"], reverse=True)

    # CSV Report
    csv_path = Path("reports/csv/a1_all_models.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rank', 'model_key', 'model_id', 'model_class', 'accuracy',
            'partial_accuracy', 'latency_seconds', 'latency_minutes'
        ])
        writer.writeheader()
        for i, result in enumerate(results, 1):
            writer.writerow({
                'rank': i,
                'model_key': result['model_key'],
                'model_id': result['model_id'],
                'model_class': result['model_class'],
                'accuracy': f"{result['accuracy']:.4f}",
                'partial_accuracy': f"{result['partial_accuracy']:.4f}",
                'latency_seconds': f"{result['latency_seconds']:.2f}",
                'latency_minutes': f"{result['latency_seconds']/60:.2f}",
            })

    # JSON Report
    json_report = {
        "experiment": "A1",
        "description": "Baseline oneshot evaluation - no RAG",
        "created_at": a1_summary["created_at"],
        "total_models": len(results),
        "best_large_model": {
            "model_key": "gpt52",
            "model_id": "openai/gpt-5.2",
            "accuracy": 0.75,
            "rank": 1
        },
        "best_small_model": {
            "model_key": "qwen3_vl_30b",
            "model_id": "qwen/qwen3-vl-30b-a3b-thinking",
            "accuracy": 0.7273,
            "rank": 2
        },
        "all_results": results,
        "statistics": {
            "mean_accuracy": a1_summary["by_arm"]["A1"]["mean_accuracy"],
            "large_models_mean": a1_summary["by_model_class"]["large"]["mean_accuracy"],
            "small_models_mean": a1_summary["by_model_class"]["small"]["mean_accuracy"],
        }
    }

    json_path = Path("reports/json/a1_full_report.json")
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)

    print(f"✓ A1 CSV: {csv_path}")
    print(f"✓ A1 JSON: {json_path}")


def generate_a3_report():
    """Generate comprehensive A3 report."""
    print("\nGenerating A3 Report...")

    # Load A3 data
    a3_summary = json.load(open("runs/matrix/summary_20260125_011535.json"))
    a3_predictions = load_predictions(Path("runs/matrix/20260125_002746_A3_gpt52/predictions.jsonl"))

    # Calculate detailed metrics
    total_tokens = sum(p.get("debug", {}).get("total_tokens", 0) for p in a3_predictions)
    total_cost = sum(p.get("debug", {}).get("total_cost_usd", 0) for p in a3_predictions)
    total_llm_calls = sum(p.get("debug", {}).get("llm_calls_count", 0) for p in a3_predictions)
    correct = sum(1 for p in a3_predictions if p.get("is_correct"))

    # Per-prediction CSV
    csv_path = Path("reports/csv/a3_predictions.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'question_id', 'expected_answer', 'predicted_answer', 'is_correct',
            'consensus_reached', 'total_rounds', 'final_confidence',
            'total_tokens', 'total_cost_usd', 'llm_calls_count'
        ])
        writer.writeheader()
        for pred in a3_predictions:
            debug = pred.get("debug", {})
            writer.writerow({
                'question_id': pred['question_id'],
                'expected_answer': pred.get('expected_answer', ''),
                'predicted_answer': pred.get('predicted_answer', ''),
                'is_correct': pred.get('is_correct', False),
                'consensus_reached': debug.get('consensus_reached', False),
                'total_rounds': debug.get('total_rounds', 0),
                'final_confidence': debug.get('final_confidence', 0),
                'total_tokens': debug.get('total_tokens', 0),
                'total_cost_usd': debug.get('total_cost_usd', 0),
                'llm_calls_count': debug.get('llm_calls_count', 0),
            })

    # JSON Report
    json_report = {
        "experiment": "A3",
        "description": "Multi-agent consensus RAG with large model (gpt52)",
        "model": {
            "model_key": "gpt52",
            "model_id": "openai/gpt-5.2",
            "model_class": "best_large"
        },
        "architecture": {
            "specialists": [
                {"role": "hepatologist", "specialty": "Hepatology"},
                {"role": "oncologist", "specialty": "Oncology"},
                {"role": "radiologist", "specialty": "Radiology"}
            ],
            "supervisor": {"role": "supervisor", "function": "Consensus coordination"},
            "rag": {
                "top_k": 5,
                "vector_store": "ChromaDB",
                "embedding_model": "text-embedding-3-large"
            }
        },
        "results": {
            "total_predictions": len(a3_predictions),
            "correct_predictions": correct,
            "accuracy": a3_summary["results"][0]["accuracy"],
            "partial_accuracy": a3_summary["results"][0]["partial_accuracy"],
            "latency_seconds": a3_summary["results"][0]["latency_seconds"],
            "latency_minutes": a3_summary["results"][0]["latency_seconds"] / 60,
        },
        "costs": {
            "total_tokens": total_tokens,
            "avg_tokens_per_prediction": total_tokens / len(a3_predictions),
            "total_cost_usd": total_cost,
            "avg_cost_per_prediction": total_cost / len(a3_predictions),
        },
        "llm_calls": {
            "total": total_llm_calls,
            "avg_per_prediction": total_llm_calls / len(a3_predictions),
        },
        "consensus_metrics": {
            "avg_rounds": sum(p.get("debug", {}).get("total_rounds", 0) for p in a3_predictions) / len(a3_predictions),
            "consensus_rate": sum(1 for p in a3_predictions if p.get("debug", {}).get("consensus_reached", False)) / len(a3_predictions),
            "avg_confidence": sum(p.get("debug", {}).get("final_confidence", 0) for p in a3_predictions) / len(a3_predictions),
        }
    }

    json_path = Path("reports/json/a3_full_report.json")
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)

    print(f"✓ A3 CSV: {csv_path}")
    print(f"✓ A3 JSON: {json_path}")


def generate_a4_report():
    """Generate comprehensive A4 report."""
    print("\nGenerating A4 Report...")

    # Load A4 data
    a4_summary = json.load(open("runs/matrix/summary_20260125_150845.json"))
    a4_predictions = load_predictions(Path("runs/matrix/20260125_083639_A4_qwen3_vl_30b/predictions.jsonl"))

    # Calculate detailed metrics
    total_tokens = sum(p.get("debug", {}).get("total_tokens", 0) for p in a4_predictions)
    total_cost = sum(p.get("debug", {}).get("total_cost_usd", 0) for p in a4_predictions)
    total_llm_calls = sum(p.get("debug", {}).get("llm_calls_count", 0) for p in a4_predictions)
    correct = sum(1 for p in a4_predictions if p.get("is_correct"))

    # Per-prediction CSV
    csv_path = Path("reports/csv/a4_predictions.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'question_id', 'expected_answer', 'predicted_answer', 'is_correct',
            'consensus_reached', 'total_rounds', 'final_confidence',
            'total_tokens', 'total_cost_usd', 'llm_calls_count'
        ])
        writer.writeheader()
        for pred in a4_predictions:
            debug = pred.get("debug", {})
            writer.writerow({
                'question_id': pred['question_id'],
                'expected_answer': pred.get('expected_answer', ''),
                'predicted_answer': pred.get('predicted_answer', ''),
                'is_correct': pred.get('is_correct', False),
                'consensus_reached': debug.get('consensus_reached', False),
                'total_rounds': debug.get('total_rounds', 0),
                'final_confidence': debug.get('final_confidence', 0),
                'total_tokens': debug.get('total_tokens', 0),
                'total_cost_usd': debug.get('total_cost_usd', 0),
                'llm_calls_count': debug.get('llm_calls_count', 0),
            })

    # JSON Report
    json_report = {
        "experiment": "A4",
        "description": "Multi-agent consensus RAG with small model (qwen3_vl_30b)",
        "model": {
            "model_key": "qwen3_vl_30b",
            "model_id": "qwen/qwen3-vl-30b-a3b-thinking",
            "model_class": "best_small"
        },
        "architecture": {
            "specialists": [
                {"role": "hepatologist", "specialty": "Hepatology"},
                {"role": "oncologist", "specialty": "Oncology"},
                {"role": "radiologist", "specialty": "Radiology"}
            ],
            "supervisor": {"role": "supervisor", "function": "Consensus coordination"},
            "rag": {
                "top_k": 5,
                "vector_store": "ChromaDB",
                "embedding_model": "text-embedding-3-large"
            }
        },
        "results": {
            "total_predictions": len(a4_predictions),
            "correct_predictions": correct,
            "accuracy": a4_summary["results"][0]["accuracy"],
            "partial_accuracy": a4_summary["results"][0]["partial_accuracy"],
            "latency_seconds": a4_summary["results"][0]["latency_seconds"],
            "latency_minutes": a4_summary["results"][0]["latency_seconds"] / 60,
        },
        "costs": {
            "total_tokens": total_tokens,
            "avg_tokens_per_prediction": total_tokens / len(a4_predictions),
            "total_cost_usd": total_cost,
            "avg_cost_per_prediction": total_cost / len(a4_predictions),
        },
        "llm_calls": {
            "total": total_llm_calls,
            "avg_per_prediction": total_llm_calls / len(a4_predictions),
        },
        "consensus_metrics": {
            "avg_rounds": sum(p.get("debug", {}).get("total_rounds", 0) for p in a4_predictions) / len(a4_predictions),
            "consensus_rate": sum(1 for p in a4_predictions if p.get("debug", {}).get("consensus_reached", False)) / len(a4_predictions),
            "avg_confidence": sum(p.get("debug", {}).get("final_confidence", 0) for p in a4_predictions) / len(a4_predictions),
        }
    }

    json_path = Path("reports/json/a4_full_report.json")
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)

    print(f"✓ A4 CSV: {csv_path}")
    print(f"✓ A4 JSON: {json_path}")


def generate_comparison_report():
    """Generate comparison report across all experiments."""
    print("\nGenerating Comparison Report...")

    # Load all summaries
    a1_summary = json.load(open("runs/matrix/summary_20260124_193025.json"))
    a3_summary = json.load(open("runs/matrix/summary_20260125_011535.json"))
    a4_summary = json.load(open("runs/matrix/summary_20260125_150845.json"))

    # Load predictions
    a3_predictions = load_predictions(Path("runs/matrix/20260125_002746_A3_gpt52/predictions.jsonl"))
    a4_predictions = load_predictions(Path("runs/matrix/20260125_083639_A4_qwen3_vl_30b/predictions.jsonl"))

    # Calculate costs
    a3_cost = sum(p.get("debug", {}).get("total_cost_usd", 0) for p in a3_predictions)
    a4_cost = sum(p.get("debug", {}).get("total_cost_usd", 0) for p in a4_predictions)

    # CSV Comparison
    csv_path = Path("reports/csv/comparison_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'experiment', 'model', 'architecture', 'accuracy', 'partial_accuracy',
            'total_cost_usd', 'avg_cost_per_pred', 'latency_minutes', 'avg_latency_per_pred_sec'
        ])
        writer.writeheader()

        # A1 best large
        a1_best_large = next(r for r in a1_summary["results"] if r["model_key"] == "gpt52")
        writer.writerow({
            'experiment': 'A1',
            'model': 'gpt52 (large)',
            'architecture': 'Oneshot (no RAG)',
            'accuracy': f"{a1_best_large['accuracy']:.4f}",
            'partial_accuracy': f"{a1_best_large['partial_accuracy']:.4f}",
            'total_cost_usd': 'N/A',
            'avg_cost_per_pred': 'N/A',
            'latency_minutes': f"{a1_best_large['latency_seconds']/60:.2f}",
            'avg_latency_per_pred_sec': f"{a1_best_large['latency_seconds']/88:.2f}",
        })

        # A1 best small
        a1_best_small = next(r for r in a1_summary["results"] if r["model_key"] == "qwen3_vl_30b")
        writer.writerow({
            'experiment': 'A1',
            'model': 'qwen3_vl_30b (small)',
            'architecture': 'Oneshot (no RAG)',
            'accuracy': f"{a1_best_small['accuracy']:.4f}",
            'partial_accuracy': f"{a1_best_small['partial_accuracy']:.4f}",
            'total_cost_usd': 'N/A',
            'avg_cost_per_pred': 'N/A',
            'latency_minutes': f"{a1_best_small['latency_seconds']/60:.2f}",
            'avg_latency_per_pred_sec': f"{a1_best_small['latency_seconds']/88:.2f}",
        })

        # A3
        writer.writerow({
            'experiment': 'A3',
            'model': 'gpt52 (large)',
            'architecture': 'Multi-agent consensus + RAG',
            'accuracy': f"{a3_summary['results'][0]['accuracy']:.4f}",
            'partial_accuracy': f"{a3_summary['results'][0]['partial_accuracy']:.4f}",
            'total_cost_usd': f"{a3_cost:.2f}",
            'avg_cost_per_pred': f"{a3_cost/88:.4f}",
            'latency_minutes': f"{a3_summary['results'][0]['latency_seconds']/60:.2f}",
            'avg_latency_per_pred_sec': f"{a3_summary['results'][0]['latency_seconds']/88:.2f}",
        })

        # A4
        writer.writerow({
            'experiment': 'A4',
            'model': 'qwen3_vl_30b (small)',
            'architecture': 'Multi-agent consensus + RAG',
            'accuracy': f"{a4_summary['results'][0]['accuracy']:.4f}",
            'partial_accuracy': f"{a4_summary['results'][0]['partial_accuracy']:.4f}",
            'total_cost_usd': f"{a4_cost:.2f}",
            'avg_cost_per_pred': f"{a4_cost/88:.4f}",
            'latency_minutes': f"{a4_summary['results'][0]['latency_seconds']/60:.2f}",
            'avg_latency_per_pred_sec': f"{a4_summary['results'][0]['latency_seconds']/88:.2f}",
        })

    # JSON Comparison
    json_report = {
        "report_type": "Comprehensive Comparison",
        "generated_at": datetime.now().isoformat(),
        "experiments": {
            "A1": {
                "description": "Baseline oneshot evaluation",
                "total_models": 10,
                "best_large": {
                    "model": "gpt52",
                    "accuracy": a1_best_large['accuracy'],
                    "partial_accuracy": a1_best_large['partial_accuracy'],
                },
                "best_small": {
                    "model": "qwen3_vl_30b",
                    "accuracy": a1_best_small['accuracy'],
                    "partial_accuracy": a1_best_small['partial_accuracy'],
                }
            },
            "A3": {
                "description": "Multi-agent consensus RAG (large model)",
                "model": "gpt52",
                "accuracy": a3_summary['results'][0]['accuracy'],
                "partial_accuracy": a3_summary['results'][0]['partial_accuracy'],
                "total_cost_usd": a3_cost,
                "improvement_over_a1": (a3_summary['results'][0]['accuracy'] - a1_best_large['accuracy']) / a1_best_large['accuracy'] * 100,
            },
            "A4": {
                "description": "Multi-agent consensus RAG (small model)",
                "model": "qwen3_vl_30b",
                "accuracy": a4_summary['results'][0]['accuracy'],
                "partial_accuracy": a4_summary['results'][0]['partial_accuracy'],
                "total_cost_usd": a4_cost,
                "improvement_over_a1": (a4_summary['results'][0]['accuracy'] - a1_best_small['accuracy']) / a1_best_small['accuracy'] * 100,
            }
        },
        "key_findings": {
            "best_overall_accuracy": "A3 (gpt52 with consensus) - 77.27%",
            "best_cost_efficiency": "A4 (qwen3_vl_30b with consensus) - $1.86 total",
            "consensus_benefit_large": f"+{((a3_summary['results'][0]['accuracy'] - a1_best_large['accuracy']) / a1_best_large['accuracy'] * 100):.2f}%",
            "consensus_benefit_small": f"+{((a4_summary['results'][0]['accuracy'] - a1_best_small['accuracy']) / a1_best_small['accuracy'] * 100):.2f}%",
        }
    }

    json_path = Path("reports/json/comprehensive_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)

    print(f"✓ Comparison CSV: {csv_path}")
    print(f"✓ Comparison JSON: {json_path}")


if __name__ == "__main__":
    print("="*80)
    print("HCC LLM Architecture Benchmark - Report Generation")
    print("="*80)

    generate_a1_report()
    generate_a3_report()
    generate_a4_report()
    generate_comparison_report()

    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  reports/csv/")
    print("    - a1_all_models.csv")
    print("    - a3_predictions.csv")
    print("    - a4_predictions.csv")
    print("    - comparison_summary.csv")
    print("  reports/json/")
    print("    - a1_full_report.json")
    print("    - a3_full_report.json")
    print("    - a4_full_report.json")
    print("    - comprehensive_comparison.json")
