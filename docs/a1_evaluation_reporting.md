# A1 Evaluation Instrumentation and Reporting

This document describes the attempt-level logging schema and the reporting workflow
used to produce paper-ready metrics for arm A1.

## Attempt Logs (Schema `eval.v1`)

Each evaluated response is stored as a JSONL record under:

`results/arm=A1/run=<run_id>/model=<model_key>.jsonl`

Required fields (shortened):

- `schema_version`: `eval.v1`
- `timestamp_utc`: ISO-8601 timestamp in UTC
- `experiment_id`, `run_id`, `model_key`, `model_id`, `model_class`, `question_id`
- `prompt_hash`: SHA-256 hash of the prompt messages
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `latency_ms`, `cost_usd`
- `raw_answer`
- `gold`: `{label, allowed_labels, rationale}`
- `scoring`: `{is_correct, is_partial, pred_label, confidence, error_type}`
- `errors`: `{provider_error, timeout, parse_error}`

## Running Reports

Generate tables, figures, and a summary markdown file:

```bash
python -m stats.report_tables --experiment A1 --runs A1_r01,A1_r02,A1_r03
```

Outputs are written to `reports/a1/`:

- CSV tables for model metrics, CIs, Friedman/Wilcoxon tests, coverage, latency, cost
- Figures for accuracy CIs, latency ECDFs, and confusion matrices
- `reports/a1/summary.md` with reproducible settings and statistical notes

## Config Snapshots

Each run writes a configuration snapshot to:

`results/run=<run_id>/config_snapshot.yaml`

The snapshot includes the default config, experiment settings, provider config,
retrieval/consensus settings, and run metadata.
