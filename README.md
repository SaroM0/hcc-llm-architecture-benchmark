# oncology-agentic-rag

Core library and experiment scaffold for agentic RAG in oncology.

## Quickstart

- Copy `.env.example` to `.env` and fill required variables.
- Configure defaults in `configs/default.yaml` and provider settings in `configs/providers/`.
- Place only the canonical HCC guideline documents in `data/raw/markdown/hcc/`.

## CLI

- Ingest: `python -m oncology_rag.cli.ingest --config configs/default.yaml`
- Eval: `python -m oncology_rag.cli.eval matrix --dataset data/eval/sct_validated_ground_truth.csv`

## Makefile Commands

Run `make help` to see all available commands. Below is a summary:

### Setup

```bash
make install    # Install package in development mode
make check-env  # Verify environment and dependencies
```

### Smoke tests

```bash
make smoke      # Run all smoke tests (A1-A3)
make smoke-a1   # A1: oneshot
make smoke-a2   # A2: consensus large
make smoke-a3   # A3: consensus small
```

### Evaluation

```bash
make eval-a1      # Baseline, all configured models
make eval-a2      # Large-model consensus RAG
make eval-a3      # Small-model consensus RAG
make eval-all     # A1, A2 and A3
```

### Reports

```bash
make report        # Generate report with SVG charts from latest runs
make report-smoke  # Generate smoke test report
```

### Maintenance

```bash
make clean-runs   # Remove all run artifacts
```

## Markdown ingestion

- Put the five canonical markdown files under `data/raw/markdown/hcc/`.
- Keep filenames and guideline years aligned with `configs/rag/corpus.yaml`.
- Run: `python -m oncology_rag.cli.ingest --config configs/default.yaml --input data/raw/markdown/hcc`
- Optional: add `--dry-run` to see how many chunks would be ingested.

## Project layout

- `src/oncology_rag/`: core library (no side effects on import)
- `configs/`: declarative configuration only
- `data/eval/`: versioned benchmark data
- `data/raw/markdown/hcc/`: local canonical source documents (ignored)
- `data/indexes/`: regenerated local indexes (ignored)
- `runs/`: ignored raw execution artifacts
- `results/`: ignored derived tables and reports
- `docs/`: architecture and evaluation notes
