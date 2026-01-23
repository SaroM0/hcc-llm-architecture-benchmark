# oncology-agentic-rag

Core library and experiment scaffold for agentic RAG in oncology.

## Quickstart

- Copy `.env.example` to `.env` and fill required variables.
- Configure defaults in `configs/default.yaml` and provider settings in `configs/providers/`.
- Place source documents in `data/raw/`.

## CLI

- Ingest: `python -m oncology_rag.cli.ingest --config configs/default.yaml`
- Chat: `python -m oncology_rag.cli.chat --config configs/default.yaml`
- Eval: `python -m oncology_rag.cli.eval --config configs/default.yaml`

## Makefile Commands

Run `make help` to see all available commands. Below is a summary:

### Setup

```bash
make install    # Install package in development mode
make check-env  # Verify environment and dependencies
```

### Smoke Tests (2 items, quick validation)

```bash
make smoke      # Run all smoke tests (A1-A4)
make smoke-a1   # A1: oneshot
make smoke-a2   # A2: oneshot + RAG
make smoke-a3   # A3: consensus
make smoke-a4   # A4: consensus + RAG
```

### Evaluation Tests (small models only)

```bash
make test-mini    # 5 items, all arms, small models
make test-small   # 50 items, all arms, small models
make test-medium  # 100 items, all arms, small models
```

### Full Evaluation

```bash
make test-full    # 200 items, all arms, all models (large + small)
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

- Put markdown files under `data/raw/markdown/` (or any folder you prefer).
- Run: `python -m oncology_rag.cli.ingest --config configs/default.yaml --input data/raw/markdown`
- Optional: add `--dry-run` to see how many chunks would be ingested.

## Project layout

- `src/oncology_rag/`: core library (no side effects on import)
- `configs/`: declarative configuration only
- `data/`: raw, processed, and indexes
- `runs/`: artifacts from executions
- `docs/`: architecture and evaluation notes
