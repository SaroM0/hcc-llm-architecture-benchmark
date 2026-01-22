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

## Project layout

- `src/oncology_rag/`: core library (no side effects on import)
- `configs/`: declarative configuration only
- `data/`: raw, processed, and indexes
- `runs/`: artifacts from executions
- `docs/`: architecture and evaluation notes
