# Contributing

## Mandatory rules

- No business logic in `scripts/`. Scripts are thin wrappers that call `src/oncology_rag/cli/`.
- No I/O inside agent nodes. Nodes operate only on state and emit events.
- Do not modify `data/raw/`. Treat it as immutable.
- All run-time behavior must be configurable via `configs/`.

## Run artifacts and manifest

- Each run must write a manifest to `runs/experiments/<run_id>/manifest.json`.
- The manifest must include:
  - `run_id`
  - `created_at`
  - `git_commit`
  - `config_resolved` (full resolved config)
  - `models` (provider and model ids)
  - `seeds`
  - `budgets`
  - `corpus_hash`
  - `index_hash`
  - `data_version`

## Naming and hygiene

- `run_id` format: `YYYYMMDD_HHMMSS_<shortlabel>`.
- Keep `runs/` reproducible and disposable. Never import from `runs/` in core code.
- Keep `configs/` as the single source of truth.
