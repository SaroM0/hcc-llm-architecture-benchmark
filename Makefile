.PHONY: help install check-env format lint test validate-consensus ingest \
	smoke smoke-a1 smoke-a2 smoke-a3 eval-a1 eval-a2 eval-a3 eval-all \
	report sct-agreement sct-validate clean clean-runs

ifneq (,$(wildcard .env))
include .env
export
endif

PYTHON_BIN := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHON := PYTHONPATH=src $(PYTHON_BIN)
DATASET := data/eval/sct_validated_ground_truth.csv
PROVIDER_CONFIG := configs/providers/openrouter.yaml
EMBEDDINGS_CONFIG := configs/rag/embeddings.yaml
CHROMA_CONFIG := configs/rag/chroma.yaml
CORPUS_DIR := data/raw/markdown/hcc
RUNS_DIR := runs
A1_GROUP ?= all
A2_GROUP ?= large
A3_GROUP ?= small
LIMIT_ARG := $(if $(LIMIT),--limit $(LIMIT),)

help:
	@echo "Oncology RAG Evaluation Framework"
	@echo ""
	@echo "Setup:       install check-env ingest"
	@echo "Development: format lint test validate-consensus"
	@echo "Smoke:       smoke smoke-a1 smoke-a2 smoke-a3"
	@echo "Evaluation:  eval-a1 eval-a2 eval-a3 eval-all"
	@echo "Analysis:    report sct-agreement sct-validate"
	@echo "Maintenance: clean clean-runs"
	@echo ""
	@echo "Optional: A1_GROUP=... A2_GROUP=... A3_GROUP=... LIMIT=N"

install:
	$(PYTHON_BIN) -m pip install -e ".[dev,report]"

check-env:
	@$(PYTHON_BIN) --version
	@test -f $(DATASET) && echo "Dataset: OK" || (echo "Dataset: MISSING"; exit 1)
	@test -f $(PROVIDER_CONFIG) && echo "Provider config: OK" || exit 1
	@test -d $(CORPUS_DIR) && echo "Canonical corpus: OK" || echo "Canonical corpus: MISSING"
	@test -d data/indexes/chroma && echo "Chroma index: OK" || echo "Chroma index: MISSING (run make ingest)"
	@test -n "$$OPENROUTER_API_KEY" && echo "OpenRouter key: set" || echo "OpenRouter key: NOT SET"
	@$(PYTHON) -c "from oncology_rag.cli.eval import main; print('Imports: OK')"

format:
	$(PYTHON) -m ruff format src scripts

lint:
	$(PYTHON) -m ruff check src scripts

test:
	$(PYTHON) -m compileall -q src scripts
	$(PYTHON) scripts/validate_consensus.py
	$(PYTHON) -m oncology_rag.cli.eval --help >/dev/null
	$(PYTHON) -m oncology_rag.cli.ingest --help >/dev/null

validate-consensus:
	$(PYTHON) scripts/validate_consensus.py

ingest:
	@test -d $(CORPUS_DIR) || (echo "Missing $(CORPUS_DIR)"; exit 1)
	$(PYTHON) -m oncology_rag.cli.ingest \
		--config configs/default.yaml \
		--input $(CORPUS_DIR)

smoke: smoke-a1 smoke-a2 smoke-a3

smoke-a1:
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET) --provider-config $(PROVIDER_CONFIG) \
		--runs-dir $(RUNS_DIR) --arms A1 --model-groups small --limit 2

smoke-a2:
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET) --provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) --chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) --arms A2 --model-groups large --limit 2

smoke-a3:
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET) --provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) --chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) --arms A3 --model-groups small --limit 2

eval-a1:
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET) --provider-config $(PROVIDER_CONFIG) \
		--runs-dir $(RUNS_DIR) --arms A1 --model-groups $(A1_GROUP) $(LIMIT_ARG)

eval-a2:
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET) --provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) --chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) --arms A2 --model-groups $(A2_GROUP) $(LIMIT_ARG)

eval-a3:
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET) --provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) --chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) --arms A3 --model-groups $(A3_GROUP) $(LIMIT_ARG)

eval-all: eval-a1 eval-a2 eval-a3

report:
	$(PYTHON) scripts/smoke_report.py \
		--runs-dir $(RUNS_DIR) --dataset $(DATASET) --output-dir results/reports

sct-agreement:
	@test -n "$(RESPONSES_CSV)" || (echo "Set RESPONSES_CSV=path/to/responses.csv"; exit 1)
	$(PYTHON) -m oncology_rag.cli.sct_agreement \
		--responses $(RESPONSES_CSV) --dataset $(DATASET) --runs-dir $(RUNS_DIR)

sct-validate:
	@test -n "$(RESPONSES_CSV)" || (echo "Set RESPONSES_CSV=path/to/responses.csv"; exit 1)
	$(PYTHON) -m oncology_rag.cli.sct_validate \
		--responses $(RESPONSES_CSV) \
		--output-csv $(DATASET) \
		--output-json data/eval/sct_validated_ground_truth.json

clean-runs:
	rm -rf runs results logs reports

clean: clean-runs
	rm -rf .pytest_cache .ruff_cache .mypy_cache .cache htmlcov .coverage
	find src scripts tests -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf src/*.egg-info
