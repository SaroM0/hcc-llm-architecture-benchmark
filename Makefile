.PHONY: help install check-env format lint test ingest \
	smoke smoke-a1 smoke-a2 smoke-a3 smoke-a4 \
	test-mini test-small test-medium test-full \
	report report-smoke clean-runs

# =============================================================================
# LOAD .env FILE (if exists)
# =============================================================================
ifneq (,$(wildcard .env))
    include .env
    export
endif

# =============================================================================
# CONFIGURATION
# =============================================================================
PYTHON := PYTHONPATH=src python3
DATASET := data/eval/sct_items_hepa_icca.json
DATASET_SMOKE := data/eval/sct_items_hepa_icca_smoke.json
PROVIDER_CONFIG := configs/providers/openrouter.yaml
EMBEDDINGS_CONFIG := configs/rag/embeddings.yaml
CHROMA_CONFIG := configs/rag/chroma.yaml
RUNS_DIR := runs

# Default model for single-model tests (first small model)
DEFAULT_SMALL_MODEL := mistral_7b

# =============================================================================
# HELP
# =============================================================================
help:
	@echo "Oncology RAG Evaluation Framework"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install package in development mode"
	@echo "  check-env     Verify environment and dependencies"
	@echo ""
	@echo "Development:"
	@echo "  format        Format code"
	@echo "  lint          Run linters"
	@echo "  test          Run unit tests"
	@echo "  ingest        Ingest documents into vector store"
	@echo ""
	@echo "Smoke Tests (2 items, quick validation):"
	@echo "  smoke         Run smoke tests for all arms (A1-A4)"
	@echo "  smoke-a1      Smoke test for A1 (oneshot)"
	@echo "  smoke-a2      Smoke test for A2 (oneshot + RAG)"
	@echo "  smoke-a3      Smoke test for A3 (consensus)"
	@echo "  smoke-a4      Smoke test for A4 (consensus + RAG)"
	@echo ""
	@echo "Evaluation Tests (small models only):"
	@echo "  test-mini     Run with 5 items, 1 small model, all arms"
	@echo "  test-small    Run with 50 items, all small models, all arms"
	@echo "  test-medium   Run with 100 items, all small models, all arms"
	@echo ""
	@echo "Full Evaluation:"
	@echo "  test-full     Run full evaluation (200 items, all models, all arms)"
	@echo ""
	@echo "Reports:"
	@echo "  report        Generate reports from latest runs"
	@echo "  report-smoke  Generate smoke test report"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean-runs    Remove all run artifacts"

# =============================================================================
# SETUP
# =============================================================================
install:
	@echo "Installing package in development mode..."
	pip install -e .

check-env:
	@echo "Checking environment..."
	@echo ""
	@echo "Python: $$(python3 --version)"
	@test -n "$$OPENROUTER_API_KEY" && echo "OPENROUTER_API_KEY: set" || echo "OPENROUTER_API_KEY: NOT SET (required for API calls)"
	@test -n "$$OPENROUTER_BASE_URL" && echo "OPENROUTER_BASE_URL: set" || echo "OPENROUTER_BASE_URL: NOT SET (will use default)"
	@echo ""
	@echo "Dataset files:"
	@test -f $(DATASET) && echo "  $(DATASET): OK" || echo "  $(DATASET): MISSING"
	@test -f $(DATASET_SMOKE) && echo "  $(DATASET_SMOKE): OK" || echo "  $(DATASET_SMOKE): MISSING"
	@echo ""
	@echo "Config files:"
	@test -f $(PROVIDER_CONFIG) && echo "  $(PROVIDER_CONFIG): OK" || echo "  $(PROVIDER_CONFIG): MISSING"
	@test -f $(EMBEDDINGS_CONFIG) && echo "  $(EMBEDDINGS_CONFIG): OK" || echo "  $(EMBEDDINGS_CONFIG): MISSING"
	@test -f $(CHROMA_CONFIG) && echo "  $(CHROMA_CONFIG): OK" || echo "  $(CHROMA_CONFIG): MISSING"
	@echo ""
	@echo "Vector store:"
	@test -d data/indexes/chroma && echo "  data/indexes/chroma: OK" || echo "  data/indexes/chroma: NOT FOUND (run 'make ingest' first)"
	@echo ""
	@$(PYTHON) -c "from oncology_rag.cli.eval import main; print('Python imports: OK')" 2>/dev/null || echo "Python imports: FAILED (run 'make install' or check PYTHONPATH)"

# =============================================================================
# DEVELOPMENT
# =============================================================================
format:
	@echo "Formatting code with ruff..."
	$(PYTHON) -m ruff format src/ scripts/

lint:
	@echo "Linting code with ruff..."
	$(PYTHON) -m ruff check src/ scripts/

test:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/ -v

ingest:
	@echo "Ingesting documents into vector store..."
	$(PYTHON) -m oncology_rag.cli.ingest --config configs/default.yaml

# =============================================================================
# SMOKE TESTS (2 items each, quick validation)
# =============================================================================
smoke: smoke-a1 smoke-a2 smoke-a3 smoke-a4
	@echo ""
	@echo "All smoke tests completed."

smoke-a1:
	@echo "Running smoke test for A1 (oneshot)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_SMOKE) \
		--provider-config $(PROVIDER_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A1 \
		--model-groups small \
		--limit 2

smoke-a2:
	@echo "Running smoke test for A2 (oneshot + RAG)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_SMOKE) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A2 \
		--model-groups small \
		--limit 2

smoke-a3:
	@echo "Running smoke test for A3 (consensus)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_SMOKE) \
		--provider-config $(PROVIDER_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A3 \
		--model-groups small \
		--limit 2

smoke-a4:
	@echo "Running smoke test for A4 (consensus + RAG)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_SMOKE) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A4 \
		--model-groups small \
		--limit 2

# =============================================================================
# EVALUATION TESTS (progressive scale, small models)
# =============================================================================
test-mini:
	@echo "Running mini test (5 items, 1 small model, all arms)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A1 A2 A3 A4 \
		--model-groups small \
		--limit 5

test-small:
	@echo "Running small test (50 items, all small models, all arms)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A1 A2 A3 A4 \
		--model-groups small \
		--limit 50

test-medium:
	@echo "Running medium test (100 items, all small models, all arms)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A1 A2 A3 A4 \
		--model-groups small \
		--limit 100

# =============================================================================
# FULL EVALUATION (all items, all models)
# =============================================================================
test-full:
	@echo "Running full evaluation (200 items, all models, all arms)..."
	@echo "This will take a significant amount of time and API credits."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A1 A2 A3 A4 \
		--model-groups large small

# =============================================================================
# REPORTS
# =============================================================================
report:
	@echo "Generating evaluation report..."
	$(PYTHON) scripts/smoke_report.py \
		--runs-dir $(RUNS_DIR) \
		--dataset $(DATASET) \
		--output-dir $(RUNS_DIR)/reports
	@echo "Report generated at $(RUNS_DIR)/reports/"

report-smoke:
	@echo "Generating smoke test report..."
	$(PYTHON) scripts/smoke_report.py \
		--runs-dir $(RUNS_DIR) \
		--dataset $(DATASET_SMOKE) \
		--output-dir $(RUNS_DIR)/reports/smoke
	@echo "Smoke report generated at $(RUNS_DIR)/reports/smoke/"

# =============================================================================
# MAINTENANCE
# =============================================================================
clean-runs:
	@echo "Removing all run artifacts..."
	rm -rf $(RUNS_DIR)/experiments $(RUNS_DIR)/matrix $(RUNS_DIR)/reports
	@echo "Run artifacts cleaned."
