.PHONY: help install check-env format lint test ingest \
	smoke smoke-a1 smoke-a2 smoke-a3 smoke-a4 \
	test-mini test-small test-medium test-full \
	eval-a1-large eval-a1-small eval-a1-all \
	eval-a2-large eval-a2-small eval-a2-all \
	eval-a3-large eval-a3-small eval-a3-all \
	eval-a4-large eval-a4-small eval-a4-all eval-a4-qwen \
	report report-smoke sct-agreement sct-validate clean-runs

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
DATASET_VALIDATED := data/eval/sct_validated_ground_truth.csv
PROVIDER_CONFIG := configs/providers/openrouter.yaml
EMBEDDINGS_CONFIG := configs/rag/embeddings.yaml
CHROMA_CONFIG := configs/rag/chroma.yaml
RUNS_DIR := runs

# Default model for single-model tests (first small model)
DEFAULT_SMALL_MODEL := qwen3_vl_30b

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
	@echo "  eval-a1-large Run A1 on validated ground truth (large models)"
	@echo "  eval-a1-small Run A1 on validated ground truth (small models)"
	@echo "  eval-a1-all   Run A1 on validated ground truth (large + small)"
	@echo "  eval-a2-large Run A2 on validated ground truth (large models)"
	@echo "  eval-a2-small Run A2 on validated ground truth (small models)"
	@echo "  eval-a2-all   Run A2 on validated ground truth (large + small)"
	@echo "  eval-a3-large Run A3 on validated ground truth (large models)"
	@echo "  eval-a3-small Run A3 on validated ground truth (small models)"
	@echo "  eval-a3-all   Run A3 on validated ground truth (large + small)"
	@echo "  eval-a4-large Run A4 on validated ground truth (large models)"
	@echo "  eval-a4-small Run A4 on validated ground truth (small models)"
	@echo "  eval-a4-qwen  Run A4 on validated ground truth (qwen3_vl_30b only)"
	@echo "  eval-a4-all   Run A4 on validated ground truth (large + small)"
	@echo ""
	@echo "Reports:"
	@echo "  report        Generate reports from latest runs"
	@echo "  report-smoke  Generate smoke test report"
	@echo ""
	@echo "Analysis:"
	@echo "  sct-agreement Analyze expert vs model agreement (requires RESPONSES_CSV)"
	@echo "                Usage: make sct-agreement RESPONSES_CSV=path/to/responses.csv"
	@echo "  sct-validate  Generate validated ground truth from expert responses"
	@echo "                Usage: make sct-validate RESPONSES_CSV=path/to/responses.csv"
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
# VALIDATED GROUND TRUTH BY ARM
# =============================================================================
eval-a1-large:
	@echo "Running A1 on validated ground truth (large models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A1 \
		--model-groups large

eval-a1-small:
	@echo "Running A1 on validated ground truth (small models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A1 \
		--model-groups small

eval-a1-all:
	@echo "Running A1 on validated ground truth (large + small models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A1 \
		--model-groups large small

eval-a2-large:
	@echo "Running A2 on validated ground truth (large models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A2 \
		--model-groups large

eval-a2-small:
	@echo "Running A2 on validated ground truth (small models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A2 \
		--model-groups small

eval-a2-all:
	@echo "Running A2 on validated ground truth (large + small models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A2 \
		--model-groups large small

eval-a3-large:
	@echo "Running A3 on validated ground truth (large models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A3 \
		--model-groups large

eval-a3-small:
	@echo "Running A3 on validated ground truth (small models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A3 \
		--model-groups small

eval-a3-all:
	@echo "Running A3 on validated ground truth (large + small models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A3 \
		--model-groups large small

eval-a4-large:
	@echo "Running A4 on validated ground truth (large models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A4 \
		--model-groups large

eval-a4-small:
	@echo "Running A4 on validated ground truth (small models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A4 \
		--model-groups small

eval-a4-all:
	@echo "Running A4 on validated ground truth (large + small models)..."
	$(PYTHON) -m oncology_rag.cli.eval matrix \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--embeddings-config $(EMBEDDINGS_CONFIG) \
		--chroma-config $(CHROMA_CONFIG) \
		--runs-dir $(RUNS_DIR) \
		--arms A4 \
		--model-groups large small

eval-a4-qwen:
	@echo "Running A4 on validated ground truth (qwen3_vl_30b only)..."
	$(PYTHON) -m oncology_rag.cli.eval single \
		--experiment configs/experiments/a4_consensus_small.yaml \
		--dataset $(DATASET_VALIDATED) \
		--provider-config $(PROVIDER_CONFIG) \
		--runs-dir $(RUNS_DIR)

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
# ANALYSIS
# =============================================================================
sct-agreement:
ifndef RESPONSES_CSV
	$(error RESPONSES_CSV is required. Usage: make sct-agreement RESPONSES_CSV=data/sct_responses/your_file.csv)
endif
	@echo "Analyzing SCT expert agreement..."
	$(PYTHON) -m oncology_rag.cli.sct_agreement \
		--responses $(RESPONSES_CSV) \
		--dataset $(DATASET) \
		--runs-dir $(RUNS_DIR)
	@echo "Agreement analysis complete. Check runs/agreement/ for results."

sct-validate:
ifndef RESPONSES_CSV
	$(error RESPONSES_CSV is required. Usage: make sct-validate RESPONSES_CSV=data/sct_responses/your_file.csv)
endif
	@echo "Generating validated ground truth from expert responses..."
	$(PYTHON) -m oncology_rag.cli.sct_validate \
		--responses $(RESPONSES_CSV) \
		--verified-emails user9@gmail.com \
		--output-csv data/eval/sct_validated_ground_truth.csv \
		--output-json data/eval/sct_validated_ground_truth.json
	@echo "Validated ground truth generated at data/eval/"

# =============================================================================
# MAINTENANCE
# =============================================================================
clean-runs:
	@echo "Removing all run artifacts..."
	rm -rf $(RUNS_DIR)/experiments $(RUNS_DIR)/matrix $(RUNS_DIR)/reports
	@echo "Run artifacts cleaned."
