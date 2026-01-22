.PHONY: format lint test ingest eval

format:
	@echo "Formatting not configured yet."

lint:
	@echo "Linting not configured yet."

test:
	@echo "Tests not configured yet."

ingest:
	python -m oncology_rag.cli.ingest --config configs/default.yaml

eval:
	python -m oncology_rag.cli.eval --config configs/default.yaml
