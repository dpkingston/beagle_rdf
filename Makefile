# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
.PHONY: test lint typecheck run install dev-install

install:
	pip install -e .

dev-install:
	pip install -e ".[test,dev]"

test:
	pytest tests/

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

run:
	python -m beagle_node --config config/node.example.yaml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
