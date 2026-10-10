# Linux/macOS/Git-Bash task runner. Windows PowerShell users: see tasks.ps1 (same targets).
PYTHON ?= python
EXPERIMENT ?= baseline

.PHONY: setup lint format test train

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

test:
	pytest -q

train:
	$(PYTHON) scripts/train.py experiment=$(EXPERIMENT)
