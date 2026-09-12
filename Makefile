.PHONY: setup test lint format

setup:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest

lint:
	ruff check .
	black --check .
	isort --check-only .

format:
	ruff check --fix .
	black .
	isort .
