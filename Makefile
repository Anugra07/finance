PYTHON ?= python3

.PHONY: install install-v1 test lint bootstrap refresh-db

install:
	$(PYTHON) -m pip install -e '.[dev]'

install-v1:
	$(PYTHON) -m pip install -e '.[v1,dev]'

test:
	pytest

lint:
	ruff check src tests

bootstrap:
	ai-analyst bootstrap

refresh-db:
	ai-analyst db refresh
