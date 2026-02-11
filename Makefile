PYTHON = uv run python


ruff:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format .
