PYTHON = uv run python


ruff:
	$(PYTHON) -m ruff check .
