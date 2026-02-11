PYTHON = uv run python

describe:
	@echo "Usage:\n\t./describe.py datasets/dataset_[train/test].csv"

ruff:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format .
