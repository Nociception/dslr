.PHONY: describe histogram scatter_plot pair_plot ruff logreg_train logreg_predict clean help


PYTHON = uv run python

describe:
	@echo "Usage:\n\t./describe.py datasets/dataset_[train/test].csv"

histogram:
	$(PYTHON) histogram.py

scatter_plot:
	$(PYTHON) scatter_plot.py
	
pair_plot:
	$(PYTHON) pair_plot.py

ruff:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format .

logreg_train:
	@echo "Usage:\n\t./logreg_train.py datasets/dataset_train.csv"

logreg_predict:
	@echo "Usage:\n\t./logreg_predict.py datasets/dataset_test.csv"

clean:
	@if [ -f describe.csv ]; then \
			rm describe.csv; \
	fi

help:
	@echo "If uv is not installed, you should fix it !"
	@echo "\n\tpip install uv"