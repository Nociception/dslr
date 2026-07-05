.PHONY: describe histogram scatter_plot pair_plot ruff logreg_train logreg_predict clean help

define VENV_HELP_MSG

Make sure your virtual environment is activated before running this command.
To activate your virtual environment, run the following command:

	source .venv/bin/activate

endef
export VENV_HELP_MSG

PYTHON = uv run python

describe:
	@echo "Usage:\n\t./describe.py datasets/dataset_train.csv"

histogram:
	$(PYTHON) histogram.py
	@echo "Take a look at histogram.png and best_hand.png in the current directory"

scatter_plot:
	$(PYTHON) scatter_plot.py

pair_plot:
	$(PYTHON) pair_plot.py
	@echo "Take a look at pair_plot.png in the current directory"

ruff:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format .

logreg_train:
	@echo "Usage:\n\t./logreg_train.py datasets/dataset_train.csv"
	@echo "$$VENV_HELP_MSG"

logreg_predict:
	@echo "Usage:\n\t./logreg_predict.py datasets/dataset_test.csv"
	@echo "$$VENV_HELP_MSG"

clean:
	@if [ -f describe.csv ]; then \
			rm describe.csv; \
	fi

help:
	@echo "If uv is not installed, you should fix it !"
	@echo "\n\tpip install uv"
