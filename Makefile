.PHONY: install data train infer test run-app

PYTHON ?= python3
PIP ?= pip3

install:
	$(PIP) install -e .

data:
	$(PYTHON) density_grid_generator.py

train:
	$(PYTHON) scripts/train.py

infer:
	$(PYTHON) scripts/infer.py

run-app:
	$(PYTHON) scripts/app.py

test:
	$(PYTHON) -c "import graviq; print('graviq OK')"
	$(PYTHON) -m unittest tests.test_dataset_pairs tests.test_noise tests.test_interferometer tests.test_baseline_threshold -v
