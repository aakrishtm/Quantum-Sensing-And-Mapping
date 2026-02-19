.PHONY: install data train infer test run-app

PYTHON ?= python3
PIP ?= pip3

install:
	$(PIP) install -e .

# Density grids first, then Qiskit-simulated Gzz (quantum sensor) grids; chain so Gzz runs only if density succeeds
data:
	$(PYTHON) density_grid_generator.py && \
	$(PYTHON) generate_gzz_grids.py --shots 500

train:
	$(PYTHON) scripts/train.py

infer:
	$(PYTHON) scripts/infer.py

run-app:
	$(PYTHON) scripts/app.py

test:
	$(PYTHON) -c "import graviq; print('graviq OK')"
	$(PYTHON) -m unittest tests.test_dataset_pairs tests.test_noise tests.test_interferometer tests.test_baseline_threshold -v
