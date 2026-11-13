PY ?= python
RUN = arrivalcast

.PHONY: install dev lint test test-fast data reproduce-syn reproduce-bts report paper clean docker

install:            ## install the package
	$(PY) -m pip install -e .

dev:                ## install with dev + data extras and git hooks
	$(PY) -m pip install -e ".[dev,notebooks]"
	pre-commit install

lint:
	ruff check src tests

test:               ## full suite incl. end-to-end smoke training (~1 min on CPU)
	pytest

test-fast:
	pytest -m "not slow"

data:               ## regenerate the 26 synthetic sources (deterministic, seed 42)
	$(RUN) data synthetic --source all --output-dir data/synthetic

reproduce-syn:      ## final synthetic study (LogNormMix 10 seeds, Neural Hawkes 5 seeds); ~3h on 1 CPU core
	$(RUN) run --run-name final_syn --lnm-lr 0.003 --nhp-max-epochs 600 --models poisson hawkes nhp --seeds 0 1 2 3 4
	$(RUN) run --run-name final_syn --lnm-lr 0.003 --nhp-max-epochs 600 --models lnm --seeds 0 1 2 3 4 5 6 7 8 9

reproduce-bts:      ## held-out BTS check with settings frozen from final_syn
	$(RUN) run --dataset bts --freeze-from results/final_syn --run-name bts_final --models poisson hawkes nhp --seeds 0 1 2 3 4
	$(RUN) run --dataset bts --freeze-from results/final_syn --run-name bts_final --models lnm --seeds 0 1 2 3 4 5 6 7 8 9

report:             ## regenerate every table/figure/number used in the paper
	$(RUN) report results/final_syn --bts results/bts_final --tuning results/tuning --out paper/generated
	$(RUN) hawkes-check --out paper/generated

paper: report       ## build paper/main.pdf
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

docker:
	docker build -t arrivalcast .

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov
	cd paper && latexmk -C || true
