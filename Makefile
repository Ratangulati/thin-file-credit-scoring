PY ?= .venv/bin/python

.PHONY: setup train evaluate serve test notebook all

setup:            ## create venv and install pinned deps
	uv venv --python 3.12 .venv || python3.12 -m venv .venv
	uv pip install --python $(PY) -r requirements.txt || $(PY) -m pip install -r requirements.txt

train:            ## clean data, simulate behavioral features, fit models -> models/
	$(PY) -m src.train

evaluate:         ## metrics tables + charts -> reports/
	$(PY) -m src.evaluate

serve:            ## run the scoring API on :8000
	$(PY) -m uvicorn src.api:app --reload --port 8000

test:
	$(PY) -m pytest -q

notebook:         ## execute the EDA notebook in place
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

all: train evaluate test
