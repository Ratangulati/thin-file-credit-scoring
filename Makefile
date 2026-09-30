PY ?= .venv/bin/python

.PHONY: setup train evaluate serve test notebook all

setup:
	uv venv --python 3.12 .venv || python3.12 -m venv .venv
	uv pip install --python $(PY) -r requirements.txt || $(PY) -m pip install -r requirements.txt

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

serve:
	$(PY) -m uvicorn src.api:app --reload --port 8000

test:
	$(PY) -m pytest -q

notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

all: train evaluate test
