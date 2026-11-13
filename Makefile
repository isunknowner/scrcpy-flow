PY ?= python3

.PHONY: install dev install-vision run test lint smoke build

install:
	$(PY) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

dev:
	.venv/bin/python -m pip install -r requirements-dev.txt

install-vision:
	.venv/bin/python -m pip install -r requirements-vision.txt

run:
	$(PY) -m scflow

test:
	$(PY) -m unittest discover -s tests -v

lint:
	ruff check scflow tests

smoke:
	xvfb-run -a $(PY) -m scflow

build:
	./build_linux.sh
