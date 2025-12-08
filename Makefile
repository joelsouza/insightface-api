.PHONY: setup run test

setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

run:
	./bin/start

test:
	./venv/bin/python -m pytest -s tests
