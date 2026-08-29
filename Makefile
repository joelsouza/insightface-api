.PHONY: setup run test docker-build docker-run

# Docker configuration
IMAGE_NAME ?= insightface-api
IMAGE_TAG ?= latest

setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

run:
	./bin/start

test:
	./venv/bin/python -m pytest -v ./tests

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# The image already contains the model files, so no volume is mounted.
docker-run:
	docker run --rm -it \
		-p 5001:5001 \
		$(IMAGE_NAME):$(IMAGE_TAG)
