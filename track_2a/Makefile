.PHONY: run build test

IMAGE ?= booklet-check:latest
# Default input: the two official example cases (one task A with its booklet, one task B).
INPUT_DIR ?= data/examples
INPUT ?= cases.jsonl
OUTPUT_DIR ?= output

# The evaluation injects BASE_URL / API_KEY; the project template names and local names work too.
BASE_URL ?= $(or $(LLM_BASE_URL),$(APERTUS_BASE_URL))
API_KEY ?= $(or $(LLM_API_KEY),$(APERTUS_API_KEY))
export BASE_URL API_KEY LLM_NAME

build:
	docker build --platform linux/amd64 -t $(IMAGE) .

# Builds the image and runs it on $(INPUT_DIR)/$(INPUT); predictions land in $(OUTPUT_DIR)/.
# Without BASE_URL/API_KEY the container runs in a clearly labelled DRY MODE (no model calls).
run: build
	mkdir -p $(OUTPUT_DIR)
	docker run --rm --platform linux/amd64 \
	  -e BASE_URL -e API_KEY -e LLM_NAME \
	  -v "$(abspath $(INPUT_DIR)):/data:ro" \
	  -v "$(abspath $(OUTPUT_DIR)):/output" \
	  $(IMAGE) --input /data/$(INPUT) --output /output/predictions.jsonl
	@echo "predictions: $(OUTPUT_DIR)/predictions.jsonl"

test:
	python -m pytest -q tests
