SHELL = /bin/bash

CONTAINER_RUNTIME := $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
COMPOSE_CMD := $(CONTAINER_RUNTIME)-compose
IMAGE_NAME := starfwi-test-image
CONTAINER_NAME := starfwi
STARVZ_SHELL := starvz-shell

.DEFAULT_GOAL := help
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## build starfwi + local StarPU fork (../starpu) into the container image
	@$(CONTAINER_RUNTIME) build \
		--build-context starpu=../starpu \
		-f Containerfile \
		-t $(IMAGE_NAME) \
		.

.PHONY: start
start: build ## spin up the two MPI test nodes (node1 + node2)
	@$(COMPOSE_CMD) up -d --force-recreate node1 node2

.PHONY: test
test: start ## run the full FWI test: modeling (true model → observed data) then inversion (2 MPI ranks: node1=2 workers, node2=3 workers)
	@mkdir -p ./.local_shared_volume/prof_files \
	          ./.local_shared_volume/observed \
	          ./.local_shared_volume/perf_models \
	          ./.local_shared_volume/viz
	@if [ ! -d ./.local_shared_volume/marmousi2 ]; then \
		echo "Extracting marmousi2.zip..."; \
		unzip -q ./.local_shared_volume/marmousi2.zip -d ./.local_shared_volume/; \
	fi
	@if [ ! -f ./.local_shared_volume/marmousi2/MODEL_P-WAVE_VELOCITY_1.25m.segy ]; then \
		echo "Extracting marmousi2 SEG-Y files..."; \
		tar -xzf ./.local_shared_volume/marmousi2/MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz -C ./.local_shared_volume/marmousi2/; \
		tar -xzf ./.local_shared_volume/marmousi2/MODEL_S-WAVE_VELOCITY_1.25m.segy.tar.gz -C ./.local_shared_volume/marmousi2/; \
		tar -xzf ./.local_shared_volume/marmousi2/MODEL_DENSITY_1.25m.segy.tar.gz -C ./.local_shared_volume/marmousi2/; \
	fi
	@chmod -R 777 ./.local_shared_volume 2>/dev/null || true
	@rm -f ./.local_shared_volume/prof_files/*
	@sleep 5
	@echo "==> Phase 1: starfwi-modeling (forward propagation on true model, writes observed data)"
	@$(CONTAINER_RUNTIME) exec -u mpiuser starfwi-test-node-1 \
		mpirun \
		--bind-to none \
		--app /shared/appfile-modeling \
		--mca btl_tcp_if_include eth0
	@echo "==> Phase 2: starfwi-fwi (inversion on perturbed model using observed data)"
	@$(CONTAINER_RUNTIME) exec -u mpiuser starfwi-test-node-1 \
		mpirun \
		--bind-to none \
		--app /shared/appfile-fwi \
		--mca btl_tcp_if_include eth0

.PHONY: trace
trace: build ## convert FxT traces with starpu_fxt_tool and render plots with StarVZ (requires starvz-shell image: https://github.com/vanderlei-filho/starvz-shell)
	@$(COMPOSE_CMD) up -d --force-recreate
	@$(CONTAINER_RUNTIME) exec -w /shared/prof_files starfwi-test-node-1 \
		bash -c 'starpu_fxt_tool -i *'
	@$(CONTAINER_RUNTIME) exec -w /shared/prof_files starvz-shell \
		starvz -1 -t .
	@$(CONTAINER_RUNTIME) exec -w /shared/prof_files starvz-shell \
		starvz -2 .

.PHONY: kill
kill: ## stop and remove all running containers
	$(COMPOSE_CMD) down

.PHONY: clean
clean: kill ## stop containers and delete the container image
	$(CONTAINER_RUNTIME) rmi $(IMAGE_NAME)
