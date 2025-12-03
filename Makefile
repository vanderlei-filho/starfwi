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
build: ## build the starfwi test container image
	@$(CONTAINER_RUNTIME) build --cpuset-cpus=0-1 -f Containerfile -t $(IMAGE_NAME) .

.PHONY: init
init: build ## spin up containers for testing starfwi
	@$(COMPOSE_CMD) up -d --force-recreate

.PHONY: init-nodes
init-nodes: build ## spin up only test nodes (without starvz)
	@$(COMPOSE_CMD) up -d --force-recreate node1 node2

.PHONY: test
test: init-nodes ## build and run the standard test workload for starfwi
	@sleep 5
	@$(CONTAINER_RUNTIME) exec -u mpiuser starfwi-test-node-1 \
		mpirun \
		--bind-to none \
		--app /shared/appfile \
		--mca btl_tcp_if_include eth0 \
		#--display-map

.PHONY: trace
trace: init ## process and plot traces using starpu_fxt_tool and starvz
	@$(CONTAINER_RUNTIME) exec -w /shared/prof_files starfwi-test-node-1 \
		bash -c 'starpu_fxt_tool -i *'
	@$(CONTAINER_RUNTIME) exec -w /shared/prof_files starvz-shell \
		starvz -1 -t .
	@$(CONTAINER_RUNTIME) exec -w /shared/prof_files starvz-shell \
		starvz -2 .

.PHONY: kill
kill: ## shutdown test containers
	$(COMPOSE_CMD) down

.PHONY: clean
clean: kill ## cleanup containers and images
	$(CONTAINER_RUNTIME) rmi $(IMAGE_NAME)
