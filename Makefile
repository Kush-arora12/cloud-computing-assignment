SUBSET ?=

.PHONY: train test-teacher clean-data test up down

# Task 2: Training targets
train:
	@if [ -n "$(SUBSET)" ]; then \
		echo "Running training on subset of $(SUBSET) records..."; \
		python training/train.py --subset $(SUBSET); \
	else \
		echo "Running full training on cleaned records..."; \
		python training/train.py; \
	fi

clean-data:
	python worker/clean.py

test-nacl:
	python tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check

# Task 6: Cluster lifecycle targets
up:
	k3d cluster create --config cluster/k3d.yaml
	kubectl apply -k k8s/

down:
	k3d cluster delete --config cluster/k3d.yaml
test-teacher:
	python tools/score.py results/teacher_predictions.jsonl data/validation.jsonl --out results/teacher_validation.json
