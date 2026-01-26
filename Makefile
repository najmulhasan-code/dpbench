# DPBench Makefile
# Common commands for development and experiments

.PHONY: install test lint clean experiments help

# Default target
help:
	@echo "DPBench - LLM Coordination Benchmark"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install package in development mode"
	@echo "  install-lock     Install with pinned dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test             Run tests"
	@echo "  lint             Run linter (ruff)"
	@echo "  clean            Remove build artifacts"
	@echo ""
	@echo "Experiments:"
	@echo "  quick-test       Run 1 episode quick test"
	@echo "  experiments      Run all 8 experimental conditions"
	@echo "  experiments-5p   Run 4 conditions with 5 philosophers"
	@echo "  experiments-3p   Run 4 conditions with 3 philosophers"

# Setup
install:
	pip install -e .

install-lock:
	pip install -r requirements-lock.txt
	pip install -e .

# Development
test:
	pytest tests/ -v

lint:
	ruff check dpbench/ scripts/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Quick test
quick-test:
	dpbench --episodes 1 --max-timesteps 10 --show-reasoning --log-dir ./logs

# All experiments (8 conditions x 30 episodes)
experiments: experiments-5p experiments-3p
	@echo "All experiments complete!"
	@echo "Results in: ./logs/"

# 5 Philosophers (4 conditions)
experiments-5p:
	@echo "Running 5-philosopher experiments..."
	dpbench --episodes 30 --mode simultaneous --log-dir ./logs
	dpbench --episodes 30 --mode simultaneous --no-communication --log-dir ./logs
	dpbench --episodes 30 --mode sequential --log-dir ./logs
	dpbench --episodes 30 --mode sequential --no-communication --log-dir ./logs

# 3 Philosophers (4 conditions)
experiments-3p:
	@echo "Running 3-philosopher experiments..."
	dpbench --episodes 30 --philosophers 3 --mode simultaneous --log-dir ./logs
	dpbench --episodes 30 --philosophers 3 --mode simultaneous --no-communication --log-dir ./logs
	dpbench --episodes 30 --philosophers 3 --mode sequential --log-dir ./logs
	dpbench --episodes 30 --philosophers 3 --mode sequential --no-communication --log-dir ./logs

# Individual experiment shortcuts
sim-5p-comm:
	dpbench --episodes 30 --mode simultaneous --log-dir ./logs -o results/sim_5p_comm.json

sim-5p-nocomm:
	dpbench --episodes 30 --mode simultaneous --no-communication --log-dir ./logs -o results/sim_5p_nocomm.json

seq-5p-comm:
	dpbench --episodes 30 --mode sequential --log-dir ./logs -o results/seq_5p_comm.json

seq-5p-nocomm:
	dpbench --episodes 30 --mode sequential --no-communication --log-dir ./logs -o results/seq_5p_nocomm.json
