.PHONY: help install install-dev install-train install-app install-all \
       test test-fast lint format typecheck type-check check all \
       demo inference data pairs train evaluate \
       docker docker-cpu docker-demo docker-train \
       docs paper clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Installation ──────────────────────────────────────────────

install: ## Install LandmarkDiff (inference only)
	pip install -e .

install-dev: ## Install with development dependencies
	pip install -e ".[dev]"
	pre-commit install

install-train: ## Install with training dependencies
	pip install -e ".[train]"

install-app: ## Install with Gradio demo dependencies
	pip install -e ".[app]"

install-all: ## Install all optional dependencies
	pip install -e ".[train,eval,app,dev]"

# ── Quality ───────────────────────────────────────────────────

test: ## Run test suite
	pytest tests/ -v

test-fast: ## Run tests (no slow tests)
	pytest tests/ -v -m "not slow"

lint: ## Run linter
	ruff check landmarkdiff/ scripts/ tests/

format: ## Auto-format code
	ruff format landmarkdiff/ scripts/ tests/
	ruff check --fix landmarkdiff/ scripts/ tests/

typecheck: type-check ## Alias for type-check
type-check: ## Run type checker
	mypy landmarkdiff/ --ignore-missing-imports

check: lint type-check test ## Run all quality checks

all: install-dev lint type-check test ## Install dev deps, then lint + typecheck + test

# ── Documentation ────────────────────────────────────────────

docs: ## Build Sphinx documentation
	sphinx-build -b html docs/ docs/_build/html

# ── Demo & Inference ──────────────────────────────────────────

demo: ## Launch Gradio demo
	python scripts/app.py

inference: ## Run inference on a sample image (set IMG=path/to/face.jpg)
	python scripts/run_inference.py $(IMG) --procedure rhinoplasty --intensity 0.6

# ── Training ──────────────────────────────────────────────────

data: ## Download FFHQ face images (5K)
	python scripts/download_ffhq.py --num 5000 --resolution 512

pairs: ## Generate synthetic training pairs
	python scripts/generate_synthetic_data.py --input data/ffhq_samples/ --output data/synthetic_pairs/ --num 5000

train: ## Train ControlNet (Phase A, 10K steps)
	python scripts/train_controlnet.py --data_dir data/synthetic_pairs/ --output_dir checkpoints/ --num_train_steps 10000

evaluate: ## Run evaluation
	python scripts/evaluate.py --data_dir data/test_pairs/ --checkpoint checkpoints/latest

# ── Docker ────────────────────────────────────────────────────

docker: ## Build Docker image (GPU)
	docker build -t landmarkdiff .

docker-cpu: ## Build Docker image (CPU-only, smaller)
	docker build -t landmarkdiff:cpu -f Dockerfile.cpu .

docker-demo: ## Run Gradio demo in Docker (CPU)
	docker compose up app

docker-train: ## Run training in Docker (GPU)
	docker compose --profile training run train

# ── Paper ─────────────────────────────────────────────────────

paper: ## Build MICCAI paper PDF
	cd paper && make

# ── Cleanup ───────────────────────────────────────────────────

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	rm -rf docs/_build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
