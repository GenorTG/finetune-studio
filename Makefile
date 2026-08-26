.PHONY: help install install-gguf install-conda check run webui test lint clean

help:
	@echo "Finetune Studio — common targets:"
	@echo "  make install         install into ./.venv"
	@echo "  make install-gguf    install + llama-cpp-python for GGUF"
	@echo "  make install-conda   install into conda env CONDA_ENV=chris-ai"
	@echo "  make check           verify install"
	@echo "  make run             start webui on :7860"
	@echo "  make test            run pytest"
	@echo "  make lint            ruff check"
	@echo "  make clean           remove .venv + caches"

install:
	bash install.sh

install-cpu:
	bash install.sh --cpu

install-conda:
	USE_CONDA=1 bash install.sh

check:
	bash install.sh --check

run:
	bash run.sh

webui: run

test:
	.venv/bin/python -m pytest tests/ -v --tb=short || python -m pytest tests/ -v --tb=short

lint:
	.venv/bin/python -m ruff check src/ || true

clean:
	rm -rf .venv data/*.db __pycache__ */__pycache__ */*/__pycache__ .pytest_cache