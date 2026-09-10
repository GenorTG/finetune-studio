.PHONY: help install install-cpu install-conda install-llama check update run webui test lint clean

help:
	@echo "Finetune Studio — common targets:"
	@echo "  make install         install into ./.venv + llama-cpp-python"
	@echo "  make install-cpu     CPU-only install (skip GPU wheel selection)"
	@echo "  make install-conda   install into conda env CONDA_ENV=chris-ai"
	@echo "  make install-llama   build ~/llama.cpp CLI (convert_hf_to_gguf + llama-quantize)"
	@echo "  make check           verify install + llama.cpp CLI"
	@echo "  make update          self-healing: pull + sync deps + llama.cpp + restart"
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

install-llama:
	bash install.sh --llama-cpp-only

check:
	bash install.sh --check

update:
	bash update.sh

run:
	bash run.sh

webui: run

test:
	.venv/bin/python -m pytest tests/ -v --tb=short || python -m pytest tests/ -v --tb=short

lint:
	.venv/bin/python -m ruff check src/ || true

clean:
	rm -rf .venv data/*.db __pycache__ */__pycache__ */*/__pycache__ .pytest_cache