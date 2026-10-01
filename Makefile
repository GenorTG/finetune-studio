.PHONY: help install install-cpu install-conda install-llama check update run webui test lint codemap codemap-check hooks clean

help:
	@echo "Finetune Studio — common targets:"
	@echo "  make install         install into ./.venv + llama-cpp-python"
	@echo "  make install-cpu     CPU-only install (skip GPU wheel selection)"
	@echo "  make install-conda   install into conda env CONDA_ENV=chris-ai"
	@echo "  make install-llama   build project-local .llama.cpp CLI (convert_hf_to_gguf + llama-quantize)"
	@echo "  make check           verify install + llama.cpp CLI"
	@echo "  make update          self-healing: pull + sync deps + llama.cpp + restart"
	@echo "  make run             start webui on :7860"
	@echo "  make test            run pytest"
	@echo "  make lint            ruff check src/ (never fails; run ruff directly)"
	@echo "  make codemap         regenerate docs/CODEMAP.md"
	@echo "  make codemap-check   fail if docs/CODEMAP.md is stale"
	@echo "  make hooks           install the per-commit VERSION bump hook"
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
	.venv/bin/python -m pytest tests/ -v --tb=short

lint:
	.venv/bin/python -m ruff check src/ || true

codemap:
	.venv/bin/python scripts/codemap.py

codemap-check:
	@.venv/bin/python scripts/codemap.py && git diff --exit-code -- docs/CODEMAP.md || (echo 'CODEMAP stale — run `make codemap` and commit it'; exit 1)

hooks:
	bash scripts/install-hooks.sh

clean:
	rm -rf .venv data/*.db __pycache__ */__pycache__ */*/__pycache__ .pytest_cache