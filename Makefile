.PHONY: help install install-cpu install-llama check update run webui test lint codemap codemap-check hooks clean release-status release-prepare release-finish

help:
	@echo "Finetune Studio — common targets:"
	@echo "  make install         install into ./.venv + llama-cpp-python"
	@echo "  make install-cpu     CPU-only install (skip GPU wheel selection)"
	@echo "  make install-llama   build project-local .llama.cpp CLI (convert_hf_to_gguf + llama-quantize)"
	@echo "  make check           verify install + llama.cpp CLI"
	@echo "  make update          self-healing: pull + sync deps + llama.cpp + restart"
	@echo "  make run             start webui on :7860"
	@echo "  make test            run pytest"
	@echo "  make lint            ruff check src/ (never fails; run ruff directly)"
	@echo "  make codemap         regenerate docs/CODEMAP.md"
	@echo "  make codemap-check   fail if docs/CODEMAP.md is stale"
	@echo "  make hooks           install the branch-rule git hooks (once per clone)"
	@echo "  make release-status  what dev holds since the last release tag"
	@echo "  make release-prepare BUMP=minor   cut release/vX.Y.Z from dev, PR -> main"
	@echo "  make release-finish V=X.Y.Z       tag main after the merge, PR sync -> dev"
	@echo "  make clean           remove .venv + caches"

install:
	bash install.sh

install-cpu:
	bash install.sh --cpu

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
	@if [ ! -f docs/CODEMAP.md ]; then \
	  .venv/bin/python scripts/codemap.py --stdout >/dev/null \
	    && echo 'no docs/CODEMAP.md (gitignored dev doc, absent in a fresh clone/CI): generator ran clean, drift not checked'; \
	  exit $$?; fi; \
	cp docs/CODEMAP.md .codemap.prev; \
	.venv/bin/python scripts/codemap.py >/dev/null; \
	if cmp -s .codemap.prev docs/CODEMAP.md; then rm -f .codemap.prev; \
	else rm -f .codemap.prev; echo 'CODEMAP was stale — regenerated (local, gitignored dev doc)'; exit 1; fi

hooks:
	bash scripts/install-hooks.sh

# Releases (CONTRIBUTING.md "Versions and releases"): prepare -> merge the PR into main -> finish.
release-status:
	python3 scripts/release.py status

release-prepare:
	@test -n "$(BUMP)" || { echo "usage: make release-prepare BUMP=major|minor|patch|X.Y.Z"; exit 2; }
	python3 scripts/release.py prepare $(BUMP)

release-finish:
	@test -n "$(V)" || { echo "usage: make release-finish V=X.Y.Z"; exit 2; }
	python3 scripts/release.py finish $(V)

clean:
	rm -rf .venv data/*.db __pycache__ */__pycache__ */*/__pycache__ .pytest_cache