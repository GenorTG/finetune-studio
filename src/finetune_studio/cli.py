"""Back-compat shim -- the real code lives in finetune_studio.cli/.

Old: `from finetune_studio.cli import main` and `python -m finetune_studio.cli`
both still work.
"""
from finetune_studio.cli._registry import main  # noqa: F401


if __name__ == "__main__":
    main()
