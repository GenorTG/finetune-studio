"""Makes finetune_studio runnable as `python -m finetune_studio`.

The `__main__.py` file is what Python runs when you invoke a package
with `python -m package_name`. Without this file, `python -m finetune_studio`
would do nothing.

KEY CONCEPTS
============
- `python -m package_name`: runs the package's __main__.py
- vs `python -m package_name.module`: runs a specific module's code
- The `__name__ == "__main__"` check: code inside this block only
  runs when the file is executed directly, not when imported.
"""

from finetune_studio.cli import main

if __name__ == "__main__":
    main()
