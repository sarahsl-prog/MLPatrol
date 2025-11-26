Pre-commit hooks

This repository includes a minimal `.pre-commit-config.yaml` to run code formatters and linters locally before committing changes.

Install (once):

```powershell
pip install pre-commit
pre-commit install
```

Run hooks manually:

```powershell
pre-commit run --all-files
```

Configured hooks:

- `black` — code formatting
- `isort` — sort imports
- `flake8` — linting checks

If you want additional hooks (e.g., ruff, safety), let me know and I can add them.