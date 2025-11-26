**Pre-commit Hooks Setup**

This project uses `pre-commit` to run formatters and linters locally before commits.

Install and enable the hooks in your developer environment (Windows PowerShell):

```powershell
# optional: activate your virtualenv first
python -m pip install --upgrade pip
pip install pre-commit

# Install the git hooks for this repo
pre-commit install

# To run hooks once across the repository (recommended after first install)
pre-commit run --all-files
```

Notes:
- The configured hooks are `black`, `isort`, `flake8`, and several small checks (trailing whitespace, end-of-file fixer, YAML check).
- If you prefer to make pre-commit part of your development dependencies, add `pre-commit` to your `requirements-dev.txt` or `pyproject.toml` dev dependencies.
- On CI we already run `black --check`, `isort --check-only` and `flake8` as part of the test workflow.
