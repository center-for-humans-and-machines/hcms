# Repository Guidelines

## Project Structure & Module Organization

- `hcms/`: main Python package. Key domains include `loading/`, `monitoring/`, `evaluation/`, `rating/`, `plot/`, and shared helpers in `utils/`.
- `tests/`: pytest suite (`test_*.py`), including `tests/test_regression.py` for model-output regression checks.
- `script/`: project entrypoints for setup, linting, and testing (`bootstrap`, `install`, `lint`, `test`).
- `data/`: datasets and paper artifacts (`data/paper/`, `data/dataset/`).
- Root scripts: `generate_dataset.py` and `evaluate_dataset.py` run core experiment workflows.

## Build, Test, and Development Commands

- `./script/bootstrap`: install dependencies, pre-commit hooks, and default `.env`.
- `./script/install`: install the package via Poetry.
- `./script/lint`: run both Docker super-linter and local pre-commit checks.
- `./script/test`: run pytest excluding regression tests; writes `.test_report.xml`.
- `./script/test -r`: run full suite including regression tests.
- Example experiment commands:
  - `poetry run python generate_dataset.py --round-number=2 --tag=acm-tist`
  - `poetry run python evaluate_dataset.py --round-number=3`

## Coding Style & Naming Conventions

- Python target: `3.13` (Poetry-managed).
- Formatting/linting enforced via pre-commit: `ruff`, `ruff-format`, `pyink`, `flake8`, `pylint`, `mypy`, plus shell/Markdown checks.
- Use `snake_case` for functions/modules, `PascalCase` for classes, and clear domain-oriented names (for example, `test_loading_constants.py`).
- Shell scripts follow Google Shell Style and use 2-space indentation via `shfmt`.

## Testing Guidelines

- Framework: `pytest`.
- Name tests as `test_*.py` and test functions as `test_*`.
- Keep fast unit checks in default suite; place expensive model/dataset assertions in regression tests.
- For non-regression test runs, export required env vars: `MODEL_API_KEY`, `MODEL_ENDPOINT`, `BATCH_MODEL_NAME`.

## Commit & Pull Request Guidelines

- Recent history uses short, scoped summaries like `Poetry: Add pytz` and `Readme: Remove link to dataset-card.md`.
- Preferred format: `Scope: imperative summary` (one topic per atomic commit).
- PRs should include: purpose, key changes, validation commands run (for example `./script/lint` and `./script/test`), and links to related issues.
- Include screenshots only when plots/notebook outputs materially change.

## Security & Configuration Tips

- Never commit secrets; keep credentials in `.env` (seed from `env.example`).
- Provider-specific templates (`.env.openai`, `.env.azure`, etc.) are references; copy values into local `.env` as needed.
