# Contributing to Mechanex

## Development Setup

1. Create and activate a virtual environment.
2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```
3. Run quality gates before opening a PR:
```bash
ruff check mechanex tests
mypy
coverage run -m unittest discover -s tests -v
coverage report --fail-under=70
```

## Branch and PR Rules

- Use short-lived feature branches from `main` (or `develop` if used by your team).
- Keep PRs focused and small enough for same-day review.
- Require passing CI before merge.
- Require at least one reviewer for non-trivial runtime/policy changes.

## Testing Rules

- Add or update tests for all behavior changes.
- For bug fixes, include a regression test that fails before the fix.
- Integration tests live under `tests/integration` and are opt-in via environment variables.

## Commit Guidelines

- Use imperative commit messages (`add`, `fix`, `refactor`).
- Include a short scope prefix where possible (`policy:`, `runtime:`, `ci:`).
- Do not bundle unrelated changes in one commit.
