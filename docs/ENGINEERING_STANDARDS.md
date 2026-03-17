# Mechanex Engineering Standards

## Quality Gates

- Lint: `ruff check mechanex tests`
- Type checks: `mypy`
- Unit tests: `python -m unittest discover -s tests -v`
- Coverage threshold: `>= 70%` for configured runtime modules

No PR should merge with failing gates.

## Code Guidelines

- Prefer explicit runtime behavior over hidden side effects.
- Keep policy behavior deterministic when possible.
- Use dependency injection for transport/runtime clients to keep tests isolated.
- Avoid top-level network calls in module import paths.

## API/SDK Compatibility Rules

- Treat public SDK method signatures as stable contracts.
- Add deprecation windows for breaking changes.
- Add contract tests for policy payload shape when behavior changes.

## Reliability Rules

- All network paths must include explicit timeout behavior.
- Retry only transient failures; do not retry validation errors.
- Include request identifiers in remote calls and logs where available.
