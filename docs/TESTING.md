# Testing Guide

## Test Layout

- Unit tests: `tests/`
- Integration tests: `tests/integration/`

## Run Unit Tests

```bash
python -m unittest discover -s tests -v
```

## Run Integration Tests

Integration tests are opt-in.

Required environment variables:
- `MECHANEX_INTEGRATION_BASE_URL`
- `MECHANEX_INTEGRATION_API_KEY`
- `MECHANEX_INTEGRATION_RUN_REMOTE=1`

Run:
```bash
python -m unittest discover -s tests/integration -v
```

## Coverage

```bash
coverage run -m unittest discover -s tests -v
coverage report --fail-under=70
```

## Test Design Rules

- Keep tests deterministic and side-effect free.
- Use stubs/mocks for remote services in unit tests.
- Keep at least one high-signal integration smoke test for remote policy execution.
