# CI/CD

## CI Workflow

Workflow: `.github/workflows/ci.yml`

On PR and push:
1. Install dependencies from `requirements-dev.txt`
2. Run `ruff`
3. Run `mypy`
4. Run unit tests with `coverage`
5. Enforce coverage threshold

Integration tests:
- Available via `workflow_dispatch`
- Require repository secrets for remote endpoint credentials

## Release Workflow

Existing release publish workflow remains in:
- `.github/workflows/workflow.yml`

Use release workflow only after CI passes on the release commit.
