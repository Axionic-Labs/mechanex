# Release Process

## Pre-release Checklist

1. CI green on target branch.
2. No failing integration smoke tests.
3. `IMPLEMENTATION_VERIFICATION.md` updated for major runtime changes.
4. Changelog/release notes drafted.

## Versioning

- Use semantic versioning.
- Breaking public SDK changes require major version bumps.
- New backward-compatible features require minor version bumps.
- Bug fixes require patch version bumps.

## Release Steps

1. Create release PR with version bump.
2. Merge after approval and green CI.
3. Tag release commit.
4. Publish GitHub release to trigger package publish workflow.
5. Validate package install and smoke test from clean environment.
