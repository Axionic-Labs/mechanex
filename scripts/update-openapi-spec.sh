#!/usr/bin/env bash
# Fetch the latest OpenAPI spec from the backend repo.
# Usage: ./scripts/update-openapi-spec.sh [branch]
set -euo pipefail

BRANCH="${1:-development}"
REPO="Axionic-Labs/axionic-mvp-backend"
OUTPUT="tests/fixtures/backend-openapi.json"

echo "Fetching openapi.json from $REPO@$BRANCH..."

gh api "repos/$REPO/contents/openapi.json?ref=$BRANCH" \
  --jq '.content' | base64 -d > "$OUTPUT"

echo "Updated $OUTPUT"
