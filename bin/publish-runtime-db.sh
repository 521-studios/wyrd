#!/usr/bin/env bash
# Publish an L4 runtime DB to S3 + atomically flip current.json.
# Thin wrapper around wyrd.generators.kenning.lexicon.runtime_db_publish.
# See that module for the architecture (D38, PR 3 of wyrd-d90t).
#
# Usage:
#   bin/publish-runtime-db.sh --env staging --source /tmp/runtime.db
#   bin/publish-runtime-db.sh --env production --source /tmp/runtime.db --dry-run
#
# Per-env conventions (overridable via --bucket / --profile):
#   staging    → s3://521studios-staging-kenning-runtime    (profile: 521-Staging-Admin)
#   production → s3://521studios-production-kenning-runtime (profile: 521-Production-Admin)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer the project's venv python (matches the existing convention —
# 521 ops scripts assume an editable install at the repo root). Falls
# back to the system python on PATH if the venv isn't materialized;
# the operator gets a clear ImportError if boto3 isn't reachable.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" -m wyrd.generators.kenning.lexicon.runtime_db_publish "$@"
