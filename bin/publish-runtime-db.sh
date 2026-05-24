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

# Pick the python interpreter. Operator-overridable via WYRD_PYTHON
# so CI / non-standard installs can point at a different binary.
# Otherwise:
#   * prefer .venv/bin/python (the 521 ops convention; editable install
#     at the repo root)
#   * fall back to the system python on PATH ONLY when the operator
#     explicitly opts in via WYRD_PUBLISH_ALLOW_SYSTEM_PYTHON=1.
# Falling back silently was hostile under pressure — operator gets a
# confusing ModuleNotFoundError far from the actual missing-venv root
# cause. Loud upfront is better.
if [[ -n "${WYRD_PYTHON:-}" ]]; then
    PYTHON="$WYRD_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
elif [[ "${WYRD_PUBLISH_ALLOW_SYSTEM_PYTHON:-0}" = "1" ]]; then
    PYTHON="python3"
else
    echo "error: $REPO_ROOT/.venv/bin/python is missing." >&2
    echo "Create the venv with 'uv sync' or set WYRD_PYTHON=/path/to/python." >&2
    echo "(To use the system python anyway, set WYRD_PUBLISH_ALLOW_SYSTEM_PYTHON=1.)" >&2
    exit 1
fi

exec "$PYTHON" -m wyrd.generators.kenning.lexicon.runtime_db_publish "$@"
