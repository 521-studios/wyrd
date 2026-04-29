#!/usr/bin/env bash
# Builds wyrd's Lambda deployment zip at <repo-root>/function.zip.
#
# Usage:  bin/build-lambda.sh
# Output: function.zip in the repo root, containing wyrd/ + Python deps.
#
# We install for arm64 + linux + Python 3.12 to match the Lambda runtime,
# regardless of the host architecture.

set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR=build/lambda
ZIP=function.zip

rm -rf "$BUILD_DIR" "$ZIP"
mkdir -p "$BUILD_DIR"

# Install wyrd + its declared deps for the Lambda's runtime.
pip install \
    --quiet \
    --target "$BUILD_DIR" \
    --platform manylinux2014_aarch64 \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --upgrade \
    .

# Drop bundled __pycache__ and dist-info to shrink the artifact.
find "$BUILD_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR" -type d -name '*.dist-info' -exec rm -rf {} + 2>/dev/null || true

(cd "$BUILD_DIR" && zip -rq "../../$ZIP" .)

printf 'Built %s (%s bytes)\n' "$ZIP" "$(stat -c%s "$ZIP")"
