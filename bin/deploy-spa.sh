#!/usr/bin/env bash
# wyrd-20pz: build the Vite SPA in spa-next/ and sync to S3.
#
# Usage: bin/deploy-spa.sh <bucket-name>
#
# Vite handles its own asset hashing (dist/assets/index-<hash>.js etc.),
# so we don't need to bake a sha into filenames anymore — the
# index.html that Vite produces already references the hashed assets
# by name. aws s3 sync copies the dist tree; hashed assets get long-
# immutable cache-control, index.html stays no-cache.

set -euo pipefail

BUCKET=${1:?bucket name required}

cd "$(dirname "$0")/../spa-next"

echo "→ Installing SPA dependencies"
npm ci

echo "→ Building SPA (Vite)"
npm run build

echo "→ Syncing dist/ to s3://$BUCKET/"
# Two-pass sync: (1) hashed assets with immutable cache, (2)
# index.html with no-cache. The --exclude/--include flags filter
# what each pass touches. --delete removes stale files (old build
# artifacts) so the bucket only carries the current build.
aws s3 sync dist/ "s3://$BUCKET/" \
    --delete \
    --exclude "index.html" \
    --cache-control "public, max-age=31536000, immutable"

aws s3 cp dist/index.html "s3://$BUCKET/index.html" \
    --cache-control "no-cache" \
    --content-type "text/html"

echo "✓ SPA deployed to s3://$BUCKET/"
