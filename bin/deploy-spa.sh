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
# Two-step upload (sync + cp, not two sync passes):
#   1. `aws s3 sync` mirrors the assets tree with --delete to clear
#      stale prior-deploy artifacts; --exclude index.html so it
#      doesn't get the immutable cache header by accident.
#   2. `aws s3 cp` uploads index.html separately with no-cache so
#      browsers always check for a fresh entry-point.
#
# CUTOVER NOTE (wyrd-20pz first deploy): --delete will purge the
# legacy spa/ paths (app.<sha>.js, style.<sha>.css) that the prior
# deploys uploaded. Any browser holding a stale index.html that
# references those paths will 404 on its bundle until the user
# refreshes. CloudFront /index.html invalidation in the next step
# minimizes the window; users with a long-lived tab from
# pre-cutover may need a Ctrl-Shift-R. One-time pain at cutover;
# steady-state deploys never hit this since the build's hash names
# rotate.
aws s3 sync dist/ "s3://$BUCKET/" \
    --delete \
    --exclude "index.html" \
    --cache-control "public, max-age=31536000, immutable"

aws s3 cp dist/index.html "s3://$BUCKET/index.html" \
    --cache-control "no-cache" \
    --content-type "text/html"

echo "✓ SPA deployed to s3://$BUCKET/"
