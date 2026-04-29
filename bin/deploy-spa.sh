#!/usr/bin/env bash
# Uploads the SPA to S3 with hashed asset names for cache-busting.
#
# Usage: bin/deploy-spa.sh <bucket-name> <sha>
#
# index.html (no-cache) gets uploaded to the bucket root with __SHA__ replaced
# by the current git sha. app.js and style.css are uploaded as
# app.<sha>.js and style.<sha>.css with long-immutable cache-control — safe
# because the URL changes per deploy.

set -euo pipefail

BUCKET=${1:?bucket name required}
SHA=${2:?sha required}

cd "$(dirname "$0")/.."

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cp spa/app.js "$WORK/app.$SHA.js"
cp spa/style.css "$WORK/style.$SHA.css"
sed "s/__SHA__/$SHA/g" spa/index.html > "$WORK/index.html"

aws s3 cp "$WORK/app.$SHA.js" "s3://$BUCKET/app.$SHA.js" \
    --cache-control "public, max-age=31536000, immutable" \
    --content-type "text/javascript"

aws s3 cp "$WORK/style.$SHA.css" "s3://$BUCKET/style.$SHA.css" \
    --cache-control "public, max-age=31536000, immutable" \
    --content-type "text/css"

aws s3 cp "$WORK/index.html" "s3://$BUCKET/index.html" \
    --cache-control "no-cache" \
    --content-type "text/html"
