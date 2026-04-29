#!/usr/bin/env bash
# Invalidates the CloudFront distribution that fronts a given alias.
#
# Usage: bin/invalidate-cloudfront.sh <site-domain>
#
# Looks up the distribution by alias (so we don't need to thread the
# distribution ID through wyrd's terraform — it lives in infra-frontend).
# Targets only the un-hashed paths; hashed assets don't need invalidation
# because new deploys reference new URLs.

set -euo pipefail

SITE_DOMAIN=${1:?site domain required}

DIST_ID=$(aws cloudfront list-distributions \
    --query "DistributionList.Items[?Aliases.Items[?@=='$SITE_DOMAIN']].Id | [0]" \
    --output text)

if [ -z "$DIST_ID" ] || [ "$DIST_ID" = "None" ]; then
    echo "no CloudFront distribution found for $SITE_DOMAIN" >&2
    exit 1
fi

INV_ID=$(aws cloudfront create-invalidation \
    --distribution-id "$DIST_ID" \
    --paths '/index.html' '/' \
    --query 'Invalidation.Id' --output text)

aws cloudfront wait invalidation-completed \
    --distribution-id "$DIST_ID" --id "$INV_ID"

echo "invalidated $DIST_ID ($SITE_DOMAIN) — $INV_ID"
