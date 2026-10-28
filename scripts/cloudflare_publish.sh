#!/usr/bin/env bash
# cloudflare_publish.sh — upload per-VRAM bundles to R2 and the download
# page to Cloudflare Pages (v5.186).
#
#   python examples/publish_cloudflare.py --packages dist/ --out cf/ \
#       --base-url https://dl.example.com --version v5.186-pilot
#   scripts/cloudflare_publish.sh cf/
#
# R2 goes through its S3 API (aws CLI), which uploads multi-GB GGUF zips in
# parts; the page goes up with `wrangler pages deploy` (direct upload, the
# Pages project has no Git integration). Credentials come from the
# environment only:
#
#   CLOUDFLARE_ACCOUNT_ID   account id
#   R2_ACCESS_KEY_ID        R2 API token (Object Read & Write on the bucket)
#   R2_SECRET_ACCESS_KEY
#   R2_BUCKET               bucket behind the --base-url download domain
#   CLOUDFLARE_API_TOKEN    token with Cloudflare Pages: Edit (for the page)
#   CF_PAGES_PROJECT        Pages project name (default helioslm)
#
# DRY_RUN=1 prints the commands without running them.
set -euo pipefail
OUT=${1:?usage: cloudflare_publish.sh <publish_cloudflare.py --out dir>}
: "${CLOUDFLARE_ACCOUNT_ID:?}" "${R2_BUCKET:?}"
PROJECT=${CF_PAGES_PROJECT:-helioslm}
ENDPOINT="https://${CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com"
run() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ $*"; else "$@"; fi; }

[ -d "$OUT/r2" ] && [ -f "$OUT/site/index.html" ] || {
  echo "$OUT has no r2/ and site/ (run examples/publish_cloudflare.py first)" >&2; exit 1; }

# 1. bundles (keys <version>/<zip>), immutable: long cache
export AWS_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID:-} AWS_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY:-}
export AWS_DEFAULT_REGION=auto
run aws s3 cp "$OUT/r2/" "s3://$R2_BUCKET/" --recursive --endpoint-url "$ENDPOINT" \
  --content-type application/zip --cache-control "public, max-age=31536000, immutable"
# packages.json next to the zips too, for scripts that read the bucket
run aws s3 cp "$OUT/site/packages.json" "s3://$R2_BUCKET/packages.json" \
  --endpoint-url "$ENDPOINT" --content-type application/json --cache-control "no-cache"

# 2. download page (direct upload; not connected to the GitHub repo)
: "${CLOUDFLARE_API_TOKEN:?}"
run npx --yes wrangler@4 pages deploy "$OUT/site" --project-name "$PROJECT" --branch main
