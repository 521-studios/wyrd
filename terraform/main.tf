terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
  backend "s3" {
    # bucket and key passed via -backend-config at init time (see deploy.yml)
    region = "us-east-2"
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  name                = "wyrd"
  spa_bucket          = "521studios-${var.env}-wyrd-spa"
  runtime_db_bucket   = "521studios-${var.env}-kenning-runtime"
  tags = {
    Project     = "wyrd"
    Environment = var.env
    ManagedBy   = "terraform"
  }
}

# ─── SPA S3 bucket — static site assets, keyed by git sha ───────────────────

resource "aws_s3_bucket" "spa" {
  bucket = local.spa_bucket
  tags   = local.tags
}

resource "aws_s3_bucket_versioning" "spa" {
  bucket = aws_s3_bucket.spa.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "spa" {
  bucket                  = aws_s3_bucket.spa.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "spa" {
  bucket = aws_s3_bucket.spa.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ─── Kenning runtime DB bucket — L4 SQLite, served to Lambda via S3 ─────────
#
# The Kenning generator reads its meanings + per-culture proportions from
# an L4 SQLite DB at cold start (wyrd-d90t). The Lambda resolves the DB via
# WYRD_RUNTIME_DB_BUCKET; this bucket holds the versioned keys (v/<ts>.db)
# plus the current.json pointer (see bin/publish-runtime-db.sh).
#
# Staging's bucket + public-access-block were hand-provisioned during the d90t
# cutover and adopted into terraform state via one-shot ``import`` blocks, now
# removed (wyrd-lnt6 cleanup — the follow-up the import comment anticipated).
# Production has no pre-existing bucket, so terraform creates it here; the
# Lambda falls back to its bundled seed DB when the bucket carries no published
# key (runtime_db.py), so an empty bucket serves fine until a DB is published.

resource "aws_s3_bucket" "runtime_db" {
  bucket = local.runtime_db_bucket
  tags   = local.tags
}

resource "aws_s3_bucket_public_access_block" "runtime_db" {
  bucket                  = aws_s3_bucket.runtime_db.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "runtime_db" {
  bucket = aws_s3_bucket.runtime_db.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ─── IAM: Lambda execution role ─────────────────────────────────────────────

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${local.name}-${var.env}-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Read access to the runtime DB bucket so the Lambda can fetch
# current.json + the versioned L4 DB on cold start. Object-level only;
# the loader knows its target key from current.json + doesn't list.
data "aws_iam_policy_document" "runtime_db_read" {
  statement {
    actions   = ["s3:GetObject", "s3:HeadObject"]
    resources = ["${aws_s3_bucket.runtime_db.arn}/*"]
  }
  # wyrd-ow4c: ListBucket on the bucket itself. Without it, a GetObject on a
  # MISSING key returns 403 AccessDenied instead of 404 NoSuchKey — which the
  # runtime DB loader still catches + falls back on, but logs as a scary
  # AccessDenied and masks "the key just isn't there yet". With ListBucket the
  # loader sees a clean miss. (Reading a PRESENT key only needs GetObject.)
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.runtime_db.arn]
  }
}

resource "aws_iam_role_policy" "lambda_runtime_db_read" {
  name   = "${local.name}-${var.env}-runtime-db-read"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.runtime_db_read.json
}

# ─── DynamoDB: defective-name reports (wyrd-dsl5) ───────────────────────────
#
# A user flags a generated name as defective in the SPA; the Lambda writes a
# report here. The operator triage CLI (`wyrd defects list/show/accept/
# dismiss`) reads + updates these rows under an admin profile.
#
# Hash key `id` (uuid). GSI `status-created_at-index` lets the CLI pull "all
# new reports, newest first" without a table scan — status is the partition,
# created_at (ISO-8601, lexicographically sortable) the range. PAY_PER_REQUEST
# because volume is human-paced (a GM clicking a flag), not throughput-bound.
resource "aws_dynamodb_table" "defects" {
  name         = "521studios-${var.env}-wyrd-defects"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }
  attribute {
    name = "status"
    type = "S"
  }
  attribute {
    name = "created_at"
    type = "S"
  }

  global_secondary_index {
    name            = "status-created_at-index"
    hash_key        = "status"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  tags = local.tags
}

# The Lambda only WRITES reports (PutItem). Triage reads/updates happen from
# the operator CLI under an admin profile, not the function role — so the
# function gets the narrowest grant that lets a flag succeed.
data "aws_iam_policy_document" "defects_write" {
  statement {
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.defects.arn]
  }
}

resource "aws_iam_role_policy" "lambda_defects_write" {
  name   = "${local.name}-${var.env}-defects-write"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.defects_write.json
}

# ─── Lambda function ────────────────────────────────────────────────────────

resource "aws_lambda_function" "api" {
  function_name    = "${local.name}-${var.env}"
  role             = aws_iam_role.lambda.arn
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)
  handler          = "wyrd.lambda_handler.handler"
  runtime          = "python3.12"
  architectures    = ["arm64"]
  # 1769MB was the full-vCPU inflection point, bumped there so the ~3.7s
  # cold-start bundle-load (SQLite walk → bundle dict → Meaning objects →
  # NameGenerator) survived the timeout — at 512MB / ~1/6 vCPU it ballooned to
  # ~20s. SnapStart (wyrd-ocs8) now amortizes that load into the snapshot
  # (restore ~1s, bundle already resident), so the full-vCPU-for-cold-load
  # rationale is gone. What remains binding is the RAM floor: measured
  # Max Memory Used ≈ 1233MB (the deserialized bundle graph). 1536MB keeps
  # ~300MB over that floor while shrinking the SnapStart snapshot (≈ memory_size,
  # ×2 retained versions) and the per-invoke billed GB-ms; #888 made per-name
  # generation fast (~130ms) so the lost fractional CPU is well within budget.
  # The real lever is lazy-loading to cut the 1233MB floor (wyrd ticket).
  memory_size      = 1536
  timeout          = 15

  # wyrd-ocs8: SnapStart is gated behind var.enable_snapstart (DEFAULT OFF).
  # SnapStart cached a 1769MB snapshot per published version, and `publish = true`
  # + staging's per-push auto-deploy cut a new version every merge — so snapshots
  # accumulated unbounded (~300+ versions), each billing snapshot storage forever.
  # That blew the cost budget and grew daily. With the flag off, apply_on = "None"
  # stops new snapshots; cold start is ~3.7s on this 1769MB config (prod has no
  # users yet — acceptable). Only flip enable_snapstart=true once version
  # retention bounds the published-version count (keep current + prior), or the
  # snapshot cost returns. `publish = true` stays so the `live` alias + Function
  # URL routing is unchanged.
  publish = true

  snap_start {
    apply_on = var.enable_snapstart ? "PublishedVersions" : "None"
  }

  environment {
    # wyrd-0gou: SPA feature flags. Staging flips WYRD_FF_ALL=true so every
    # gated config option shows for validation; production defaults all off
    # and enables validated flags one-by-one via var.enabled_feature_flags
    # (each → WYRD_FF_<NAME>=true) and option default-value overrides via
    # var.feature_flag_defaults (each → WYRD_DEFAULT_<OPTION>). The Flask app
    # resolves these onto /api/manifest; see wyrd/feature_flags.py.
    variables = merge(
      {
        ENV                    = var.env
        WYRD_RUNTIME_DB_BUCKET = aws_s3_bucket.runtime_db.bucket
        WYRD_DEFECTS_TABLE     = aws_dynamodb_table.defects.name
        LOG_LEVEL              = var.log_level
        WYRD_FF_ALL            = var.env == "staging" ? "true" : "false"
        # wyrd-rogd.13: bare word-placement threshold (D43); load-time knob.
        WYRD_BARE_POSITION_THRESHOLD = var.bare_position_threshold
      },
      {
        # Skip "all" so a stray entry can't shadow the env-based WYRD_FF_ALL
        # conditional above (merge() is last-wins).
        for name in var.enabled_feature_flags :
        "WYRD_FF_${upper(replace(replace(name, ".", "_"), "-", "_"))}" => "true"
        if lower(name) != "all"
      },
      {
        # Normalize keys the same way as flag names so 'priors-path' /
        # 'priors.path' → WYRD_DEFAULT_PRIORS_PATH (the server lowercases the
        # suffix → 'priors_path', matching the SPA's snake_case field key).
        for opt, value in var.feature_flag_defaults :
        "WYRD_DEFAULT_${upper(replace(replace(opt, ".", "_"), "-", "_"))}" => value
      },
    )
  }

  tags = local.tags
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${aws_lambda_function.api.function_name}"
  retention_in_days = 14
  tags              = local.tags
}

# ─── SnapStart alias (wyrd-g1wp) ──────────────────────────────────────────
# SnapStart applies only to PUBLISHED versions, so the Function URL + CloudFront
# permissions invoke this alias (→ a published, snapshot-restored version)
# rather than $LATEST. terraform tracks the version published at apply time; the
# deploy then republishes with the new code + repoints this alias once the
# snapshot is Active (see .github/workflows/deploy.yml).
resource "aws_lambda_alias" "live" {
  name             = "live"
  function_name    = aws_lambda_function.api.function_name
  function_version = aws_lambda_function.api.version
}

# ─── Lambda Function URL — CloudFront uses this as the API origin via OAC ──

resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  qualifier          = aws_lambda_alias.live.name
  authorization_type = "AWS_IAM"
}

# Grants CloudFront OAC permission to invoke this Function URL.
# Both actions are required: InvokeFunctionUrl (Function URL auth layer) and
# InvokeFunction (underlying Lambda invocation).  OAC signs requests with SigV4;
# the Lambda is not publicly accessible — only reachable through CloudFront.
resource "aws_lambda_permission" "cloudfront_url" {
  statement_id  = "AllowCloudFrontInvokeFunctionUrl"
  action        = "lambda:InvokeFunctionUrl"
  function_name = aws_lambda_function.api.function_name
  qualifier     = aws_lambda_alias.live.name
  principal     = "cloudfront.amazonaws.com"
}

resource "aws_lambda_permission" "cloudfront_invoke" {
  statement_id  = "AllowCloudFrontInvokeFunction"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  qualifier     = aws_lambda_alias.live.name
  principal     = "cloudfront.amazonaws.com"
}
