terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
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
# The bucket + its public-access-block were initially provisioned by hand
# during the d90t cutover so the rodney crash-hunt could publish a runtime
# DB before terraform was wired up. The ``import`` blocks below adopt those
# pre-existing resources into terraform state on the next apply, after which
# this directive can be deleted in a follow-up cleanup PR (terraform supports
# ``import`` blocks as one-shot adoption sugar in 1.5+).
import {
  to = aws_s3_bucket.runtime_db
  id = local.runtime_db_bucket
}

import {
  to = aws_s3_bucket_public_access_block.runtime_db
  id = local.runtime_db_bucket
}

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
}

resource "aws_iam_role_policy" "lambda_runtime_db_read" {
  name   = "${local.name}-${var.env}-runtime-db-read"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.runtime_db_read.json
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
  # 1769MB is the Lambda inflection point where the function gets a
  # full vCPU (below that, CPU is fractional and scales linearly with
  # memory). The cold-start bundle-load (SQLite walk → bundle dict →
  # Meaning objects → NameGenerator) takes ~3.7s on a full vCPU; on
  # the previous 512MB / ~1/6 vCPU config it ballooned to ~20s and
  # hit the 10s timeout wall on every cold start. Lazy-loading is the
  # proper architectural fix (only fetch the morphemes the request
  # actually samples); until then, the memory bump is what keeps the
  # generator on the right side of the timeout budget.
  memory_size      = 1769
  timeout          = 15

  environment {
    variables = {
      ENV                    = var.env
      WYRD_RUNTIME_DB_BUCKET = aws_s3_bucket.runtime_db.bucket
      LOG_LEVEL              = var.log_level
    }
  }

  tags = local.tags
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${aws_lambda_function.api.function_name}"
  retention_in_days = 14
  tags              = local.tags
}

# ─── Lambda Function URL — CloudFront uses this as the API origin via OAC ──

resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
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
  principal     = "cloudfront.amazonaws.com"
}

resource "aws_lambda_permission" "cloudfront_invoke" {
  statement_id  = "AllowCloudFrontInvokeFunction"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "cloudfront.amazonaws.com"
}
