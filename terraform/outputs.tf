output "lambda_function_url" {
  description = "Lambda Function URL (used as CloudFront API origin)"
  value       = aws_lambda_function_url.api.function_url
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.api.function_name
}

output "spa_bucket_name" {
  description = "wyrd SPA S3 bucket name"
  value       = aws_s3_bucket.spa.id
}

output "spa_bucket_arn" {
  description = "wyrd SPA S3 bucket ARN"
  value       = aws_s3_bucket.spa.arn
}

output "spa_bucket_regional_domain" {
  description = "wyrd SPA S3 bucket regional domain (used as CloudFront origin)"
  value       = aws_s3_bucket.spa.bucket_regional_domain_name
}

output "defects_table_name" {
  description = "DynamoDB table holding defective-name reports (wyrd-dsl5)"
  value       = aws_dynamodb_table.defects.name
}
