variable "env" {
  description = "Deployment environment (staging | production)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-2"
}

variable "lambda_zip_path" {
  description = "Path to the compiled Lambda zip"
  type        = string
  default     = "../function.zip"
}

variable "active_sha" {
  description = "Git SHA of the currently deployed SPA build (used as CloudFront origin_path for atomic deploys)"
  type        = string
}
