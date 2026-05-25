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

variable "log_level" {
  description = "Python logging level for the Lambda runtime (DEBUG | INFO | WARNING | ERROR)"
  type        = string
  default     = "INFO"
}
