output "backend_bucket_arn" {
  value = aws_s3_bucket.backend_bucket.arn
  description = "The ARN of the S3 bucket for the application"
}

output "backend_bucket_name" {
  value = aws_s3_bucket.backend_bucket.id
  description = "The name of the S3 bucket for the application"
}
