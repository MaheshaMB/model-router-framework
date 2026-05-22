####################################
# Provider
####################################
terraform {
  required_providers {
    aws = {
        source = "hashicorp/aws"
        version = "4.45.0"
    }
    github = {  
        source = "hashicorp/github"
        version = "~> 6.3.0" # Check for the latest version
    }
  }
  required_version = ">=1.5.0"
}

provider "aws" {
  alias = "acm"
  region = var.aws_region
  default_tags {
    tags = {
      env = var.deploy_env
      product = var.product
      productversion = var.product_version
      customer = var.customer_name
      revenue = var.revenue_type
      requestor = var.requestor_name
      managedby = "Terraform"
    }
  }
}

provider "github" {
  token = var.pipeline_token
  owner = var.github_owner
}

########################################################################
# data: Fetch AWS account ID using aws_caller_identity
########################################################################
data "aws_caller_identity" "current" {}

########################################################################
# aws s3 bucket : This is for deploy static web page 
########################################################################
resource "aws_s3_bucket" "backend_bucket" {
  bucket = "${var.stack_name}-${data.aws_caller_identity.current.account_id}-${var.product}-backend"

  lifecycle {
    prevent_destroy = true
  }

}

resource "aws_s3_bucket_public_access_block" "backend_bucket_public_access_block" {
  bucket = aws_s3_bucket.backend_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "backend_bucket_versioning" {
  bucket = aws_s3_bucket.backend_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backend_bucket_server_side_encryption_configuration" {
  bucket = aws_s3_bucket.backend_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
