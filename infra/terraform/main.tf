data "aws_caller_identity" "current" {}

resource "random_id" "suffix" {
  byte_length = 3
}

locals {
  account_id  = data.aws_caller_identity.current.account_id
  name        = "${var.project_name}-${random_id.suffix.hex}"
  embed_arn   = "arn:aws:bedrock:${var.region}::foundation-model/${var.embedding_model_id}"
  build_agent = fileexists(var.agent_package_path)
}

# ---- Corpus bucket: processed text + KB metadata sidecars under kb/ -------------------------------------
resource "aws_s3_bucket" "corpus" {
  bucket        = "${local.name}-corpus"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "corpus" {
  bucket                  = aws_s3_bucket.corpus.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---- S3 Vectors: one vector bucket, one index per approach -----------------------------------------------
resource "aws_s3vectors_vector_bucket" "this" {
  vector_bucket_name = "${local.name}-vectors"
  force_destroy      = true
}

# Approach 1 (DIY): we write vectors ourselves. Chunk text is stored as non-filterable metadata.
resource "aws_s3vectors_index" "diy" {
  vector_bucket_name = aws_s3vectors_vector_bucket.this.vector_bucket_name
  index_name         = "diy-index"
  data_type          = "float32"
  dimension          = var.embedding_dimensions
  distance_metric    = "cosine"

  metadata_configuration {
    non_filterable_metadata_keys = ["text"]
  }
}

# Approach 2 (Knowledge Base): Bedrock writes vectors here during ingestion. These two keys hold the chunk
# text and Bedrock's internal metadata; they exceed the filterable-metadata size limit, so mark them non-filterable.
resource "aws_s3vectors_index" "kb" {
  vector_bucket_name = aws_s3vectors_vector_bucket.this.vector_bucket_name
  index_name         = "kb-index"
  data_type          = "float32"
  dimension          = var.embedding_dimensions
  distance_metric    = "cosine"

  metadata_configuration {
    non_filterable_metadata_keys = ["AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"]
  }
}
