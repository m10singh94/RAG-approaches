# Approach 2: Amazon Bedrock Knowledge Base backed by S3 Vectors.

data "aws_iam_policy_document" "kb_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:bedrock:${var.region}:${local.account_id}:knowledge-base/*"]
    }
  }
}

resource "aws_iam_role" "kb" {
  name               = "${local.name}-kb"
  assume_role_policy = data.aws_iam_policy_document.kb_trust.json
}

data "aws_iam_policy_document" "kb" {
  statement {
    sid       = "Embed"
    actions   = ["bedrock:InvokeModel"]
    resources = [local.embed_arn]
  }
  statement {
    sid       = "ReadCorpus"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.corpus.arn, "${aws_s3_bucket.corpus.arn}/kb/*"]
  }
  statement {
    sid = "VectorStore"
    actions = [
      "s3vectors:GetIndex", "s3vectors:PutVectors", "s3vectors:GetVectors",
      "s3vectors:DeleteVectors", "s3vectors:QueryVectors", "s3vectors:ListVectors",
    ]
    resources = [aws_s3vectors_index.kb.index_arn]
  }
}

resource "aws_iam_role_policy" "kb" {
  role   = aws_iam_role.kb.id
  policy = data.aws_iam_policy_document.kb.json
}

# IAM is eventually consistent; KB creation validates the role immediately and fails without a short wait.
resource "time_sleep" "kb_iam" {
  depends_on      = [aws_iam_role_policy.kb]
  create_duration = "20s"
}

resource "aws_bedrockagent_knowledge_base" "this" {
  name     = "${local.name}-kb"
  role_arn = aws_iam_role.kb.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = local.embed_arn
      embedding_model_configuration {
        bedrock_embedding_model_configuration {
          dimensions          = var.embedding_dimensions
          embedding_data_type = "FLOAT32"
        }
      }
    }
  }

  storage_configuration {
    type = "S3_VECTORS"
    s3_vectors_configuration {
      index_arn = aws_s3vectors_index.kb.index_arn
    }
  }

  depends_on = [time_sleep.kb_iam]
}

resource "aws_bedrockagent_data_source" "corpus" {
  name                 = "${local.name}-corpus"
  knowledge_base_id    = aws_bedrockagent_knowledge_base.this.id
  data_deletion_policy = "DELETE"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn         = aws_s3_bucket.corpus.arn
      inclusion_prefixes = ["kb/"]
    }
  }

  # Same chunking as the DIY indexer (src/rag_approaches/chunking.py) so retrieval is compared fairly.
  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "FIXED_SIZE"
      fixed_size_chunking_configuration {
        max_tokens         = var.chunk_max_tokens
        overlap_percentage = var.chunk_overlap_pct
      }
    }
  }
}
