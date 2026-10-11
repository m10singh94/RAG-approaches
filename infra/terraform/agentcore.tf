# Approach 3: the LangGraph agent hosted on Amazon Bedrock AgentCore Runtime (direct code deployment).
# Build the package first:  uv run python scripts/build_agent_package.py
# If build/agent_package.zip doesn't exist, everything in this file is skipped.

resource "aws_s3_bucket" "artifacts" {
  bucket        = "${local.name}-artifacts"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Versioning lets the runtime pin an exact object version, so a rebuilt zip triggers a runtime update.
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_object" "agent_package" {
  count      = local.build_agent ? 1 : 0
  bucket     = aws_s3_bucket.artifacts.id
  key        = "agent/agent_package.zip"
  source     = var.agent_package_path
  etag       = filemd5(var.agent_package_path)
  depends_on = [aws_s3_bucket_versioning.artifacts]
}

data "aws_iam_policy_document" "agent_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock-agentcore.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:bedrock-agentcore:${var.region}:${local.account_id}:*"]
    }
  }
}

resource "aws_iam_role" "agent" {
  name               = "${local.name}-agent"
  assume_role_policy = data.aws_iam_policy_document.agent_trust.json
}

data "aws_iam_policy_document" "agent" {
  statement {
    sid     = "InvokeModels"
    actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    # Geo inference profiles (au.*) route to foundation models in other Regions of the geography.
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:${var.region}:${local.account_id}:inference-profile/*",
    ]
  }
  statement {
    sid       = "RetrieveFromKb"
    actions   = ["bedrock:Retrieve"]
    resources = [aws_bedrockagent_knowledge_base.this.arn]
  }
  statement {
    sid       = "QueryDiyIndex"
    actions   = ["s3vectors:QueryVectors", "s3vectors:GetVectors"]
    resources = [aws_s3vectors_index.diy.index_arn]
  }
  statement {
    sid       = "ReadPackage"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.artifacts.arn}/agent/*"]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams", "logs:DescribeLogGroups"]
    resources = ["arn:aws:logs:${var.region}:${local.account_id}:log-group:/aws/bedrock-agentcore/runtimes/*"]
  }
  statement {
    sid       = "Observability"
    actions   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords", "xray:GetSamplingRules", "xray:GetSamplingTargets"]
    resources = ["*"]
  }
  statement {
    sid       = "Metrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["bedrock-agentcore"]
    }
  }
}

resource "aws_iam_role_policy" "agent" {
  role   = aws_iam_role.agent.id
  policy = data.aws_iam_policy_document.agent.json
}

resource "time_sleep" "agent_iam" {
  depends_on      = [aws_iam_role_policy.agent]
  create_duration = "20s"
}

resource "aws_bedrockagentcore_agent_runtime" "rag_agent" {
  count              = local.build_agent ? 1 : 0
  agent_runtime_name = replace("${var.project_name}_agent", "-", "_") # letters, digits and underscores only
  description        = "Agentic RAG (LangGraph) over the Well-Architected / Prescriptive Guidance / SRE corpus"
  role_arn           = aws_iam_role.agent.arn

  agent_runtime_artifact {
    code_configuration {
      entry_point = ["main.py"]
      runtime     = "PYTHON_3_12" # keep in sync with scripts/build_agent_package.py
      code {
        s3 {
          bucket     = aws_s3_bucket.artifacts.id
          prefix     = aws_s3_object.agent_package[0].key
          version_id = aws_s3_object.agent_package[0].version_id
        }
      }
    }
  }

  network_configuration {
    network_mode = "PUBLIC"
  }

  environment_variables = {
    RAG_REGION           = var.region
    GENERATION_MODEL_ID  = var.generation_model_id
    EMBEDDING_MODEL_ID   = var.embedding_model_id
    EMBEDDING_DIMENSIONS = tostring(var.embedding_dimensions)
    KB_ID                = aws_bedrockagent_knowledge_base.this.id
    VECTOR_BUCKET        = aws_s3vectors_vector_bucket.this.vector_bucket_name
    DIY_INDEX_NAME       = aws_s3vectors_index.diy.index_name
    AGENT_RETRIEVER      = var.agent_retriever
  }

  depends_on = [time_sleep.agent_iam]
}
