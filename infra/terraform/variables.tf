variable "region" {
  description = "AWS Region. Sydney supports Bedrock Knowledge Bases, S3 Vectors and all AgentCore features."
  type        = string
  default     = "ap-southeast-2"
}

variable "aws_profile" {
  description = "AWS CLI profile (null = default credential chain)."
  type        = string
  default     = null
}

variable "project_name" {
  type    = string
  default = "rag-approaches"
}

variable "embedding_model_id" {
  type    = string
  default = "amazon.titan-embed-text-v2:0"
}

variable "embedding_dimensions" {
  type    = number
  default = 1024
}

variable "generation_model_id" {
  description = "Claude model or inference profile id used by the agent. List options with `rag models`."
  type        = string
  default     = "au.anthropic.claude-sonnet-4-5-20250929-v1:0"
}

variable "chunk_max_tokens" {
  description = "Fixed-size chunk size for the KB; the DIY indexer reads the same value from outputs."
  type        = number
  default     = 300
}

variable "chunk_overlap_pct" {
  type    = number
  default = 20
}

variable "agent_package_path" {
  description = "Zip built by scripts/build_agent_package.py. The AgentCore runtime is skipped if it doesn't exist."
  type        = string
  default     = "../../build/agent_package.zip"
}

variable "agent_retriever" {
  description = "Which retriever the deployed agent's search tool uses: \"kb\" or \"diy\"."
  type        = string
  default     = "kb"
}
