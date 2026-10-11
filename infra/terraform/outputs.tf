# `rag tf-outputs` saves these to config/terraform-outputs.json for the Python code.

output "region" {
  value = var.region
}

output "corpus_bucket" {
  value = aws_s3_bucket.corpus.id
}

output "vector_bucket" {
  value = aws_s3vectors_vector_bucket.this.vector_bucket_name
}

output "diy_index_name" {
  value = aws_s3vectors_index.diy.index_name
}

output "kb_id" {
  value = aws_bedrockagent_knowledge_base.this.id
}

output "kb_data_source_id" {
  value = aws_bedrockagent_data_source.corpus.data_source_id
}

output "agent_runtime_arn" {
  value = try(aws_bedrockagentcore_agent_runtime.rag_agent[0].agent_runtime_arn, null)
}

output "generation_model_id" {
  value = var.generation_model_id
}

output "embedding_model_id" {
  value = var.embedding_model_id
}

output "embedding_dimensions" {
  value = var.embedding_dimensions
}

output "chunk_max_tokens" {
  value = var.chunk_max_tokens
}

output "chunk_overlap_pct" {
  value = var.chunk_overlap_pct
}
