"""Settings resolved from (in priority order) environment variables, then Terraform outputs.

`rag tf-outputs` writes `config/terraform-outputs.json` after `terraform apply`, so locally you rarely need
to set anything except AWS_PROFILE. Inside AgentCore Runtime everything comes from environment variables
that Terraform injects.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TF_OUTPUTS_FILE = ROOT / "config" / "terraform-outputs.json"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

try:  # optional: not installed inside the AgentCore package
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _tf_outputs() -> dict:
    if not TF_OUTPUTS_FILE.exists():
        return {}
    raw = json.loads(TF_OUTPUTS_FILE.read_text(encoding="utf-8"))
    return {k: v.get("value") for k, v in raw.items()}


@dataclass(frozen=True)
class Settings:
    region: str
    profile: str | None
    generation_model_id: str
    judge_model_id: str
    embedding_model_id: str
    embedding_dimensions: int
    corpus_bucket: str | None
    vector_bucket: str | None
    diy_index_name: str
    kb_id: str | None
    kb_data_source_id: str | None
    agent_runtime_arn: str | None
    agent_retriever: str  # "kb" or "diy": which retriever the agent's search tool uses
    top_k: int
    max_tokens: int
    chunk_max_tokens: int
    chunk_overlap_pct: int


@lru_cache
def get_settings() -> Settings:
    tf = _tf_outputs()

    def pick(env: str, tf_key: str | None = None, default=None):
        return os.getenv(env) or (tf.get(tf_key) if tf_key else None) or default

    generation_model_id = pick("GENERATION_MODEL_ID", "generation_model_id", "au.anthropic.claude-sonnet-4-6")
    return Settings(
        region=pick("RAG_REGION", "region", os.getenv("AWS_REGION", "ap-southeast-2")),
        profile=os.getenv("AWS_PROFILE") or None,
        generation_model_id=generation_model_id,
        judge_model_id=pick("JUDGE_MODEL_ID", None, generation_model_id),
        embedding_model_id=pick("EMBEDDING_MODEL_ID", "embedding_model_id", "amazon.titan-embed-text-v2:0"),
        embedding_dimensions=int(pick("EMBEDDING_DIMENSIONS", "embedding_dimensions", 1024)),
        corpus_bucket=pick("CORPUS_BUCKET", "corpus_bucket"),
        vector_bucket=pick("VECTOR_BUCKET", "vector_bucket"),
        diy_index_name=pick("DIY_INDEX_NAME", "diy_index_name", "diy-index"),
        kb_id=pick("KB_ID", "kb_id"),
        kb_data_source_id=pick("KB_DATA_SOURCE_ID", "kb_data_source_id"),
        agent_runtime_arn=pick("AGENT_RUNTIME_ARN", "agent_runtime_arn"),
        agent_retriever=pick("AGENT_RETRIEVER", None, "kb"),
        top_k=int(pick("TOP_K", None, 5)),
        max_tokens=int(pick("MAX_TOKENS", None, 2048)),  # 1024 truncated multi-source agent answers
        chunk_max_tokens=int(pick("CHUNK_MAX_TOKENS", "chunk_max_tokens", 300)),
        chunk_overlap_pct=int(pick("CHUNK_OVERLAP_PCT", "chunk_overlap_pct", 20)),
    )


def require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is not configured. Run `terraform apply` then `rag tf-outputs`, or set it in .env."
        )
    return value
