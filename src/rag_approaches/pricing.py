"""Per-query cost estimates from measured usage + unit prices in config/pricing.yaml."""

from __future__ import annotations

from functools import lru_cache

import yaml

from rag_approaches.config import ROOT, get_settings
from rag_approaches.types import RagResult

HOURS_PER_MONTH = 730


@lru_cache
def prices() -> dict:
    return yaml.safe_load((ROOT / "config" / "pricing.yaml").read_text(encoding="utf-8"))


def _model_prices(model_id: str) -> dict:
    p = prices()
    for key, val in p["models"].items():
        if key in model_id:
            premium = 1 + p["geo_profile_premium"] if model_id.split(".")[0] in {"au", "apac", "eu", "us", "jp"} else 1
            return {k: v * premium for k, v in val.items()}
    raise KeyError(f"No price for model {model_id!r}; add it to config/pricing.yaml")


def cost_breakdown(result: RagResult) -> dict[str, float]:
    p, s, u = prices(), get_settings(), result.usage
    m = _model_prices(s.generation_model_id)
    out = {
        "llm_input": u.input_tokens / 1e6 * m["input_per_mtok"],
        "llm_output": u.output_tokens / 1e6 * m["output_per_mtok"],
        "embedding": u.embedding_tokens / 1e3 * p["embeddings"]["titan-embed-text-v2"]["per_1k_tokens"],
        # Both the DIY index and the KB live in S3 Vectors, so a retrieval costs the same request fee either way.
        "vector_query": u.vector_queries * p["s3_vectors"]["query_per_million_requests"] / 1e6,
    }
    if result.approach == "agentcore":
        rt = p["agentcore_runtime"]
        hours = result.latency_ms / 1000 / 3600
        out["agentcore_runtime"] = hours * (
            rt["assumed_vcpu"] * rt["assumed_active_cpu_fraction"] * rt["vcpu_per_hour"]
            + rt["assumed_memory_gb"] * rt["gb_memory_per_hour"]
        )
    out["total"] = sum(out.values())
    return {k: round(v, 7) for k, v in out.items()}


def fixed_monthly(n_vectors: int, dims: int, metadata_bytes: int = 2000) -> dict[str, float]:
    """Idle cost per month: what you pay with zero queries."""
    p = prices()
    gb = n_vectors * (dims * 4 + metadata_bytes) / 1e9
    ref = p["reference_alternatives"]
    return {
        "s3_vectors_storage_per_index": round(gb * p["s3_vectors"]["storage_per_gb_month"], 4),
        "knowledge_base_service": 0.0,  # no charge for the KB itself; you pay for embeddings + vector store
        "agentcore_runtime_idle": 0.0,  # consumption-based; no charge while no session is active
        "reference_opensearch_serverless_min": round(
            ref["opensearch_serverless_min_ocus"] * ref["opensearch_serverless_per_ocu_hour"] * HOURS_PER_MONTH, 2),
    }
