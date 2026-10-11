"""Approach registry. Imports are lazy so the AgentCore package only pulls in what it needs."""

from __future__ import annotations

APPROACHES = {
    "diy": "DIY: Titan embeddings + S3 Vectors + Converse",
    "kb": "Knowledge Base: Retrieve API + same prompt as DIY",
    "kb-rag": "Knowledge Base: managed RetrieveAndGenerate",
    "agentic": "Agentic: LangGraph agent, run locally",
    "agentcore": "Agentic: same LangGraph agent on AgentCore Runtime",
}


def get_approach(name: str):
    if name == "diy":
        from rag_approaches.approaches.diy import DiyRag

        return DiyRag()
    if name in ("kb", "kb-rag"):
        from rag_approaches.approaches.knowledge_base import KnowledgeBaseRag

        return KnowledgeBaseRag(managed_generation=name == "kb-rag")
    if name == "agentic":
        from rag_approaches.approaches.agentic import AgenticRag

        return AgenticRag()
    if name == "agentcore":
        from rag_approaches.approaches.agentcore_remote import AgentCoreRag

        return AgentCoreRag()
    raise ValueError(f"Unknown approach {name!r}; choose from {', '.join(APPROACHES)}")
