"""Approach 2 - Amazon Bedrock Knowledge Bases: managed ingestion (chunk + embed + store) and retrieval.

Two modes, because the blog compares both:
  * managed_generation=False -> Retrieve API + our own prompt (apples-to-apples with DIY: only retrieval differs)
  * managed_generation=True  -> RetrieveAndGenerate (least code; you give up prompt control and token usage data)
"""

from __future__ import annotations

import time
from pathlib import Path

from rag_approaches.aws import client, model_arn
from rag_approaches.config import get_settings, require
from rag_approaches.llm import generate
from rag_approaches.types import Chunk, RagResult, Tracer, Usage

KB_PREFIX = "kb/"


class KnowledgeBaseRag:
    def __init__(self, managed_generation: bool = False) -> None:
        s = get_settings()
        self.kb_id = require(s.kb_id, "KB_ID")
        self.managed_generation = managed_generation
        self.name = "kb-rag" if managed_generation else "kb"

    # ---- ingestion ------------------------------------------------------------------------------------
    @staticmethod
    def upload_corpus() -> int:
        from rag_approaches.corpus import PROCESSED_DIR  # local: corpus deps aren't in the AgentCore package

        bucket = require(get_settings().corpus_bucket, "CORPUS_BUCKET")
        s3 = client("s3")
        files = [p for p in PROCESSED_DIR.rglob("*") if p.is_file()]
        for path in files:
            key = KB_PREFIX + path.relative_to(PROCESSED_DIR).as_posix()
            s3.upload_file(str(path), bucket, key)
        return len(files)

    def sync(self, poll_seconds: int = 15) -> dict:
        data_source_id = require(get_settings().kb_data_source_id, "KB_DATA_SOURCE_ID")
        agent = client("bedrock-agent")
        job = agent.start_ingestion_job(knowledgeBaseId=self.kb_id, dataSourceId=data_source_id)["ingestionJob"]
        while job["status"] in ("STARTING", "IN_PROGRESS"):
            time.sleep(poll_seconds)
            job = agent.get_ingestion_job(
                knowledgeBaseId=self.kb_id, dataSourceId=data_source_id, ingestionJobId=job["ingestionJobId"]
            )["ingestionJob"]
            print(f"  ingestion {job['status']}: {job.get('statistics', {})}")
        return job

    # ---- querying -------------------------------------------------------------------------------------
    def _retrieval_config(self, top_k: int, source: str | None) -> dict:
        vector_cfg: dict = {"numberOfResults": top_k}
        if source:
            vector_cfg["filter"] = {"equals": {"key": "source", "value": source}}
        return {"vectorSearchConfiguration": vector_cfg}

    def retrieve(self, query: str, usage: Usage, top_k: int, source: str | None = None) -> list[Chunk]:
        resp = client("bedrock-agent-runtime").retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration=self._retrieval_config(top_k, source),
        )
        usage.vector_queries += 1
        usage.embedding_tokens += len(query) // 4  # KB embeds the query for you (billed, but not reported)
        return [_to_chunk(r["content"]["text"], r.get("metadata", {}), r.get("location", {}), r.get("score"))
                for r in resp["retrievalResults"]]

    def query(self, question: str, top_k: int | None = None, source: str | None = None) -> RagResult:
        top_k = top_k or get_settings().top_k
        if self.managed_generation:
            return self._retrieve_and_generate(question, top_k, source)
        tracer, usage = Tracer(), Usage()
        with tracer.step("kb.Retrieve", top_k=top_k, source_filter=source) as d:
            chunks = self.retrieve(question, usage, top_k, source)
            d["hits"] = [f"{c.title} ({c.score})" for c in chunks]
        with tracer.step("generate (Converse)", model=get_settings().generation_model_id) as d:
            answer = generate(question, chunks, usage)
            d["input_tokens"], d["output_tokens"] = usage.input_tokens, usage.output_tokens
        return RagResult(self.name, question, answer, chunks, usage, tracer.events, tracer.total_ms())

    def _retrieve_and_generate(self, question: str, top_k: int, source: str | None) -> RagResult:
        tracer, usage = Tracer(), Usage(estimated=True)
        model_id = get_settings().generation_model_id
        with tracer.step("kb.RetrieveAndGenerate", model=model_id, top_k=top_k, source_filter=source) as d:
            resp = client("bedrock-agent-runtime").retrieve_and_generate(
                input={"text": question},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.kb_id,
                        "modelArn": model_arn(model_id),
                        "retrievalConfiguration": self._retrieval_config(top_k, source),
                    },
                },
            )
            chunks: list[Chunk] = []
            for citation in resp.get("citations", []):
                for ref in citation.get("retrievedReferences", []):
                    chunk = _to_chunk(ref["content"]["text"], ref.get("metadata", {}), ref.get("location", {}))
                    if all(chunk.text != c.text for c in chunks):
                        chunks.append(chunk)
            d["citations"] = len(resp.get("citations", []))
            d["note"] = "RetrieveAndGenerate does not return token usage; tokens are estimated (chars / 4)."
        answer = resp["output"]["text"]
        # The managed prompt template isn't visible either, so this is a lower bound on input tokens.
        usage.input_tokens = (len(question) + sum(len(c.text) for c in chunks)) // 4
        usage.output_tokens = len(answer) // 4
        usage.vector_queries, usage.llm_calls = 1, 1
        return RagResult(self.name, question, answer, chunks, usage, tracer.events, tracer.total_ms())


def _to_chunk(text: str, metadata: dict, location: dict, score: float | None = None) -> Chunk:
    uri = location.get("s3Location", {}).get("uri")
    return Chunk(
        text=text,
        source=metadata.get("source", "unknown"),
        title=metadata.get("title") or (Path(uri).name if uri else "unknown"),
        doc_id=metadata.get("doc_id"),
        score=round(score, 4) if score is not None else None,
        uri=uri,
    )
