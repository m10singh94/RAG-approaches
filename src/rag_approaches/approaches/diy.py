"""Approach 1 - DIY RAG: you own chunking, embedding, the vector index and the prompt.

Vector store is Amazon S3 Vectors: pay-per-use with no idle cluster, which keeps the cost comparison honest
at low volume. Swapping in pgvector or OpenSearch only changes `_search` and `build_index`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from rag_approaches.aws import client
from rag_approaches.chunking import chunk_text
from rag_approaches.config import get_settings, require
from rag_approaches.llm import embed, embed_many, generate
from rag_approaches.types import Chunk, RagResult, Tracer, Usage

PUT_BATCH = 200  # S3 Vectors accepts up to 500 vectors per PutVectors call


class DiyRag:
    name = "diy"

    def __init__(self) -> None:
        s = get_settings()
        self.bucket = require(s.vector_bucket, "VECTOR_BUCKET")
        self.index = s.diy_index_name

    # ---- indexing -------------------------------------------------------------------------------------
    def build_index(self) -> Usage:
        from rag_approaches.corpus import iter_documents  # local: corpus deps aren't in the AgentCore package

        s = get_settings()
        usage = Usage()
        s3v = client("s3vectors")
        for doc in iter_documents():
            pieces = chunk_text(doc.text, s.chunk_max_tokens, s.chunk_overlap_pct)
            # Resumable: batches are written in order, so if the last chunk exists the document is done.
            last_key = f"{doc.doc_id}#{len(pieces) - 1}"
            if s3v.get_vectors(vectorBucketName=self.bucket, indexName=self.index, keys=[last_key])["vectors"]:
                print(f"  skip {doc.doc_id} (already indexed)")
                continue
            for start in range(0, len(pieces), PUT_BATCH):
                batch = pieces[start : start + PUT_BATCH]
                vectors = embed_many(batch, usage)
                s3v.put_vectors(
                    vectorBucketName=self.bucket,
                    indexName=self.index,
                    vectors=[
                        {
                            "key": f"{doc.doc_id}#{start + i}",
                            "data": {"float32": vec},
                            # "text" is configured as non-filterable in Terraform (it's too big to filter on).
                            "metadata": {"source": doc.source, "title": doc.title, "doc_id": doc.doc_id, "text": text},
                        }
                        for i, (text, vec) in enumerate(zip(batch, vectors))
                    ],
                )
            print(f"  indexed {doc.doc_id}: {len(pieces)} chunks")
        return usage

    # ---- querying -------------------------------------------------------------------------------------
    def retrieve(self, query: str, usage: Usage, top_k: int, source: str | None = None) -> list[Chunk]:
        return self._search(embed(query, usage), usage, top_k, source)

    def _search(self, vector: list[float], usage: Usage, top_k: int, source: str | None = None) -> list[Chunk]:
        kwargs = {"filter": {"source": source}} if source else {}
        resp = client("s3vectors").query_vectors(
            vectorBucketName=self.bucket,
            indexName=self.index,
            queryVector={"float32": vector},
            topK=top_k,
            returnMetadata=True,
            returnDistance=True,
            **kwargs,
        )
        usage.vector_queries += 1
        return [
            Chunk(
                text=v["metadata"]["text"],
                source=v["metadata"]["source"],
                title=v["metadata"]["title"],
                doc_id=v["metadata"]["doc_id"],
                score=round(1 - v["distance"], 4),  # cosine distance -> similarity
            )
            for v in resp["vectors"]
        ]

    def query(self, question: str, top_k: int | None = None, source: str | None = None) -> RagResult:
        top_k = top_k or get_settings().top_k
        tracer, usage = Tracer(), Usage()
        with tracer.step("embed + vector search", top_k=top_k, source_filter=source) as d:
            chunks = self.retrieve(question, usage, top_k, source)
            d["hits"] = [f"{c.title} ({c.score})" for c in chunks]
        with tracer.step("generate (Converse)", model=get_settings().generation_model_id) as d:
            answer = generate(question, chunks, usage)
            d["input_tokens"], d["output_tokens"] = usage.input_tokens, usage.output_tokens
        return RagResult(self.name, question, answer, chunks, usage, tracer.events, tracer.total_ms())


class DiyFanoutRag(DiyRag):
    """DIY + one filtered search per source, merged, then a single LLM call.

    The control for the agent: single searches let the source with the most similar wording crowd out the
    others (Q1 returned 0 SRE chunks even at k=30). Fan-out guarantees every source a share, without an agent
    loop. If this matches the agent on cross-source questions, the agent's edge there is the *search
    strategy*, not agency. The trade-off: it always pays for every source, even on single-source questions.
    """

    name = "diy-fanout"
    SOURCES = ("wellarchitected", "prescriptive-guidance", "sre")

    def query(self, question: str, top_k: int | None = None, source: str | None = None) -> RagResult:
        if source:  # an explicit filter means one source, so fan-out is the same as a plain search
            return super().query(question, top_k, source)
        top_k = top_k or get_settings().top_k
        tracer, usage = Tracer(), Usage()
        with tracer.step("embed (once)") as d:
            vector = embed(question, usage)
            d["embedding_tokens"] = usage.embedding_tokens
        with (
            ThreadPoolExecutor(max_workers=len(self.SOURCES)) as pool,
            tracer.step("vector search per source", top_k_each=top_k, sources=list(self.SOURCES)) as d,
        ):
            # Throwaway Usage per thread (+= isn't atomic); count the queries once afterwards.
            per_source = list(pool.map(lambda s: self._search(vector, Usage(), top_k, s), self.SOURCES))
            usage.vector_queries += len(self.SOURCES)
            d["hits"] = {s: [f"{c.title} ({c.score})" for c in hits] for s, hits in zip(self.SOURCES, per_source)}
        # Grouped by source, best first, so the prompt reads as one block per source.
        chunks = [c for hits in per_source for c in hits]
        with tracer.step("generate (Converse)", model=get_settings().generation_model_id, chunks=len(chunks)) as d:
            answer = generate(question, chunks, usage)
            d["input_tokens"], d["output_tokens"] = usage.input_tokens, usage.output_tokens
        return RagResult(self.name, question, answer, chunks, usage, tracer.events, tracer.total_ms())
