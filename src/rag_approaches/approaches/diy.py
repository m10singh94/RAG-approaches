"""Approach 1 - DIY RAG: you own chunking, embedding, the vector index and the prompt.

Vector store is Amazon S3 Vectors: pay-per-use with no idle cluster, which keeps the cost comparison honest
at low volume. Swapping in pgvector or OpenSearch only changes `_search` and `build_index`.
"""

from __future__ import annotations

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
        vector = embed(query, usage)
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
