"""Generation and embedding calls shared by the DIY and Knowledge Base (retrieve-only) approaches.

Using one prompt and one model for both means differences in answer quality come from retrieval, not from
prompt tweaks.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from rag_approaches.aws import client
from rag_approaches.config import get_settings
from rag_approaches.types import Chunk, Usage

SYSTEM_PROMPT = """You are an assistant that answers questions about cloud architecture, migration and \
site reliability engineering.
Answer ONLY from the numbered context passages provided. Cite the passages you use inline, like [1] or [2][4].
If the passages do not contain the answer, reply exactly: "I couldn't find this in the knowledge base." \
Do not use outside knowledge."""

NOT_FOUND = "I couldn't find this in the knowledge base."


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.title})\n{c.text}" for i, c in enumerate(chunks, start=1))


def generate(question: str, chunks: list[Chunk], usage: Usage) -> str:
    s = get_settings()
    resp = client("bedrock-runtime").converse(
        modelId=s.generation_model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [{"text": f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"}],
            }
        ],
        inferenceConfig={"maxTokens": s.max_tokens, "temperature": 0},
    )
    usage.input_tokens += resp["usage"]["inputTokens"]
    usage.output_tokens += resp["usage"]["outputTokens"]
    usage.llm_calls += 1
    return "".join(block.get("text", "") for block in resp["output"]["message"]["content"])


class RateLimiter:
    """Spaces calls evenly across threads. Pacing below the quota beats bursting and relying on retries:
    new accounts get 600 Titan V2 requests/minute, and a KB ingestion job draws from the same quota."""

    def __init__(self, per_minute: int) -> None:
        self.interval = 60.0 / per_minute
        self._next = time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        time.sleep(max(0.0, slot - now))


_embed_limiter = RateLimiter(int(os.getenv("EMBED_RPM", "500")))


def embed(text: str, usage: Usage | None = None) -> list[float]:
    s = get_settings()
    _embed_limiter.wait()
    resp = client("bedrock-runtime").invoke_model(
        modelId=s.embedding_model_id,
        body=json.dumps({"inputText": text, "dimensions": s.embedding_dimensions, "normalize": True}),
    )
    body = json.loads(resp["body"].read())
    if usage is not None:
        usage.embedding_tokens += body["inputTextTokenCount"]
    return body["embedding"]


def embed_many(texts: list[str], usage: Usage, workers: int = 8) -> list[list[float]]:
    """Titan V2 embeds one text per call, so parallelise. Usage is summed after to stay thread-safe."""
    per_call = [Usage() for _ in texts]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        vectors = list(pool.map(embed, texts, per_call))
    usage.embedding_tokens += sum(u.embedding_tokens for u in per_call)
    return vectors
