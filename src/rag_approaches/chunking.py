"""Fixed-size chunking that mirrors the Knowledge Base data source config (300 tokens, 20% overlap).

Titan's tokenizer isn't public, so tokens are approximated as words / 0.75. Close enough to keep chunk sizes
comparable between the DIY index and the Knowledge Base.
"""

from __future__ import annotations

WORDS_PER_TOKEN = 0.75


def chunk_text(text: str, max_tokens: int = 300, overlap_pct: int = 20) -> list[str]:
    words = text.split()
    size = max(1, int(max_tokens * WORDS_PER_TOKEN))
    step = max(1, size - int(size * overlap_pct / 100))
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks
