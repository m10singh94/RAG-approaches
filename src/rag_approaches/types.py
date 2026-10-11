"""The shared result shape. Every approach returns a RagResult, so the eval harness and the UI never branch."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    text: str
    source: str
    title: str
    score: float | None = None
    doc_id: str | None = None
    uri: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    vector_queries: int = 0  # S3 Vectors QueryVectors calls (DIY) or KB Retrieve calls
    llm_calls: int = 0
    estimated: bool = False  # True when the API doesn't report tokens (e.g. RetrieveAndGenerate)


@dataclass
class TraceEvent:
    step: str
    elapsed_ms: float
    detail: dict = field(default_factory=dict)


@dataclass
class RagResult:
    approach: str
    question: str
    answer: str
    chunks: list[Chunk]
    usage: Usage
    trace: list[TraceEvent]
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RagResult:
        return cls(
            approach=d["approach"],
            question=d["question"],
            answer=d["answer"],
            chunks=[Chunk(**c) for c in d["chunks"]],
            usage=Usage(**d["usage"]),
            trace=[TraceEvent(**t) for t in d["trace"]],
            latency_ms=d["latency_ms"],
        )


class Tracer:
    """Times named steps; each step yields a dict the caller can fill with details for the logs panel."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._start = time.perf_counter()

    @contextmanager
    def step(self, name: str, **detail):
        t0 = time.perf_counter()
        d = dict(detail)
        try:
            yield d
        finally:
            self.events.append(TraceEvent(name, round((time.perf_counter() - t0) * 1000, 1), d))

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 1)
