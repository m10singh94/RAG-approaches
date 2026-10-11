"""Invoke the agent deployed on Amazon Bedrock AgentCore Runtime (see agent/main.py and infra/terraform)."""

from __future__ import annotations

import json
import time
import uuid

from rag_approaches.aws import client
from rag_approaches.config import get_settings, require
from rag_approaches.types import RagResult, TraceEvent


class AgentCoreRag:
    name = "agentcore"

    def __init__(self) -> None:
        self.runtime_arn = require(get_settings().agent_runtime_arn, "AGENT_RUNTIME_ARN")

    def query(self, question: str, top_k: int | None = None, source: str | None = None,
              session_id: str | None = None) -> RagResult:
        start = time.perf_counter()
        resp = client("bedrock-agentcore").invoke_agent_runtime(
            agentRuntimeArn=self.runtime_arn,
            runtimeSessionId=session_id or f"rag-approaches-{uuid.uuid4()}",  # must be >= 33 chars
            payload=json.dumps({"prompt": question, "top_k": top_k, "source": source}).encode(),
        )
        data = json.loads(resp["response"].read())
        round_trip = round((time.perf_counter() - start) * 1000, 1)

        result = RagResult.from_dict(data)
        result.approach = self.name
        # Everything beyond the agent's own runtime is AgentCore overhead: network, auth, (cold) start.
        result.trace.insert(0, TraceEvent("invoke_agent_runtime (round trip)", round_trip, {
            "agent_latency_ms": result.latency_ms,
            "overhead_ms": round(round_trip - result.latency_ms, 1),
            "session_id": resp.get("runtimeSessionId"),
        }))
        result.latency_ms = round_trip
        return result
