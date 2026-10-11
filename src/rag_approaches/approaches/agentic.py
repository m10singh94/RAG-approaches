"""Approach 3 - Agentic RAG with LangGraph. Retrieval is a *tool*; the model decides when and how to search.

The same graph runs locally (`agentic`) and inside AgentCore Runtime (`agentcore`), so the blog can separate
"what does an agent loop cost/buy you" from "what does AgentCore hosting cost/buy you".

Graph:   START -> agent --(tool calls?)--> tools -> agent ... -> END
"""

from __future__ import annotations

import json
import time
from typing import Literal

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag_approaches.aws import client
from rag_approaches.config import get_settings
from rag_approaches.types import Chunk, RagResult, TraceEvent, Usage

AGENT_PROMPT = """You are a research assistant for cloud architecture, migration and site reliability \
engineering. Your knowledge base contains three sources:
  - "wellarchitected": the AWS Well-Architected Framework
  - "prescriptive-guidance": AWS Prescriptive Guidance (migration strategy, cloud design patterns, \
decomposing monoliths, security reference architecture, RAG options, logging/monitoring)
  - "sre": Google's Site Reliability Engineering book and SRE Workbook

Always use the search_knowledge_base tool before answering. For comparisons across sources, search each \
source separately using the `source` filter. Rephrase and search again if results look off-topic. Use at \
most {max_searches} searches.
Answer only from search results and cite document titles in square brackets, e.g. [AWS Well-Architected \
Framework]. If the knowledge base doesn't cover the question, reply: "I couldn't find this in the knowledge base."
"""

MAX_SEARCHES = 4
Source = Literal["wellarchitected", "prescriptive-guidance", "sre"]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


class AgenticRag:
    name = "agentic"

    def __init__(self) -> None:
        s = get_settings()
        if s.agent_retriever == "diy":
            from rag_approaches.approaches.diy import DiyRag

            self.retriever = DiyRag()
        else:
            from rag_approaches.approaches.knowledge_base import KnowledgeBaseRag

            self.retriever = KnowledgeBaseRag()
        self.llm = ChatBedrockConverse(
            model=s.generation_model_id,
            client=client("bedrock-runtime"),
            region_name=s.region,
            temperature=0,
            max_tokens=s.max_tokens,
        )

    def _build_graph(self, collected: list[Chunk], usage: Usage, top_k: int):
        retriever = self.retriever

        @tool
        def search_knowledge_base(query: str, source: Source | None = None) -> str:
            """Search the knowledge base. Optionally restrict to one source: "wellarchitected",
            "prescriptive-guidance" or "sre". Returns the most relevant passages with their titles."""
            chunks = retriever.retrieve(query, usage, top_k, source)
            collected.extend(chunks)
            return "\n\n".join(f"[{c.title}] (score {c.score})\n{c.text}" for c in chunks) or "No results."

        tools = [search_knowledge_base]
        llm_with_tools = self.llm.bind_tools(tools)

        def agent(state: MessagesState):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent)
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        return graph.compile()

    def query(self, question: str, top_k: int | None = None, source: str | None = None) -> RagResult:
        top_k = top_k or get_settings().top_k
        collected: list[Chunk] = []
        usage = Usage()
        graph = self._build_graph(collected, usage, top_k)
        prompt = question if not source else f"{question}\n\n(Only use the '{source}' source.)"
        messages = [SystemMessage(AGENT_PROMPT.format(max_searches=MAX_SEARCHES)), HumanMessage(prompt)]

        trace: list[TraceEvent] = []
        start = last = time.perf_counter()
        final = ""
        # stream_mode="updates" yields after every node, which gives per-step timings for the logs panel.
        for update in graph.stream({"messages": messages}, {"recursion_limit": 2 * MAX_SEARCHES + 3},
                                   stream_mode="updates"):
            now = time.perf_counter()
            for node, payload in update.items():
                for msg in payload["messages"]:
                    trace.append(TraceEvent(f"{node}", round((now - last) * 1000, 1), _describe(msg)))
                    if isinstance(msg, AIMessage):
                        meta = msg.usage_metadata or {}
                        usage.input_tokens += meta.get("input_tokens", 0)
                        usage.output_tokens += meta.get("output_tokens", 0)
                        usage.llm_calls += 1
                        if not msg.tool_calls:
                            final = _text(msg.content)
            last = now
        latency = round((time.perf_counter() - start) * 1000, 1)
        return RagResult(self.name, question, final, _dedupe(collected), usage, trace, latency)


def _describe(msg) -> dict:
    if isinstance(msg, AIMessage):
        d = {"type": "llm", **({"input_tokens": msg.usage_metadata.get("input_tokens"),
                                 "output_tokens": msg.usage_metadata.get("output_tokens")} if msg.usage_metadata else {})}
        if msg.tool_calls:
            d["tool_calls"] = [json.dumps({"tool": c["name"], **c["args"]}) for c in msg.tool_calls]
        else:
            d["final_answer_chars"] = len(_text(msg.content))
        return d
    if isinstance(msg, ToolMessage):
        return {"type": "tool_result", "tool": msg.name, "chars": len(_text(msg.content))}
    return {"type": type(msg).__name__}


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    seen, out = set(), []
    for c in chunks:
        if c.text not in seen:
            seen.add(c.text)
            out.append(c)
    return out
