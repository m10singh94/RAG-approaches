"""Chat with any approach and see what happened under the hood: retrieved chunks, step timings, tokens, cost.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from rag_approaches.approaches import APPROACHES, get_approach
from rag_approaches.config import LOGS_DIR, get_settings
from rag_approaches.pricing import cost_breakdown
from rag_approaches.types import RagResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-app")
SOURCES = {"All sources": None, "Well-Architected": "wellarchitected",
           "Prescriptive Guidance": "prescriptive-guidance", "SRE": "sre"}

st.set_page_config(page_title="RAG approaches on AWS", page_icon="🔎", layout="wide")


@st.cache_resource
def approach(name: str):
    return get_approach(name)


def log_result(result: RagResult) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    with (LOGS_DIR / f"chat-{datetime.now():%Y%m%d}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), **result.to_dict()}) + "\n")
    log.info("%s | %.0f ms | in=%d out=%d | %s", result.approach, result.latency_ms,
             result.usage.input_tokens, result.usage.output_tokens, result.question[:80])


def render_result(result: RagResult) -> None:
    st.markdown(result.answer)
    cost = cost_breakdown(result)
    u = result.usage
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latency", f"{result.latency_ms / 1000:.2f} s")
    c2.metric("LLM calls", u.llm_calls)
    c3.metric("Tokens in / out", f"{u.input_tokens:,} / {u.output_tokens:,}" + (" (est.)" if u.estimated else ""))
    c4.metric("Est. cost", f"${cost['total']:.5f}")

    with st.expander(f"Retrieved chunks ({len(result.chunks)})"):
        for i, c in enumerate(result.chunks, 1):
            st.markdown(f"**[{i}] {c.title}** · `{c.source}` · score `{c.score}`")
            st.caption(c.text[:700] + ("…" if len(c.text) > 700 else ""))
    with st.expander("Trace / logs"):
        st.dataframe(pd.DataFrame([{"step": e.step, "ms": e.elapsed_ms, "detail": json.dumps(e.detail, default=str)}
                                   for e in result.trace]), hide_index=True, use_container_width=True)
    with st.expander("Cost breakdown (USD)"):
        st.json(cost)


def run(name: str, question: str, top_k: int, source: str | None) -> RagResult | None:
    try:
        # One AgentCore session per chat: only the first message pays the ~5 s microVM cold start.
        extra = {"session_id": st.session_state.agentcore_session} if name == "agentcore" else {}
        result = approach(name).query(question, top_k=top_k, source=source, **extra)
        log_result(result)
        return result
    except Exception as exc:
        log.exception("query failed")
        st.error(f"{name}: {exc}")
        return None


# ---- sidebar ----------------------------------------------------------------------------------------------
s = get_settings()
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Chat", "Compare approaches"])
    if mode == "Chat":
        selected = [st.selectbox("Approach", list(APPROACHES), format_func=lambda k: APPROACHES[k])]
    else:
        selected = st.multiselect("Approaches", list(APPROACHES), default=["diy", "kb", "agentic"],
                                  format_func=lambda k: APPROACHES[k])
    top_k = st.slider("Top-k chunks", 1, 15, s.top_k)
    source = SOURCES[st.selectbox("Source filter", list(SOURCES))]
    st.caption(f"Region `{s.region}` · model `{s.generation_model_id}` · KB `{s.kb_id or 'not deployed'}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.agentcore_session = f"rag-approaches-chat-{uuid.uuid4()}"

st.title("RAG on AWS: DIY vs. Knowledge Bases vs. AgentCore")

# ---- chat -------------------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agentcore_session" not in st.session_state:
    st.session_state.agentcore_session = f"rag-approaches-chat-{uuid.uuid4()}"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            results = [RagResult.from_dict(r) for r in msg["results"]]
            if len(results) == 1:
                render_result(results[0])
            elif results:
                for col, r in zip(st.columns(len(results)), results):
                    with col:
                        st.subheader(r.approach)
                        render_result(r)

if question := st.chat_input("Ask about Well-Architected, Prescriptive Guidance or SRE…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        results = []
        cols = st.columns(len(selected)) if len(selected) > 1 else [st.container()]
        for col, name in zip(cols, selected):
            with col:
                if len(selected) > 1:
                    st.subheader(name)
                with st.spinner(f"{APPROACHES[name]}…"):
                    result = run(name, question, top_k, source)
                if result:
                    render_result(result)
                    results.append(result.to_dict())
    st.session_state.messages.append({"role": "assistant", "results": results})
