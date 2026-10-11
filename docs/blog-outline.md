# Blog outline: DIY RAG vs. Bedrock Knowledge Bases vs. agentic RAG on AgentCore

Working title: **"Three ways to do RAG on AWS, and why they aren't really three alternatives"**

## 1. Hook (about 150 words)
- Everyone asks "Knowledge Bases or AgentCore?". They solve different layers: retrieval vs. orchestration.
- What we built: one corpus (Well-Architected, Prescriptive Guidance, SRE book), 31 questions, five configurations, measured results.

## 2. The three approaches in one picture
- The mermaid diagram from the README. Explain "you own / AWS owns" for each.
- Snippets:
  - DIY retrieval: `DiyRag.retrieve` (embed, then `query_vectors`): `src/rag_approaches/approaches/diy.py`
  - KB retrieval: `KnowledgeBaseRag.retrieve` (one `retrieve` call), then `_retrieve_and_generate`
  - Agent: the `search_knowledge_base` tool and the 5-line graph in `agentic.py`
  - AgentCore: `agent/main.py`, about 10 lines wrapping the same graph

## 3. Setup and fairness
- Corpus and licensing (why Phoenix Project and TOGAF are out, and how that became the "unanswerable" test).
- Same chunks, model and prompt, and the same vector store (S3 Vectors) for DIY and KB.
- Eval method: LLM judge (1-5), source hit rate, refusal accuracy, p50/p95 latency, tokens, cost.

## 4. Infrastructure as code
- Terraform for everything, including `aws_bedrockagentcore_agent_runtime` with direct code deployment (no Docker).
- Gotchas: S3 Vectors non-filterable metadata keys for the KB index; IAM propagation delay; agent runtime names
  allow only underscores; ARM64 wheels for the agent package.

## 5. Results (fill in from results/summary-latest.csv)
| Approach | Correctness | Multi-hop | Source hit | Refusal acc. | p50 | p95 | LLM calls | $/query |
|---|---|---|---|---|---|---|---|---|
| diy | | | | | | | | |
| kb | | | | | | | | |
| kb-rag | | | | | | | | |
| agentic (local) | | | | | | | | |
| agentcore | | | | | | | | |

Things to look for:
- `diy` vs. `kb`: same prompt, so any gap is retrieval and chunking implementation.
- `kb` vs. `kb-rag`: what you give up (prompt control, token visibility) for zero generation code.
- single-hop vs. multi-hop: where the agent earns its extra LLM calls.
- `agentic` vs. `agentcore`: pure hosting overhead (cold start, network). The trace shows `overhead_ms`.

## 6. Cost
- Fixed vs. variable split; `results/cost-model.png` (monthly cost vs. queries per month, log-log).
- The vector store dominates idle cost: S3 Vectors (~0) vs. OpenSearch Serverless minimum OCUs.
- The agent multiplies LLM tokens; AgentCore Runtime bills active CPU and memory-time, not idle.
- RetrieveAndGenerate doesn't report token usage, which makes cost attribution harder.
- Put "prices as of <date>, ap-southeast-2" on every number.

## 7. Decision guide
- **DIY:** custom chunking or hybrid search or reranking, multi-tenant filtering, an existing vector DB, cost tuning at scale.
- **Knowledge Bases:** fastest path to production RAG over S3, Confluence, SharePoint or web content; standard chunking is fine.
- **Agentic on AgentCore:** multi-step or comparative questions, retrieval as one tool among many, need for sessions,
  memory, identity and observability. Not for simple FAQ lookups.
- **Hybrid (common in practice):** a KB as the retrieval tool inside an AgentCore agent, which is what this repo deploys.

## 8. Demo
- Streamlit "Compare approaches" mode: one question, three columns, trace and cost under each. GIF or screenshots.

## 9. Wrap-up and repo link
