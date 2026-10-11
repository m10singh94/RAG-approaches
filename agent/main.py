"""AgentCore Runtime entrypoint. Wraps the same LangGraph agent the local `agentic` approach uses.

Packaged by scripts/build_agent_package.py (deps + src/rag_approaches + this file) and deployed by Terraform
as a direct-code-deployment runtime (no container image needed).
"""

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from rag_approaches.approaches.agentic import AgenticRag

app = BedrockAgentCoreApp()
rag = AgenticRag()  # built once per container; reused across invocations in the same session


@app.entrypoint
def invoke(payload: dict) -> dict:
    result = rag.query(payload["prompt"], top_k=payload.get("top_k"), source=payload.get("source"))
    return result.to_dict()


if __name__ == "__main__":
    app.run()
