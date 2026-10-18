# Progress log

What has been done so far, in order, with the commands that worked. Use it to pick the work up again on any
machine. Findings for the blog itself are in [blog-outline.md](blog-outline.md).

> **Keep this file current:** update it whenever something is added to or removed from the repo or the AWS deployment.

**Docs:**
- `blog-outline.md`: committed; structure and findings.
- `blog.md`: the blog draft. **Local only, never committed** (listed in `.git/info/exclude`, git's local-only
  ignore list). Back it up separately; it isn't on GitHub.

**Status (2026-10-18):** infrastructure deployed in `ap-southeast-2`; both vector indexes loaded; all six
approaches pass a smoke test (`diy-fanout` added the same day). **Next:** check the eval questions by hand, then run `rag eval`.

---

## 1. Local tooling

| Tool | Version used | Notes |
|---|---|---|
| Python | 3.12 | |
| uv | 0.9.x | manages the venv and dependencies |
| AWS CLI | v2 (2.35+) | needed for `aws login` |
| Terraform | 1.16.5 | installed with `winget install Hashicorp.Terraform` |

On Windows, if a shell opened before the install doesn't find `terraform`, open a new terminal so it picks up
the updated PATH.

```powershell
uv sync        # creates .venv from uv.lock
```

`boto3[crt]` is a dependency on purpose: credentials from `aws login` need it.

## 2. AWS account and access

- Account type: **new AWS experience** (Google sign-up), **FREE plan**. Projects are locked to one Region:
  `ap-southeast-2` (Sydney). Check the plan with
  `aws freetier get-account-plan-state --region us-east-1 --profile default`.
- AWS CLI profile: `default`, Region `ap-southeast-2`.

```powershell
aws configure set region ap-southeast-2 --profile default
aws login --region ap-southeast-2 --profile default     # browser sign-in
aws sts get-caller-identity --profile default           # check
```

Sessions last 12 hours and refresh automatically for up to 90 days. If anything reports *"session has
expired"*, run `aws login` again.

### Bedrock model access
- The **Anthropic use-case form** has been submitted (Bedrock console → Model catalog → any Anthropic model).
- Models that work for this account:
  - generation: `au.anthropic.claude-sonnet-4-6` (default) and `au.anthropic.claude-sonnet-4-5-20250929-v1:0`
  - embeddings: `amazon.titan-embed-text-v2:0`, which runs inside Sydney
- `au.*` profiles keep inference within Australia (Sydney and Melbourne). List the options with `uv run rag models`.
- Titan V2 quota: **600 requests/minute**. The DIY indexer paces itself to 500 (`EMBED_RPM`).

## 3. AWS Agent Toolkit (for Claude Code)

```powershell
aws configure agent-toolkit --yes --region us-east-1 --profile default    # interactive wizard
aws agent-toolkit list-available-skills --region us-east-1 --profile default
claude mcp remove aws-mcp -s user
claude mcp add aws-mcp -s user -e AWS_MCP_PROXY_PROFILES=default -- uvx mcp-proxy-for-aws@latest https://aws-mcp.us-east-1.api.aws/mcp --metadata INSTALL_SOURCE=aws-cli
claude mcp get aws-mcp      # should report "Connected"
```

The AWS rules for the new AWS experience are in `CLAUDE.md`, between the `BEGIN/END AWS Agent Toolkit rules`
markers. Help level is set to HIGH.

## 4. Corpus (public documents only)

```powershell
uv run rag download     # -> data/raw       (gitignored, never committed)
uv run rag prepare      # -> data/processed (.txt + .metadata.json sidecars)
```

Result: **79 documents**: the Well-Architected Framework, 8 Prescriptive Guidance guides, and the SRE book and
workbook chapters. The sources are listed in `corpus/manifest.yaml`. The Phoenix Project and TOGAF are excluded
because they aren't openly licensed.

## 5. Infrastructure (Terraform)

```powershell
uv run python scripts/build_agent_package.py          # build/agent_package.zip (ARM64 wheels, ~43 MB)
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform plan -out=tfplan     # review: expect "19 to add, 0 to change, 0 to destroy"
terraform -chdir=infra/terraform apply tfplan
uv run rag tf-outputs                                 # -> config/terraform-outputs.json (gitignored)
```

Created: corpus and artifact S3 buckets, an S3 Vectors bucket with `diy-index` and `kb-index`, the Knowledge
Base and its S3 data source (fixed-size chunks of 300 tokens with 20% overlap), the AgentCore runtime (direct code
deployment, Python 3.12), and two IAM roles. All names share one random suffix; read the actual names and IDs
from `config/terraform-outputs.json`.

Terraform uses the `aws login` session directly; no extra credential setup is needed. State is local
(`infra/terraform/terraform.tfstate`, gitignored).

### Updating the deployed agent after code changes
```powershell
uv run python scripts/build_agent_package.py
terraform -chdir=infra/terraform plan -out=tfplan     # expect "0 to add, 2 to change": the zip + the runtime
terraform -chdir=infra/terraform apply tfplan
```

## 6. Loading the indexes

Run these **one after the other**, not in parallel: both draw on the same Titan quota.

```powershell
uv run rag kb-upload     # uploads 158 files (79 docs + 79 metadata) to s3://<corpus_bucket>/kb/
uv run rag kb-sync       # KB ingestion job -> COMPLETE, 79 indexed, 0 failed (~8 min)
uv run rag index-diy     # ~4,190 chunks, ~956k embedding tokens (~$0.02), ~9 min at 500 req/min
```

`index-diy` is resumable: it skips documents whose last chunk is already in the index.

Vector counts after loading: `kb-index` 4,492 and `diy-index` about 4,190. They differ because Bedrock's
tokenizer splits chunks differently from our word-based approximation.

## 7. Smoke tests (one question per approach)

```powershell
uv run rag ask -a diy       "What are the four golden signals of monitoring?"
uv run rag ask -a kb        "What are the four golden signals of monitoring?"
uv run rag ask -a kb-rag    "What are the 7 Rs of migration strategies?"
uv run rag ask -a agentic   "Compare how the AWS Well-Architected Framework and the Google SRE book approach testing disaster recovery."
uv run rag ask -a agentcore "What are the four golden signals of monitoring?"
uv run rag ask -a diy-fanout "How do the AWS Well-Architected Framework and the Google SRE book each recommend learning from incidents?"
```

Good test questions (cross-source, false-premise, near-miss) and the volume-vs-targeting experiment are
written up in `blog-outline.md`.

| Approach | Latency | Cost/query | Observation |
|---|---|---|---|
| `diy` | 3.5 s | ~$0.009 | correct |
| `kb` | 4.0 s | ~$0.010 | correct |
| `kb-rag` | 3.9 s | ~$0.007 (estimated) | correct; no scores or token usage returned |
| `agentic` | 14.3 s | ~$0.035 | made two parallel, source-filtered searches |
| `agentcore` | ~6.5 s warm / ~11 s cold | ~$0.020 | session cold start ~4.9 s, warm ~0.1 s |
| `diy-fanout` | ~7 s | ~$0.022 | one search per source; covers both sources on cross-source Q1 |

The Streamlit app runs locally and keeps one AgentCore session per chat:
```powershell
uv run streamlit run app/streamlit_app.py
```

### Cross-source experiments (retrieval only, ~free)
- A single search for a Well-Architected + SRE question returned **0 SRE chunks even at k=30**; one search per
  source fixed coverage. This led to the `diy-fanout` approach (one filtered search per source, one LLM call).
- Recommended questions for manual testing (cross-source, false-premise, near-miss, not-in-corpus) are listed in
  `blog-outline.md` and `blog.md`. Throwaway probe scripts were not kept in the repo.

### Cost and credit checks
```powershell
aws freetier get-account-plan-state --region us-east-1 --profile default   # remaining credits
```
- Idle cost of the whole deployment: **under $0.01/week** (S3 + S3 Vectors storage; KB and AgentCore cost nothing idle).
- Per-question costs: ~$0.007 (`kb-rag`) to ~$0.06 (`agentic`); one 6-approach compare in Streamlit is ~$0.12.
- Cost Explorer is **not enabled** on the account. Enable it in Billing and Cost Management to see actual spend
  (data appears after ~24 h).

## 8. Settings in effect

| Setting | Value | Where |
|---|---|---|
| Region | `ap-southeast-2` | `variables.tf`, `.env.example` |
| Generation model | `au.anthropic.claude-sonnet-4-6` | `config.py`, `variables.tf` |
| Embeddings | Titan V2, 1024 dims, normalized | `variables.tf` |
| Chunking | fixed 300 tokens, 20% overlap (KB and DIY) | `variables.tf`, `chunking.py` |
| top-k | 5 | `config.py` (`TOP_K`) |
| max_tokens | 2048 | `config.py` (`MAX_TOKENS`) |
| Embedding pacing | 500 req/min | `llm.py` (`EMBED_RPM`) |
| Agent retriever | Knowledge Base | `variables.tf` (`agent_retriever`) |

## 9. Next steps

0. Optional, recommended: set a spend limit in AWS Settings → Billing and enable Cost Explorer.
1. Check questions by hand (Streamlit "Compare approaches" mode). Review the reference answers in
   `eval/questions.jsonl`, since the judge scores against them.
2. Eval design for AgentCore: warm up once, reuse a session, and report cold-start latency separately.
3. Verify `config/pricing.yaml` against the Sydney pricing pages (several values are us-east-1 drafts).
4. Run the eval: `uv run rag eval -a diy,diy-fanout,kb,kb-rag,agentic,agentcore`, then `uv run rag cost-model --vectors 4500`.
5. Possible follow-up: larger, hierarchical or semantic chunking (the "errors" golden signal fell outside the top 5).

## 10. Teardown (when finished)

```powershell
terraform -chdir=infra/terraform destroy
```

While deployed, the idle cost is a few cents a month: S3 and S3 Vectors storage plus logs. AgentCore charges
only while a request is being handled.
