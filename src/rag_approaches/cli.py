"""`rag` command line. Typical order:

    rag download && rag prepare          # public corpus -> data/processed
    (terraform apply)  &&  rag tf-outputs
    rag index-diy                        # approach 1: embed + write to S3 Vectors
    rag kb-upload && rag kb-sync         # approach 2: upload to S3 + KB ingestion job
    rag ask -a agentic "..."             # try any approach
    rag eval -a diy,kb,kb-rag,agentic,agentcore
    rag cost-model --vectors 30000
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess

from rag_approaches.config import ROOT, TF_OUTPUTS_FILE


def _tf_outputs(_args) -> None:
    tf = shutil.which("terraform") or "terraform"
    out = subprocess.run([tf, "-chdir=infra/terraform", "output", "-json"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    TF_OUTPUTS_FILE.write_text(out, encoding="utf-8")
    print(f"Wrote {TF_OUTPUTS_FILE.relative_to(ROOT)}")


def _download(args) -> None:
    from rag_approaches import corpus

    corpus.download(force=args.force)


def _prepare(_args) -> None:
    from rag_approaches import corpus

    corpus.prepare()


def _index_diy(_args) -> None:
    from rag_approaches.approaches.diy import DiyRag

    usage = DiyRag().build_index()
    print(f"Embedded {usage.embedding_tokens:,} tokens")


def _kb_upload(_args) -> None:
    from rag_approaches.approaches.knowledge_base import KnowledgeBaseRag

    print(f"Uploaded {KnowledgeBaseRag.upload_corpus()} files")


def _kb_sync(_args) -> None:
    from rag_approaches.approaches.knowledge_base import KnowledgeBaseRag

    job = KnowledgeBaseRag().sync()
    print(json.dumps({k: job.get(k) for k in ("status", "statistics", "failureReasons")}, indent=2, default=str))


def _ask(args) -> None:
    from rag_approaches.approaches import get_approach
    from rag_approaches.pricing import cost_breakdown

    result = get_approach(args.approach).query(args.question, top_k=args.top_k, source=args.source)
    print(result.answer, "\n")
    for i, c in enumerate(result.chunks, 1):
        print(f"  [{i}] {c.title}  score={c.score}")
    print()
    for e in result.trace:
        print(f"  {e.elapsed_ms:>8.0f} ms  {e.step}  {json.dumps(e.detail, default=str)[:160]}")
    print(f"\n  latency {result.latency_ms:.0f} ms | usage {result.usage} | cost {cost_breakdown(result)}")


def _eval(args) -> None:
    from rag_approaches import evaluation

    evaluation.run(args.approaches.split(","), args.ids.split(",") if args.ids else None)


def _cost_model(args) -> None:
    from rag_approaches import cost_model

    cost_model.run(args.vectors)


def _models(_args) -> None:
    """List Claude inference profiles usable in the configured Region (to pick GENERATION_MODEL_ID)."""
    from rag_approaches.aws import client

    resp = client("bedrock").list_inference_profiles(maxResults=1000)
    for p in sorted(resp["inferenceProfileSummaries"], key=lambda p: p["inferenceProfileId"]):
        if "anthropic" in p["inferenceProfileId"]:
            print(f"  {p['inferenceProfileId']:<55} {p['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("download", help="download the public corpus into data/raw")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_download)
    sub.add_parser("prepare", help="extract text into data/processed").set_defaults(func=_prepare)
    sub.add_parser("tf-outputs", help="save terraform outputs to config/").set_defaults(func=_tf_outputs)
    sub.add_parser("index-diy", help="build the DIY S3 Vectors index").set_defaults(func=_index_diy)
    sub.add_parser("kb-upload", help="upload processed corpus to the KB bucket").set_defaults(func=_kb_upload)
    sub.add_parser("kb-sync", help="run a Knowledge Base ingestion job").set_defaults(func=_kb_sync)
    sub.add_parser("models", help="list Claude inference profiles in the Region").set_defaults(func=_models)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("-a", "--approach", default="diy")
    p.add_argument("-k", "--top-k", type=int)
    p.add_argument("-s", "--source", choices=["wellarchitected", "prescriptive-guidance", "sre"])
    p.set_defaults(func=_ask)

    p = sub.add_parser("eval", help="run the eval set")
    p.add_argument("-a", "--approaches", default="diy,kb,kb-rag,agentic")
    p.add_argument("--ids", help="comma-separated question ids (default: all)")
    p.set_defaults(func=_eval)

    p = sub.add_parser("cost-model", help="project monthly cost vs. volume")
    p.add_argument("--vectors", type=int, required=True, help="number of vectors in the index")
    p.set_defaults(func=_cost_model)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
