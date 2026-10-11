"""Run every question through each approach, judge the answers with an LLM, and write results/ files.

Metrics per approach:
  correctness   1-5 from an LLM judge against the reference answer
  source_hit    did retrieval return a chunk from at least one expected source?
  refusal_ok    for unanswerable questions: did the approach decline instead of making something up?
  latency, tokens, estimated cost per query
"""

from __future__ import annotations

import json
import re
import statistics
from datetime import datetime

from rag_approaches.approaches import get_approach
from rag_approaches.aws import client
from rag_approaches.config import RESULTS_DIR, ROOT, get_settings
from rag_approaches.pricing import cost_breakdown

QUESTIONS = ROOT / "eval" / "questions.jsonl"

JUDGE_PROMPT = """You are grading a RAG system's answer against a reference answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Score correctness 1-5 (5 = covers all key points of the reference with nothing wrong; 1 = wrong or missing).
If the reference says NOT IN CORPUS, the correct behaviour is to say the information wasn't found:
score 5 if the system declined, 1 if it answered from outside knowledge.
Reply with JSON only: {{"correctness": <1-5>, "declined": <true|false>, "reason": "<one sentence>"}}"""


def load_questions(ids: list[str] | None = None) -> list[dict]:
    qs = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [q for q in qs if not ids or q["id"] in ids]


def judge(question: dict, answer: str) -> dict:
    resp = client("bedrock-runtime").converse(
        modelId=get_settings().judge_model_id,
        messages=[{"role": "user", "content": [{"text": JUDGE_PROMPT.format(
            question=question["question"], reference=question["reference_answer"], answer=answer)}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0},
    )
    text = resp["output"]["message"]["content"][0]["text"]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {"correctness": None, "declined": None, "reason": text}


def run(approaches: list[str], question_ids: list[str] | None = None) -> None:
    questions = load_questions(question_ids)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    detail_path = RESULTS_DIR / f"eval-{stamp}.jsonl"
    summary = []

    for name in approaches:
        rag = get_approach(name)
        rows = []
        for q in questions:
            source = q["expected_sources"][0] if q["category"] == "filtered" else None
            try:
                result = rag.query(q["question"], source=source)
            except Exception as exc:  # noqa: BLE001 - keep going; one throttled call shouldn't sink a 30-question run
                print(f"  [{name}] {q['id']} FAILED: {exc}")
                continue
            verdict = judge(q, result.answer)
            row = {
                "approach": name, "id": q["id"], "category": q["category"],
                "correctness": verdict.get("correctness"), "declined": verdict.get("declined"),
                "judge_reason": verdict.get("reason"),
                "source_hit": (not q["expected_sources"]) or any(c.source in q["expected_sources"] for c in result.chunks),
                "latency_ms": result.latency_ms, "cost": cost_breakdown(result), "result": result.to_dict(),
            }
            rows.append(row)
            with detail_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  [{name}] {q['id']}: correctness={row['correctness']} hit={row['source_hit']} "
                  f"{result.latency_ms:.0f}ms ${row['cost']['total']:.5f}")
        summary.append(_summarise(name, rows))

    _write_summary(summary, stamp)


def _summarise(name: str, rows: list[dict]) -> dict:
    if not rows:
        return {"approach": name, "n": 0}
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    negatives = [r for r in rows if r["category"] == "unanswerable"]
    lat = sorted(r["latency_ms"] for r in rows)
    scores = [r["correctness"] for r in answerable if r["correctness"] is not None]
    return {
        "approach": name,
        "n": len(rows),
        "correctness_mean": round(statistics.mean(scores), 2) if scores else None,
        "multi_hop_correctness": _mean([r["correctness"] for r in answerable if r["category"] == "multi-hop"]),
        "source_hit_rate": round(sum(r["source_hit"] for r in answerable) / max(1, len(answerable)), 2),
        "refusal_accuracy": round(sum(bool(r["declined"]) for r in negatives) / len(negatives), 2) if negatives else None,
        "latency_p50_ms": lat[len(lat) // 2],
        "latency_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
        "avg_input_tokens": round(statistics.mean(r["result"]["usage"]["input_tokens"] for r in rows)),
        "avg_output_tokens": round(statistics.mean(r["result"]["usage"]["output_tokens"] for r in rows)),
        "avg_llm_calls": round(statistics.mean(r["result"]["usage"]["llm_calls"] for r in rows), 2),
        "avg_cost_usd": round(statistics.mean(r["cost"]["total"] for r in rows), 6),
    }


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def _write_summary(summary: list[dict], stamp: str) -> None:
    import pandas as pd

    df = pd.DataFrame(summary)
    path = RESULTS_DIR / f"summary-{stamp}.csv"
    df.to_csv(path, index=False)
    df.to_csv(RESULTS_DIR / "summary-latest.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nWrote {path.relative_to(ROOT)}")
