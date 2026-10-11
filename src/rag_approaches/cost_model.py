"""Project monthly cost vs. query volume from measured per-query cost (results/summary-latest.csv).

Total monthly = fixed (idle) cost + volume x measured cost per query. Produces the chart for the blog.
"""

from __future__ import annotations

import pandas as pd

from rag_approaches.config import RESULTS_DIR, ROOT, get_settings
from rag_approaches.pricing import fixed_monthly

VOLUMES = [1_000, 10_000, 100_000, 1_000_000]


def run(n_vectors: int) -> None:
    import matplotlib.pyplot as plt

    summary = pd.read_csv(RESULTS_DIR / "summary-latest.csv")
    fixed = fixed_monthly(n_vectors, get_settings().embedding_dimensions)
    index_storage = fixed["s3_vectors_storage_per_index"]

    rows = []
    for _, r in summary.iterrows():
        for v in VOLUMES:
            rows.append({"approach": r["approach"], "queries_per_month": v,
                         "monthly_usd": round(index_storage + v * r["avg_cost_usd"], 2)})
    for v in VOLUMES:  # what the same KB would cost on the "classic" OpenSearch Serverless default
        kb = summary.loc[summary["approach"] == "kb", "avg_cost_usd"]
        if not kb.empty:
            rows.append({"approach": "kb (OpenSearch Serverless store)", "queries_per_month": v,
                         "monthly_usd": round(fixed["reference_opensearch_serverless_min"] + v * kb.iloc[0], 2)})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "cost-model.csv", index=False)
    print(f"Fixed monthly costs: {fixed}\n")
    print(df.pivot(index="queries_per_month", columns="approach", values="monthly_usd").to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in df.groupby("approach"):
        ax.plot(g["queries_per_month"], g["monthly_usd"], marker="o", label=name)
    ax.set(xscale="log", yscale="log", xlabel="Queries per month", ylabel="USD per month",
           title="Monthly cost vs. query volume")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "cost-model.png"
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out.relative_to(ROOT)}")
