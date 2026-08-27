"""
RAGAS Evaluation (Groq llama-3.3-70b judge + local bge embeddings)

Runs RAGAS generation-side metrics on a SUBSET of the risk assessments:

- Faithfulness: are the explanation's claims grounded in the retrieved context?
  (LLM-judged; complements our free verbatim grounding proxy in quality_eval.)
- LLMContextPrecisionWithoutReference: are the retrieved chunks actually
  relevant to the question/response? (Judges our retrieval + gating quality.)

The judge is llama-3.3-70b-versatile — deliberately STRONGER than the 8B model
that produced the assessments, so the judgment isn't circular. Embeddings are
the local bge model (free). A small default sample size + serial execution keep
us within Groq free-tier rate limits.

Run:  python3 -m src.ragas_eval 2025            # default 6 samples
      python3 -m src.ragas_eval 2025 10         # 10 samples
"""

from __future__ import annotations

import os
import sys
import json
import glob
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RISK_CATEGORIES, EMBEDDING_MODEL, get_risk_profiles_dir, DEFAULT_YEAR,
    LLM_EVIDENCE_CHUNKS,
)

# Judge model (free Groq tier, stronger than the evaluated 8B model)
RAGAS_JUDGE_MODEL = "llama-3.3-70b-versatile"
DEFAULT_SAMPLE_LIMIT = 6      # keep small for free-tier rate limits
CTX_CHAR_LIMIT = 800          # truncate each context chunk to save tokens


def _category_query(cat_name: str) -> str:
    for c in RISK_CATEGORIES:
        if c["name"] == cat_name:
            return c["query_templates"][0]
    return cat_name


def _build_samples(year: int, limit: int):
    """Build RAGAS samples from saved profiles, re-retrieving contexts.

    Samples are spread across CATEGORIES (round-robin over the 8 risk
    categories, rotating companies within each) so the subset covers the whole
    taxonomy rather than piling onto the first few categories. With limit=8 you
    get ~one per category; limit=16 ~two per category.
    """
    from collections import defaultdict, deque
    from ragas import SingleTurnSample
    from src.retriever import SemanticRetriever

    retriever = SemanticRetriever(year=year)
    cat_by_name = {c["name"]: c for c in RISK_CATEGORIES}

    # Load per-company profiles (skip the combined file).
    files = sorted(
        f for f in glob.glob(os.path.join(get_risk_profiles_dir(year), "*_risk_profile.json"))
    )
    profiles = [json.load(open(f, encoding="utf-8")) for f in files]

    # Group eligible assessments by category.
    by_cat = defaultdict(list)
    for p in profiles:
        for a in p.get("risk_assessments", []):
            if a.get("is_present") and not a.get("extraction_failed") \
                    and a["risk_category"] in cat_by_name:
                by_cat[a["risk_category"]].append((p["company"], a))

    cat_order = [c["name"] for c in RISK_CATEGORIES if c["name"] in by_cat]
    queues = {name: deque(by_cat[name]) for name in cat_order}

    samples, meta = [], []
    ci = 0
    while len(samples) < limit and any(queues.values()):
        name = cat_order[ci % len(cat_order)]
        ci += 1
        q = queues[name]
        if not q:
            continue
        company, a = q.popleft()
        cat = cat_by_name[name]
        results = retriever.retrieve_for_risk_category(cat, company=company, top_k=LLM_EVIDENCE_CHUNKS)
        contexts = [r["text"][:CTX_CHAR_LIMIT] for r in results] or list(a.get("evidence_snippets", []))
        if not contexts:
            continue
        samples.append(SingleTurnSample(
            user_input=_category_query(name),
            response=a.get("explanation", ""),
            retrieved_contexts=contexts,
        ))
        meta.append({"company": company, "category": name})
    return samples, meta


def run_ragas(year: int = DEFAULT_YEAR, sample_limit: int = DEFAULT_SAMPLE_LIMIT) -> dict:
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY missing in .env — cannot run RAGAS.")
        return {}

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"Building samples (limit={sample_limit})...")
    samples, meta = _build_samples(year, sample_limit)
    if not samples:
        print("No eligible samples found.")
        return {}
    print(f"Evaluating {len(samples)} samples with judge={RAGAS_JUDGE_MODEL}")

    judge = LangchainLLMWrapper(ChatGroq(model=RAGAS_JUDGE_MODEL, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )
    # Serial + patient: free-tier rate limits make parallelism counterproductive.
    run_config = RunConfig(max_workers=1, timeout=180, max_retries=5, max_wait=60)

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), LLMContextPrecisionWithoutReference()],
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns
                   if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    aggregates = {c: round(float(df[c].mean(skipna=True)), 4) for c in metric_cols}
    valid = {c: int(df[c].notna().sum()) for c in metric_cols}

    print(f"\n{'='*60}\nRAGAS RESULTS (judge={RAGAS_JUDGE_MODEL}, N={len(samples)})\n{'='*60}")
    for c, v in aggregates.items():
        flag = "" if valid[c] == len(samples) else f"  ⚠️ only {valid[c]}/{len(samples)} scored (rest failed — likely rate limit)"
        print(f"  {c:42s}: {v}{flag}")
    print(f"{'='*60}")
    if min(valid.values(), default=0) < len(samples):
        print("WARNING: some samples failed (NaN). Aggregates cover only scored "
              "samples — re-run with a smaller N or after the rate limit resets.")

    report = {
        "year": year, "judge_model": RAGAS_JUDGE_MODEL, "n_samples": len(samples),
        "n_scored": valid, "aggregates": aggregates,
        "per_sample": [
            {**meta[i], **{c: (None if df[c].isna().iloc[i] else round(float(df[c].iloc[i]), 4))
                           for c in metric_cols}}
            for i in range(len(meta))
        ],
    }
    out = os.path.join(get_risk_profiles_dir(year), "ragas_report.json")
    if max(valid.values(), default=0) == 0:
        # Total failure (rate limit): don't overwrite a previously-good report.
        print(f"All samples failed — NOT overwriting {out} (kept previous results, if any).")
        return report
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out}")
    return report


if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_YEAR
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SAMPLE_LIMIT
    run_ragas(yr, n)
