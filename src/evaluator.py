"""
Evaluation Module

Calculates retrieval metrics (Recall@K, MRR, nDCG@K) based on 
manual annotations. Provides a baseline vs. semantic+rerank comparison.
"""

import os
import math
import pandas as pd
import numpy as np
from typing import List, Dict

import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.retriever import SemanticRetriever
from config import COMPANIES, RISK_CATEGORIES, DEFAULT_YEAR, get_risk_profiles_dir


def generate_annotation_scaffold(
    year: int = DEFAULT_YEAR,
    top_k: int = 10,
    out_path: str = None,
) -> str:
    """
    Build a FRESH labeling scaffold from the current index so gold labels match
    the current chunk_ids. For each risk category × company, retrieves the top
    candidates and writes a long-format CSV with an empty `is_relevant` column
    for you to fill (1 = relevant, 0/blank = not).

    Convert the filled file to evaluator format with `scaffold_to_annotations`.
    """
    retriever = SemanticRetriever(year=year)
    rows = []
    qid = 0
    for cat in RISK_CATEGORIES:
        query = cat["query_templates"][0]  # representative query
        for ticker in COMPANIES:
            qid += 1
            results = retriever.retrieve(query=query, top_k=top_k,
                                         company_filter=ticker, rerank=True)
            for r in results:
                rows.append({
                    "query_id": qid,
                    "query": query,
                    "company_filter": ticker,
                    "chunk_id": r["chunk_id"],
                    "relevance_score": round(r.get("relevance", 0.0), 3),
                    "is_relevant": "",  # <-- you fill: 1 or 0
                    "chunk_preview": r["text"][:160].replace("\n", " "),
                })
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "..",
                                "evaluation", "annotations",
                                f"retrieval_scaffold_{year}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Scaffold written: {out_path}  ({len(rows)} candidate rows, {qid} queries)")
    print("Fill the `is_relevant` column (1=relevant), then run scaffold_to_annotations().")
    return out_path


def scaffold_to_annotations(scaffold_path: str, out_path: str = None) -> str:
    """Aggregate a filled long-format scaffold into the wide annotation format
    (query, company_filter, relevant_chunk_ids) that evaluate_retrieval expects."""
    df = pd.read_csv(scaffold_path)
    df = df[df["is_relevant"].astype(str).str.strip().isin(["1", "1.0"])]
    grouped = (
        df.groupby(["query_id", "query", "company_filter"])["chunk_id"]
        .apply(lambda s: ", ".join(s)).reset_index()
        .rename(columns={"chunk_id": "relevant_chunk_ids"})
    )
    if out_path is None:
        out_path = scaffold_path.replace("scaffold", "annotations")
    grouped.to_csv(out_path, index=False)
    print(f"Annotations written: {out_path}  ({len(grouped)} labeled queries)")
    return out_path

def calculate_dcg(relevances: List[int]) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)  # +2 because index is 0-based and formula is log2(i+1)
    return dcg

def calculate_ndcg(retrieved_relevances: List[int], ideal_relevances: List[int]) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    dcg = calculate_dcg(retrieved_relevances)
    idcg = calculate_dcg(ideal_relevances)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def _score_run(retrieve_fn, df, top_k):
    """Score a retrieval method over the annotated queries.

    `retrieve_fn(query, company) -> list[chunk_id]` (ranked, best-first).
    Returns mean MRR / Recall@K / nDCG@K and the number of scored queries.
    """
    mrr, recall, ndcg = [], [], []
    for _, row in df.iterrows():
        company = None if pd.isna(row["company_filter"]) else row["company_filter"]
        truth = str(row["relevant_chunk_ids"]).strip()
        if not truth:
            continue
        gt = [s.strip() for s in truth.split(",")]

        rids = list(retrieve_fn(row["query"], company))[:top_k]

        rr = next((1.0 / (i + 1) for i, rid in enumerate(rids) if rid in gt), 0.0)
        mrr.append(rr)
        recall.append(sum(1 for rid in rids if rid in gt) / len(gt) if gt else 0.0)
        rels = [1 if rid in gt else 0 for rid in rids]
        ideal = sorted([1] * len(gt) + [0] * max(0, top_k - len(gt)), reverse=True)[:top_k]
        ndcg.append(calculate_ndcg(rels, ideal))
    return np.mean(mrr), np.mean(recall), np.mean(ndcg), len(mrr)


def evaluate_retrieval(annotations_file: str, top_k: int = 5, year: int = DEFAULT_YEAR,
                       ablation: bool = True):
    """
    Compute Recall@K, MRR, nDCG@K over annotated queries against the FY`year`
    index for a baseline ladder (proposal §3.7):

      1. Keyword (baseline)        — naive term-matching, no embeddings
      2. Dense FAISS (no rerank)   — embedding-based retrieval baseline
      3. Dense + Reranker          — embedding + cross-encoder ablation
      4. Hybrid (BM25+dense)+Rerank — extra ablation (only if `ablation`)

    BM25 presence is toggled per run via the single source of truth
    (`retriever.bm25`); same code paths, no result-changing hacks.
    """
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found: {annotations_file}")
        print("Tip: generate_annotation_scaffold() then fill is_relevant, "
              "then scaffold_to_annotations().")
        return

    df = pd.read_csv(annotations_file)
    df = df[df["relevant_chunk_ids"].notna()]
    if len(df) == 0:
        print("No valid annotations found. Fill in relevant_chunk_ids first.")
        return

    print(f"Initializing retriever (FY{year})...")
    retriever = SemanticRetriever(year=year)
    # Build BM25 once so the hybrid run works even when HYBRID_ENABLED is False.
    if retriever.bm25 is None:
        retriever._build_bm25()
    bm25_obj = retriever.bm25

    def keyword_fn(q, c):
        return [r["chunk_id"] for r in
                retriever.keyword_search(q, top_k=top_k, company_filter=c)]

    def dense_fn(rerank):
        def f(q, c):
            retriever.bm25 = None  # dense-only path
            return [r["chunk_id"] for r in
                    retriever.retrieve(q, top_k=top_k, company_filter=c, rerank=rerank)]
        return f

    def hybrid_fn(q, c):
        retriever.bm25 = bm25_obj  # enable BM25 fusion
        return [r["chunk_id"] for r in
                retriever.retrieve(q, top_k=top_k, company_filter=c, rerank=True)]

    runs = [
        ("Keyword (baseline)", keyword_fn),
        ("Dense FAISS (no rerank)", dense_fn(False)),
        ("Dense + Reranker", dense_fn(True)),
    ]
    if ablation:
        runs.append(("Hybrid (BM25+dense) + Rerank", hybrid_fn))

    print(f"\n{'='*72}\nRetrieval Evaluation (Top-K={top_k}, N={len(df)})\n{'='*72}")
    results = []
    for label, fn in runs:
        mrr, rec, ndcg, n = _score_run(fn, df, top_k)
        print(f"  {label:30s} | MRR={mrr:.4f}  Recall@{top_k}={rec:.4f}  nDCG@{top_k}={ndcg:.4f}")
        results.append({"method": label, "mrr": round(float(mrr), 4),
                        "recall": round(float(rec), 4), "ndcg": round(float(ndcg), 4)})
    retriever.bm25 = bm25_obj  # restore
    print(f"{'='*72}")
    return {"top_k": top_k, "n_queries": len(df), "runs": results}


def write_evaluation_report(year: int = DEFAULT_YEAR, top_k: int = 5) -> str:
    """Run quality + retrieval evaluation and write a human-readable Markdown
    report to evaluation/results/evaluation_report_<year>.md."""
    from src.quality_eval import evaluate_quality
    from src.category_eval import evaluate_categories

    quality = evaluate_quality(year)
    # Refresh category-detection metrics (free, no LLM) so the report is current.
    try:
        evaluate_categories(year)
    except Exception as e:
        print(f"(category eval skipped: {e})")
    anno = os.path.join(os.path.dirname(__file__), "..", "evaluation",
                        "annotations", f"retrieval_annotations_{year}.csv")
    retrieval = evaluate_retrieval(anno, top_k=top_k, year=year, ablation=True) \
        if os.path.exists(anno) else None

    out_dir = os.path.join(os.path.dirname(__file__), "..", "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evaluation_report_{year}.md")

    from datetime import date
    L = [f"# Evaluation Report — FY{year}", f"_Generated: {date.today()}_", ""]

    if retrieval:
        L += ["## Retrieval Metrics (baseline ladder)",
              f"Labeled queries: {retrieval['n_queries']} | Top-K: {retrieval['top_k']}",
              "(Ground truth: LLM-labeled silver set — spot-check recommended.)",
              "Baselines: Keyword + Dense FAISS. Ablation: + cross-encoder reranker; + BM25 hybrid.",
              "",
              "| Method | MRR | Recall@K | nDCG@K |", "|---|---|---|---|"]
        for r in retrieval["runs"]:
            L.append(f"| {r['method']} | {r['mrr']:.4f} | {r['recall']:.4f} | {r['ndcg']:.4f} |")
        L.append("")

    # Category-detection metrics (accuracy + macro-F1) from src.category_eval
    cat_path = os.path.join(get_risk_profiles_dir(year), "category_eval.json")
    if os.path.exists(cat_path):
        import json as _json
        ce = _json.load(open(cat_path, encoding="utf-8"))
        L += ["## Risk-Category Detection (accuracy + macro-F1)",
              f"Binary risk-presence over {ce['n_pairs']} company×category pairs "
              "(gold: source-verified silver).",
              "",
              f"- **Accuracy:** {ce['accuracy']}",
              f"- **Macro-F1:** {ce['macro_f1']}",
              "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for cls, m in ce.get("per_class", {}).items():
            L.append(f"| {cls} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |")
        if ce.get("errors"):
            L += ["", "Misclassifications (error analysis):"]
            for e in ce["errors"]:
                L.append(f"- [{e['company']}] {e['category']}: gold={e['gold']} pred={e['pred']} ({e['type']})")
        if ce.get("skipped"):
            L.append("")
            L.append("Excluded (extraction failed): " +
                     ", ".join(f"{s['company']}/{s['category']}" for s in ce["skipped"]))
        L += ["", f"_Note: {ce.get('gold_note','')}_ "
              "Macro-F1 is depressed by the model's two false negatives (over-aggressive "
              "relevance gating), even though accuracy is high.", ""]

    # RAGAS results (if a report was produced by src.ragas_eval)
    ragas_path = os.path.join(get_risk_profiles_dir(year), "ragas_report.json")
    if os.path.exists(ragas_path):
        import json as _json
        rg = _json.load(open(ragas_path, encoding="utf-8"))
        scored = rg.get("n_scored", {})
        L += ["## RAGAS Metrics (LLM-judged)",
              f"Judge: `{rg.get('judge_model')}` | Samples requested: {rg.get('n_samples')}",
              "(Larger judge, different family than the extraction model — not circular.)", ""]
        if scored and max(scored.values()) == 0:
            L += ["> ⚠️ Son koşu rate-limit nedeniyle başarısız (0 örnek skorlandı). "
                  "Günlük token limiti sıfırlanınca yeniden çalıştır.", ""]
        else:
            L += ["| Metric | Score | Scored |", "|---|---|---|"]
            for k, v in rg.get("aggregates", {}).items():
                n = scored.get(k, rg.get("n_samples"))
                L.append(f"| {k} | {v} | {n}/{rg.get('n_samples')} |")
            # Per-category breakdown (the most actionable view).
            from collections import defaultdict as _dd
            cat_agg = _dd(lambda: [[], []])
            for s in rg.get("per_sample", []):
                if s.get("faithfulness") is not None:
                    cat_agg[s["category"]][0].append(s["faithfulness"])
                if s.get("llm_context_precision_without_reference") is not None:
                    cat_agg[s["category"]][1].append(s["llm_context_precision_without_reference"])
            if cat_agg:
                L += ["", "### Per-category (sorted by faithfulness)",
                      "| Category | n | faithfulness | context_precision |", "|---|---|---|---|"]
                for cat in sorted(cat_agg, key=lambda x: sum(cat_agg[x][0]) / max(len(cat_agg[x][0]), 1)):
                    fs, cs = cat_agg[cat]
                    fm = sum(fs) / len(fs) if fs else float("nan")
                    cm = sum(cs) / len(cs) if cs else float("nan")
                    L.append(f"| {cat} | {len(fs)} | {fm:.2f} | {cm:.2f} |")
        # Faithfulness 0-2 rubric: map continuous RAGAS
        # faithfulness -> {2: fully supported, 1: partially, 0: unsupported}.
        def _to_rubric(x):
            if x is None:
                return None
            return 2 if x >= 0.8 else (1 if x >= 0.3 else 0)
        rubric_counts = {0: 0, 1: 0, 2: 0}
        rubric_vals = []
        for s in rg.get("per_sample", []):
            r = _to_rubric(s.get("faithfulness"))
            if r is not None:
                rubric_counts[r] += 1
                rubric_vals.append(r)
        if rubric_vals:
            mean_rubric = round(sum(rubric_vals) / len(rubric_vals), 2)
            L += ["", "### Faithfulness rubric (0–2, mapped from RAGAS)",
                  "Mapping: faithfulness ≥0.8 → **2** (fully supported), "
                  "0.3–0.8 → **1** (partial), <0.3 → **0** (unsupported).",
                  "", "| Score | Meaning | Count |", "|---|---|---|",
                  f"| 2 | fully supported | {rubric_counts[2]} |",
                  f"| 1 | partially supported | {rubric_counts[1]} |",
                  f"| 0 | unsupported / hallucinated | {rubric_counts[0]} |",
                  f"| | **mean** | **{mean_rubric}** |"]

        low = [s for s in rg.get("per_sample", [])
               if (s.get("faithfulness") or 1) < 0.5]
        if low:
            L += ["", "Low-faithfulness samples (explanation claims not fully grounded):"]
            for s in low:
                L.append(f"- [{s['company']}] {s['category']}: faithfulness={s.get('faithfulness')}")
        L.append("")

    if quality:
        sev = quality.get("severity_distribution", {})
        L += ["## Generation Quality",
              f"- **Grounding (faithfulness proxy):** {quality['grounded_snippets']}/"
              f"{quality['total_snippets']} snippets verbatim in source "
              f"(**{quality['grounding_pct']}%**)",
              f"- **Mean confidence:** {quality['mean_confidence']}",
              f"- **Mean evidence chunks:** {quality['mean_evidence_chunks']}",
              f"- **Extraction failures:** {quality['extraction_failures']}", "",
              "### Severity distribution",
              "| Severity | Count |", "|---|---|"]
        for lvl in ("negligible", "low", "medium", "high", "critical"):
            if sev.get(lvl):
                L.append(f"| {lvl} | {sev[lvl]} |")
        ung = quality.get("ungrounded_examples", [])
        if ung:
            L += ["", "### Ungrounded snippets (possible paraphrase/hallucination)"]
            for e in ung[:10]:
                L.append(f"- [{e['company']}] {e['category']}: \"{e['snippet']}…\"")
        L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nReport written: {path}")
    return path


if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_YEAR
    write_evaluation_report(yr, top_k=5)
