"""
Offline Quality Evaluation (no LLM, no labels required)

Computes a free, deterministic quality report over the generated risk profiles:

1. GROUNDING (faithfulness proxy): what fraction of each profile's
   evidence_snippets actually appear in the source Item 1A text? Ungrounded
   snippets are likely hallucinated or paraphrased — a cheap proxy for RAGAS
   faithfulness that needs no LLM judge.
2. AGGREGATE QUALITY: severity / confidence distributions, evidence counts,
   and extraction-failure counts — to spot regressions like "all high",
   "no low", or silent failures.

Run:  python3 -m src.quality_eval            # default year
      python3 -m src.quality_eval 2025
"""

from __future__ import annotations

import os
import re
import sys
import json
import glob
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import get_risk_profiles_dir, get_extracted_dir, DEFAULT_YEAR, COMPANIES


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace so trivial formatting diffs don't count
    as ungrounded. Also normalizes the quote/dash variants the extractor emits."""
    text = text.lower()
    for a, b in [("’", "'"), ("‘", "'"), ("“", '"'),
                 ("”", '"'), ("–", "-"), ("—", "-"), ("\xa0", " ")]:
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def _load_source(year: int, ticker: str) -> str | None:
    path = os.path.join(get_extracted_dir(year), f"{ticker}_item1a.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def evaluate_quality(year: int = DEFAULT_YEAR) -> dict:
    profiles_dir = get_risk_profiles_dir(year)
    # Restrict to the evaluated dataset (the active COMPANIES), so eval stats
    # stay consistent with category detection. Live-mode demo tickers (e.g.
    # GOOGL/NFLX/PLTR) have profiles on disk but are not part of the dataset.
    files = sorted(
        os.path.join(profiles_dir, f"{t}_risk_profile.json") for t in COMPANIES
        if os.path.exists(os.path.join(profiles_dir, f"{t}_risk_profile.json"))
    )
    if not files:
        print(f"No risk profiles found in {profiles_dir}")
        return {}

    sev_dist = Counter()
    conf_values = []
    evidence_counts = []
    total_snippets = 0
    grounded_snippets = 0
    failed = 0
    ungrounded_examples = []
    norm_source_cache: dict[str, str | None] = {}

    print(f"\n{'='*64}\nQUALITY REPORT (FY{year})\n{'='*64}")

    for f in files:
        prof = json.load(open(f, encoding="utf-8"))
        ticker = prof.get("company", "?")
        src = norm_source_cache.get(ticker)
        if ticker not in norm_source_cache:
            raw = _load_source(year, ticker)
            src = _normalize(raw) if raw else None
            norm_source_cache[ticker] = src

        comp_total = comp_grounded = comp_failed = 0
        for a in prof.get("risk_assessments", []):
            sev_dist[a.get("severity", "?")] += 1
            conf_values.append(a.get("confidence", 0.0))
            if a.get("extraction_failed"):
                failed += 1
                comp_failed += 1
                continue
            snippets = a.get("evidence_snippets", [])
            evidence_counts.append(len(snippets))
            for s in snippets:
                if not s:
                    continue
                total_snippets += 1
                comp_total += 1
                if src is not None and _normalize(s) in src:
                    grounded_snippets += 1
                    comp_grounded += 1
                else:
                    ungrounded_examples.append((ticker, a.get("risk_category", "?"), s[:90]))

        g = f"{comp_grounded}/{comp_total}" if comp_total else "n/a"
        flag = f"  [{comp_failed} failed]" if comp_failed else ""
        print(f"  {ticker:6s} grounded={g}{flag}")

    pct = (100 * grounded_snippets / total_snippets) if total_snippets else 0.0
    mean_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
    mean_ev = sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0

    print(f"\n--- GROUNDING (faithfulness proxy) ---")
    print(f"  Snippets grounded in source: {grounded_snippets}/{total_snippets} ({pct:.1f}%)")
    if ungrounded_examples:
        print(f"  Ungrounded examples (possible hallucination / paraphrase):")
        for tic, cat, s in ungrounded_examples[:8]:
            print(f"    - [{tic}] {cat}: \"{s}…\"")

    print(f"\n--- SEVERITY DISTRIBUTION ---")
    for level in ("negligible", "low", "medium", "high", "critical"):
        if sev_dist.get(level):
            print(f"  {level:11s}: {sev_dist[level]}")

    print(f"\n--- OTHER ---")
    print(f"  Mean confidence: {mean_conf:.2f} | Mean evidence chunks: {mean_ev:.1f}")
    print(f"  Extraction failures: {failed}")
    print(f"{'='*64}")

    report = {
        "year": year,
        "grounding_pct": round(pct, 1),
        "grounded_snippets": grounded_snippets,
        "total_snippets": total_snippets,
        "severity_distribution": dict(sev_dist),
        "mean_confidence": round(mean_conf, 3),
        "mean_evidence_chunks": round(mean_ev, 2),
        "extraction_failures": failed,
        "ungrounded_examples": [
            {"company": t, "category": c, "snippet": s} for t, c, s in ungrounded_examples
        ],
    }
    out_path = os.path.join(profiles_dir, "quality_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")
    return report


if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_YEAR
    evaluate_quality(yr)
