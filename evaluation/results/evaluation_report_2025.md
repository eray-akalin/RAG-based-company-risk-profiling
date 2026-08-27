# Evaluation Report — FY2025
_Generated: 2026-06-06_

## Retrieval Metrics (baseline ladder)
Labeled queries: 40 | Top-K: 5
(Ground truth: LLM-labeled silver set — spot-check recommended.)
Baselines: Keyword + Dense FAISS. Ablation: + cross-encoder reranker; + BM25 hybrid.

| Method | MRR | Recall@K | nDCG@K |
|---|---|---|---|
| Keyword (baseline) | 0.3958 | 0.4680 | 0.3569 |
| Dense FAISS (no rerank) | 0.5750 | 0.4064 | 0.4190 |
| Dense + Reranker | 0.8500 | 0.8487 | 0.8109 |
| Hybrid (BM25+dense) + Rerank | 0.8154 | 0.8079 | 0.7603 |

## Risk-Category Detection (accuracy + macro-F1)
Binary risk-presence over 39 company×category pairs (gold: source-verified silver).

- **Accuracy:** 0.9487
- **Macro-F1:** 0.4868

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| absent (0) | 0.0 | 0.0 | 0.0 | 0 |
| present (1) | 1.0 | 0.949 | 0.974 | 39 |

Misclassifications (error analysis):
- [AMZN] Cybersecurity Risk: gold=1 pred=0 (false_negative)
- [AMZN] Supply Chain Risk: gold=1 pred=0 (false_negative)

Excluded (extraction failed): NVDA/Operational Risk

_Note: Source-verified silver labels; all categories present in these large-cap filings._ Macro-F1 is depressed by the model's two false negatives (over-aggressive relevance gating), even though accuracy is high.

## RAGAS Metrics (LLM-judged)
Judge: `llama-3.3-70b-versatile` | Samples requested: 16
(Stronger judge than the evaluated 8B model — not circular.)

| Metric | Score | Scored |
|---|---|---|
| faithfulness | 0.6069 | 15/16 |
| llm_context_precision_without_reference | 0.8936 | 15/16 |

### Per-category (sorted by faithfulness)
| Category | n | faithfulness | context_precision |
|---|---|---|---|
| Demand / Market Risk | 2 | 0.42 | 1.00 |
| Macroeconomic Risk | 2 | 0.50 | 0.92 |
| Operational Risk | 2 | 0.54 | 0.88 |
| Cybersecurity Risk | 2 | 0.55 | 1.00 |
| Supply Chain Risk | 2 | 0.62 | 0.88 |
| IP / Technology Risk | 1 | 0.67 | 0.45 |
| Competition Risk | 2 | 0.78 | 1.00 |
| Regulatory / Legal Risk | 2 | 0.81 | 0.80 |

### Faithfulness rubric (0–2, mapped from RAGAS)
Mapping: faithfulness ≥0.8 → **2** (fully supported), 0.3–0.8 → **1** (partial), <0.3 → **0** (unsupported).

| Score | Meaning | Count |
|---|---|---|
| 2 | fully supported | 4 |
| 1 | partially supported | 9 |
| 0 | unsupported / hallucinated | 2 |
| | **mean** | **1.13** |

Low-faithfulness samples (explanation claims not fully grounded):
- [AAPL] Supply Chain Risk: faithfulness=0.25
- [AAPL] Cybersecurity Risk: faithfulness=0.4286
- [AAPL] Operational Risk: faithfulness=0.3333
- [AMD] Demand / Market Risk: faithfulness=0.3333

## Generation Quality
- **Grounding (faithfulness proxy):** 75/77 snippets verbatim in source (**97.4%**)
- **Mean confidence:** 0.65
- **Mean evidence chunks:** 1.97
- **Extraction failures:** 1

### Severity distribution
| Severity | Count |
|---|---|
| negligible | 3 |
| low | 2 |
| medium | 23 |
| high | 12 |

### Ungrounded snippets (possible paraphrase/hallucination)
- [AMD] Regulatory / Legal Risk: "We may incur costs and resources in order to comply with various new or proposed climate-r…"
- [AMD] Competition Risk: "Intel uses its microprocessor market position to price its products aggressively and targe…"
