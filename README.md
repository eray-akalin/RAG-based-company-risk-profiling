# Automated Risk Profiling of Public Companies Using Multi-Document RAG

A personal implementation exploring Retrieval-Augmented Generation (RAG) for financial
document intelligence, inspired by Lewis et al. (2020) and evaluated with the RAGAS
framework (Es et al., 2023).

** Live Demo:** https://rag-based-company-risk-profiling.streamlit.app/

An LLM-based financial document-intelligence system that automatically extracts, structures, and
compares company-level **risk profiles** from SEC Form 10-K filings using a multi-document
Retrieval-Augmented Generation (RAG) pipeline. For each of eight risk categories the system
retrieves evidence, prompts an instruction-tuned LLM under a strict JSON schema, and produces a
presence flag, a five-level severity, an explanation, supporting evidence snippets, and a
retrieval-grounded confidence score — surfaced through an interactive Streamlit dashboard.

---

## Pipeline Architecture

```
10-K Collection → Item 1A Extraction → Token-aware Chunking → Embedding + FAISS Index
                                                                        ↓
Dashboard + Comparison ← Risk Profile Aggregation ← LLM Structured Extraction ← Category Retrieval
```

| Component | Choice |
|-----------|--------|
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim, normalized) |
| Vector index | FAISS `IndexFlatIP` (cosine similarity) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation LLM | `qwen/qwen3.8-27b` (Groq API); local `Qwen2.5-3B-Instruct` fallback |
| Eval judge (RAGAS) | `openai/gpt-oss-120b` (Groq) |
| Chunking | 400 tokens, 80-token overlap (tiktoken), heading-aware |

---

## Dataset

FY2025 Form 10-K filings from **SEC EDGAR**, focusing on **Item 1A — Risk Factors**.
Evaluated dataset = 5 large-cap U.S. companies (233 chunks). The dashboard's live mode also
profiles arbitrary tickers on demand (e.g., GOOGL, NFLX, PLTR were profiled as demonstrations).

| Ticker | Company | Chunks |
|--------|---------|--------|
| AAPL | Apple Inc. | 31 |
| TSLA | Tesla, Inc. | 44 |
| NVDA | NVIDIA Corporation | 63 |
| AMZN | Amazon.com, Inc. | 33 |
| AMD | Advanced Micro Devices, Inc. | 62 |

> **Note:** Microsoft is excluded — its 10-K body has no matchable “Item 1A / Risk Factors”
> heading (only a TOC entry), so heading-based extraction fails. The 5–6 company scope is still met.

---

## Risk Taxonomy

1. Supply Chain Risk
2. Regulatory / Legal Risk
3. Competition Risk
4. Cybersecurity Risk
5. Demand / Market Risk
6. Macroeconomic Risk
7. Operational Risk
8. IP / Technology Risk

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
SEC_USER_AGENT=YourAppName your-email@example.com
```

`GROQ_API_KEY` powers the generation LLM and the RAGAS judge (get a free key at
https://console.groq.com). `SEC_USER_AGENT` is required by SEC EDGAR's fair-access
policy — it must carry a real contact email, or requests are rejected.

> To run fully offline, set `USE_API = False` in `config.py` to use the local Qwen2.5-3B model.

---

## Usage — Reproducing the Pipeline (FY2025)

```bash
# 1. Collect 10-K filings from SEC EDGAR
python3 -c "from src.collector import collect_10k_for_year; collect_10k_for_year(2025)"

# 2. Extract Item 1A Risk Factors
python3 -c "from src.extractor import extract_all_item_1a_for_year; extract_all_item_1a_for_year(2025)"

# 3. Token-aware chunking
python3 -c "from src.chunker import chunk_all_for_year; chunk_all_for_year(2025)"

# 4. Embedding + FAISS index
python3 -c "from src.embedder import build_index_for_year; build_index_for_year(2025)"

# 5. LLM structured risk extraction (requires GROQ_API_KEY)
python3 -c "from src.risk_extractor import RiskExtractor; e=RiskExtractor(); e.load_model(); e.load_retriever(year=2025); e.extract_all_profiles(year=2025)"

# 6. Launch the interactive dashboard locally
streamlit run app.py
```

> A hosted version is available — no setup required:
> **https://rag-based-company-risk-profiling.streamlit.app/**

**Dashboard features:** risk heatmap, top risks per company, evidence explorer, pairwise company
comparison, multi-year (FY2021–2025) selection, and **live on-demand analysis** — type any ticker
and the full pipeline runs in real time to add that company.

---

## Evaluation

```bash
# Retrieval baseline ladder (Recall@5 / MRR / nDCG@5) + category accuracy/macro-F1 + faithfulness
python3 -m src.evaluator 2025

# Generation quality (verbatim grounding %, severity distribution)
python3 -m src.quality_eval 2025

# RAGAS faithfulness + context precision (LLM-judged; uses GROQ_API_KEY)
python3 -m src.ragas_eval 2025 16
```

A consolidated, human-readable report is written to
`evaluation/results/evaluation_report_2025.md`.

### Headline Results (FY2025)

| Axis | Result |
|------|--------|
| Retrieval (Dense + Reranker) | MRR 0.85 · Recall@5 0.85 · nDCG@5 0.81 |
| Retrieval (Keyword baseline) | MRR 0.40 — reranker is the single largest contributor |
| Risk-category detection | Accuracy 0.95 · macro-F1 0.49 |
| Faithfulness | 97% snippets verbatim-grounded · RAGAS faithfulness 0.61 · context-precision 0.89 |

> **Note on generation-side numbers:** the retrieval metrics are model-independent, but the
> category-detection and faithfulness rows were measured when the generation LLM was
> `llama-3.1-8b-instant` and the judge was `llama-3.3-70b-versatile`. Groq has since retired both,
> so the pipeline now runs `qwen/qwen3.8-27b` and `openai/gpt-oss-120b`. Re-run the evaluation
> scripts to regenerate these two rows against the current models.

---

## Project Structure

```
company-risk-profiling-with-rag/
├── config.py                     # Configuration (companies, taxonomy, models, thresholds)
├── requirements.txt              # Pinned Python dependencies
├── app.py                        # Streamlit dashboard
├── .streamlit/config.toml        # Streamlit runtime settings
├── data/                         # raw / extracted / chunks / embeddings / risk_profiles (gitignored)
├── src/
│   ├── collector.py              # SEC EDGAR data collection
│   ├── extractor.py              # Item 1A section extraction
│   ├── chunker.py                # Token-aware chunking + metadata
│   ├── embedder.py               # Embedding + FAISS indexing
│   ├── retriever.py              # Category-aware retrieval (multi-query, rerank, gating)
│   ├── risk_extractor.py         # LLM-based structured risk extraction
│   ├── comparator.py             # Company + year-over-year comparison
│   ├── live_pipeline.py          # On-demand live analysis (ticker + year)
│   ├── model_cache.py            # Process-wide model caching
│   ├── evaluator.py              # Retrieval metrics + baseline ladder + report builder
│   ├── quality_eval.py           # Grounding proxy + severity/quality stats
│   ├── category_eval.py          # Category-detection accuracy + macro-F1
│   └── ragas_eval.py             # RAGAS faithfulness + context precision
├── prompts/
│   └── risk_extraction.py        # LLM prompt templates + JSON schema + few-shot
└── evaluation/
    ├── annotations/              # Retrieval gold + category gold (silver-labeled)
    └── results/                  # evaluation_report_2025.md
```

---

## AI Usage

The system architecture, risk taxonomy, evaluation design, and all key technical decisions were made
by me. An AI coding assistant was used in a supporting role only — for implementation help, suggesting
alternatives I reviewed, and drafting parts of the documentation. All AI-assisted code and text were
reviewed, tested, and validated by me.

---

## References

- [SEC EDGAR API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [FAISS](https://faiss.ai)
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
- Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.*
