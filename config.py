"""
Configuration and constants for the Risk Profiling RAG pipeline.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
RISK_PROFILES_DIR = os.path.join(DATA_DIR, "risk_profiles")

# ============================================================
# Multi-Year Configuration
# ============================================================
AVAILABLE_YEARS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_YEAR = 2025

def get_year_dir(base_dir: str, year: int) -> str:
    """Get the year-specific subdirectory path."""
    return os.path.join(base_dir, str(year))

def get_raw_dir(year: int) -> str:
    return get_year_dir(RAW_DIR, year)

def get_extracted_dir(year: int) -> str:
    return get_year_dir(EXTRACTED_DIR, year)

def get_chunks_dir(year: int) -> str:
    return get_year_dir(CHUNKS_DIR, year)

def get_embeddings_dir(year: int) -> str:
    return get_year_dir(EMBEDDINGS_DIR, year)

def get_risk_profiles_dir(year: int) -> str:
    return get_year_dir(RISK_PROFILES_DIR, year)

# ============================================================
# Target Companies (Ticker → Company Name)
# ============================================================
COMPANIES = {
    "AAPL": "Apple Inc.",
    # MSFT temporarily excluded: its 10-K body has no matchable "Item 1A /
    # Risk Factors" heading (only a TOC entry), so extraction needs a separate
    # structure-based approach. Re-enable once extractor handles it.
    # "MSFT": "Microsoft Corporation",
    "TSLA": "Tesla, Inc.",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
}

# ============================================================
# SEC EDGAR Configuration
# ============================================================
SEC_EDGAR_BASE_URL = "https://efts.sec.gov/LATEST"
SEC_EDGAR_FILINGS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
# SEC requires a User-Agent carrying a real name and contact email, so this is
# read from the environment rather than committed. Format: "AppName email@domain".
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "CompanyRiskProfilingRAG contact@example.com"
)

# ============================================================
# Risk Taxonomy
# ============================================================
RISK_CATEGORIES = [
    {
        "id": "supply_chain",
        "name": "Supply Chain Risk",
        "description": "Risks from supply chain disruptions, single-source supplier dependencies, raw material shortages, manufacturing concentration, and logistics bottlenecks. Look for mentions of specific suppliers, geographic concentration, component shortages, or inventory issues.",
        "query_templates": [
            "What supply chain disruptions or single-source supplier dependencies does the company face?",
            "How could component shortages or manufacturing concentration affect the company?",
            "What logistics, inventory, or raw material risks does the company disclose?",
        ],
    },
    {
        "id": "regulatory_legal",
        "name": "Regulatory / Legal Risk",
        "description": "Risks from government regulations, active legal proceedings, regulatory investigations, compliance failures, fines, and policy changes. Look for mentions of specific lawsuits, settlement amounts, regulatory agencies (FTC, SEC, DOJ), or new legislation.",
        "query_templates": [
            "What government regulations or compliance requirements pose risks to the company?",
            "What legal proceedings, lawsuits, or settlements is the company exposed to?",
            "What regulatory investigations, fines, or penalties could affect the company?",
        ],
    },
    {
        "id": "competition",
        "name": "Competition Risk",
        "description": "Risks from market competition leading to pricing pressure, loss of market share, and competitive disadvantage. Look for mentions of specific competitors, margin compression, customer switching, or market entry barriers eroding.",
        "query_templates": [
            "How does competition threaten the company's market share?",
            "What competitive pricing pressures could compress the company's margins?",
            "What risks come from new market entrants or competing products?",
        ],
    },
    {
        "id": "cybersecurity",
        "name": "Cybersecurity Risk",
        "description": "Risks from data breaches, cyberattacks, ransomware, system intrusions, and privacy violations. Look for mentions of past security incidents, breach notification costs, data protection regulations (GDPR, CCPA), or specific threat vectors.",
        "query_templates": [
            "What cybersecurity, data breach, or cyberattack risks does the company face?",
            "How could a security incident or ransomware attack affect the company?",
            "What data privacy or data protection risks does the company disclose?",
        ],
    },
    {
        "id": "demand_market",
        "name": "Demand / Market Risk",
        "description": "Risks from demand uncertainty, shifting consumer preferences, market saturation, product adoption failure, and revenue concentration. Look for customer concentration percentages, seasonal dependency, or declining product lines.",
        "query_templates": [
            "What risks come from changes in customer demand or consumer preferences?",
            "How could market saturation or customer revenue concentration affect the company?",
            "What risks relate to product adoption, seasonality, or declining demand?",
        ],
    },
    {
        "id": "macroeconomic",
        "name": "Macroeconomic Risk",
        "description": "Risks from economic downturns, inflation, interest rate changes, currency fluctuations, and geopolitical instability. Look for quantified foreign exchange exposure, specific country risks, tariff impacts, or recession scenario analysis.",
        "query_templates": [
            "How could an economic downturn or recession affect the company?",
            "What risks come from inflation, interest rate changes, or currency fluctuations?",
            "How do geopolitical instability, tariffs, or trade tensions pose risks to the company?",
        ],
    },
    {
        "id": "operational",
        "name": "Operational Risk",
        "description": "Risks from internal process failures, system outages, workforce challenges, quality control issues, and business continuity threats. Look for mentions of specific outage incidents, employee turnover rates, safety violations, or operational KPI impacts.",
        "query_templates": [
            "What operational risks such as system outages or process failures does the company face?",
            "How could workforce turnover or talent retention problems affect the company?",
            "What risks relate to quality control, safety, or business continuity?",
        ],
    },
    {
        "id": "ip_technology",
        "name": "IP / Technology Risk",
        "description": "Risks from intellectual property disputes, patent infringement claims, technology obsolescence, R&D failures, and AI/ML governance risks. Look for specific patent cases, IP litigation costs, technology migration challenges, or R&D write-offs.",
        "query_templates": [
            "What intellectual property or patent infringement risks does the company face?",
            "How could technology obsolescence or failed innovation affect the company?",
            "What risks relate to research and development, AI governance, or platform changes?",
        ],
    },
]

# ============================================================
# Chunking Configuration
# ============================================================
# NOTE: sizes are now measured in *tokens* (tiktoken / cl100k_base),
# not characters. The chunker packs whole paragraphs (semantic units)
# up to CHUNK_SIZE tokens with CHUNK_OVERLAP tokens of trailing context.
CHUNK_SIZE = 400          # max tokens per chunk
CHUNK_OVERLAP = 80        # overlap tokens between consecutive chunks
CHUNK_TOKENIZER = "cl100k_base"  # tiktoken encoding used for token counting
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # fallback separators
# Treat short, non-terminated paragraphs as risk-factor headings and start
# a fresh chunk at them so a single risk factor is not split across chunks.
CHUNK_HEADING_AWARE = True
CHUNK_HEADING_MAX_WORDS = 14

# ============================================================
# Embedding Configuration
# ============================================================
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384  # bge-small-en-v1.5 output dimension

# ============================================================
# Retrieval Configuration
# ============================================================
TOP_K = 5                 # number of chunks to retrieve per query
RETRIEVE_TOP_K = 20       # initial retrieval pool for reranking
RERANK_ENABLED = True     # enable cross-encoder reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Hybrid retrieval (BM25 lexical + dense, fused via RRF) -------------
# Disabled by default: on the FY2025 labeled eval, dense-only consistently beat
# hybrid (MRR/Recall/nDCG ~0.83 vs ~0.79) — natural-language queries + strong
# dense embeddings + cross-encoder rerank leave BM25 adding only noise.
# Flip to True to re-enable; evaluator's ablation builds BM25 regardless.
HYBRID_ENABLED = False    # fuse BM25 and dense results before reranking
BM25_TOP_K = 20           # lexical candidate pool size
RRF_K = 60                # Reciprocal Rank Fusion constant (standard default)

# --- Relevance gating ---------------------------------------------------
# A chunk's `relevance` is sigmoid(cross-encoder rerank score) in [0, 1].
# Cross-encoder scores are NOT comparable across risk categories (broad
# categories like macro/competition score far lower than focused ones like
# cybersecurity), so we gate RELATIVELY: keep chunks whose relevance is at
# least RELEVANCE_KEEP_RATIO of the best chunk for that category. This makes
# the number of returned chunks vary by evidence strength instead of a fixed
# TOP_K, without zeroing-out genuinely-present-but-diffuse risks.
RELEVANCE_KEEP_RATIO = 0.5
# Absolute noise floor: if even the best chunk is below this, the category
# has no real evidence and returns nothing.
RELEVANCE_FLOOR = 0.05
# If fewer than this many chunks survive gating, the risk is treated as not
# present (LLM is skipped, severity = negligible).
MIN_RELEVANT_CHUNKS = 1
# Severity guardrail: if even the BEST retrieved chunk for a category is weak
# (relevance below this), the evidence is too thin to justify medium/high — the
# severity is capped at "low". This only fires on genuinely weak evidence; a
# single strong chunk (e.g. relevance 0.9) is never capped, so the LLM keeps
# full freedom on real evidence.
LOW_EVIDENCE_RELEVANCE = 0.30

# ============================================================
# LLM Configuration
# ============================================================
USE_API = True            # Set to False to use local Qwen model, True for Groq API
GROQ_MODEL = "llama-3.1-8b-instant"

LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.1     # low temperature for structured output

# --- Request-size control (keep within Groq free-tier 6000 TPM limit) ---
# Chunks are ~400-480 tokens each, so all TOP_K can be sent in full: 5 chunks
# (~2.3k tokens) + few-shot + system + output ≈ 3.9k tokens, safely < 6000.
# The char cap is only a safety net for rare oversized chunks, not routine
# truncation (a normal chunk is ~1900 chars, below the cap).
LLM_EVIDENCE_CHUNKS = 5        # evidence chunks included in the prompt
LLM_EVIDENCE_CHAR_LIMIT = 2200 # per-chunk safety cap (~550 tokens)

# ============================================================
# Risk Profile JSON Schema
# ============================================================
RISK_PROFILE_SCHEMA = {
    "company": "string",
    "risk_category": "string",
    "is_present": "boolean",
    "severity": "low | medium | high",
    "explanation": "string (1-3 sentences)",
    "evidence_snippets": ["string"],
    "confidence": "float (0.0 - 1.0)",
}

# ============================================================
# Evaluation Configuration
# ============================================================
EVAL_RECALL_K = 5
EVAL_NDCG_K = 5
FAITHFULNESS_RUBRIC = {
    0: "Unsupported or hallucinated",
    1: "Partially supported by retrieved evidence",
    2: "Fully supported by retrieved evidence",
}
