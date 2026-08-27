"""
Text Chunker with Metadata

Splits extracted Item 1A Risk Factors text into overlapping chunks
and attaches metadata (company, year, section, chunk_id).
"""

from __future__ import annotations

import os
import re
import json
import glob
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    COMPANIES, EXTRACTED_DIR, CHUNKS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_TOKENIZER,
    CHUNK_HEADING_AWARE, CHUNK_HEADING_MAX_WORDS,
    get_chunks_dir, get_extracted_dir,
)

# ------------------------------------------------------------------
# Token counting (tiktoken). Falls back to a char-based estimate if
# tiktoken is unavailable, so the pipeline never hard-fails.
# ------------------------------------------------------------------
_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding(CHUNK_TOKENIZER)
        except Exception:  # pragma: no cover - defensive fallback
            _ENCODER = False  # sentinel: tiktoken unavailable
    return _ENCODER


def count_tokens(text: str) -> int:
    """Count tokens in `text`. Falls back to ~4 chars/token if needed."""
    enc = _get_encoder()
    if enc is False:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on terminal punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _hard_token_split(text: str, chunk_size: int) -> list[str]:
    """Last-resort split of an oversized unit by raw token windows."""
    enc = _get_encoder()
    if enc is False:
        # char-based approximation (~4 chars/token)
        step = chunk_size * 4
        return [text[i:i + step].strip() for i in range(0, len(text), step)]
    toks = enc.encode(text)
    return [
        enc.decode(toks[i:i + chunk_size]).strip()
        for i in range(0, len(toks), chunk_size)
    ]


def _looks_like_heading(paragraph: str) -> bool:
    """
    Heuristic: a short paragraph that is NOT terminated with sentence
    punctuation is likely a risk-factor heading/sub-heading. Plain-text
    SEC extracts lose bold formatting, so this is a best-effort signal.
    """
    p = paragraph.strip()
    if not p or "\n" in p:
        return False
    if len(p.split()) > CHUNK_HEADING_MAX_WORDS:
        return False
    return not p.rstrip().endswith((".", "!", "?", ":", ";", ","))


def _make_semantic_units(text: str, chunk_size: int) -> list[str]:
    """
    Break text into paragraph-level units, never exceeding chunk_size.
    Oversized paragraphs are split into sentence groups (then token windows).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if count_tokens(para) <= chunk_size:
            units.append(para)
            continue
        # Paragraph too big: pack sentences greedily.
        current = ""
        for sent in _split_sentences(para):
            candidate = f"{current} {sent}".strip() if current else sent
            if count_tokens(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    units.append(current)
                if count_tokens(sent) > chunk_size:
                    units.extend(_hard_token_split(sent, chunk_size))
                    current = ""
                else:
                    current = sent
        if current:
            units.append(current)
    return units


def _tail_tokens(text: str, overlap_tokens: int) -> str:
    """
    Return the trailing `overlap_tokens` tokens of `text` as a string.

    Used to carry a small, bounded slice of the previous chunk into the next
    one. (The old unit-based approach pulled an entire ~chunk_size unit as
    "overlap", which doubled chunk sizes.)
    """
    if overlap_tokens <= 0 or not text:
        return ""
    enc = _get_encoder()
    if enc is False:
        approx = overlap_tokens * 4  # ~4 chars/token fallback
        return text[-approx:] if len(text) > approx else text
    toks = enc.encode(text)
    if len(toks) <= overlap_tokens:
        return text
    return enc.decode(toks[-overlap_tokens:])


def split_text_recursive(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: list = None,  # kept for backward compatibility (unused)
) -> list[str]:
    """
    Token-aware, paragraph/heading-aware chunker.

    - Sizes are measured in tokens (tiktoken), not characters.
    - Whole paragraphs are kept intact and packed greedily up to
      `chunk_size` tokens, preserving semantic boundaries.
    - Consecutive chunks share `chunk_overlap` tokens of trailing context.
    - When CHUNK_HEADING_AWARE is on, a heading-like paragraph forces a
      fresh chunk so a single risk factor stays together.

    Args:
        text: The text to split
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Token overlap between consecutive chunks
        separators: Ignored; retained so existing callers don't break.

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    if count_tokens(text) <= chunk_size:
        return [text.strip()]

    units = _make_semantic_units(text, chunk_size)

    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        force_boundary = (
            CHUNK_HEADING_AWARE
            and current_units
            and _looks_like_heading(unit)
        )

        if force_boundary:
            chunks.append("\n\n".join(current_units).strip())
            # Start a clean section at the heading (no overlap across boundary).
            current_units = [unit]
            current_tokens = unit_tokens
            continue

        if current_units and current_tokens + unit_tokens > chunk_size:
            chunk_text = "\n\n".join(current_units).strip()
            chunks.append(chunk_text)
            # Carry only a small token-bounded tail into the next chunk.
            overlap_text = _tail_tokens(chunk_text, chunk_overlap)
            current_units = ([overlap_text] if overlap_text else []) + [unit]
            current_tokens = sum(count_tokens(u) for u in current_units)
        else:
            current_units.append(unit)
            current_tokens += unit_tokens

    if current_units:
        chunks.append("\n\n".join(current_units).strip())

    return [c for c in chunks if c]


def extract_year_from_filename(filename: str) -> str:
    """Extract the fiscal year from the extracted text filename."""
    # Filenames like AAPL_item1a.txt — we'll check collection metadata
    return "2025"  # Will be overridden if metadata is available


def chunk_single_company(
    ticker: str,
    company_name: str,
    text_path: str,
    year: str = "2025",
) -> list[dict]:
    """
    Chunk a single company's Item 1A text and attach metadata.

    Returns:
        List of chunk dicts with metadata
    """
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"  Text length: {len(text):,} chars")

    # Split into chunks
    raw_chunks = split_text_recursive(text)
    print(f"  Generated {len(raw_chunks)} chunks")

    # Create chunk objects with metadata
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk = {
            "chunk_id": f"{ticker}_{year}_item1a_{i:04d}",
            "company": ticker,
            "company_name": company_name,
            "year": year,
            "section": "Item 1A - Risk Factors",
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "text": chunk_text,
            "char_count": len(chunk_text),
        }
        chunks.append(chunk)

    return chunks


def chunk_all_companies() -> list[dict]:
    """
    Chunk all extracted Item 1A texts.
    Returns a combined list of all chunks with metadata.
    """
    all_chunks = []

    # Try to load collection metadata for filing dates
    filing_years = {}
    meta_paths = [
        os.path.join(EXTRACTED_DIR, "extraction_metadata.json"),
        os.path.join(os.path.dirname(EXTRACTED_DIR), "raw", "collection_metadata.json"),
    ]
    for meta_path in meta_paths:
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for entry in meta:
                ticker = entry.get("ticker", "")
                filing_date = entry.get("filing_date", "")
                if filing_date:
                    filing_years[ticker] = filing_date[:4]

    # Process each extracted file
    for ticker, company_name in COMPANIES.items():
        text_path = os.path.join(EXTRACTED_DIR, f"{ticker}_item1a.txt")
        if not os.path.exists(text_path):
            print(f"  SKIP: {text_path} not found")
            continue

        year = filing_years.get(ticker, "2025")

        print(f"\n{'='*60}")
        print(f"Chunking: {company_name} ({ticker}) — Year: {year}")
        print(f"{'='*60}")

        chunks = chunk_single_company(ticker, company_name, text_path, year)
        all_chunks.extend(chunks)

    # Save all chunks
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    # Save as a single JSON file
    chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Also save per-company files
    company_chunks = {}
    for chunk in all_chunks:
        ticker = chunk["company"]
        if ticker not in company_chunks:
            company_chunks[ticker] = []
        company_chunks[ticker].append(chunk)

    for ticker, chunks in company_chunks.items():
        company_path = os.path.join(CHUNKS_DIR, f"{ticker}_chunks.json")
        with open(company_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Save chunking metadata
    meta = {
        "total_chunks": len(all_chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "companies": {
            ticker: len(chunks)
            for ticker, chunks in company_chunks.items()
        },
        "avg_chunk_length": round(
            sum(c["char_count"] for c in all_chunks) / len(all_chunks), 1
        ) if all_chunks else 0,
    }
    meta_path = os.path.join(CHUNKS_DIR, "chunking_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Chunking Summary")
    print(f"{'='*60}")
    print(f"Total chunks: {meta['total_chunks']}")
    print(f"Avg chunk length: {meta['avg_chunk_length']} chars")
    for ticker, count in meta["companies"].items():
        print(f"  {ticker}: {count} chunks")

    return all_chunks


def chunk_all_for_year(target_year: int) -> list[dict]:
    """
    Chunk all extracted Item 1A texts for a specific fiscal year.
    Returns a combined list of all chunks with metadata.
    """
    all_chunks = []
    year_ext_dir = get_extracted_dir(target_year)
    year_chunks_dir = get_chunks_dir(target_year)
    
    # Load filing metadata for this year
    filing_years = {}
    meta_path = os.path.join(year_ext_dir, "extraction_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        for entry in meta:
            filing_years[entry.get("ticker", "")] = str(target_year)

    # Process each extracted file
    for ticker, company_name in COMPANIES.items():
        text_path = os.path.join(year_ext_dir, f"{ticker}_item1a.txt")
        if not os.path.exists(text_path):
            print(f"  SKIP: {text_path} not found")
            continue

        year = str(target_year)

        print(f"\n{'='*60}")
        print(f"Chunking: {company_name} ({ticker}) — FY{year}")
        print(f"{'='*60}")

        chunks = chunk_single_company(ticker, company_name, text_path, year)
        all_chunks.extend(chunks)

    # Save all chunks to year-specific dir
    os.makedirs(year_chunks_dir, exist_ok=True)

    chunks_path = os.path.join(year_chunks_dir, "all_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Save per-company files
    company_chunks = {}
    for chunk in all_chunks:
        ticker = chunk["company"]
        if ticker not in company_chunks:
            company_chunks[ticker] = []
        company_chunks[ticker].append(chunk)

    for ticker, chunks in company_chunks.items():
        company_path = os.path.join(year_chunks_dir, f"{ticker}_chunks.json")
        with open(company_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Chunking Summary (FY{target_year})")
    print(f"{'='*60}")
    print(f"Total chunks: {len(all_chunks)}")
    for ticker, chunks in company_chunks.items():
        print(f"  {ticker}: {len(chunks)} chunks")

    return all_chunks


if __name__ == "__main__":
    print("=" * 60)
    print("Text Chunker with Metadata")
    print("=" * 60)
    chunk_all_companies()
