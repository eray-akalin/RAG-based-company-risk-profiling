"""
SEC EDGAR 10-K Report Collector

Collects Form 10-K annual reports from SEC EDGAR for target companies.
Uses the EDGAR Full-Text Search API and Company Filings API.
"""

import os
import json
import time
import requests
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COMPANIES, RAW_DIR, SEC_USER_AGENT, get_raw_dir


# EDGAR API endpoints
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"


def get_headers() -> dict:
    """Return headers required by SEC EDGAR (they require a User-Agent)."""
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """
    Look up the CIK (Central Index Key) number for a given ticker symbol.
    Returns CIK as a zero-padded 10-digit string.
    """
    print(f"  Looking up CIK for {ticker}...")
    resp = requests.get(COMPANY_TICKERS_URL, headers=get_headers())
    resp.raise_for_status()
    data = resp.json()

    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            print(f"  Found CIK: {cik} for {ticker}")
            return cik

    print(f"  WARNING: CIK not found for ticker {ticker}")
    return None


def get_latest_10k_filing(cik: str, ticker: str) -> Optional[dict]:
    """
    Find the most recent 10-K filing for a given CIK.
    Returns a dict with accession number and primary document filename.
    """
    print(f"  Fetching filing history for CIK {cik}...")
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=get_headers())
    resp.raise_for_status()
    data = resp.json()

    recent_filings = data.get("filings", {}).get("recent", {})
    forms = recent_filings.get("form", [])
    accessions = recent_filings.get("accessionNumber", [])
    primary_docs = recent_filings.get("primaryDocument", [])
    filing_dates = recent_filings.get("filingDate", [])

    for i, form in enumerate(forms):
        # Only accept original 10-K filings, NOT 10-K/A amendments
        # (amendments often contain only partial corrections without Item 1A)
        if form == "10-K":
            accession = accessions[i].replace("-", "")
            filing_info = {
                "ticker": ticker,
                "cik": cik,
                "form": form,
                "accession_number": accessions[i],
                "accession_clean": accession,
                "primary_document": primary_docs[i],
                "filing_date": filing_dates[i],
            }
            print(f"  Found {form} filed on {filing_dates[i]}")
            return filing_info

    print(f"  WARNING: No 10-K filing found for CIK {cik}")
    return None


def get_10k_for_year(cik: str, ticker: str, target_year: int) -> Optional[dict]:
    """
    Find the 10-K filing for a specific fiscal year.
    SEC 10-K filings are typically filed within a few months after fiscal year end.
    E.g., FY2024 reports are usually filed between Jan-Apr 2025.
    
    Strategy: Look for 10-K filings where:
      - filing_date year == target_year (fiscal year ends same year), OR
      - filing_date year == target_year + 1 AND month <= 6 (filed early next year)
    """
    print(f"  Fetching filing history for CIK {cik} (target year: {target_year})...")
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=get_headers())
    resp.raise_for_status()
    data = resp.json()

    recent_filings = data.get("filings", {}).get("recent", {})
    forms = recent_filings.get("form", [])
    accessions = recent_filings.get("accessionNumber", [])
    primary_docs = recent_filings.get("primaryDocument", [])
    filing_dates = recent_filings.get("filingDate", [])

    candidates = []
    for i, form in enumerate(forms):
        if form != "10-K":
            continue
        filing_date = filing_dates[i]
        filing_year = int(filing_date[:4])
        filing_month = int(filing_date[5:7])
        
        # Match: filed in target_year (Jul-Dec) or target_year+1 (Jan-Jun)
        is_match = (
            (filing_year == target_year and filing_month >= 7) or
            (filing_year == target_year + 1 and filing_month <= 6)
        )
        
        if is_match:
            accession = accessions[i].replace("-", "")
            candidates.append({
                "ticker": ticker,
                "cik": cik,
                "form": form,
                "accession_number": accessions[i],
                "accession_clean": accession,
                "primary_document": primary_docs[i],
                "filing_date": filing_date,
                "fiscal_year": target_year,
            })
    
    if candidates:
        # Prefer earliest filing date (original, not amendment)
        candidates.sort(key=lambda x: x["filing_date"])
        chosen = candidates[0]
        print(f"  Found 10-K for FY{target_year} filed on {chosen['filing_date']}")
        return chosen
    
    print(f"  WARNING: No 10-K filing found for FY{target_year}")
    return None


def download_10k_document(filing_info: dict) -> Optional[str]:
    """
    Download the primary 10-K document (usually HTML).
    Returns the local file path where it was saved.
    """
    cik = filing_info["cik"].lstrip("0")
    accession = filing_info["accession_clean"]
    filename = filing_info["primary_document"]
    ticker = filing_info["ticker"]

    url = ARCHIVES_URL.format(cik=cik, accession=accession, filename=filename)
    print(f"  Downloading from: {url}")

    resp = requests.get(url, headers=get_headers())
    resp.raise_for_status()

    # Determine file extension and year-specific directory
    ext = os.path.splitext(filename)[1] or ".html"
    fiscal_year = filing_info.get("fiscal_year", filing_info['filing_date'][:4])
    local_filename = f"{ticker}_10K_{filing_info['filing_date']}{ext}"
    
    # Save to year-specific directory
    year_raw_dir = get_raw_dir(int(fiscal_year))
    os.makedirs(year_raw_dir, exist_ok=True)
    local_path = os.path.join(year_raw_dir, local_filename)

    with open(local_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

    print(f"  Saved to: {local_path} ({len(resp.text):,} chars)")
    return local_path


def collect_all_10k_reports() -> dict:
    """
    Main collection function.
    Downloads the latest 10-K for each company in COMPANIES.
    Returns a dict mapping ticker → local file path.
    """
    results = {}
    metadata = []

    for ticker, company_name in COMPANIES.items():
        print(f"\n{'='*60}")
        print(f"Processing: {company_name} ({ticker})")
        print(f"{'='*60}")

        try:
            # Step 1: Get CIK
            cik = get_cik_for_ticker(ticker)
            if not cik:
                continue
            time.sleep(0.2)  # SEC rate limit: 10 requests/second

            # Step 2: Find latest 10-K
            filing_info = get_latest_10k_filing(cik, ticker)
            if not filing_info:
                continue
            time.sleep(0.2)

            # Step 3: Download document
            local_path = download_10k_document(filing_info)
            if local_path:
                results[ticker] = local_path
                filing_info["local_path"] = local_path
                filing_info["company_name"] = company_name
                metadata.append(filing_info)

            time.sleep(0.5)  # Be respectful to SEC servers

        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            continue

    # Save metadata
    meta_path = os.path.join(RAW_DIR, "collection_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")

    return results


def collect_10k_for_year(target_year: int) -> dict:
    """
    Download the 10-K for each company in COMPANIES for a specific fiscal year.
    Returns a dict mapping ticker → local file path.
    """
    results = {}
    metadata = []

    for ticker, company_name in COMPANIES.items():
        print(f"\n{'='*60}")
        print(f"Processing: {company_name} ({ticker}) — FY{target_year}")
        print(f"{'='*60}")

        try:
            cik = get_cik_for_ticker(ticker)
            if not cik:
                continue
            time.sleep(0.2)

            filing_info = get_10k_for_year(cik, ticker, target_year)
            if not filing_info:
                continue
            time.sleep(0.2)

            local_path = download_10k_document(filing_info)
            if local_path:
                results[ticker] = local_path
                filing_info["local_path"] = local_path
                filing_info["company_name"] = company_name
                metadata.append(filing_info)

            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            continue

    # Save metadata in year-specific dir
    year_raw_dir = get_raw_dir(target_year)
    os.makedirs(year_raw_dir, exist_ok=True)
    meta_path = os.path.join(year_raw_dir, "collection_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("SEC EDGAR 10-K Report Collector")
    print("=" * 60)

    collected = collect_all_10k_reports()

    print(f"\n{'='*60}")
    print(f"Collection Summary")
    print(f"{'='*60}")
    print(f"Total companies attempted: {len(COMPANIES)}")
    print(f"Successfully collected: {len(collected)}")
    for ticker, path in collected.items():
        print(f"  {ticker}: {path}")
