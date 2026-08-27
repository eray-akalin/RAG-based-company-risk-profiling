from __future__ import annotations

import os
import json
import numpy as np
from src.collector import get_cik_for_ticker, get_10k_for_year, get_latest_10k_filing, download_10k_document
from src.extractor import process_single_file
from src.chunker import chunk_single_company
from src.embedder import load_chunks, generate_embeddings, build_faiss_index, save_index, save_index_for_year
from src.risk_extractor import RiskExtractor
from config import EMBEDDINGS_DIR, CHUNKS_DIR, get_embeddings_dir, get_chunks_dir, get_risk_profiles_dir

def run_live_analysis(ticker: str, year: int = None, status_placeholder=None):
    """
    Run the full live analysis pipeline for a given ticker and optional year.
    
    Args:
        ticker: Stock ticker symbol (e.g., "NFLX")
        year: Fiscal year to analyze. If None, fetches the latest available.
        status_placeholder: Streamlit placeholder for status updates
    """
    ticker = ticker.upper()
    company_name = f"{ticker} Inc."
    year_str = f" (FY{year})" if year else ""
    
    def update_status(msg):
        if status_placeholder:
            status_placeholder.info(msg)
        print(msg)
    
    try:
        update_status(f"1/5: SEC'den {ticker} 10-K raporu aranıyor{year_str}...")
        cik = get_cik_for_ticker(ticker)
        if not cik: return False, f"CIK kodu bulunamadı: {ticker}"
        
        if year:
            filing_info = get_10k_for_year(cik, ticker, year)
        else:
            filing_info = get_latest_10k_filing(cik, ticker)
        
        if not filing_info: return False, f"10-K raporu bulunamadı{year_str}."
        
        # Determine fiscal year from filing info
        fiscal_year = filing_info.get("fiscal_year", None)
        if fiscal_year is None and year:
            fiscal_year = year
            filing_info["fiscal_year"] = year
        
        local_path = download_10k_document(filing_info)
        
        update_status(f"2/5: Risk Faktörleri (Item 1A) çekiliyor...")
        result = process_single_file(local_path, ticker, company_name, year=fiscal_year)
        if not result: return False, "Item 1A metni çıkarılamadı."
        
        update_status(f"3/5: Metin parçalanıyor (Chunking)...")
        text_path = result["output_file"]
        new_chunks = chunk_single_company(ticker, company_name, text_path, str(fiscal_year) if fiscal_year else filing_info["filing_date"][:4])
        
        update_status(f"4/5: Vektörleştirme ve FAISS indexleme yapılıyor...")
        
        # Determine which embeddings directory to use
        if fiscal_year:
            emb_dir = get_embeddings_dir(fiscal_year)
            chunks_dir = get_chunks_dir(fiscal_year)
        else:
            emb_dir = EMBEDDINGS_DIR
            chunks_dir = CHUNKS_DIR
        
        try:
            # Try loading existing chunks and embeddings for this year
            chunks_path = os.path.join(chunks_dir, "all_chunks.json")
            if os.path.exists(chunks_path):
                with open(chunks_path, "r", encoding="utf-8") as f:
                    old_chunks = json.load(f)
            else:
                old_chunks = []
            
            emb_path = os.path.join(emb_dir, "embeddings.npy")
            if os.path.exists(emb_path):
                old_embeddings = np.load(emb_path)
            else:
                old_embeddings = np.empty((0, 384), dtype=np.float32)
        except Exception:
            old_chunks = []
            old_embeddings = np.empty((0, 384), dtype=np.float32)
            
        # Drop any previously-indexed chunks for THIS ticker so re-running a
        # ticker replaces its data instead of duplicating it in the index.
        # (embeddings.npy is row-aligned with all_chunks.json order.)
        if old_chunks:
            keep = [c.get("company") != ticker for c in old_chunks]
            old_chunks = [c for c, k in zip(old_chunks, keep) if k]
            if len(old_embeddings) == len(keep):
                old_embeddings = old_embeddings[np.array(keep, dtype=bool)]

        new_embeddings = generate_embeddings(new_chunks)
        all_embeddings = np.vstack([old_embeddings, new_embeddings])
        all_chunks = old_chunks + new_chunks
        
        # Save chunks to year-specific directory
        os.makedirs(chunks_dir, exist_ok=True)
        with open(os.path.join(chunks_dir, "all_chunks.json"), "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)
            
        index = build_faiss_index(all_embeddings)
        
        if fiscal_year:
            save_index_for_year(index, all_chunks, all_embeddings, fiscal_year)
        else:
            save_index(index, all_chunks, all_embeddings)
        
        update_status(f"5/5: Llama-3 API ile Yapay Zeka Risk Analizi Yapılıyor...")
        extractor = RiskExtractor()
        extractor.load_model()
        extractor.load_retriever(year=fiscal_year)
        
        profile = extractor.extract_company_profile(ticker, year=fiscal_year)
        
        # Save to year-specific directory
        if fiscal_year:
            output_dir = get_risk_profiles_dir(fiscal_year)
        else:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "risk_profiles")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual
        with open(os.path.join(output_dir, f"{ticker}_risk_profile.json"), "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False)
            
        # Append to all
        combined_path = os.path.join(output_dir, "all_risk_profiles.json")
        if os.path.exists(combined_path):
            with open(combined_path, "r", encoding="utf-8") as f:
                all_profiles = json.load(f)
            all_profiles = [p for p in all_profiles if p["company"] != ticker]
            all_profiles.append(profile)
        else:
            all_profiles = [profile]
            
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, ensure_ascii=False)
            
        update_status(f"✅ Analiz Başarılı!")
        return True, profile
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Hata oluştu: {str(e)}"
