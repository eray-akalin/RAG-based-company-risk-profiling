import os
import json
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.retriever import SemanticRetriever
from config import RISK_CATEGORIES, TOP_K

def fix_json_snippets():
    profiles_dir = os.path.join(os.path.dirname(__file__), "data", "risk_profiles")
    if not os.path.exists(profiles_dir):
        print("Profiles directory not found.")
        return

    retriever = None
    fixed_count = 0

    # Process individual files
    for filename in os.listdir(profiles_dir):
        if not filename.endswith("_risk_profile.json") or filename == "all_risk_profiles.json":
            continue

        filepath = os.path.join(profiles_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            profile = json.load(f)

        ticker = profile["company"]
        modified = False

        for risk in profile.get("risk_assessments", []):
            cat_name = risk["risk_category"]
            snippets = risk.get("evidence_snippets", [])
            
            for i, snippet in enumerate(snippets):
                # If it looks like just "[Chunk X]" or doesn't have an ID
                if "[Chunk" in snippet and "ID:" not in snippet:
                    match = re.search(r"\[Chunk\s*(\d+)\]", snippet)
                    if match:
                        chunk_idx = int(match.group(1)) - 1 # 1-based to 0-based
                        
                        # Lazy load retriever
                        if retriever is None:
                            print("Loading retriever to fix missing IDs...")
                            retriever = SemanticRetriever()
                            
                        # Find the category config
                        cat_config = next((c for c in RISK_CATEGORIES if c["name"] == cat_name), None)
                        if cat_config:
                            # Recreate the chunks the LLM saw
                            chunks = retriever.retrieve_for_risk_category(cat_config, company=ticker, top_k=TOP_K)
                            
                            if 0 <= chunk_idx < len(chunks):
                                real_id = chunks[chunk_idx]["chunk_id"]
                                # Replace with proper ID format
                                snippets[i] = f"[Chunk {chunk_idx+1}] (ID: {real_id})"
                                modified = True
                                fixed_count += 1

        if modified:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            print(f"Fixed missing IDs in {filename}")

    # Re-combine all_risk_profiles.json
    all_profiles = []
    for filename in os.listdir(profiles_dir):
        if filename.endswith("_risk_profile.json") and filename != "all_risk_profiles.json":
            with open(os.path.join(profiles_dir, filename), "r", encoding="utf-8") as f:
                all_profiles.append(json.load(f))
                
    if all_profiles:
        with open(os.path.join(profiles_dir, "all_risk_profiles.json"), "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, indent=2, ensure_ascii=False)
            
    print(f"Done! Fixed {fixed_count} missing chunk IDs across all files.")

if __name__ == "__main__":
    fix_json_snippets()
