"""
Generate Mock Risk Profiles for Dashboard Testing

Since running the 3B LLM locally on all data might take a long time,
this script generates realistic mock JSON output using the exact same schema
so we can build and test the Streamlit dashboard immediately.
"""

import os
import json
import random

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import COMPANIES, RISK_CATEGORIES

def generate_mock_profiles():
    output_dir = os.path.join(os.path.dirname(__file__), "data", "risk_profiles")
    os.makedirs(output_dir, exist_ok=True)
    
    all_profiles = []
    
    for ticker, company_name in COMPANIES.items():
        risk_assessments = []
        
        for category in RISK_CATEGORIES:
            cat_name = category["name"]
            
            # Add some logical domain-specific mock biases
            is_present = random.random() > 0.15 # 85% chance a risk is mentioned
            
            if not is_present:
                severity = "low"
                explanation = "No significant mention of this specific risk found in the recent 10-K filing."
                snippets = []
                conf = 0.9
            else:
                if cat_name == "Supply Chain Risk" and ticker in ["AAPL", "TSLA", "NVDA", "AMD"]:
                    severity = random.choice(["medium", "high"])
                elif cat_name == "Regulatory / Legal Risk" and ticker in ["MSFT", "AMZN", "AAPL"]:
                    severity = random.choice(["medium", "high"])
                elif cat_name == "Cybersecurity Risk":
                    severity = random.choice(["low", "medium", "high"])
                else:
                    severity = random.choice(["low", "medium"])
                    
                explanations = {
                    "high": f"{company_name} explicitly notes that {cat_name.lower()} is a critical threat that could materially impact revenue and margins. Mitigation efforts are ongoing but uncertain.",
                    "medium": f"The filing discusses {cat_name.lower()} as a notable ongoing concern, though they believe current strategies provide adequate protection.",
                    "low": f"Briefly mentioned as a general industry risk, but not highlighted as an immediate or material threat to {company_name}."
                }
                explanation = explanations[severity]
                
                snippets = [
                    f"...we face significant uncertainties related to {cat_name.lower()} which could adversely affect our financial condition...",
                    f"...changes in this area may require substantial additional investments or alter our business practices..."
                ]
                # Keep 1 or 2 snippets
                snippets = snippets[:random.randint(1, 2)]
                conf = round(random.uniform(0.65, 0.95), 2)
                
            risk_assessments.append({
                "company": ticker,
                "risk_category": cat_name,
                "is_present": is_present,
                "severity": severity,
                "explanation": explanation,
                "evidence_snippets": snippets,
                "confidence": conf
            })
            
        profile = {
            "company": ticker,
            "company_name": company_name,
            "risk_assessments": risk_assessments,
            "total_categories": len(RISK_CATEGORIES),
            "risks_found": sum(1 for r in risk_assessments if r["is_present"])
        }
        all_profiles.append(profile)
        
        # Save individual
        with open(os.path.join(output_dir, f"{ticker}_risk_profile.json"), "w") as f:
            json.dump(profile, f, indent=2)
            
    # Save combined
    with open(os.path.join(output_dir, "all_risk_profiles.json"), "w") as f:
        json.dump(all_profiles, f, indent=2)
        
    print(f"✅ Generated mock risk profiles for {len(all_profiles)} companies in data/risk_profiles/")

if __name__ == "__main__":
    generate_mock_profiles()
