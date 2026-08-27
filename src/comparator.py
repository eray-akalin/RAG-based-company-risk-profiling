"""
Company Risk Comparator

Provides logic to load extracted risk profiles, aggregate data,
and generate comparison matrices between companies.
Supports multi-year analysis and cross-year comparisons.
"""

from __future__ import annotations
import os
import json
import pandas as pd
from typing import List, Dict, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COMPANIES, RISK_CATEGORIES, get_risk_profiles_dir, AVAILABLE_YEARS

# Severity mapping for numeric scoring (5-point scale)
SEVERITY_SCORE = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "negligible": 0,
    "none": 0  # if is_present == False
}


class RiskComparator:
    """Loads risk profiles and provides data structures for comparison."""

    def __init__(self, profiles_dir: str = None, year: int = None):
        """
        Initialize the comparator.
        
        Args:
            profiles_dir: Override path for profiles directory
            year: Fiscal year to load profiles for
        """
        if profiles_dir is not None:
            self.profiles_dir = profiles_dir
        elif year is not None:
            self.profiles_dir = get_risk_profiles_dir(year)
        else:
            # Default: try flat directory (backward compat)
            self.profiles_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "risk_profiles"
            )
            
        self.year = year
        self.profiles = []
        self._load_profiles()

    def _load_profiles(self):
        """Load all available risk profiles."""
        if not os.path.exists(self.profiles_dir):
            return

        combined_path = os.path.join(self.profiles_dir, "all_risk_profiles.json")
        if os.path.exists(combined_path):
            with open(combined_path, "r", encoding="utf-8") as f:
                self.profiles = json.load(f)
        else:
            # Try to load individual files if combined doesn't exist
            for ticker in COMPANIES:
                path = os.path.join(self.profiles_dir, f"{ticker}_risk_profile.json")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        self.profiles.append(json.load(f))

    def get_available_companies(self) -> List[str]:
        """Return list of tickers with available profiles."""
        return [p["company"] for p in self.profiles]

    def get_company_profile(self, ticker: str) -> Optional[Dict]:
        """Get the full profile for a specific company."""
        for p in self.profiles:
            if p["company"] == ticker:
                return p
        return None

    def get_risk_heatmap_data(self) -> pd.DataFrame:
        """
        Generate a matrix of companies vs risk categories with severity scores.
        Useful for building a heatmap.
        """
        data = []
        
        for profile in self.profiles:
            ticker = profile["company"]
            row = {"Company": ticker}
            
            for risk in profile["risk_assessments"]:
                cat = risk["risk_category"]
                if risk["is_present"]:
                    score = SEVERITY_SCORE.get(risk["severity"], 1)
                else:
                    score = 0
                row[cat] = score
                
            data.append(row)
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data).set_index("Company")
        return df
    
    def get_severity_labels_matrix(self) -> pd.DataFrame:
        """Generate a matrix of text severity labels (Low, Medium, High)."""
        data = []
        
        for profile in self.profiles:
            ticker = profile["company"]
            row = {"Company": ticker}
            
            for risk in profile["risk_assessments"]:
                cat = risk["risk_category"]
                if risk["is_present"]:
                    row[cat] = risk["severity"].capitalize()
                else:
                    row[cat] = "None"
                
            data.append(row)
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data).set_index("Company")
        return df

    def compare_two_companies(self, ticker1: str, ticker2: str) -> pd.DataFrame:
        """
        Compare two companies side-by-side across all risk categories.
        """
        p1 = self.get_company_profile(ticker1)
        p2 = self.get_company_profile(ticker2)
        
        if not p1 or not p2:
            raise ValueError("One or both company profiles not found.")
            
        data = []
        
        # Build dictionary for quick lookup
        r1 = {r["risk_category"]: r for r in p1["risk_assessments"]}
        r2 = {r["risk_category"]: r for r in p2["risk_assessments"]}
        
        for cat in RISK_CATEGORIES:
            cat_name = cat["name"]
            
            risk1 = r1.get(cat_name, {})
            risk2 = r2.get(cat_name, {})
            
            sev1 = risk1.get("severity", "none").capitalize() if risk1.get("is_present") else "None"
            sev2 = risk2.get("severity", "none").capitalize() if risk2.get("is_present") else "None"
            
            data.append({
                "Risk Category": cat_name,
                f"{ticker1} Severity": sev1,
                f"{ticker2} Severity": sev2,
                f"{ticker1} Explanation": risk1.get("explanation", ""),
                f"{ticker2} Explanation": risk2.get("explanation", "")
            })
            
        return pd.DataFrame(data)

    def get_top_risks_for_company(self, ticker: str, top_n: int = 3) -> List[Dict]:
        """Get the highest severity risks for a specific company."""
        profile = self.get_company_profile(ticker)
        if not profile:
            return []
            
        risks = [r for r in profile["risk_assessments"] if r["is_present"]]
        
        # Sort by severity score (High=3, Med=2, Low=1) then confidence
        risks.sort(key=lambda x: (SEVERITY_SCORE.get(x["severity"], 0), x["confidence"]), reverse=True)
        
        return risks[:top_n]

    # ==================================================================
    # Multi-Year / Cross-Year Methods
    # ==================================================================

    @staticmethod
    def get_available_years() -> List[int]:
        """Return list of years that have risk profile data available."""
        available = []
        for year in AVAILABLE_YEARS:
            year_dir = get_risk_profiles_dir(year)
            combined_path = os.path.join(year_dir, "all_risk_profiles.json")
            if os.path.exists(combined_path):
                available.append(year)
        
        # Also check flat directory (legacy data)
        flat_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "risk_profiles"
        )
        flat_combined = os.path.join(flat_dir, "all_risk_profiles.json")
        if os.path.exists(flat_combined) and not available:
            # Legacy data exists but no year-specific data — treat as 2025
            available.append(2025)
        
        return sorted(available)

    @staticmethod
    def compare_years(ticker: str, year1: int, year2: int) -> pd.DataFrame:
        """
        Compare the same company across two different fiscal years.
        Returns a DataFrame with risk categories and severity for each year.
        """
        comp1 = RiskComparator(year=year1)
        comp2 = RiskComparator(year=year2)
        
        p1 = comp1.get_company_profile(ticker)
        p2 = comp2.get_company_profile(ticker)
        
        if not p1 and not p2:
            return pd.DataFrame()
        
        data = []
        r1 = {r["risk_category"]: r for r in (p1 or {}).get("risk_assessments", [])}
        r2 = {r["risk_category"]: r for r in (p2 or {}).get("risk_assessments", [])}
        
        for cat in RISK_CATEGORIES:
            cat_name = cat["name"]
            risk1 = r1.get(cat_name, {})
            risk2 = r2.get(cat_name, {})
            
            sev1 = risk1.get("severity", "none").capitalize() if risk1.get("is_present") else "None"
            sev2 = risk2.get("severity", "none").capitalize() if risk2.get("is_present") else "None"
            
            score1 = SEVERITY_SCORE.get(sev1.lower(), 0)
            score2 = SEVERITY_SCORE.get(sev2.lower(), 0)
            
            change = score2 - score1
            if change > 0:
                trend = "📈 Increased"
            elif change < 0:
                trend = "📉 Decreased"
            else:
                trend = "➡️ Stable"
            
            data.append({
                "Risk Category": cat_name,
                f"FY{year1}": sev1,
                f"FY{year2}": sev2,
                "Trend": trend,
                f"FY{year1} Explanation": risk1.get("explanation", "N/A"),
                f"FY{year2} Explanation": risk2.get("explanation", "N/A"),
            })
        
        return pd.DataFrame(data)

    @staticmethod
    def get_risk_trend(ticker: str, years: List[int] = None) -> pd.DataFrame:
        """
        Get risk severity trend for a company across multiple years.
        Returns a DataFrame suitable for line chart plotting.
        
        Columns: Year, risk_category_1_score, risk_category_2_score, ...
        """
        if years is None:
            years = RiskComparator.get_available_years()
        
        data = []
        for year in sorted(years):
            comp = RiskComparator(year=year)
            profile = comp.get_company_profile(ticker)
            
            if not profile:
                continue
            
            row = {"Year": year}
            for risk in profile["risk_assessments"]:
                cat = risk["risk_category"]
                if risk["is_present"]:
                    row[cat] = SEVERITY_SCORE.get(risk["severity"], 1)
                else:
                    row[cat] = 0
            data.append(row)
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data).set_index("Year")


if __name__ == "__main__":
    # Quick test
    comp = RiskComparator()
    print("Available companies:", comp.get_available_companies())
    print("Available years:", RiskComparator.get_available_years())
