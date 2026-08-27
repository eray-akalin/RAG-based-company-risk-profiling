"""
Category-Detection Evaluation (accuracy + macro-F1)

Proposal §3.7 asks us to label a set of company-risk pairs with expected risk
categories and compare the model's output using accuracy and macro-F1.

We frame this as binary risk-presence detection per (company × category):
- Gold: `evaluation/annotations/category_gold_2025.csv` (is_present_gold ∈ {0,1});
  source-verified silver labels (every category is genuinely discussed in these
  large-cap Item 1A sections — verified by keyword presence in the extracted text).
- Prediction: the `is_present` field of each saved risk profile.

Pairs whose extraction failed (LLM error) are excluded and reported separately.

Run:  python3 -m src.category_eval 2025
"""

from __future__ import annotations

import os
import sys
import json
import glob
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import get_risk_profiles_dir, DEFAULT_YEAR


def _load_gold(year: int) -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "evaluation",
                        "annotations", f"category_gold_{year}.csv")
    gold = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold[(row["company"], row["category"])] = int(row["is_present_gold"])
    return gold


def _load_predictions(year: int) -> tuple[dict, dict]:
    """Return (pred_map, failed_map) keyed by (company, category)."""
    pred, failed = {}, {}
    for fp in glob.glob(os.path.join(get_risk_profiles_dir(year), "*_risk_profile.json")):
        prof = json.load(open(fp, encoding="utf-8"))
        comp = prof.get("company")
        for a in prof.get("risk_assessments", []):
            key = (comp, a["risk_category"])
            if a.get("extraction_failed"):
                failed[key] = True
            else:
                pred[key] = 1 if a.get("is_present") else 0
    return pred, failed


def evaluate_categories(year: int = DEFAULT_YEAR) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
    )

    gold = _load_gold(year)
    pred, failed = _load_predictions(year)

    y_true, y_pred, errors, skipped = [], [], [], []
    for key, g in sorted(gold.items()):
        if key in failed or key not in pred:
            skipped.append({"company": key[0], "category": key[1],
                            "reason": "extraction_failed" if key in failed else "no_prediction"})
            continue
        p = pred[key]
        y_true.append(g)
        y_pred.append(p)
        if g != p:
            errors.append({"company": key[0], "category": key[1],
                           "gold": g, "pred": p,
                           "type": "false_negative" if g == 1 else "false_positive"})

    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    macro_f1 = round(float(f1_score(y_true, y_pred, labels=[0, 1],
                                    average="macro", zero_division=0)), 4)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    per_class = {
        "absent (0)": {"precision": round(float(prec[0]), 3), "recall": round(float(rec[0]), 3),
                       "f1": round(float(f1[0]), 3), "support": int(support[0])},
        "present (1)": {"precision": round(float(prec[1]), 3), "recall": round(float(rec[1]), 3),
                        "f1": round(float(f1[1]), 3), "support": int(support[1])},
    }

    print(f"\n{'='*60}\nCATEGORY DETECTION (FY{year}, N={len(y_true)} pairs)\n{'='*60}")
    print(f"  Accuracy : {acc}")
    print(f"  Macro-F1 : {macro_f1}")
    print(f"  Confusion [rows=gold 0/1, cols=pred 0/1]: {cm}")
    for cls, m in per_class.items():
        print(f"    {cls:12s} P={m['precision']} R={m['recall']} F1={m['f1']} (n={m['support']})")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors:
            print(f"    - [{e['company']}] {e['category']}: gold={e['gold']} pred={e['pred']} ({e['type']})")
    if skipped:
        print(f"  Skipped ({len(skipped)}): " +
              ", ".join(f"{s['company']}/{s['category']}({s['reason']})" for s in skipped))
    print(f"{'='*60}")

    report = {
        "year": year, "n_pairs": len(y_true), "accuracy": acc, "macro_f1": macro_f1,
        "per_class": per_class, "confusion_matrix": cm,
        "errors": errors, "skipped": skipped,
        "gold_note": "Source-verified silver labels; all categories present in these large-cap filings.",
    }
    out = os.path.join(get_risk_profiles_dir(year), "category_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out}")
    return report


if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_YEAR
    evaluate_categories(yr)
