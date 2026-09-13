import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Error Analysis -- Post Phase 7
================================
After Phase 7 fine-tuning is complete, this script:
1. Loads predictions from all three models (mBERT, MuRIL, adapted-mBERT)
2. Compares predictions on the held-out validation sets
3. Finds concrete examples where adaptation:
   (a) improved classification  -- baseline wrong, adapted correct
   (b) made no difference       -- both correct or both wrong
   (c) made things worse        -- baseline correct, adapted wrong
4. Analyses patterns: sentence length, CMI score, code-mixing intensity

Outputs:
  results/tables/table7_error_analysis.csv
  Appends findings section to RESULTS_SUMMARY.md

NOTE: Run this AFTER phase7_classification.py has completed.
"""

import csv
import datetime
from pathlib import Path

import pandas as pd
import numpy as np

ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
TABLES         = ROOT / "results" / "tables"
LOGS           = ROOT / "logs"

print("=" * 60)
print("ERROR ANALYSIS -- Post Phase 7")
print("=" * 60)

# ── Load required files ────────────────────────────────────────────────────────
pred_path = TABLES / "phase7_predictions.csv"  # written by phase7 if enabled
dev_path  = DATA_PROCESSED / "dev_subset.csv"

if not pred_path.exists():
    print(f"\n  [!] {pred_path.name} not found.")
    print("      Phase 7 must be run with save_predictions=True before error analysis.")
    print("      Re-run phase7_classification.py -- it will save predictions automatically.")
    print("\n  Attempting fallback: loading per-fold results from checkpoint...")

    fold_path = LOGS / "phase7_fold_results.csv"
    if not fold_path.exists():
        print("  [!] No Phase 7 results found at all. Run phase7_classification.py first.")
        raise SystemExit(1)

    fold_df = pd.read_csv(fold_path, encoding="utf-8")
    print(f"  Loaded fold results: {len(fold_df)} fold entries")
    print(fold_df.to_string(index=False))
    print("\n  Error analysis requires sentence-level predictions (not just fold-level metrics).")
    print("  Please ensure phase7_classification.py saves predictions and re-run this script.")
    raise SystemExit(0)

# ── Load predictions ───────────────────────────────────────────────────────────
df_pred = pd.read_csv(pred_path, encoding="utf-8")
df_dev  = pd.read_csv(dev_path,  encoding="utf-8")

print(f"\n[1/4] Loaded {len(df_pred):,} sentence predictions")
print(f"      Columns: {list(df_pred.columns)}")

# Expected columns: text, true_label, mBERT_pred, MuRIL_pred, adapted_pred, cmi_score, word_count

label2id = {lbl: i for i, lbl in enumerate(sorted(df_pred["true_label"].unique()))}

def correct(pred_col, true_col):
    return (df_pred[pred_col] == df_pred[true_col]).astype(int)

df_pred["mbert_correct"]   = correct("mBERT_pred",   "true_label")
df_pred["adapted_correct"] = correct("adapted_pred", "true_label")

# ── Categorise each example ────────────────────────────────────────────────────
print("\n[2/4] Categorising examples...")

def categorise(row):
    b = row["mbert_correct"]
    a = row["adapted_correct"]
    if b == 0 and a == 1:
        return "improved"
    elif b == 1 and a == 0:
        return "degraded"
    elif b == 1 and a == 1:
        return "both_correct"
    else:
        return "both_wrong"

df_pred["error_category"] = df_pred.apply(categorise, axis=1)

counts = df_pred["error_category"].value_counts()
total  = len(df_pred)
print(f"\n  Error category breakdown:")
for cat, cnt in counts.items():
    print(f"    {cat:20}: {cnt:5,}  ({cnt/total*100:.1f}%)")

# ── Pattern analysis ───────────────────────────────────────────────────────────
print("\n[3/4] Pattern analysis (sentence length, CMI score)...")

df_pred["word_count"] = df_pred["text"].astype(str).apply(lambda x: len(x.split()))

for cat in ["improved", "degraded", "both_correct", "both_wrong"]:
    sub = df_pred[df_pred["error_category"] == cat]
    if len(sub) == 0:
        continue
    print(f"\n  [{cat.upper()}] n={len(sub)}")
    print(f"    Avg word count  : {sub['word_count'].mean():.1f}")
    if "cmi_score" in sub.columns:
        print(f"    Avg CMI score   : {sub['cmi_score'].mean():.2f}")

# ── Pull concrete examples ────────────────────────────────────────────────────
print("\n[4/4] Pulling concrete examples (5-10 per category)...")

N_EXAMPLES = 7
example_rows = []

for cat in ["improved", "degraded", "both_wrong"]:
    sub = df_pred[df_pred["error_category"] == cat].copy()
    if "cmi_score" in sub.columns:
        # Sort by CMI score to get diverse examples
        sub = sub.sort_values("cmi_score", ascending=False)
    sampled = sub.head(N_EXAMPLES)

    for _, row in sampled.iterrows():
        example_rows.append({
            "category":        cat,
            "text":            row["text"],
            "true_label":      row["true_label"],
            "mBERT_pred":      row.get("mBERT_pred", "N/A"),
            "adapted_pred":    row.get("adapted_pred", "N/A"),
            "word_count":      row["word_count"],
            "cmi_score":       row.get("cmi_score", "N/A"),
            "analysis_note":   "",   # to be filled manually
        })

df_examples = pd.DataFrame(example_rows)
df_examples.to_csv(TABLES / "table7_error_analysis.csv", index=False, encoding="utf-8")
print(f"    Saved {len(df_examples)} examples -> results/tables/table7_error_analysis.csv")

# ── Pattern summary for RESULTS_SUMMARY ───────────────────────────────────────
summary_text = f"""
## Error Analysis

### Category Breakdown
| Category | Count | % |
|---|---|---|
"""
for cat, cnt in counts.items():
    summary_text += f"| {cat} | {cnt:,} | {cnt/total*100:.1f}% |\n"

summary_text += """
### Pattern Observations
*(To be completed after reviewing table7_error_analysis.csv)*

Preliminary observations based on mean values:
"""
for cat in ["improved", "degraded"]:
    sub = df_pred[df_pred["error_category"] == cat]
    if len(sub) == 0:
        continue
    avg_len = sub["word_count"].mean()
    avg_cmi = sub["cmi_score"].mean() if "cmi_score" in sub.columns else "N/A"
    summary_text += f"\n- **{cat.title()}** examples: avg word count = {avg_len:.1f}, avg CMI = {avg_cmi:.2f if isinstance(avg_cmi, float) else avg_cmi}"

# Append to RESULTS_SUMMARY if it exists
results_summary_path = ROOT / "RESULTS_SUMMARY.md"
if results_summary_path.exists():
    with open(results_summary_path, "a", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"    Appended error analysis -> RESULTS_SUMMARY.md")

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)
print(f"  Output: results/tables/table7_error_analysis.csv")
print(f"  Review the CSV and fill in 'analysis_note' column with your observations.")
