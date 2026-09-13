import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Phase 2 -- Clean the Dataset
============================
Loads the filtered dataset from Phase 1, removes exact duplicates,
empty/broken rows, and pure-URL/spam rows.

Intentionally does NOT:
  - Normalize spelling variation (e.g. "chhe" vs "che") — this is
    signal, not noise, in code-mixed NLP research.
  - Transliterate or standardize scripts.

Outputs:
  - data/processed/filtered_clean.csv      (full cleaned set)
  - data/processed/dev_subset.csv          (3,000-row stratified sample)
  - results/tables/table2_cleaning_report.csv
"""

import re
import csv
import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent
DATA_PROCESSED  = ROOT / "data" / "processed"
TABLES          = ROOT / "results" / "tables"
LOGS            = ROOT / "logs"

RANDOM_SEED = 42

# ── Logging helper ─────────────────────────────────────────────────────────────
def append_log(row: dict):
    log_file = LOGS / "experiment_log.csv"
    fieldnames = [
        "date", "phase", "model", "tokenizer", "dataset_version",
        "sample_size", "vocab_size", "learning_rate", "batch_size",
        "epochs", "seed", "result"
    ]
    write_header = not log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

# ── Load ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 2 — Dataset Cleaning")
print("=" * 60)

input_path = DATA_PROCESSED / "filtered_dataset.csv"
if not input_path.exists():
    raise FileNotFoundError(
        f"Phase 1 output not found at {input_path}. Run phase1_dataset.py first."
    )

df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
n_start = len(df)
print(f"\n[1/5] Loaded {n_start:,} rows from Phase 1 output.")

# Track removal counts for the report
removals = {}

# ── Step 1: Remove rows with empty/null text ───────────────────────────────────
df["text"] = df["text"].astype(str).str.strip()
mask_empty = df["text"].isin(["", "nan", "None", "NaN", "null"])
n_empty = mask_empty.sum()
df = df[~mask_empty].reset_index(drop=True)
removals["empty_or_null_text"] = int(n_empty)
print(f"[2/5] Removed {n_empty:,} empty/null text rows. Remaining: {len(df):,}")

# ── Step 2: Remove rows with missing sentiment label ───────────────────────────
mask_no_label = df["sentiment_label"].isna() | (df["sentiment_label"].astype(str).str.strip() == "")
n_no_label = mask_no_label.sum()
df = df[~mask_no_label].reset_index(drop=True)
removals["missing_sentiment_label"] = int(n_no_label)
print(f"[2/5] Removed {n_no_label:,} rows with missing sentiment label. Remaining: {len(df):,}")

# ── Step 3: Remove pure-URL / spam rows ───────────────────────────────────────
# A row is "pure URL" if ≥80% of its non-whitespace tokens look like URLs.
# A row is "spam" if it has fewer than 2 real words (after stripping URLs).
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|bit\.ly/\S+|youtu\.be/\S+", re.IGNORECASE
)

def is_spam_or_url(text: str) -> bool:
    tokens = text.split()
    if len(tokens) == 0:
        return True
    url_tokens = [t for t in tokens if URL_PATTERN.fullmatch(t)]
    # Pure URL row: all tokens are URLs
    if len(url_tokens) == len(tokens):
        return True
    # After removing URLs, fewer than 2 real words remain
    real_words = [t for t in tokens if not URL_PATTERN.fullmatch(t)]
    if len(real_words) < 2:
        return True
    return False

mask_spam = df["text"].apply(is_spam_or_url)
n_spam = mask_spam.sum()
df = df[~mask_spam].reset_index(drop=True)
removals["spam_or_pure_url"] = int(n_spam)
print(f"[3/5] Removed {n_spam:,} spam/pure-URL rows. Remaining: {len(df):,}")

# ── Step 4: Remove exact duplicates (on text column) ──────────────────────────
n_before_dedup = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
n_dedup = n_before_dedup - len(df)
removals["exact_duplicates_removed"] = int(n_dedup)
print(f"[4/5] Removed {n_dedup:,} exact duplicate rows. Remaining: {len(df):,}")

n_final = len(df)
total_removed = n_start - n_final

# ── Step 5: Create dev subset (3,000 rows, stratified) ────────────────────────
print(f"\n[5/5] Creating stratified dev subset of 3,000 rows...")
DEV_SIZE = 3000
unique_labels = df["sentiment_label"].unique()
n_labels = len(unique_labels)
per_class = DEV_SIZE // n_labels

dev_frames = []
for lbl in unique_labels:
    pool = df[df["sentiment_label"] == lbl]
    n_sample = min(per_class, len(pool))
    dev_frames.append(pool.sample(n=n_sample, random_state=RANDOM_SEED))

df_dev = pd.concat(dev_frames, ignore_index=True)
# Top up to exactly DEV_SIZE if possible
if len(df_dev) < DEV_SIZE and len(df) > len(df_dev):
    already_in = set(df_dev.index)
    # use original df index approach
    df_dev_idx = set(df_dev["text"])
    remaining = df[~df["text"].isin(df_dev_idx)]
    shortfall = min(DEV_SIZE - len(df_dev), len(remaining))
    if shortfall > 0:
        extra = remaining.sample(n=shortfall, random_state=RANDOM_SEED)
        df_dev = pd.concat([df_dev, extra], ignore_index=True)

df_dev = df_dev.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# ── Save outputs ───────────────────────────────────────────────────────────────
df.to_csv(DATA_PROCESSED / "filtered_clean.csv", index=False, encoding="utf-8")
df_dev.to_csv(DATA_PROCESSED / "dev_subset.csv", index=False, encoding="utf-8")
print(f"    Full cleaned set : {n_final:,} rows → data/processed/filtered_clean.csv")
print(f"    Dev subset       : {len(df_dev):,} rows → data/processed/dev_subset.csv")

# ── Cleaning report table ──────────────────────────────────────────────────────
report_rows = [
    {"step": "rows_before_cleaning",       "count": n_start,        "pct_of_original": "100.00%"},
    {"step": "removed_empty_null",          "count": removals["empty_or_null_text"],       "pct_of_original": f"{removals['empty_or_null_text']/n_start*100:.2f}%"},
    {"step": "removed_missing_label",       "count": removals["missing_sentiment_label"],  "pct_of_original": f"{removals['missing_sentiment_label']/n_start*100:.2f}%"},
    {"step": "removed_spam_url",            "count": removals["spam_or_pure_url"],          "pct_of_original": f"{removals['spam_or_pure_url']/n_start*100:.2f}%"},
    {"step": "removed_exact_duplicates",    "count": removals["exact_duplicates_removed"], "pct_of_original": f"{removals['exact_duplicates_removed']/n_start*100:.2f}%"},
    {"step": "total_removed",               "count": total_removed,  "pct_of_original": f"{total_removed/n_start*100:.2f}%"},
    {"step": "rows_after_cleaning",         "count": n_final,        "pct_of_original": f"{n_final/n_start*100:.2f}%"},
    {"step": "dev_subset_size",             "count": len(df_dev),    "pct_of_original": f"{len(df_dev)/n_final*100:.2f}% of cleaned"},
]

# Add class distribution after cleaning
label_counts = df["sentiment_label"].value_counts()
for lbl, cnt in label_counts.items():
    report_rows.append({
        "step": f"class_{lbl}_after_cleaning",
        "count": cnt,
        "pct_of_original": f"{cnt/n_final*100:.2f}% of cleaned"
    })

df_report = pd.DataFrame(report_rows)
df_report.to_csv(TABLES / "table2_cleaning_report.csv", index=False, encoding="utf-8")
print(f"\n    Cleaning report  : results/tables/table2_cleaning_report.csv")

# ── Log ────────────────────────────────────────────────────────────────────────
append_log({
    "date": datetime.datetime.now().isoformat(),
    "phase": "Phase2",
    "sample_size": n_final,
    "seed": RANDOM_SEED,
    "result": (f"start={n_start}, after_clean={n_final}, removed={total_removed} "
               f"(empty={removals['empty_or_null_text']}, "
               f"no_label={removals['missing_sentiment_label']}, "
               f"spam={removals['spam_or_pure_url']}, "
               f"dupes={removals['exact_duplicates_removed']}), "
               f"dev_subset={len(df_dev)}")
})

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
print(f"  Rows before cleaning : {n_start:,}")
print(f"  Total removed        : {total_removed:,}  ({total_removed/n_start*100:.2f}%)")
print(f"    └─ empty/null      : {removals['empty_or_null_text']:,}")
print(f"    └─ missing label   : {removals['missing_sentiment_label']:,}")
print(f"    └─ spam/url        : {removals['spam_or_pure_url']:,}")
print(f"    └─ exact dupes     : {removals['exact_duplicates_removed']:,}")
print(f"  Rows after cleaning  : {n_final:,}  ({n_final/n_start*100:.2f}% retained)")
print(f"  Dev subset           : {len(df_dev):,} rows")
print(f"\n  Class distribution after cleaning:")
for lbl, cnt in label_counts.items():
    print(f"    {str(lbl):20s}: {cnt:6,}  ({cnt/n_final*100:.1f}%)")
print("\n  ⏸️  CHECKPOINT A: Review table1_dataset_statistics.csv and this cleaning report.")
print("  If numbers look reasonable, confirm and we will proceed to Stage 2.")
