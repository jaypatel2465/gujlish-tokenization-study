import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Phase 1 -- Acquire and Verify the Dataset
=========================================
Downloads the Gujarati-English code-mixed sentiment dataset from HuggingFace,
applies a Gujarati-English dominance filter, reports label quality, generates
a human-audit sample, and produces table1_dataset_statistics.csv.

Dominance filter (documented):
    Keep rows where:
        (gujarati_token_count + english_token_count) > 1.5 * hindi_token_count
    AND NOT (gujarati_token_count == 0 AND english_token_count == 0)

    Rationale: A multiplier of 1.5x ensures the text is clearly Gujlish-dominant,
    not merely Hindi-English with a few Gujarati words sprinkled in. Rows where
    both Gujarati and English counts are zero are excluded as they have no
    Gujlish signal. Rows where hindi_token_count == 0 automatically pass.
"""

import os
import sys
import csv
import json
import random
import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths & Environment ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache")
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
TABLES          = ROOT / "results" / "tables"
LOGS            = ROOT / "logs"

for d in [DATA_RAW, DATA_PROCESSED, TABLES, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS / "experiment_log.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Logging helper ─────────────────────────────────────────────────────────────
def append_log(row: dict):
    fieldnames = [
        "date", "phase", "model", "tokenizer", "dataset_version",
        "sample_size", "vocab_size", "learning_rate", "batch_size",
        "epochs", "seed", "result"
    ]
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row_full = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(row_full)

# ── Step 1: Download ───────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — Dataset Acquisition & Verification")
print("=" * 60)

print("\n[1/5] Downloading dataset from HuggingFace...")
try:
    from datasets import load_dataset
    ds = load_dataset(
        "ShrutiPatel3011/gujarati-english-codemixed-sentiment"
    )
    # Combine all splits into one DataFrame
    frames = []
    for split_name, split_data in ds.items():
        df_split = split_data.to_pandas()
        df_split["_split"] = split_name
        frames.append(df_split)
    df_raw = pd.concat(frames, ignore_index=True)
    print(f"    Downloaded {len(df_raw):,} total rows across splits: {list(ds.keys())}")
    df_raw.to_csv(DATA_RAW / "raw_dataset.csv", index=False, encoding="utf-8")
    print(f"    Saved raw data → data/raw/raw_dataset.csv")
except Exception as e:
    print(f"    ERROR downloading from HuggingFace: {e}")
    print("    Attempting to load from local cache if available...")
    cache_path = DATA_RAW / "raw_dataset.csv"
    if cache_path.exists():
        df_raw = pd.read_csv(cache_path, encoding="utf-8")
        print(f"    Loaded {len(df_raw):,} rows from local cache.")
    else:
        print("    FATAL: No local cache found. Cannot proceed with Phase 1.")
        sys.exit(1)

print(f"\n    Columns: {list(df_raw.columns)}")
print(f"    Sample row:\n{df_raw.iloc[0].to_dict()}")

# ── Standardise column names ────────────────────────────────────────────────────
# Detect token count column names flexibly
col_map = {}
for col in df_raw.columns:
    cl = col.lower().replace("-", "_").replace(" ", "_")
    if "gujarati" in cl and "token" in cl:
        col_map["gujarati_token_count"] = col
    elif "english" in cl and "token" in cl:
        col_map["english_token_count"] = col
    elif "hindi" in cl and "token" in cl:
        col_map["hindi_token_count"] = col
    elif "cmi" in cl:
        col_map["cmi_score"] = col
    elif "sentiment" in cl and "label" in cl:
        col_map["sentiment_label"] = col
    elif "label_source" in cl:
        col_map["label_source"] = col

print(f"\n    Detected column mapping: {col_map}")

# Rename for consistent downstream use
df_raw = df_raw.rename(columns={v: k for k, v in col_map.items()})

# Fill missing token count columns with 0
for tc in ["gujarati_token_count", "english_token_count", "hindi_token_count"]:
    if tc not in df_raw.columns:
        print(f"    WARNING: '{tc}' column not found — filling with 0.")
        df_raw[tc] = 0
    else:
        df_raw[tc] = pd.to_numeric(df_raw[tc], errors="coerce").fillna(0)

if "cmi_score" not in df_raw.columns:
    print("    WARNING: 'cmi_score' column not found — filling with NaN.")
    df_raw["cmi_score"] = np.nan

# Detect text column
text_col = None
for candidate in ["text", "sentence", "comment", "content", "utterance"]:
    if candidate in df_raw.columns:
        text_col = candidate
        break
if text_col is None:
    # Use the first string-type column
    for col in df_raw.columns:
        if df_raw[col].dtype == object:
            text_col = col
            break
print(f"    Text column detected: '{text_col}'")
df_raw = df_raw.rename(columns={text_col: "text"})

# ── Step 2: Gujarati-English dominance filter ──────────────────────────────────
print("\n[2/5] Applying Gujarati-English dominance filter...")
MULTIPLIER = 1.5
print(f"    Filter: (gujarati_tokens + english_tokens) > {MULTIPLIER} x hindi_tokens")
print(f"         AND NOT (gujarati_tokens == 0 AND english_tokens == 0)")

total_before = len(df_raw)
mask_dominant = (
    (df_raw["gujarati_token_count"] + df_raw["english_token_count"])
    > (MULTIPLIER * df_raw["hindi_token_count"])
)
mask_has_gujeng = ~(
    (df_raw["gujarati_token_count"] == 0) & (df_raw["english_token_count"] == 0)
)
df_filtered = df_raw[mask_dominant & mask_has_gujeng].copy().reset_index(drop=True)
total_after = len(df_filtered)

print(f"    Before filter : {total_before:,} rows")
print(f"    After filter  : {total_after:,} rows  ({total_after/total_before*100:.1f}% retained)")

# ── Step 3: Label quality audit ────────────────────────────────────────────────
print("\n[3/5] Label quality audit...")

if "label_source" in df_filtered.columns:
    label_src_counts = df_filtered["label_source"].value_counts(dropna=False)
    print("    label_source breakdown (after filter):")
    for src, cnt in label_src_counts.items():
        print(f"        {str(src):30s}: {cnt:6,}  ({cnt/total_after*100:.1f}%)")
else:
    print("    WARNING: 'label_source' column not found in dataset.")
    df_filtered["label_source"] = "unknown"

if "sentiment_label" in df_filtered.columns:
    label_counts = df_filtered["sentiment_label"].value_counts(dropna=False)
    print("    sentiment_label distribution (after filter):")
    for lbl, cnt in label_counts.items():
        print(f"        {str(lbl):20s}: {cnt:6,}  ({cnt/total_after*100:.1f}%)")
else:
    print("    WARNING: 'sentiment_label' column not found.")
    df_filtered["sentiment_label"] = "unknown"

# ── Step 4: Human-audit sample ─────────────────────────────────────────────────
print("\n[4/5] Generating human-audit sample...")

AUDIT_SIZE = 150
# Stratified sampling by sentiment_label
unique_labels = df_filtered["sentiment_label"].unique()
n_labels = len(unique_labels)
per_class = max(1, AUDIT_SIZE // n_labels)
audit_frames = []
for lbl in unique_labels:
    pool = df_filtered[df_filtered["sentiment_label"] == lbl]
    n_sample = min(per_class, len(pool))
    audit_frames.append(pool.sample(n=n_sample, random_state=RANDOM_SEED))
df_audit = pd.concat(audit_frames, ignore_index=True)
# Top up to AUDIT_SIZE if needed
if len(df_audit) < AUDIT_SIZE:
    remaining = df_filtered.drop(df_audit.index, errors="ignore")
    extra = remaining.sample(n=min(AUDIT_SIZE - len(df_audit), len(remaining)), random_state=RANDOM_SEED)
    df_audit = pd.concat([df_audit, extra], ignore_index=True)
df_audit = df_audit.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle

audit_out = df_audit[["text", "sentiment_label", "cmi_score",
                        "gujarati_token_count", "english_token_count",
                        "hindi_token_count"]].copy()
audit_out["my_label"] = ""  # blank column for student to fill in
audit_out["my_notes"] = ""  # optional notes column
audit_out.to_csv(DATA_PROCESSED / "human_audit_sample.csv", index=False, encoding="utf-8")
print(f"    Saved {len(audit_out)} rows -> data/processed/human_audit_sample.csv")
print("    [!] Please fill in the 'my_label' column before Phase 7 final evaluation.")

# ── Step 5: Table 1 — Dataset Statistics ──────────────────────────────────────
print("\n[5/5] Computing Table 1 — Dataset Statistics...")

# Sentence length (words)
df_filtered["word_count"] = df_filtered["text"].astype(str).apply(lambda x: len(x.split()))

stats = {
    "metric": [],
    "value": []
}

def add_stat(k, v):
    stats["metric"].append(k)
    stats["value"].append(v)

add_stat("total_rows_raw", total_before)
add_stat("total_rows_after_filter", total_after)
add_stat("filter_retention_pct", f"{total_after/total_before*100:.2f}%")
add_stat("filter_multiplier_used", MULTIPLIER)

# Class distribution
for lbl, cnt in df_filtered["sentiment_label"].value_counts().items():
    add_stat(f"class_{lbl}_count", cnt)
    add_stat(f"class_{lbl}_pct", f"{cnt/total_after*100:.2f}%")

add_stat("avg_word_count", f"{df_filtered['word_count'].mean():.2f}")
add_stat("median_word_count", f"{df_filtered['word_count'].median():.2f}")
add_stat("min_word_count", int(df_filtered["word_count"].min()))
add_stat("max_word_count", int(df_filtered["word_count"].max()))

if df_filtered["cmi_score"].notna().any():
    add_stat("avg_cmi_score", f"{df_filtered['cmi_score'].mean():.4f}")
    add_stat("median_cmi_score", f"{df_filtered['cmi_score'].median():.4f}")
    add_stat("min_cmi_score", f"{df_filtered['cmi_score'].min():.4f}")
    add_stat("max_cmi_score", f"{df_filtered['cmi_score'].max():.4f}")

avg_guj = df_filtered["gujarati_token_count"].mean()
avg_eng = df_filtered["english_token_count"].mean()
avg_hin = df_filtered["hindi_token_count"].mean()
total_avg = avg_guj + avg_eng + avg_hin if (avg_guj + avg_eng + avg_hin) > 0 else 1
add_stat("avg_gujarati_tokens_per_sentence", f"{avg_guj:.2f}")
add_stat("avg_english_tokens_per_sentence",  f"{avg_eng:.2f}")
add_stat("avg_hindi_tokens_per_sentence",    f"{avg_hin:.2f}")
add_stat("gujarati_token_share_pct", f"{avg_guj/total_avg*100:.1f}%")
add_stat("english_token_share_pct",  f"{avg_eng/total_avg*100:.1f}%")
add_stat("hindi_token_share_pct",    f"{avg_hin/total_avg*100:.1f}%")

# Label source breakdown
if "label_source" in df_filtered.columns:
    for src, cnt in df_filtered["label_source"].value_counts(dropna=False).items():
        add_stat(f"label_source_{str(src)}_count", cnt)
        add_stat(f"label_source_{str(src)}_pct", f"{cnt/total_after*100:.2f}%")

df_stats = pd.DataFrame(stats)
df_stats.to_csv(TABLES / "table1_dataset_statistics.csv", index=False, encoding="utf-8")
print(f"    Saved -> results/tables/table1_dataset_statistics.csv")

# Save the filtered dataset for Phase 2
df_filtered.to_csv(DATA_PROCESSED / "filtered_dataset.csv", index=False, encoding="utf-8")
print(f"    Saved filtered dataset -> data/processed/filtered_dataset.csv")

# ── Draft academic access email ────────────────────────────────────────────────
email_text = """To: [Researcher's email — find on ResearchGate profile]
Subject: Academic Access Request — Gujarati-English Code-Mixed Dataset

Dear Researcher,

I am a Computer Engineering student working on empirical NLP research focused on
subword tokenization and its impact on semantic representation in Gujarati-English
code-mixed (Gujlish) text.

I came across your paper "Language Identification and Translation of English and
Gujarati code-mixed data" which describes a 44,672-sentence five-class dataset for
Gujarati-English code-mixed NLP. I am writing to respectfully request academic access
to this dataset for the purpose of comparison and validation in my research.

My research is non-commercial and will be submitted as part of my academic coursework.
I am happy to:
- Cite your work appropriately in any resulting publication
- Share my findings with you upon completion
- Comply with any data usage restrictions you specify

Could you please let me know if access can be arranged, or if there is an official
request process I should follow?

Thank you very much for your time and your contribution to the field.

Best regards,
[Your Name]
[Your Institution]
[Your Email]
"""
with open(LOGS / "academic_access_email.txt", "w", encoding="utf-8") as f:
    f.write(email_text)
print(f"\n    Drafted academic access email -> logs/academic_access_email.txt")

# ── Log this run ───────────────────────────────────────────────────────────────
append_log({
    "date": datetime.datetime.now().isoformat(),
    "phase": "Phase1",
    "dataset_version": "ShrutiPatel3011/gujarati-english-codemixed-sentiment",
    "sample_size": total_after,
    "seed": RANDOM_SEED,
    "result": f"filtered={total_after}/{total_before}, audit_sample={len(audit_out)}"
})

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE")
print("=" * 60)
print(f"  Raw rows         : {total_before:,}")
print(f"  After filter     : {total_after:,}  ({total_after/total_before*100:.1f}%)")
print(f"  Audit sample     : {len(audit_out)} rows -> data/processed/human_audit_sample.csv")
print(f"  Table 1          : results/tables/table1_dataset_statistics.csv")
print(f"  Filtered data    : data/processed/filtered_dataset.csv")
print("\n  [CHECKPOINT A] Review table1_dataset_statistics.csv before proceeding.")
print("  [ACTION]  Fill in 'my_label' in human_audit_sample.csv at your convenience.")
