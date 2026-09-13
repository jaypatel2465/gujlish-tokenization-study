import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Phase 4 -- Build the Gujlish Vocabulary
=========================================
Combines word frequency (from Phase 2 data) with mBERT fragmentation severity
(from Phase 3 output) to rank and select the top 75 Gujlish words to add
to the vocabulary.

Ranking formula:
    score = log2(frequency) * mbert_pieces
    where log2 is used (rather than log10) for a gentler penalty on high-freq words.

Constraints:
    - Minimum frequency threshold: 10 occurrences (user-confirmed)
    - Only words where mbert_pieces >= 2 (fragmented words only)
    - Excludes pure-ASCII or pure-number words (these are already handled well by mBERT)

Outputs:
    results/tables/table3_top_fragmented_words.csv
    results/figures/fig2_top20_fragmented_words.png
"""

import csv
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
TABLES         = ROOT / "results" / "tables"
FIGURES        = ROOT / "results" / "figures"
LOGS           = ROOT / "logs"

FREQ_THRESHOLD = 10    # user-confirmed minimum frequency
VOCAB_SIZE     = 75    # user-confirmed vocabulary addition size

def append_log(row: dict):
    log_file = LOGS / "experiment_log.csv"
    fieldnames = ["date","phase","model","tokenizer","dataset_version",
                  "sample_size","vocab_size","learning_rate","batch_size",
                  "epochs","seed","result"]
    write_header = not log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

print("=" * 60)
print("PHASE 4 -- Gujlish Vocabulary Selection")
print("=" * 60)

# ── Load Phase 3 outputs ───────────────────────────────────────────────────────
freq_path   = DATA_PROCESSED / "word_frequencies.csv"
pieces_path = DATA_PROCESSED / "word_piece_counts.csv"

if not freq_path.exists() or not pieces_path.exists():
    raise FileNotFoundError("Phase 3 outputs not found. Run phase3_fragmentation.py first.")

df_freq   = pd.read_csv(freq_path,   encoding="utf-8")
df_pieces = pd.read_csv(pieces_path, encoding="utf-8")

print(f"\n[1/4] Loaded word frequencies : {len(df_freq):,} words")
print(f"      Loaded piece counts      : {len(df_pieces):,} words")

# Merge
df = df_freq.merge(df_pieces, on="word", how="inner")
print(f"      After merge              : {len(df):,} words")

# ── Tokenization string for display ───────────────────────────────────────────
# Load mBERT tokenizer to show actual wordpiece split in the table
print("\n[2/4] Loading mBERT tokenizer for subword display...")
from transformers import AutoTokenizer
mbert_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def get_subword_display(word: str, tokenizer) -> str:
    """Return the wordpiece split as a readable string, e.g. 'sa ##ras'"""
    tokens = tokenizer.tokenize(word)
    return " | ".join(tokens) if tokens else word

# ── Filter candidates ──────────────────────────────────────────────────────────
print(f"\n[3/4] Applying selection filters...")
print(f"      Min frequency threshold : {FREQ_THRESHOLD}")
print(f"      Only fragmented words   : mbert_pieces >= 2")

import re
from matplotlib import font_manager

def is_pure_ascii(word: str) -> bool:
    """True if word can be fully encoded as ASCII (no Indic/non-ASCII characters).
    This correctly excludes English contractions like don't, it's, I'm."""
    try:
        str(word).encode('ascii')
        return True
    except (UnicodeEncodeError, AttributeError):
        return False

def find_gujarati_font():
    """Try to find a font that supports Gujarati script."""
    candidates = [
        "Noto Sans Gujarati", "Noto Sans", "Lohit Gujarati",
        "Arial Unicode MS", "Segoe UI", "FreeSans"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None  # fallback: use rank labels

n_before = len(df)
df = df[df["frequency"] >= FREQ_THRESHOLD].copy()
print(f"      After freq filter       : {len(df):,} words (removed {n_before - len(df):,})")

df = df[df["mbert_pieces"] >= 2].copy()
print(f"      After frag filter       : {len(df):,} words (fragmented only)")

# Optionally exclude pure ASCII words
df["is_pure_ascii"] = df["word"].apply(is_pure_ascii)
n_ascii = df["is_pure_ascii"].sum()
df = df[~df["is_pure_ascii"]].copy()
print(f"      After ASCII filter      : {len(df):,} words (removed {n_ascii:,} pure-ASCII/English)")
print(f"      Note: 'pure ASCII' includes contractions like don't, it's (not Gujlish vocabulary)")

# ── Compute combined score ─────────────────────────────────────────────────────
# score = log2(frequency) * mbert_pieces
# Rationale: log2 keeps frequent words from dominating; mbert_pieces rewards
#            words that are genuinely hard for the tokenizer.
df["log2_freq"] = np.log2(df["frequency"].astype(float))
df["combined_score"] = df["log2_freq"] * df["mbert_pieces"]

# ── Select top 75 ─────────────────────────────────────────────────────────────
df_sorted = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
actual_size = min(VOCAB_SIZE, len(df_sorted))
df_top = df_sorted.head(actual_size).copy()
if actual_size < VOCAB_SIZE:
    print(f"      [NOTE] Only {actual_size} candidates after all filters (target was {VOCAB_SIZE}).")
    print(f"             Using all {actual_size} available candidates.")

print(f"\n      Selected {len(df_top)} words by combined score (log2(freq) x mbert_pieces)")
print(f"      Score range: {df_top['combined_score'].min():.2f} -- {df_top['combined_score'].max():.2f}")

# Add subword display column
df_top["mbert_tokenization"] = df_top["word"].apply(
    lambda w: get_subword_display(w, mbert_tok)
)
df_top["rank"] = range(1, len(df_top) + 1)

# ── Save Table 3 ───────────────────────────────────────────────────────────────
out_cols = ["rank", "word", "frequency", "mbert_pieces", "muril_pieces",
            "log2_freq", "combined_score", "mbert_tokenization"]
df_out = df_top[out_cols].copy()
df_out["log2_freq"]       = df_out["log2_freq"].round(3)
df_out["combined_score"]  = df_out["combined_score"].round(3)

df_out.to_csv(TABLES / "table3_top_fragmented_words.csv", index=False, encoding="utf-8")
print(f"\n    Saved -> results/tables/table3_top_fragmented_words.csv")

# Also save just the word list for Phase 5 to consume
words_to_add = df_top["word"].tolist()
with open(DATA_PROCESSED / "gujlish_vocab_additions.txt", "w", encoding="utf-8") as f:
    for w in words_to_add:
        f.write(w + "\n")
print(f"    Vocab list saved -> data/processed/gujlish_vocab_additions.txt")

# Print top 20 for inspection
print("\n    Top 20 selected words:")
print(f"    {'Rank':>4} {'Word':>25} {'Freq':>8} {'mBERT pcs':>10} {'Score':>8}  Tokenization")
print("    " + "-" * 90)
for _, row in df_top.head(20).iterrows():
    print(f"    {int(row['rank']):>4} {row['word']:>25} {int(row['frequency']):>8} "
          f"{int(row['mbert_pieces']):>10} {row['combined_score']:>8.2f}  {row['mbert_tokenization']}")

# ── Figure 2: Top 20 fragmented words ─────────────────────────────────────────
print("\n[4/4] Generating fig2_top20_fragmented_words.png...")
top20 = df_top.head(20).copy()

# Find a Gujarati-capable font; fall back to rank-based labels
gujarati_font = find_gujarati_font()
if gujarati_font:
    print(f"    Using font: {gujarati_font} (supports Gujarati script)")
    plt.rcParams['font.family'] = gujarati_font
else:
    print("    No Gujarati font found -- using rank + piece count labels instead of word text.")

# Build display labels: try word text; if font missing show rank
# We always build rank+pieces label as fallback
top20["display_label"] = [
    f"#{int(r['rank'])}  ({int(r['mbert_pieces'])} pcs)"
    for _, r in top20.iterrows()
]
# If a Gujarati font is available, use the actual word
if gujarati_font:
    top20["display_label"] = top20["word"]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')

# Left: combined score bar chart
colors_score = plt.cm.plasma(np.linspace(0.3, 0.9, len(top20)))
axes[0].barh(
    range(len(top20)), top20["combined_score"].values,
    color=colors_score, edgecolor='white', linewidth=0.4
)
axes[0].set_yticks(range(len(top20)))
axes[0].set_yticklabels(top20["display_label"].values, fontsize=10, color='white')
axes[0].invert_yaxis()
axes[0].set_xlabel("Combined Score (log2(freq) x mBERT pieces)", fontsize=11, color='white')
axes[0].set_title("Top 20 Words by Selection Score", fontsize=13, color='white', fontweight='bold')
axes[0].tick_params(colors='white')
for spine in axes[0].spines.values():
    spine.set_edgecolor('#444466')
# Annotate scores
for i, (_, row) in enumerate(top20.iterrows()):
    axes[0].text(row["combined_score"] + 0.1, i,
                 f'{row["combined_score"]:.1f}', va='center', fontsize=8, color='#aaaacc')

# Right: mBERT pieces bar chart
colors_pieces = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top20)))
axes[1].barh(
    range(len(top20)), top20["mbert_pieces"].values,
    color=colors_pieces, edgecolor='white', linewidth=0.4
)
axes[1].set_yticks(range(len(top20)))
axes[1].set_yticklabels(top20["display_label"].values, fontsize=10, color='white')
axes[1].invert_yaxis()
axes[1].set_xlabel("mBERT Subword Pieces", fontsize=11, color='white')
axes[1].set_title("Top 20 Words: mBERT Fragmentation Severity", fontsize=13, color='white', fontweight='bold')
axes[1].tick_params(colors='white')
axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
for spine in axes[1].spines.values():
    spine.set_edgecolor('#444466')
for i, (_, row) in enumerate(top20.iterrows()):
    axes[1].text(row["mbert_pieces"] + 0.05, i, str(int(row["mbert_pieces"])),
                 va='center', fontsize=9, color='white')

font_note = f"(word labels use {gujarati_font})" if gujarati_font else "(labels show rank + piece count; see table3_top_fragmented_words.csv for Gujarati text)"
fig.suptitle(f"Gujlish Vocabulary Candidates: Top 20 Most Fragmented High-Frequency Words\n{font_note}",
             fontsize=13, color='white', fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIGURES / "fig2_top20_fragmented_words.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"    Saved -> results/figures/fig2_top20_fragmented_words.png")

# ── Log ────────────────────────────────────────────────────────────────────────
append_log({
    "date": datetime.datetime.now().isoformat(),
    "phase": "Phase4",
    "tokenizer": "mBERT",
    "sample_size": len(df),
    "vocab_size": VOCAB_SIZE,
    "result": (f"candidates_after_filter={len(df)}, "
               f"selected={VOCAB_SIZE}, "
               f"freq_threshold={FREQ_THRESHOLD}, "
               f"score_range={df_top['combined_score'].min():.2f}-{df_top['combined_score'].max():.2f}")
})

print("\n" + "=" * 60)
print("PHASE 4 COMPLETE")
print("=" * 60)
print(f"  Freq threshold           : >= {FREQ_THRESHOLD} occurrences")
print(f"  Fragmentation filter     : mBERT pieces >= 2")
print(f"  Non-ASCII only           : Yes")
print(f"  Candidates after filters : {len(df_sorted):,}")
print(f"  Words selected           : {VOCAB_SIZE}")
print(f"  Output: table3_top_fragmented_words.csv, fig2_top20_fragmented_words.png")
