import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Phase 3 -- Measure Tokenizer Fragmentation Baseline
=====================================================
Loads mBERT and MuRIL tokenizers. For every word in the cleaned dataset,
records how many subword pieces each tokenizer produces. Computes:
  - Average subwords/word
  - % of words fragmented (split into 2+ pieces)
  - Average fragmentation ratio
  - % of words represented as a single token

Outputs:
  results/tables/table2_fragmentation_baseline.csv
  results/figures/fig1_subwords_per_word_distribution.png
"""

import csv
import datetime
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os
ROOT           = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache")
DATA_PROCESSED = ROOT / "data" / "processed"
TABLES         = ROOT / "results" / "tables"
FIGURES        = ROOT / "results" / "figures"
LOGS           = ROOT / "logs"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
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
print("PHASE 3 -- Tokenizer Fragmentation Baseline")
print("=" * 60)

# ── Load cleaned dataset ───────────────────────────────────────────────────────
input_path = DATA_PROCESSED / "filtered_clean.csv"
if not input_path.exists():
    raise FileNotFoundError("Run phase2_clean.py first.")

df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
print(f"\n[1/4] Loaded {len(df):,} rows from filtered_clean.csv")

# ── Tokenize-safe word splitting ───────────────────────────────────────────────
def split_into_words(text: str) -> list:
    """
    Split text into word tokens using whitespace + basic punctuation stripping.
    Preserves Gujarati/Devanagari/Latin script words. Discards pure-punctuation tokens.
    """
    # split on whitespace
    tokens = str(text).split()
    words = []
    for tok in tokens:
        # strip leading/trailing punctuation but keep internal punctuation (apostrophes etc.)
        word = tok.strip('.,!?;:"\'()[]{}|<>/*@#$%^&~`')
        if word and not re.fullmatch(r'[\W_]+', word):
            words.append(word)
    return words

print("[2/4] Extracting unique word vocabulary from dataset...")
word_freq: Counter = Counter()
for text in df["text"].astype(str):
    for word in split_into_words(text):
        word_freq[word] += 1

all_words = list(word_freq.keys())
print(f"    Total unique words (types) : {len(all_words):,}")
print(f"    Total word tokens          : {sum(word_freq.values()):,}")

# Save word frequency for Phase 4
freq_df = pd.DataFrame(
    [{"word": w, "frequency": c} for w, c in word_freq.most_common()],
    columns=["word", "frequency"]
)
freq_df.to_csv(DATA_PROCESSED / "word_frequencies.csv", index=False, encoding="utf-8")
print(f"    Word frequencies saved -> data/processed/word_frequencies.csv")

# ── Load tokenizers ────────────────────────────────────────────────────────────
print("\n[3/4] Loading tokenizers (downloading if needed)...")
from transformers import AutoTokenizer

print("    Loading bert-base-multilingual-cased (mBERT)...")
mbert_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
print(f"    mBERT vocab size: {len(mbert_tok):,}")

print("    Loading google/muril-base-cased (MuRIL)...")
muril_tok = AutoTokenizer.from_pretrained("google/muril-base-cased")
print(f"    MuRIL vocab size: {len(muril_tok):,}")

# ── Fragmentation analysis ─────────────────────────────────────────────────────
print("\n[4/4] Computing fragmentation for all unique words...")

def analyze_fragmentation(tokenizer, words: list, tok_name: str) -> dict:
    """
    For each word, tokenize and record the number of subword pieces.
    Returns a dict with per-word piece counts and aggregate stats.
    """
    piece_counts = []
    word_piece_map = {}   # word -> n_pieces (for Phase 4 to reuse)

    for word in words:
        # tokenize as a single word (no special tokens)
        ids = tokenizer.encode(word, add_special_tokens=False)
        n_pieces = len(ids)
        piece_counts.append(n_pieces)
        word_piece_map[word] = n_pieces

    piece_counts = np.array(piece_counts, dtype=np.float32)

    stats = {
        "tokenizer": tok_name,
        "vocab_size": len(tokenizer),
        "unique_words_analyzed": len(words),
        "avg_subwords_per_word": float(np.mean(piece_counts)),
        "median_subwords_per_word": float(np.median(piece_counts)),
        "pct_words_fragmented": float(np.mean(piece_counts >= 2) * 100),
        "pct_words_single_token": float(np.mean(piece_counts == 1) * 100),
        "pct_words_3plus_pieces": float(np.mean(piece_counts >= 3) * 100),
        "max_pieces_any_word": int(np.max(piece_counts)),
        "avg_fragmentation_ratio": float(
            np.mean(piece_counts[piece_counts > 1])
            if np.any(piece_counts > 1) else 1.0
        ),
    }
    return stats, piece_counts, word_piece_map

mbert_stats, mbert_counts, mbert_word_map = analyze_fragmentation(
    mbert_tok, all_words, "mBERT (bert-base-multilingual-cased)"
)
muril_stats, muril_counts, muril_word_map = analyze_fragmentation(
    muril_tok, all_words, "MuRIL (google/muril-base-cased)"
)

# Print summary
for stats in [mbert_stats, muril_stats]:
    print(f"\n  [{stats['tokenizer']}]")
    print(f"    Vocab size                   : {stats['vocab_size']:,}")
    print(f"    Unique words analyzed        : {stats['unique_words_analyzed']:,}")
    print(f"    Avg subwords/word            : {stats['avg_subwords_per_word']:.3f}")
    print(f"    Median subwords/word         : {stats['median_subwords_per_word']:.1f}")
    print(f"    % words fragmented (2+ pcs)  : {stats['pct_words_fragmented']:.1f}%")
    print(f"    % words single token         : {stats['pct_words_single_token']:.1f}%")
    print(f"    % words 3+ pieces            : {stats['pct_words_3plus_pieces']:.1f}%")
    print(f"    Max pieces (any word)        : {stats['max_pieces_any_word']}")
    print(f"    Avg ratio (fragmented only)  : {stats['avg_fragmentation_ratio']:.3f}")

# ── Save Table 2 ───────────────────────────────────────────────────────────────
rows = []
for stats in [mbert_stats, muril_stats]:
    rows.append({
        "tokenizer":                  stats["tokenizer"],
        "vocab_size":                 stats["vocab_size"],
        "unique_words_analyzed":      stats["unique_words_analyzed"],
        "avg_subwords_per_word":      round(stats["avg_subwords_per_word"], 4),
        "median_subwords_per_word":   round(stats["median_subwords_per_word"], 1),
        "pct_words_fragmented":       round(stats["pct_words_fragmented"], 2),
        "pct_words_single_token":     round(stats["pct_words_single_token"], 2),
        "pct_words_3plus_pieces":     round(stats["pct_words_3plus_pieces"], 2),
        "max_pieces_any_word":        stats["max_pieces_any_word"],
        "avg_fragmentation_ratio":    round(stats["avg_fragmentation_ratio"], 4),
    })

df_table = pd.DataFrame(rows)
df_table.to_csv(TABLES / "table2_fragmentation_baseline.csv", index=False, encoding="utf-8")
print(f"\n    Saved -> results/tables/table2_fragmentation_baseline.csv")

# Save word-level piece counts for Phase 4
mbert_word_df = pd.DataFrame(
    [{"word": w, "mbert_pieces": mbert_word_map[w], "muril_pieces": muril_word_map[w]}
     for w in all_words],
    columns=["word", "mbert_pieces", "muril_pieces"]
)
mbert_word_df.to_csv(DATA_PROCESSED / "word_piece_counts.csv", index=False, encoding="utf-8")
print(f"    Saved word-level piece counts -> data/processed/word_piece_counts.csv")

# ── Figure 1: Subwords-per-word distribution ───────────────────────────────────
print("\n    Generating fig1_subwords_per_word_distribution.png...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

max_pieces = max(int(mbert_counts.max()), int(muril_counts.max()), 8)
bins = np.arange(0.5, min(max_pieces + 1.5, 15.5), 1.0)

# Weight by word frequency so the plot reflects actual token usage
mbert_weights = np.array([word_freq[w] for w in all_words], dtype=np.float64)
muril_weights = mbert_weights.copy()
mbert_weights /= mbert_weights.sum()
muril_weights /= muril_weights.sum()

ax.hist(mbert_counts, bins=bins, weights=mbert_weights * 100,
        alpha=0.75, label='mBERT', color='#e94560', edgecolor='white', linewidth=0.5)
ax.hist(muril_counts, bins=bins, weights=muril_weights * 100,
        alpha=0.75, label='MuRIL', color='#0f3460', edgecolor='white', linewidth=0.5)

ax.set_xlabel("Subword Pieces per Word", fontsize=13, color='white', labelpad=10)
ax.set_ylabel("% of Word Tokens (frequency-weighted)", fontsize=13, color='white', labelpad=10)
ax.set_title("Subword Fragmentation Distribution: mBERT vs MuRIL\n(Gujarati-English Code-Mixed Dataset)",
             fontsize=14, color='white', pad=15, fontweight='bold')
ax.tick_params(colors='white', labelsize=11)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
for spine in ax.spines.values():
    spine.set_edgecolor('#444466')

# Annotate avg lines
ax.axvline(mbert_stats["avg_subwords_per_word"], color='#ff6b6b', linestyle='--', linewidth=1.5,
           label=f"mBERT avg ({mbert_stats['avg_subwords_per_word']:.2f})")
ax.axvline(muril_stats["avg_subwords_per_word"], color='#74b9ff', linestyle='--', linewidth=1.5,
           label=f"MuRIL avg ({muril_stats['avg_subwords_per_word']:.2f})")

legend = ax.legend(fontsize=11, framealpha=0.3, facecolor='#1a1a2e', labelcolor='white')
plt.tight_layout()
fig.savefig(FIGURES / "fig1_subwords_per_word_distribution.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"    Saved -> results/figures/fig1_subwords_per_word_distribution.png")

# ── Log ────────────────────────────────────────────────────────────────────────
append_log({
    "date": datetime.datetime.now().isoformat(),
    "phase": "Phase3",
    "tokenizer": "mBERT + MuRIL",
    "sample_size": len(all_words),
    "result": (f"mBERT: avg={mbert_stats['avg_subwords_per_word']:.3f} "
               f"pct_frag={mbert_stats['pct_words_fragmented']:.1f}% | "
               f"MuRIL: avg={muril_stats['avg_subwords_per_word']:.3f} "
               f"pct_frag={muril_stats['pct_words_fragmented']:.1f}%")
})

print("\n" + "=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)
print(f"  mBERT avg subwords/word  : {mbert_stats['avg_subwords_per_word']:.3f}")
print(f"  MuRIL avg subwords/word  : {muril_stats['avg_subwords_per_word']:.3f}")
print(f"  mBERT % fragmented       : {mbert_stats['pct_words_fragmented']:.1f}%")
print(f"  MuRIL % fragmented       : {muril_stats['pct_words_fragmented']:.1f}%")
print(f"  Output: table2_fragmentation_baseline.csv, fig1_subwords_per_word_distribution.png")
