import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Phase 5b -- Fragmentation Re-measurement Only
==============================================
The adapted model is already saved to models/mbert_adapted/.
This script only re-runs the fragmentation analysis and generates
Table 4 and Figure 3.
"""

import csv, datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
TABLES         = ROOT / "results" / "tables"
FIGURES        = ROOT / "results" / "figures"
MODELS         = ROOT / "models"
LOGS           = ROOT / "logs"

print("=" * 60)
print("PHASE 5b -- Fragmentation Re-measurement (adapted tokenizer)")
print("=" * 60)

# Load both tokenizers
print("\n[1/3] Loading tokenizers...")
baseline_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
adapted_tok  = AutoTokenizer.from_pretrained(str(MODELS / "mbert_adapted"))
baseline_vocab = len(baseline_tok)
adapted_vocab  = len(adapted_tok)
print(f"  Baseline vocab : {baseline_vocab:,}")
print(f"  Adapted vocab  : {adapted_vocab:,}  (+{adapted_vocab - baseline_vocab} tokens added)")

# Load word data
df_pieces = pd.read_csv(DATA_PROCESSED / "word_piece_counts.csv", encoding="utf-8")
df_freq   = pd.read_csv(DATA_PROCESSED / "word_frequencies.csv",  encoding="utf-8")
df_merged = df_pieces.merge(df_freq, on="word", how="inner")

# Drop rows where word is NaN
df_merged = df_merged.dropna(subset=["word"]).copy()
df_merged["word"] = df_merged["word"].astype(str)
all_words  = df_merged["word"].tolist()
freq_map   = dict(zip(df_merged["word"], df_merged["frequency"]))
print(f"  Words to analyze: {len(all_words):,}")

# Count pieces with adapted tokenizer
print("\n[2/3] Re-counting subword pieces with adapted tokenizer...")

def count_pieces_safe(tokenizer, words):
    counts = []
    for word in words:
        try:
            ids = tokenizer.encode(str(word), add_special_tokens=False)
            counts.append(max(len(ids), 1))
        except Exception:
            counts.append(1)
    return np.array(counts, dtype=np.float32)

baseline_counts = df_merged["mbert_pieces"].values.astype(np.float32)
adapted_counts  = count_pieces_safe(adapted_tok, all_words)
freq_weights    = np.array([freq_map[w] for w in all_words], dtype=np.float64)
freq_weights   /= freq_weights.sum()

def compute_stats(counts, name):
    return {
        "tokenizer":              name,
        "avg_subwords_per_word":  float(np.mean(counts)),
        "pct_words_fragmented":   float(np.mean(counts >= 2) * 100),
        "pct_words_single_token": float(np.mean(counts == 1) * 100),
        "pct_words_3plus":        float(np.mean(counts >= 3) * 100),
        "max_pieces":             int(np.max(counts)),
        "wavg_subwords":          float(np.sum(counts * freq_weights)),
        "wpct_fragmented":        float(np.sum((counts >= 2) * freq_weights) * 100),
    }

before = compute_stats(baseline_counts, "mBERT baseline")
after  = compute_stats(adapted_counts,  "mBERT adapted")

def pct_chg(b, a):
    if b == 0:
        return "N/A"
    return f"{(a - b) / b * 100:+.2f}%"

# Print results
print("\n  Before vs After Adaptation:")
metric_pairs = [
    ("avg_subwords_per_word", "Avg subwords/word"),
    ("pct_words_fragmented",  "% words fragmented (2+ pieces)"),
    ("pct_words_single_token","% words single token"),
    ("pct_words_3plus",       "% words 3+ pieces"),
    ("max_pieces",            "Max pieces (any word)"),
    ("wavg_subwords",         "Freq-weighted avg subwords/word"),
    ("wpct_fragmented",       "Freq-weighted % fragmented"),
]
rows = []
print(f"  {'Metric':50s} {'Before':>10} {'After':>10} {'Change':>10}")
print("  " + "-" * 85)
for key, label in metric_pairs:
    bv = before[key]
    av = after[key]
    chg = pct_chg(bv, av)
    abs_chg = round(av - bv, 4) if isinstance(bv, (int, float)) else "N/A"
    print(f"  {label:50s} {str(round(bv,4) if isinstance(bv,float) else bv):>10} "
          f"{str(round(av,4) if isinstance(av,float) else av):>10} {chg:>10}")
    rows.append({
        "metric":          label,
        "mBERT_baseline":  round(bv, 4) if isinstance(bv, float) else bv,
        "mBERT_adapted":   round(av, 4) if isinstance(av, float) else av,
        "absolute_change": abs_chg,
        "pct_change":      chg,
    })

# Save Table 4
df_t4 = pd.DataFrame(rows)
df_t4.to_csv(TABLES / "table4_before_after_fragmentation.csv", index=False, encoding="utf-8")
print(f"\n  Saved -> results/tables/table4_before_after_fragmentation.csv")

# Figure 3
print("\n[3/3] Generating fig3_fragmentation_before_after.png...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")

# Left: grouped bar
mlabels = ["Avg\nsub/word", "% Frag\n(2+)", "% Single\ntoken", "% 3+\npieces"]
bvals = [before["avg_subwords_per_word"], before["pct_words_fragmented"],
         before["pct_words_single_token"], before["pct_words_3plus"]]
avals = [after["avg_subwords_per_word"],  after["pct_words_fragmented"],
         after["pct_words_single_token"],  after["pct_words_3plus"]]

x, w = np.arange(len(mlabels)), 0.35
b1 = axes[0].bar(x - w/2, bvals, w, label="mBERT Baseline", color="#e94560", alpha=0.85, edgecolor="white", linewidth=0.5)
b2 = axes[0].bar(x + w/2, avals, w, label="mBERT Adapted",  color="#53d8fb", alpha=0.85, edgecolor="white", linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(mlabels, color="white", fontsize=10)
axes[0].set_ylabel("Value", color="white")
axes[0].tick_params(colors="white")
axes[0].set_title("Fragmentation Metrics: Before vs After", color="white", fontweight="bold")
axes[0].legend(fontsize=10, framealpha=0.3, facecolor="#1a1a2e", labelcolor="white")
for spine in axes[0].spines.values():
    spine.set_edgecolor("#444466")
for bars in [b1, b2]:
    for rect in bars:
        h = rect.get_height()
        axes[0].text(rect.get_x() + rect.get_width() / 2., h + 0.3,
                     f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="white")

# Right: histogram overlay
bins = np.arange(0.5, 14.5, 1.0)
axes[1].hist(baseline_counts, bins=bins, weights=freq_weights * 100, alpha=0.7,
             label="Baseline", color="#e94560", edgecolor="white", linewidth=0.5)
axes[1].hist(adapted_counts, bins=bins, weights=freq_weights * 100, alpha=0.7,
             label="Adapted",  color="#53d8fb", edgecolor="white", linewidth=0.5)
axes[1].set_xlabel("Subword Pieces per Word", color="white")
axes[1].set_ylabel("% Word Tokens (freq-weighted)", color="white")
axes[1].set_title("Distribution Shift After Adaptation", color="white", fontweight="bold")
axes[1].tick_params(colors="white")
axes[1].legend(fontsize=10, framealpha=0.3, facecolor="#1a1a2e", labelcolor="white")
for spine in axes[1].spines.values():
    spine.set_edgecolor("#444466")

fig.suptitle("mBERT Vocabulary Adaptation: Fragmentation Before vs After",
             color="white", fontweight="bold", fontsize=13)
plt.tight_layout()
fig.savefig(FIGURES / "fig3_fragmentation_before_after.png", dpi=300,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved -> results/figures/fig3_fragmentation_before_after.png")

# Log
log_file = LOGS / "experiment_log.csv"
fieldnames = ["date","phase","model","tokenizer","dataset_version",
              "sample_size","vocab_size","learning_rate","batch_size","epochs","seed","result"]
write_header = not log_file.exists()
with open(log_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow({
        "date":      datetime.datetime.now().isoformat(),
        "phase":     "Phase5",
        "model":     "bert-base-multilingual-cased",
        "tokenizer": "mBERT adapted",
        "sample_size": len(all_words),
        "vocab_size": f"{baseline_vocab}->{adapted_vocab}",
        "result": (f"tokens_added={adapted_vocab - baseline_vocab}, "
                   f"avg: {before['avg_subwords_per_word']:.3f}->{after['avg_subwords_per_word']:.3f}, "
                   f"frag%: {before['pct_words_fragmented']:.1f}%->{after['pct_words_fragmented']:.1f}%")
    })

print("\n" + "=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)
print(f"  Tokens added           : {adapted_vocab - baseline_vocab}")
print(f"  Avg subwords/word      : {before['avg_subwords_per_word']:.3f} -> {after['avg_subwords_per_word']:.3f}  ({pct_chg(before['avg_subwords_per_word'], after['avg_subwords_per_word'])})")
print(f"  % fragmented           : {before['pct_words_fragmented']:.1f}% -> {after['pct_words_fragmented']:.1f}%  ({pct_chg(before['pct_words_fragmented'], after['pct_words_fragmented'])})")
print(f"  % single token         : {before['pct_words_single_token']:.1f}% -> {after['pct_words_single_token']:.1f}%  ({pct_chg(before['pct_words_single_token'], after['pct_words_single_token'])})")
print("\n  [CHECKPOINT B READY] Review table4_before_after_fragmentation.csv and fig3.")
