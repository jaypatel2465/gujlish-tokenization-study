import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Phase 5 -- Vocabulary Adaptation and Re-measurement
=====================================================
1. Adds the top 75 Gujlish words (selected in Phase 4) to the mBERT tokenizer.
2. Loads the mBERT model and resizes its embedding matrix to cover the new tokens.
   New token embeddings are randomly initialized (standard HuggingFace behaviour).
3. Saves the adapted tokenizer and model to models/mbert_adapted/.
4. Re-runs the exact same fragmentation analysis from Phase 3 using the adapted
   tokenizer.
5. Produces a before/after comparison table and figure.

Outputs:
    models/mbert_adapted/               (tokenizer + model weights)
    results/tables/table4_before_after_fragmentation.csv
    results/figures/fig3_fragmentation_before_after.png
"""

import csv
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
ROOT           = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache")
DATA_PROCESSED = ROOT / "data" / "processed"
TABLES         = ROOT / "results" / "tables"
FIGURES        = ROOT / "results" / "figures"
MODELS         = ROOT / "models"
LOGS           = ROOT / "logs"
MODELS.mkdir(parents=True, exist_ok=True)

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
print("PHASE 5 -- Vocabulary Adaptation & Re-measurement")
print("=" * 60)

# ── Load vocab addition list ───────────────────────────────────────────────────
vocab_path = DATA_PROCESSED / "gujlish_vocab_additions.txt"
if not vocab_path.exists():
    raise FileNotFoundError("Run phase4_vocab_selection.py first.")

with open(vocab_path, "r", encoding="utf-8") as f:
    new_tokens = [line.strip() for line in f if line.strip()]

print(f"\n[1/5] Loaded {len(new_tokens)} Gujlish tokens to add to mBERT vocab.")

# ── Load baseline mBERT tokenizer and model ────────────────────────────────────
print("\n[2/5] Loading baseline mBERT tokenizer and model (downloading if needed)...")
print("      Note: bert-base-multilingual-cased is ~714MB. Please wait.")
from transformers import AutoTokenizer, BertForSequenceClassification
import torch

baseline_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
baseline_vocab_size = len(baseline_tok)
print(f"      Baseline mBERT vocab size  : {baseline_vocab_size:,}")

print("      Loading mBERT model weights...")
baseline_model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=3,          # positive / neutral / negative
    ignore_mismatched_sizes=True
)
print(f"      Model embedding size (before): {baseline_model.get_input_embeddings().weight.shape}")

# ── Add new tokens ─────────────────────────────────────────────────────────────
print(f"\n[3/5] Adding {len(new_tokens)} tokens to tokenizer...")
adapted_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Check which tokens are genuinely new (not already in vocab)
already_in_vocab = [t for t in new_tokens if t in adapted_tok.get_vocab()]
truly_new        = [t for t in new_tokens if t not in adapted_tok.get_vocab()]

print(f"      Tokens already in mBERT vocab : {len(already_in_vocab)}")
if already_in_vocab:
    print(f"      Examples: {already_in_vocab[:10]}")
print(f"      Genuinely new tokens          : {len(truly_new)}")

n_added = adapted_tok.add_tokens(truly_new)
new_vocab_size = len(adapted_tok)
print(f"      Tokens successfully added     : {n_added}")
print(f"      Adapted tokenizer vocab size  : {new_vocab_size:,}  (+{new_vocab_size - baseline_vocab_size:,})")

# ── Resize model embeddings ────────────────────────────────────────────────────
print(f"\n[4/5] Resizing model embedding matrix to cover {new_vocab_size:,} tokens...")
adapted_model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=3,
    ignore_mismatched_sizes=True
)
adapted_model.resize_token_embeddings(new_vocab_size)
print(f"      Model embedding size (after) : {adapted_model.get_input_embeddings().weight.shape}")
print("      New token embeddings are randomly initialized (standard HuggingFace behaviour).")

# ── Save adapted tokenizer and model ──────────────────────────────────────────
adapted_dir = MODELS / "mbert_adapted"
adapted_dir.mkdir(parents=True, exist_ok=True)
adapted_tok.save_pretrained(str(adapted_dir))
adapted_model.save_pretrained(str(adapted_dir))
print(f"\n      Saved adapted tokenizer + model -> models/mbert_adapted/")

# ── Re-run fragmentation analysis ─────────────────────────────────────────────
print("\n[5/5] Re-running fragmentation analysis with adapted tokenizer...")

# Load word list from Phase 3
pieces_path = DATA_PROCESSED / "word_piece_counts.csv"
freq_path   = DATA_PROCESSED / "word_frequencies.csv"
df_pieces   = pd.read_csv(pieces_path, encoding="utf-8")
df_freq     = pd.read_csv(freq_path,   encoding="utf-8")
df_merged   = df_pieces.merge(df_freq, on="word", how="inner")

all_words     = df_merged["word"].tolist()
word_freq_map = dict(zip(df_merged["word"], df_merged["frequency"]))

def count_pieces(tokenizer, words: list) -> np.ndarray:
    counts = []
    for word in words:
        try:
            ids = tokenizer.encode(str(word), add_special_tokens=False)
            counts.append(len(ids))
        except Exception:
            counts.append(1)  # fallback: treat as single token
    return np.array(counts, dtype=np.float32)

print(f"      Analyzing {len(all_words):,} unique words...")
baseline_counts = df_merged["mbert_pieces"].values.astype(np.float32)
adapted_counts  = count_pieces(adapted_tok, all_words)
freq_weights    = np.array([word_freq_map[w] for w in all_words], dtype=np.float64)
freq_weights   /= freq_weights.sum()

def compute_stats(counts: np.ndarray, name: str) -> dict:
    return {
        "tokenizer": name,
        "avg_subwords_per_word":     float(np.mean(counts)),
        "pct_words_fragmented":      float(np.mean(counts >= 2) * 100),
        "pct_words_single_token":    float(np.mean(counts == 1) * 100),
        "pct_words_3plus_pieces":    float(np.mean(counts >= 3) * 100),
        "max_pieces_any_word":       int(np.max(counts)),
        "avg_fragmentation_ratio":   float(
            np.mean(counts[counts > 1]) if np.any(counts > 1) else 1.0
        ),
        # frequency-weighted versions (more representative of actual text)
        "wavg_subwords_per_word":    float(np.sum(counts * freq_weights)),
        "wpct_words_fragmented":     float(np.sum((counts >= 2) * freq_weights) * 100),
    }

before_stats = compute_stats(baseline_counts, "mBERT (baseline)")
after_stats  = compute_stats(adapted_counts,  "mBERT (adapted)")

# Compute % change
def pct_change(before, after):
    if before == 0:
        return "N/A"
    return f"{(after - before) / before * 100:+.2f}%"

# ── Save Table 4 ───────────────────────────────────────────────────────────────
table4_rows = []
metrics = [
    ("avg_subwords_per_word",   "Avg subwords/word"),
    ("pct_words_fragmented",    "% words fragmented (2+ pieces)"),
    ("pct_words_single_token",  "% words single token"),
    ("pct_words_3plus_pieces",  "% words 3+ pieces"),
    ("max_pieces_any_word",     "Max pieces (any word)"),
    ("avg_fragmentation_ratio", "Avg fragmentation ratio (frag. words only)"),
    ("wavg_subwords_per_word",  "Freq-weighted avg subwords/word"),
    ("wpct_words_fragmented",   "Freq-weighted % fragmented"),
]
for key, label in metrics:
    bv = before_stats[key]
    av = after_stats[key]
    table4_rows.append({
        "metric":          label,
        "mBERT_baseline":  round(bv, 4) if isinstance(bv, float) else bv,
        "mBERT_adapted":   round(av, 4) if isinstance(av, float) else av,
        "absolute_change": round(av - bv, 4) if isinstance(bv, (int, float)) else "N/A",
        "pct_change":      pct_change(bv, av),
    })

df_table4 = pd.DataFrame(table4_rows)
df_table4.to_csv(TABLES / "table4_before_after_fragmentation.csv", index=False, encoding="utf-8")
print(f"\n    Saved -> results/tables/table4_before_after_fragmentation.csv")

# Print comparison
print("\n    Before vs After Adaptation:")
print(f"    {'Metric':50s} {'Before':>10} {'After':>10} {'Change':>10}")
print("    " + "-" * 85)
for row in table4_rows:
    print(f"    {row['metric']:50s} {str(row['mBERT_baseline']):>10} "
          f"{str(row['mBERT_adapted']):>10} {str(row['pct_change']):>10}")

# ── Figure 3: Before/After Fragmentation ──────────────────────────────────────
print("\n    Generating fig3_fragmentation_before_after.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')

# Left: Grouped bar chart for key metrics
metric_labels = ["Avg subwords/word", "% Fragmented\n(2+ pieces)", "% Single token", "% 3+ pieces"]
before_vals   = [before_stats["avg_subwords_per_word"],
                 before_stats["pct_words_fragmented"],
                 before_stats["pct_words_single_token"],
                 before_stats["pct_words_3plus_pieces"]]
after_vals    = [after_stats["avg_subwords_per_word"],
                 after_stats["pct_words_fragmented"],
                 after_stats["pct_words_single_token"],
                 after_stats["pct_words_3plus_pieces"]]

x = np.arange(len(metric_labels))
width = 0.35
b1 = axes[0].bar(x - width/2, before_vals, width, label='mBERT Baseline',
                 color='#e94560', edgecolor='white', linewidth=0.5, alpha=0.85)
b2 = axes[0].bar(x + width/2, after_vals,  width, label='mBERT Adapted',
                 color='#0f3460', edgecolor='white', linewidth=0.5, alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metric_labels, fontsize=10, color='white')
axes[0].set_ylabel("Value", fontsize=11, color='white')
axes[0].set_title("Fragmentation Metrics: Before vs After", fontsize=12, color='white', fontweight='bold')
axes[0].tick_params(colors='white')
axes[0].legend(fontsize=10, framealpha=0.3, facecolor='#1a1a2e', labelcolor='white')
for spine in axes[0].spines.values():
    spine.set_edgecolor('#444466')
# Add value labels
for bar in [b1, b2]:
    for rect in bar:
        h = rect.get_height()
        axes[0].text(rect.get_x() + rect.get_width()/2., h + 0.3,
                     f'{h:.2f}', ha='center', va='bottom', fontsize=8, color='white')

# Right: Histogram overlay — before vs after
bins = np.arange(0.5, min(int(max(baseline_counts.max(), adapted_counts.max())) + 2.5, 12.5), 1.0)
axes[1].hist(baseline_counts, bins=bins, weights=freq_weights * 100, alpha=0.7,
             label='mBERT Baseline', color='#e94560', edgecolor='white', linewidth=0.5)
axes[1].hist(adapted_counts, bins=bins, weights=freq_weights * 100, alpha=0.7,
             label='mBERT Adapted', color='#53d8fb', edgecolor='white', linewidth=0.5)
axes[1].set_xlabel("Subword Pieces per Word", fontsize=11, color='white')
axes[1].set_ylabel("% of Word Tokens (freq-weighted)", fontsize=11, color='white')
axes[1].set_title("Distribution Shift After Adaptation", fontsize=12, color='white', fontweight='bold')
axes[1].tick_params(colors='white')
axes[1].legend(fontsize=10, framealpha=0.3, facecolor='#1a1a2e', labelcolor='white')
for spine in axes[1].spines.values():
    spine.set_edgecolor('#444466')

fig.suptitle("mBERT Tokenizer: Vocabulary Adaptation Effect on Fragmentation",
             fontsize=14, color='white', fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIGURES / "fig3_fragmentation_before_after.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"    Saved -> results/figures/fig3_fragmentation_before_after.png")

# ── Log ────────────────────────────────────────────────────────────────────────
append_log({
    "date": datetime.datetime.now().isoformat(),
    "phase": "Phase5",
    "model": "bert-base-multilingual-cased",
    "tokenizer": "mBERT adapted",
    "sample_size": len(all_words),
    "vocab_size": f"{baseline_vocab_size} -> {new_vocab_size}",
    "result": (f"tokens_added={n_added}, "
               f"avg_subwords: {before_stats['avg_subwords_per_word']:.3f} -> {after_stats['avg_subwords_per_word']:.3f}, "
               f"pct_frag: {before_stats['pct_words_fragmented']:.1f}% -> {after_stats['pct_words_fragmented']:.1f}%")
})

print("\n" + "=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)
print(f"  Tokens added            : {n_added}  ({len(already_in_vocab)} already in vocab)")
print(f"  New vocab size          : {new_vocab_size:,}  (was {baseline_vocab_size:,})")
print(f"  Avg subwords/word       : {before_stats['avg_subwords_per_word']:.3f} -> {after_stats['avg_subwords_per_word']:.3f}  ({pct_change(before_stats['avg_subwords_per_word'], after_stats['avg_subwords_per_word'])})")
print(f"  % fragmented            : {before_stats['pct_words_fragmented']:.1f}% -> {after_stats['pct_words_fragmented']:.1f}%  ({pct_change(before_stats['pct_words_fragmented'], after_stats['pct_words_fragmented'])})")
print(f"  % single token          : {before_stats['pct_words_single_token']:.1f}% -> {after_stats['pct_words_single_token']:.1f}%  ({pct_change(before_stats['pct_words_single_token'], after_stats['pct_words_single_token'])})")
print(f"  Saved model             : models/mbert_adapted/")
print(f"  Output: table4_before_after_fragmentation.csv, fig3_fragmentation_before_after.png")
