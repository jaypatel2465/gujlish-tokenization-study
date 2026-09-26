"""
expand_vocab_500.py
===================
Expands the mBERT adapted vocabulary from 75 → 500 tokens.

Strategy (research-validated):
  1. Load word_piece_counts.csv (fragmentation per word for mBERT)
  2. Filter to Gujarati script words only (unicode range 0A80–0AFF)
  3. Rank by fragmentation * frequency (highest impact first)
  4. Pick top 500 not already in mBERT vocab
  5. Add to the existing mbert_adapted tokenizer/model
  6. Save the new model to models/mbert_adapted_500/

Run:
  python expand_vocab_500.py
"""
import sys, io, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import pandas as pd
import numpy as np

ROOT           = Path(__file__).parent
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_TOKENS = 500

print("=" * 60)
print(f"  Vocabulary Expansion: 75 → {TARGET_TOKENS} tokens")
print("=" * 60)

# ── Load existing vocab additions (75 tokens already added) ───────────────────
existing_file = DATA_PROCESSED / "gujlish_vocab_additions.txt"
existing_tokens = set()
if existing_file.exists():
    existing_tokens = {
        line.strip() for line in existing_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    }
print(f"\n[1/5] Existing tokens: {len(existing_tokens)}")

# ── Load fragmentation + frequency data ──────────────────────────────────────
print(f"\n[2/5] Loading word data...")
df_pieces = pd.read_csv(DATA_PROCESSED / "word_piece_counts.csv")
df_freq   = pd.read_csv(DATA_PROCESSED / "word_frequencies.csv")

df = df_pieces.merge(df_freq, on="word", how="left")
df["frequency"] = df["frequency"].fillna(1).astype(int)

print(f"  Total unique words: {len(df):,}")

# ── Gujarati script detection ─────────────────────────────────────────────────
# Gujarati unicode: U+0A80 to U+0AFF
GUJARATI_RE = re.compile(r'[\u0A80-\u0AFF]')

def is_gujarati(w):
    return bool(GUJARATI_RE.search(str(w)))

df["is_gujarati"] = df["word"].apply(is_gujarati)

# Also include high-fragmentation romanized Gujarati words (transliterated)
# These are words that appear frequently but mBERT splits heavily
# We'll keep them if mbert_pieces >= 2 and frequency >= 50
df_gujarati = df[df["is_gujarati"]].copy()
df_roman_high_frag = df[
    (~df["is_gujarati"]) &
    (df["mbert_pieces"] >= 3) &
    (df["frequency"] >= 100)
].copy()

print(f"  Gujarati script words: {len(df_gujarati):,}")
print(f"  High-frag romanized words: {len(df_roman_high_frag):,}")

# ── Score: fragmentation × frequency (impact score) ──────────────────────────
# Words that are fragmented AND frequent have the highest impact on training
for df_sub in [df_gujarati, df_roman_high_frag]:
    df_sub["impact_score"] = (df_sub["mbert_pieces"] - 1) * df_sub["frequency"]

df_candidates = pd.concat([df_gujarati, df_roman_high_frag], ignore_index=True)
df_candidates = df_candidates.sort_values("impact_score", ascending=False)

# ── Filter: valid tokens only ─────────────────────────────────────────────────
def is_valid_token(w):
    w = str(w).strip()
    if not w or len(w) < 2:
        return False
    # Skip URLs, numbers-only, emoji-heavy strings
    if re.match(r'^https?://', w):
        return False
    if re.match(r'^[\d\s\W]+$', w):
        return False
    if len(w) > 30:  # skip very long garbage words
        return False
    return True

df_candidates = df_candidates[df_candidates["word"].apply(is_valid_token)]
df_candidates = df_candidates[~df_candidates["word"].isin(existing_tokens)]
df_candidates = df_candidates.drop_duplicates(subset="word")

print(f"\n[3/5] Candidate tokens after filtering: {len(df_candidates):,}")

# ── Select top tokens ─────────────────────────────────────────────────────────
# Primary: Gujarati script (high impact)
top_gujarati = df_gujarati[
    df_gujarati["word"].apply(is_valid_token) &
    ~df_gujarati["word"].isin(existing_tokens)
].sort_values("impact_score", ascending=False).head(400)

# Secondary: high-fragmentation romanized Gujlish words
top_roman = df_roman_high_frag[
    df_roman_high_frag["word"].apply(is_valid_token) &
    ~df_roman_high_frag["word"].isin(existing_tokens)
].sort_values("impact_score", ascending=False).head(150)

new_tokens = list(top_gujarati["word"].tolist()) + list(top_roman["word"].tolist())
# Deduplicate while preserving order
seen = set()
new_tokens_dedup = []
for t in new_tokens:
    if t not in seen and t not in existing_tokens:
        seen.add(t)
        new_tokens_dedup.append(t)

new_tokens_500 = new_tokens_dedup[:TARGET_TOKENS]

print(f"\n[4/5] Selected {len(new_tokens_500)} new tokens")

# Stats breakdown
guj_count  = sum(1 for t in new_tokens_500 if is_gujarati(t))
rom_count  = len(new_tokens_500) - guj_count
print(f"  Gujarati script tokens: {guj_count}")
print(f"  Romanized Gujlish:      {rom_count}")

# Show top 20 by impact
print("\n  Top 20 tokens by impact score:")
top20 = df_candidates[df_candidates["word"].isin(new_tokens_500)].head(20)
for _, row in top20.iterrows():
    print(f"    {row['word']:<25}  pieces={row['mbert_pieces']}  freq={row['frequency']:,}  impact={row['impact_score']:.0f}")

# ── Save new vocabulary file ──────────────────────────────────────────────────
all_tokens = sorted(existing_tokens) + new_tokens_500  # existing 75 + new 500
vocab_file = DATA_PROCESSED / "gujlish_vocab_additions_500.txt"
vocab_file.write_text("\n".join(all_tokens), encoding="utf-8")
print(f"\n  Saved: data/processed/gujlish_vocab_additions_500.txt")
print(f"  Total tokens (existing + new): {len(all_tokens)}")

# ── Add tokens to mBERT and save new adapted model ────────────────────────────
print(f"\n[5/5] Adding tokens to mBERT...")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Start from the original mBERT (so we have a clean 500-token model, not 75+500)
print("  Loading bert-base-multilingual-cased...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=3,
    ignore_mismatched_sizes=True,
)

# Filter to tokens genuinely NOT in mBERT's vocabulary
new_tok_filtered = [t for t in new_tokens_500 if t not in tokenizer.vocab]
existing_tok_filtered = [t for t in existing_tokens if t not in tokenizer.vocab]

all_new = list(dict.fromkeys(existing_tok_filtered + new_tok_filtered))  # preserve order, deduplicate
print(f"  Tokens not already in mBERT vocab: {len(all_new)}")

n_added = tokenizer.add_tokens(all_new)
print(f"  Actually added: {n_added} tokens")
print(f"  New vocab size: {len(tokenizer):,}  (was 119,547)")

# Resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# Initialize new token embeddings using mean of existing embeddings
# This is much better than random init — gives the new tokens a reasonable starting point
print("  Initializing new embeddings with corpus mean...")
with torch.no_grad():
    existing_embed = model.bert.embeddings.word_embeddings.weight[:-n_added].mean(dim=0)
    model.bert.embeddings.word_embeddings.weight[-n_added:] = existing_embed.unsqueeze(0).expand(n_added, -1)

# Save
out_path = MODELS_DIR / "mbert_adapted_500"
tokenizer.save_pretrained(str(out_path))
model.save_pretrained(str(out_path))
print(f"  Saved → models/mbert_adapted_500/")

# Verify fragmentation improvement
print("\n  Fragmentation check on sample Gujarati words:")
sample_words = ["ભગવાન", "ભાઈ", "ખૂબ", "ગુજ્જુ", "મહિપતસિંહ", "जोरदार", "tamara", "saras"]
old_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
for word in sample_words:
    old_pieces = old_tok.tokenize(word)
    new_pieces = tokenizer.tokenize(word)
    flag = "✓ improved" if len(new_pieces) < len(old_pieces) else ("= same" if len(new_pieces) == len(old_pieces) else "")
    print(f"    {word:<20}  mBERT:{len(old_pieces)}  adapted_500:{len(new_pieces)}  {flag}")

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)
print(f"  New model:   models/mbert_adapted_500/")
print(f"  New vocab:   data/processed/gujlish_vocab_additions_500.txt")
print(f"  Added:       {n_added} new tokens to mBERT")
print(f"  Vocab size:  {len(tokenizer):,}")
print(f"\n  Next step:")
print(f"    Update run_phase7_v2.py to use models/mbert_adapted_500 and retrain.")
