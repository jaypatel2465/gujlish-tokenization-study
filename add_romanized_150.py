"""
add_romanized_150.py
====================
Adds 150 high-frequency romanized Gujlish words to the existing
552-token adapted model -> new total ~700 tokens.

These words are romanized Gujarati (Latin script) that mBERT splits
into 2+ subwords. Adding them helps the 93% of sentences that are
romanized, not just the 7% that use Gujarati script.

Run:
  python add_romanized_150.py
"""
import sys, io, re, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import pandas as pd

ROOT           = Path(__file__).parent
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
MODEL_PATH     = MODELS_DIR / "mbert_adapted_500"   # update this in-place

TARGET = 150

print("=" * 60)
print(f"  Adding {TARGET} Romanized Gujlish tokens")
print("=" * 60)

# ── Load existing vocab ────────────────────────────────────────────────────────
vocab_file = DATA_PROCESSED / "gujlish_vocab_additions_500.txt"
existing = {t.strip() for t in vocab_file.read_text(encoding='utf-8').splitlines() if t.strip()}
print(f"\n[1/4] Existing tokens in vocab file: {len(existing)}")

# ── Load word data ─────────────────────────────────────────────────────────────
print(f"\n[2/4] Analysing word data...")
df_pieces = pd.read_csv(DATA_PROCESSED / "word_piece_counts.csv")
df_freq   = pd.read_csv(DATA_PROCESSED / "word_frequencies.csv")
df = df_pieces.merge(df_freq, on="word", how="left")
df["frequency"] = df["frequency"].fillna(1).astype(int)

GUJARATI_RE = re.compile(r'[\u0A80-\u0AFF]')

# Clear romanized Gujlish indicators — common Gujarati romanization patterns
GUJLISH_PATTERN = re.compile(
    r'\b(che|chhe|nathi|nahi|tame|tamara|tamari|tamaro|tamne|tamaru|'
    r'khub|khubj|mast|saras|saru|saro|sara|maja|majja|bahu|bahuj|'
    r'sathe|mate|vagar|vaat|bov|jordar|kem|kai|koi|karo|kari|karu|'
    r'karta|karte|karva|karyu|thay|thai|thayu|thase|thoda|chalo|chalu|'
    r'aave|aavi|aavo|aavse|aavu|ame|amne|amara|mane|mare|mara|mari|maro|'
    r'tari|taro|tara|tane|tamne|ene|ena|ema|etle|etale|'
    r'pan|pachi|pade|pela|pase|pani|'
    r'ghar|ghare|gaya|gayu|gai|gayo|gaye|gayi|'
    r'jova|jovu|jov|joya|joyela|jovani|joi|jodi|'
    r'bhai|bhavin|bhavna|bhajan|'
    r'lavari|loko|lage|lagta|lavo|lai|'
    r'sudhi|suvichar|surat|'
    r'vadhare|vaat|vaar|var|varsad|'
    r'tamru|tamru|avaj|atle|alag|'
    r'nai|nhi|nthi|nahin|'
    r'hu|chu|chhu|cho|chho|'
    r'badha|badhi|badhu|bdha|'
    r'khabar|khali|khush|keri|'
    r'roj|rakhe|rakho|raho|rahi|rahe|'
    r'divas|dil|didi|'
    r'hata|hati|hato|hatu|hase|hain|'
    r'mehnat|mummy|mama|masi|masI|'
    r'papa|patel|'
    r'jigar|jaldi|'
    r'evu|evi|jevu|jevi|jem|'
    r'tyare|tya|te|toh|'
    r'kyare|kyu|kya|'
    r'bolavo|bolo|bole|bol|'
    r'sachu|sachi|sab|sath|saro|'
    r'aa|ae|aaj|aaje|aje|avi|ave|avu|'
    r'nu|na|ne|ni|no|'
    r'su|so|se|'
    r'ko|ke|ki|ka|'
    r'ma|me|mo|'
    r're|ro|ri|ra)\b',
    re.IGNORECASE
)

def is_romanized_gujlish(word):
    w = str(word).strip().lower()
    # Must be Latin script only
    if GUJARATI_RE.search(w):
        return False
    # Must be alphabetic (no numbers, emojis, URLs)
    if not re.match(r'^[a-zA-Z\'\-]+$', w):
        return False
    # Length 2-15
    if len(w) < 2 or len(w) > 15:
        return False
    # Must match Gujlish pattern OR be a known Gujlish word
    return bool(GUJLISH_PATTERN.search(w))

# Score by impact
df["is_roman_gujlish"] = df["word"].apply(is_romanized_gujlish)
df["impact"] = (df["mbert_pieces"] - 1) * df["frequency"]

df_roman = df[
    df["is_roman_gujlish"] &
    (df["mbert_pieces"] >= 2) &        # must actually be fragmented
    (df["frequency"] >= 50) &           # must appear at least 50 times
    (~df["word"].isin(existing))        # not already added
].sort_values("impact", ascending=False).drop_duplicates(subset="word")

print(f"  Qualifying romanized Gujlish words: {len(df_roman):,}")
print(f"\n  Top 30 candidates:")
for _, row in df_roman.head(30).iterrows():
    print(f"    {str(row['word']):<20}  pieces={row['mbert_pieces']}  freq={row['frequency']:,}  impact={row['impact']:.0f}")

# Pick top 150
new_roman_tokens = df_roman["word"].tolist()[:TARGET]
print(f"\n  Selected: {len(new_roman_tokens)} romanized tokens")

# ── Update vocab file ──────────────────────────────────────────────────────────
print(f"\n[3/4] Updating vocab file...")
all_tokens = sorted(existing) + new_roman_tokens
new_vocab_file = DATA_PROCESSED / "gujlish_vocab_additions_700.txt"
new_vocab_file.write_text("\n".join(all_tokens), encoding="utf-8")
print(f"  Total tokens in new vocab file: {len(all_tokens)}")
print(f"  Saved: data/processed/gujlish_vocab_additions_700.txt")

# ── Rebuild model with all ~700 tokens ────────────────────────────────────────
print(f"\n[4/4] Rebuilding mbert_adapted_500 with ~700 tokens...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load base mBERT (not the adapted one — rebuild clean from scratch)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=3, ignore_mismatched_sizes=True
)

# Add ALL tokens (script + romanized) that aren't already in mBERT
tokens_to_add = [t for t in all_tokens if t not in tokenizer.vocab]
print(f"  Tokens not in base mBERT vocab: {len(tokens_to_add)}")

n_added = tokenizer.add_tokens(tokens_to_add)
model.resize_token_embeddings(len(tokenizer))

# Mean-embedding init for new tokens
with torch.no_grad():
    mean_emb = model.bert.embeddings.word_embeddings.weight[:-n_added].mean(dim=0)
    model.bert.embeddings.word_embeddings.weight[-n_added:] = \
        mean_emb.unsqueeze(0).expand(n_added, -1)

print(f"  Added {n_added} tokens  |  New vocab size: {len(tokenizer):,}")

# Overwrite models/mbert_adapted_500 with new version
MODEL_PATH.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(str(MODEL_PATH))
model.save_pretrained(str(MODEL_PATH))
print(f"  Saved → models/mbert_adapted_500/  (overwritten with ~700-token version)")

# Quick fragmentation check
print(f"\n  Sample fragmentation check:")
base_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
check_words = ["tamara", "saras", "nathi", "khub", "sathe", "vadhare",
               "ભગવાન", "ભાઈ", "mehnat", "bolavo"]
for word in check_words:
    old = len(base_tok.tokenize(word))
    new = len(tokenizer.tokenize(word))
    flag = "✓" if new < old else "="
    print(f"    {word:<20}  base:{old}  adapted:{new}  {flag}")

print(f"\n{'='*60}")
print(f"  DONE")
print(f"{'='*60}")
print(f"  Vocab file : data/processed/gujlish_vocab_additions_700.txt")
print(f"  Model      : models/mbert_adapted_500/  (~700 tokens total)")
print(f"  Added      : {n_added} tokens to base mBERT")
print(f"  Vocab size : {len(tokenizer):,}")
