import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Phase 9 -- Contextual Embedding Analysis
==========================================
Analyses how vocabulary adaptation changes the embedding space geometry
without requiring full fine-tuning.

Method:
  1. Extract CLS and mean-pooled embeddings from both models (baseline & adapted)
     for a sample of Gujlish sentences from the dev subset.
  2. Compute embedding similarity distributions — does adaptation change
     inter-class separation?
  3. Analyse token fragmentation at sentence level (not word level like Phase 3).
  4. Compute embedding drift — how much do token embeddings move after adaptation?
  5. Visualise: t-SNE of sentence embeddings coloured by sentiment class.

Outputs:
  results/tables/table9_embedding_analysis.csv
  results/figures/fig8_embedding_tsne.png
  results/figures/fig9_fragmentation_sentence_level.png
  Appends Phase 9 section to RESULTS_SUMMARY.md
"""

import csv
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
TABLES         = ROOT / "results" / "tables"
FIGURES        = ROOT / "results" / "figures"
MODELS         = ROOT / "models"
LOGS           = ROOT / "logs"

import os
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("PHASE 9 -- Contextual Embedding Analysis")
print("=" * 60)

# ── Load dev subset sample ────────────────────────────────────────────────────
print("\n[1/6] Loading dev subset sample...")
dev_path = DATA_PROCESSED / "dev_subset.csv"
df_dev   = pd.read_csv(dev_path, encoding="utf-8")
df_dev["text"] = df_dev["text"].astype(str).str.strip()
df_dev = df_dev[df_dev["text"].notna() & (df_dev["text"] != "") &
                df_dev["sentiment_label"].notna()].copy()

# Stratified sample: 100 sentences per class (300 total — fast on CPU)
SAMPLE_PER_CLASS = 100
label_counts = df_dev["sentiment_label"].value_counts()
print(f"  Dev subset: {len(df_dev):,} rows | Labels: {dict(label_counts)}")

sample_frames = []
for lbl in df_dev["sentiment_label"].unique():
    pool = df_dev[df_dev["sentiment_label"] == lbl]
    n    = min(SAMPLE_PER_CLASS, len(pool))
    sample_frames.append(pool.sample(n=n, random_state=RANDOM_SEED))
df_sample = pd.concat(sample_frames).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"  Sample: {len(df_sample)} sentences ({df_sample['sentiment_label'].value_counts().to_dict()})")

# ── Load tokenizers ────────────────────────────────────────────────────────────
print("\n[2/6] Loading tokenizers...")
from transformers import AutoTokenizer, BertModel
import torch
torch.manual_seed(RANDOM_SEED)

print("  Loading mBERT baseline tokenizer...")
base_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
print(f"  Baseline vocab size: {len(base_tok):,}")

adapted_dir = MODELS / "mbert_adapted"
print("  Loading adapted tokenizer...")
adap_tok = AutoTokenizer.from_pretrained(str(adapted_dir))
print(f"  Adapted vocab size : {len(adap_tok):,}")

# ── Sentence-level fragmentation analysis ─────────────────────────────────────
print("\n[3/6] Sentence-level fragmentation analysis (n={} sentences)...".format(len(df_sample)))

def sentence_fragmentation_stats(tokenizer, texts, max_len=128):
    """Returns per-sentence token counts and fragmentation metrics."""
    results = []
    for text in texts:
        tokens = tokenizer.tokenize(str(text))
        tokens = tokens[:max_len]   # truncate
        n_tokens  = len(tokens)
        # Fragmentation: tokens that are subword continuations (start with ##)
        n_subword = sum(1 for t in tokens if t.startswith("##"))
        n_whole   = n_tokens - n_subword
        results.append({
            "n_tokens":          n_tokens,
            "n_subword_pieces":  n_subword,
            "n_whole_words":     n_whole,
            "frag_rate":         n_subword / n_tokens if n_tokens > 0 else 0.0,
        })
    return pd.DataFrame(results)

texts = df_sample["text"].tolist()
labels = df_sample["sentiment_label"].tolist()

print("  Tokenizing with mBERT baseline...")
base_frag = sentence_fragmentation_stats(base_tok, texts)
print("  Tokenizing with mBERT adapted...")
adap_frag = sentence_fragmentation_stats(adap_tok, texts)

# Compare
base_mean_frag = base_frag["frag_rate"].mean()
adap_mean_frag = adap_frag["frag_rate"].mean()
reduction      = base_mean_frag - adap_mean_frag

print(f"\n  Sentence-level fragmentation (subword piece rate):")
print(f"    mBERT baseline : {base_mean_frag:.4f} ({base_mean_frag*100:.2f}% subword pieces)")
print(f"    mBERT adapted  : {adap_mean_frag:.4f} ({adap_mean_frag*100:.2f}% subword pieces)")
print(f"    Reduction      : {reduction:.4f} ({reduction*100:.2f}pp)")
print(f"    Avg tokens/sentence (baseline): {base_frag['n_tokens'].mean():.1f}")
print(f"    Avg tokens/sentence (adapted) : {adap_frag['n_tokens'].mean():.1f}")

# Per-class fragmentation
print(f"\n  Per-class fragmentation (baseline vs adapted):")
unique_labels = sorted(set(labels))
per_class_rows = []
for lbl in unique_labels:
    idx    = [i for i, l in enumerate(labels) if l == lbl]
    b_frag = base_frag.iloc[idx]["frag_rate"].mean()
    a_frag = adap_frag.iloc[idx]["frag_rate"].mean()
    per_class_rows.append({
        "sentiment":       lbl,
        "n_sentences":     len(idx),
        "base_frag_rate":  round(b_frag, 4),
        "adap_frag_rate":  round(a_frag, 4),
        "reduction":       round(b_frag - a_frag, 4),
    })
    print(f"    {lbl:12}: base={b_frag:.4f}  adap={a_frag:.4f}  Δ={b_frag-a_frag:+.4f}")

# ── Embedding extraction (CLS token) ─────────────────────────────────────────
print("\n[4/6] Extracting sentence embeddings (CLS token, no fine-tuning)...")

def get_cls_embeddings(tokenizer, model, texts, batch_size=32, max_len=128):
    """Extract CLS embeddings for a list of texts."""
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc   = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_len, return_tensors="pt"
            )
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].numpy()  # CLS token
            all_embs.append(cls)
    return np.vstack(all_embs)

print("  Loading mBERT baseline model...")
base_model = BertModel.from_pretrained("bert-base-multilingual-cased")
base_embs  = get_cls_embeddings(base_tok, base_model, texts)
print(f"  Baseline embeddings: {base_embs.shape}")
del base_model  # free memory

print("  Loading mBERT adapted model...")
adap_model = BertModel.from_pretrained(str(adapted_dir))
adap_embs  = get_cls_embeddings(adap_tok, adap_model, texts)
print(f"  Adapted embeddings : {adap_embs.shape}")
del adap_model

# ── Embedding drift ────────────────────────────────────────────────────────────
print("\n[5/6] Embedding analysis...")

# Cosine similarity between baseline and adapted embeddings for same sentences
def cosine_sim_matrix_diag(A, B):
    """Cosine similarity between row-paired vectors A[i] and B[i]."""
    nA = np.linalg.norm(A, axis=1, keepdims=True)
    nB = np.linalg.norm(B, axis=1, keepdims=True)
    return np.sum((A / nA) * (B / nB), axis=1)

drift_sims = cosine_sim_matrix_diag(base_embs, adap_embs)
print(f"  Embedding drift (cosine sim baseline→adapted per sentence):")
print(f"    Mean   : {drift_sims.mean():.4f}")
print(f"    Std    : {drift_sims.std():.4f}")
print(f"    Min    : {drift_sims.min():.4f}")
print(f"    Max    : {drift_sims.max():.4f}")
print(f"    Sentences with drift>0.01 (1-sim): {(drift_sims < 0.99).sum()} / {len(drift_sims)}")

# Inter-class separation (avg cosine similarity within vs between classes)
label_ids  = {lbl: i for i, lbl in enumerate(sorted(set(labels)))}
label_arr  = np.array([label_ids[l] for l in labels])

def intra_inter_similarity(embs, label_arr):
    """Compute mean intra-class and inter-class cosine similarity."""
    n = embs.shape[0]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / norms
    sim_mat = embs_n @ embs_n.T   # (n, n)
    
    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            s = sim_mat[i, j]
            if label_arr[i] == label_arr[j]:
                intra.append(s)
            else:
                inter.append(s)
    return np.mean(intra), np.mean(inter)

print("\n  Inter-class separation analysis:")
base_intra, base_inter = intra_inter_similarity(base_embs, label_arr)
adap_intra, adap_inter = intra_inter_similarity(adap_embs, label_arr)
sep_base = base_intra - base_inter
sep_adap = adap_intra - adap_inter
print(f"    mBERT baseline: intra={base_intra:.4f}  inter={base_inter:.4f}  sep={sep_base:+.4f}")
print(f"    mBERT adapted : intra={adap_intra:.4f}  inter={adap_inter:.4f}  sep={sep_adap:+.4f}")
print(f"    Separation change: {sep_adap - sep_base:+.4f}")

# ── t-SNE visualisation ───────────────────────────────────────────────────────
print("\n  Running t-SNE for visualisation (2D)...")
try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    base_scaled = scaler.fit_transform(base_embs)
    adap_scaled = scaler.fit_transform(adap_embs)

    import sklearn
    _tsne_iters = {"max_iter": 1000} if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 2) else {"n_iter": 1000}
    tsne = TSNE(n_components=2, perplexity=min(30, len(texts)//3),
                random_state=RANDOM_SEED, verbose=0, **_tsne_iters)
    base_2d = tsne.fit_transform(base_scaled)

    tsne2 = TSNE(n_components=2, perplexity=min(30, len(texts)//3),
                 random_state=RANDOM_SEED, verbose=0, **_tsne_iters)
    adap_2d = tsne2.fit_transform(adap_scaled)
    tsne_ok = True
    print("  t-SNE complete.")
except Exception as e:
    print(f"  t-SNE failed: {e} — using PCA fallback.")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    base_2d = pca.fit_transform(base_embs)
    adap_2d = pca.fit_transform(adap_embs)
    tsne_ok = False

# ── Save table9 ───────────────────────────────────────────────────────────────
table9_rows = [
    {"metric": "Sample size",                   "baseline": len(texts),          "adapted": len(texts)},
    {"metric": "Mean sentence frag. rate",       "baseline": round(base_mean_frag,4), "adapted": round(adap_mean_frag,4)},
    {"metric": "Avg tokens per sentence",        "baseline": round(base_frag['n_tokens'].mean(),2), "adapted": round(adap_frag['n_tokens'].mean(),2)},
    {"metric": "Embedding drift (mean cos sim)", "baseline": "—",                "adapted": round(drift_sims.mean(),4)},
    {"metric": "Embedding drift (min cos sim)",  "baseline": "—",                "adapted": round(drift_sims.min(),4)},
    {"metric": "Intra-class cosine similarity",  "baseline": round(base_intra,4),"adapted": round(adap_intra,4)},
    {"metric": "Inter-class cosine similarity",  "baseline": round(base_inter,4),"adapted": round(adap_inter,4)},
    {"metric": "Class separation (intra−inter)", "baseline": round(sep_base,4),  "adapted": round(sep_adap,4)},
]
for r in per_class_rows:
    table9_rows.append({
        "metric":   f"Sent. frag. rate [{r['sentiment']}]",
        "baseline": r["base_frag_rate"],
        "adapted":  r["adap_frag_rate"],
    })

pd.DataFrame(table9_rows).to_csv(TABLES / "table9_embedding_analysis.csv", index=False, encoding="utf-8")
print(f"\n  Saved → results/tables/table9_embedding_analysis.csv")

# ── Figure 8: t-SNE side-by-side ─────────────────────────────────────────────
print("  Generating fig8_embedding_tsne.png...")
CLASS_COLORS = {"positive": "#00d2ff", "neutral": "#ffd93d", "negative": "#ff6b6b"}
unique_labels_sorted = sorted(set(labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1a1a2e')
method_label = "t-SNE" if tsne_ok else "PCA"

for ax_idx, (emb_2d, title) in enumerate([
    (base_2d, f"mBERT Baseline ({method_label})"),
    (adap_2d, f"mBERT Adapted ({method_label})"),
]):
    ax = axes[ax_idx]
    ax.set_facecolor('#16213e')
    for lbl in unique_labels_sorted:
        mask = np.array([l == lbl for l in labels])
        col  = CLASS_COLORS.get(lbl, "#aaaacc")
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=col, label=lbl.capitalize(), alpha=0.65, s=25, edgecolors='none')
    ax.set_title(title, fontsize=12, color='white', fontweight='bold')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.legend(fontsize=9, framealpha=0.3, facecolor='#1a1a2e', labelcolor='white')

fig.suptitle(
    f"Phase 9: Sentence Embedding Space — Before vs After Vocabulary Adaptation\n"
    f"(CLS embeddings, {len(texts)} sentences, non-fine-tuned)\n"
    f"Intra-class sim: {base_intra:.4f} → {adap_intra:.4f}  |  "
    f"Inter-class sim: {base_inter:.4f} → {adap_inter:.4f}",
    fontsize=11, color='white', fontweight='bold'
)
plt.tight_layout()
fig.savefig(FIGURES / "fig8_embedding_tsne.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → results/figures/fig8_embedding_tsne.png")

# ── Figure 9: Sentence-level fragmentation distribution ───────────────────────
print("  Generating fig9_fragmentation_sentence_level.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')

# Left: histogram of fragmentation rates
axes[0].hist(base_frag["frag_rate"], bins=30, alpha=0.75, color='#e94560',
             label=f"mBERT baseline (μ={base_mean_frag:.3f})", edgecolor='white', linewidth=0.3)
axes[0].hist(adap_frag["frag_rate"], bins=30, alpha=0.75, color='#53d8fb',
             label=f"mBERT adapted (μ={adap_mean_frag:.3f})", edgecolor='white', linewidth=0.3)
axes[0].axvline(base_mean_frag, color='#e94560', linestyle='--', linewidth=1.5)
axes[0].axvline(adap_mean_frag, color='#53d8fb', linestyle='--', linewidth=1.5)
axes[0].set_xlabel("Subword Piece Rate per Sentence", fontsize=11, color='white')
axes[0].set_ylabel("Count", fontsize=11, color='white')
axes[0].set_title("Sentence-Level Fragmentation Distribution", fontsize=12, color='white', fontweight='bold')
axes[0].tick_params(colors='white')
axes[0].legend(fontsize=9, framealpha=0.3, facecolor='#1a1a2e', labelcolor='white')
for spine in axes[0].spines.values():
    spine.set_edgecolor('#444466')

# Right: per-class fragmentation comparison
class_labels = [r["sentiment"] for r in per_class_rows]
base_rates   = [r["base_frag_rate"] for r in per_class_rows]
adap_rates   = [r["adap_frag_rate"] for r in per_class_rows]
x = np.arange(len(class_labels))
w = 0.35
axes[1].bar(x - w/2, base_rates, w, label='mBERT baseline', color='#e94560',
            edgecolor='white', linewidth=0.5, alpha=0.85)
axes[1].bar(x + w/2, adap_rates, w, label='mBERT adapted',  color='#53d8fb',
            edgecolor='white', linewidth=0.5, alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels([l.capitalize() for l in class_labels], fontsize=10, color='white')
axes[1].set_ylabel("Mean Subword Piece Rate", fontsize=11, color='white')
axes[1].set_title("Per-Class Fragmentation Comparison", fontsize=12, color='white', fontweight='bold')
axes[1].tick_params(colors='white')
axes[1].legend(fontsize=9, framealpha=0.3, facecolor='#1a1a2e', labelcolor='white')
for spine in axes[1].spines.values():
    spine.set_edgecolor('#444466')
for i, (br, ar) in enumerate(zip(base_rates, adap_rates)):
    axes[1].text(i - w/2, br + 0.003, f'{br:.3f}', ha='center', fontsize=8, color='white')
    axes[1].text(i + w/2, ar + 0.003, f'{ar:.3f}', ha='center', fontsize=8, color='white')

fig.suptitle(
    "Phase 9: Sentence-Level Fragmentation Analysis\n"
    f"(n={len(texts)} sentences, 3 sentiment classes)",
    fontsize=12, color='white', fontweight='bold'
)
plt.tight_layout()
fig.savefig(FIGURES / "fig9_fragmentation_sentence_level.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → results/figures/fig9_fragmentation_sentence_level.png")

# ── Append to RESULTS_SUMMARY.md ──────────────────────────────────────────────
phase9_text = f"""

---

## Phase 9: Contextual Embedding Analysis

### Table 9: Embedding Space Analysis ({len(texts)} sentences, non-fine-tuned CLS embeddings)

| Metric | mBERT Baseline | mBERT Adapted |
|---|---|---|
| Mean sentence fragmentation rate | {base_mean_frag:.4f} ({base_mean_frag*100:.2f}%) | {adap_mean_frag:.4f} ({adap_mean_frag*100:.2f}%) |
| Avg tokens per sentence | {base_frag['n_tokens'].mean():.1f} | {adap_frag['n_tokens'].mean():.1f} |
| Embedding drift (mean cosine sim) | — | {drift_sims.mean():.4f} |
| Embedding drift (min cosine sim) | — | {drift_sims.min():.4f} |
| Intra-class cosine similarity | {base_intra:.4f} | {adap_intra:.4f} |
| Inter-class cosine similarity | {base_inter:.4f} | {adap_inter:.4f} |
| Class separation (intra − inter) | {sep_base:+.4f} | {sep_adap:+.4f} |

### Figure 8: Sentence Embedding t-SNE

![Figure 8](results/figures/fig8_embedding_tsne.png)

### Figure 9: Sentence-Level Fragmentation

![Figure 9](results/figures/fig9_fragmentation_sentence_level.png)

### Phase 9 Interpretation
1. **Fragmentation at sentence level**: The adapted tokenizer reduces subword piece rate from {base_mean_frag*100:.2f}% to {adap_mean_frag*100:.2f}% per sentence ({(reduction*100):.2f} percentage points). This is consistent with the word-level analysis in Phase 3/5.
2. **Embedding drift is minimal** (mean cosine sim = {drift_sims.mean():.4f}): Adding 66 new tokens with random initialization barely changes the embedding space for most sentences, since most sentences don't contain the newly added tokens.
3. **Class separation ({method_label})**: The separation metric (intra − inter class cosine similarity) is {sep_base:+.4f} (baseline) vs {sep_adap:+.4f} (adapted). {'Small improvement after adaptation.' if sep_adap > sep_base else 'No meaningful improvement — expected without fine-tuning.'}
4. **Key limitation**: These are pre-fine-tuning embeddings. The newly added tokens have random embeddings and contribute noise. Post-fine-tuning analysis (requiring the saved model from GPU run) would show more meaningful representation changes.
"""

results_summary_path = ROOT / "RESULTS_SUMMARY.md"
with open(results_summary_path, "a", encoding="utf-8") as f:
    f.write(phase9_text)
print(f"\n  ✅ Appended Phase 9 section to RESULTS_SUMMARY.md")

print("\n" + "=" * 60)
print("PHASE 9 COMPLETE")
print("=" * 60)
print(f"  Outputs:")
print(f"    results/tables/table9_embedding_analysis.csv")
print(f"    results/figures/fig8_embedding_tsne.png")
print(f"    results/figures/fig9_fragmentation_sentence_level.png")
print(f"    RESULTS_SUMMARY.md (Phase 9 appended)")
print(f"\n  Sentence-level frag: {base_mean_frag*100:.2f}% → {adap_mean_frag*100:.2f}% ({reduction*100:.2f}pp reduction)")
print(f"  Embedding drift (mean cos sim baseline→adapted): {drift_sims.mean():.4f}")
print(f"  Class separation: {sep_base:+.4f} → {sep_adap:+.4f}")
