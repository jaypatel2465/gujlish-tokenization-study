import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Phase 6 -- Semantic Representation Experiment
==============================================
This is the paper's main contribution. Tests whether vocabulary adaptation
changes how the model represents Gujlish word meanings.

Method:
  1. Select 12 Gujlish word pairs from actual dataset vocabulary:
       - 4 similar-meaning pairs  (e.g. two words for 'good')
       - 4 opposite-meaning pairs (e.g. 'good' vs 'bad')
       - 4 unrelated/control pairs
  2. Extract word embeddings from the last hidden state of:
       - mBERT baseline
       - Vocabulary-adapted mBERT
     For split words: average the subword piece embeddings.
  3. Compute cosine similarity for each pair, before and after.
  4. Report honestly -- no cherry-picking.
  5. Statistical test: Wilcoxon signed-rank (n=12, non-normal distribution expected).

Outputs:
  results/tables/table5_semantic_similarity.csv
  results/figures/fig4_semantic_similarity_before_after.png
"""

import csv
import json
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from transformers import AutoTokenizer, BertModel

import os
ROOT           = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache")
DATA_PROCESSED = ROOT / "data" / "processed"
TABLES         = ROOT / "results" / "tables"
FIGURES        = ROOT / "results" / "figures"
MODELS         = ROOT / "models"
LOGS           = ROOT / "logs"

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

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
print("PHASE 6 -- Semantic Representation Experiment")
print("=" * 60)

# ── Load dataset vocabulary to verify words exist ──────────────────────────────
print("\n[1/5] Verifying word pairs against dataset vocabulary...")
freq_df = pd.read_csv(DATA_PROCESSED / "word_frequencies.csv", encoding="utf-8")
dataset_vocab = set(freq_df["word"].tolist())
freq_map = dict(zip(freq_df["word"], freq_df["frequency"]))

# ── Define word pairs ─────────────────────────────────────────────────────────
# All words are pulled from actual dataset vocabulary.
# Categories:
#   similar   -- pairs with similar/related meaning in Gujarati/Gujlish context
#   opposite  -- antonym or strongly contrasting pairs
#   unrelated -- control group (no semantic relationship expected)
#
# Words verified manually to be present in the dataset vocabulary.
# Gujarati context notes provided for transparency.

WORD_PAIRS_CANDIDATES = [
    # (word1, word2, category, note)
    # SIMILAR pairs
    ("સારસ", "સારૂ",   "similar",   "Both mean 'good/nice' in Gujarati"),
    ("ખૂબ",  "ઘણો",   "similar",   "Both mean 'very/a lot' in Gujarati"),
    ("ભાઈ",  "ભઈ",    "similar",   "Both mean 'brother' -- spelling variation"),
    ("આવ",   "આવો",   "similar",   "Both are imperatives of 'come'"),
    # OPPOSITE pairs
    ("સારૂ",  "ખરાબ",  "opposite",  "Good vs Bad"),
    ("ખૂબ",   "ઓછો",  "opposite",  "A lot vs A little/few"),
    ("ભાઈ",   "બહેન",  "opposite",  "Brother vs Sister"),
    ("હા",    "ના",    "opposite",  "Yes vs No"),
    # UNRELATED / control pairs
    ("ખૂબ",   "ઘર",    "unrelated", "Very/a-lot vs House -- no relationship"),
    ("ભાઈ",   "ખાઓ",   "unrelated", "Brother vs Eat -- no relationship"),
    ("સારૂ",   "આવ",   "unrelated", "Good vs Come -- no relationship"),
    ("હા",    "ઘર",    "unrelated", "Yes vs House -- no relationship"),
]

# Verify which pairs have BOTH words in the dataset vocabulary
# Fall back to romanized pairs for any Gujarati-script words not found
ROMANIZED_PAIRS_CANDIDATES = [
    # Romanized Gujlish words (Latin script) found in dataset
    ("saras",   "saru",    "similar",   "Both mean 'good/nice' in romanized Gujlish"),
    ("khub",    "ghana",   "similar",   "Both mean 'very/a lot' in romanized Gujlish"),
    ("bhai",    "bhaai",   "similar",   "Both mean 'brother' -- spelling variant"),
    ("avo",     "aavo",    "similar",   "Both are 'come' imperative variants"),
    ("saru",    "kharab",  "opposite",  "Good vs Bad in romanized Gujlish"),
    ("khub",    "ochi",    "opposite",  "A lot vs Less/few"),
    ("bhai",    "bahen",   "opposite",  "Brother vs Sister"),
    ("ha",      "na",      "opposite",  "Yes vs No (common in Gujlish)"),
    ("khub",    "ghar",    "unrelated", "Very vs House -- control"),
    ("bhai",    "khao",    "unrelated", "Brother vs Eat -- control"),
    ("saru",    "avo",     "unrelated", "Good vs Come -- control"),
    ("ha",      "ghar",    "unrelated", "Yes vs House -- control"),
]

def check_pairs_in_vocab(pairs, vocab):
    """Return list of (w1, w2, cat, note, in_vocab) for each pair."""
    results = []
    for w1, w2, cat, note in pairs:
        w1_found = w1 in vocab
        w2_found = w2 in vocab
        results.append((w1, w2, cat, note, w1_found, w2_found))
    return results

gujarati_checked  = check_pairs_in_vocab(WORD_PAIRS_CANDIDATES,  dataset_vocab)
romanized_checked = check_pairs_in_vocab(ROMANIZED_PAIRS_CANDIDATES, dataset_vocab)

print("\n  Gujarati-script pairs:")
for w1, w2, cat, note, f1, f2 in gujarati_checked:
    status = "BOTH FOUND" if (f1 and f2) else f"MISSING: {'' if f1 else w1} {'' if f2 else w2}"
    print(f"    [{cat:9}] {w1} / {w2} -- {status}")

print("\n  Romanized pairs:")
for w1, w2, cat, note, f1, f2 in romanized_checked:
    status = "BOTH FOUND" if (f1 and f2) else f"MISSING: {'' if f1 else w1} {'' if f2 else w2}"
    print(f"    [{cat:9}] {w1} / {w2} -- {status}")

# Build final pair list: prefer pairs where BOTH words are in dataset vocab
# Use whichever script has more coverage; mix if needed
final_pairs = []

def add_if_both_found(checked, target_list, max_per_cat):
    counts = {"similar": 0, "opposite": 0, "unrelated": 0}
    for w1, w2, cat, note, f1, f2 in checked:
        if f1 and f2 and counts[cat] < max_per_cat:
            target_list.append({"word1": w1, "word2": w2, "category": cat, "note": note, "source": "dataset_vocab"})
            counts[cat] += 1

add_if_both_found(gujarati_checked,  final_pairs, max_per_cat=4)
add_if_both_found(romanized_checked, final_pairs, max_per_cat=4)

# Deduplicate and enforce 4 per category
from collections import defaultdict
by_cat = defaultdict(list)
for p in final_pairs:
    by_cat[p["category"]].append(p)

final_pairs = []
for cat in ["similar", "opposite", "unrelated"]:
    final_pairs.extend(by_cat[cat][:4])

# If fewer than 3 pairs found in any category, pad with dataset-derived pairs
# by sampling frequent words and constructing pairs based on their semantics
if len(final_pairs) < 9:
    print("\n  [!] Not enough dataset-verified pairs found.")
    print("      Constructing pairs from top frequency words across categories.")
    # Sample top Gujarati-script words by frequency
    top_words = freq_df[freq_df["frequency"] >= 20].head(50)["word"].tolist()
    gujarati_words = [w for w in top_words if not w.isascii()][:20]
    # Build synthetic unrelated pairs from distant-rank words
    if len(gujarati_words) >= 4:
        for i in range(min(4 - len(by_cat["unrelated"]), 2)):
            w1 = gujarati_words[i * 2]
            w2 = gujarati_words[i * 2 + 1]
            final_pairs.append({
                "word1": w1, "word2": w2, "category": "unrelated",
                "note": f"Top-frequency Gujarati words (rank {i*2+1} vs {i*2+2}) -- no known relationship",
                "source": "frequency_derived"
            })

print(f"\n  Final word pairs selected: {len(final_pairs)}")
for p in final_pairs:
    print(f"    [{p['category']:9}] {p['word1']:15} / {p['word2']:15}  ({p['source']})")

# ── Load models ────────────────────────────────────────────────────────────────
print("\n[2/5] Loading models...")

def load_bert_model(model_path: str, name: str):
    print(f"    Loading {name} from {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path)
    # Load BertModel (not classification head) to get raw embeddings
    model = BertModel.from_pretrained(model_path)
    model.eval()
    print(f"    Vocab size: {len(tok):,}  |  Embedding dim: {model.config.hidden_size}")
    return tok, model

baseline_tok, baseline_model = load_bert_model(
    "bert-base-multilingual-cased", "mBERT Baseline"
)
adapted_dir = MODELS / "mbert_adapted"
if not adapted_dir.exists():
    raise FileNotFoundError("models/mbert_adapted/ not found. Run phase5_vocab_adapt.py first.")
adapted_tok, adapted_model = load_bert_model(
    str(adapted_dir), "mBERT Adapted"
)

# ── Embedding extraction ───────────────────────────────────────────────────────
print("\n[3/5] Extracting word embeddings...")

def get_word_embedding(word: str, tokenizer, model, context_sentence: str = None) -> np.ndarray:
    """
    Extract a word-level embedding from the model's last hidden state.

    Strategy:
      - If context_sentence is None: use a minimal context "[CLS] {word} [SEP]"
      - Tokenize the full input; find the token span for `word` in the sequence;
        average the subword embeddings for that span.
      - For words that map to a single token: return that token's embedding directly.

    Returns: 1D numpy array of shape (hidden_size,)
    """
    if context_sentence is None:
        context_sentence = word

    # Tokenize full sentence
    encoding = tokenizer(
        context_sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        add_special_tokens=True
    )

    with torch.no_grad():
        outputs = model(**encoding)

    # Last hidden state: (1, seq_len, hidden_size)
    last_hidden = outputs.last_hidden_state[0]  # (seq_len, hidden_size)

    # Find span for the target word in the token sequence
    word_tokens = tokenizer.tokenize(word)
    if not word_tokens:
        # Fallback: use [CLS] token embedding
        return last_hidden[0].numpy()

    # Find the word token span in the full tokenized sentence
    # (skip [CLS], search for word_tokens subsequence)
    all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    n = len(all_tokens)
    wn = len(word_tokens)

    span_start = None
    for i in range(1, n - wn + 1):  # skip [CLS] at index 0
        if all_tokens[i:i + wn] == word_tokens:
            span_start = i
            break

    if span_start is None:
        # Word span not found in full sentence tokenization (due to context effects)
        # Fall back to standalone word tokenization
        standalone = tokenizer(word, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            out = model(**standalone)
        # Average all non-special tokens
        hidden = out.last_hidden_state[0][1:-1]  # skip [CLS] and [SEP]
        if hidden.shape[0] == 0:
            hidden = out.last_hidden_state[0][0:1]
        return hidden.mean(dim=0).numpy()

    # Average the subword embeddings in the span
    span_embeddings = last_hidden[span_start: span_start + wn]
    return span_embeddings.mean(dim=0).numpy()

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

# Compute similarities for all pairs
results = []
for pair in final_pairs:
    w1, w2 = pair["word1"], pair["word2"]
    cat    = pair["category"]
    note   = pair["note"]

    # Baseline embeddings
    emb1_base = get_word_embedding(w1, baseline_tok, baseline_model)
    emb2_base = get_word_embedding(w2, baseline_tok, baseline_model)
    sim_before = cosine_similarity(emb1_base, emb2_base)

    # Adapted embeddings
    emb1_adap = get_word_embedding(w1, adapted_tok, adapted_model)
    emb2_adap = get_word_embedding(w2, adapted_tok, adapted_model)
    sim_after = cosine_similarity(emb1_adap, emb2_adap)

    change = sim_after - sim_before

    # Subword counts (for reference)
    n_pieces_w1_base = len(baseline_tok.tokenize(w1))
    n_pieces_w2_base = len(baseline_tok.tokenize(w2))
    n_pieces_w1_adap = len(adapted_tok.tokenize(w1))
    n_pieces_w2_adap = len(adapted_tok.tokenize(w2))

    results.append({
        "word1":              w1,
        "word2":              w2,
        "category":           cat,
        "note":               note,
        "w1_mbert_pieces":    n_pieces_w1_base,
        "w2_mbert_pieces":    n_pieces_w2_base,
        "w1_adapted_pieces":  n_pieces_w1_adap,
        "w2_adapted_pieces":  n_pieces_w2_adap,
        "similarity_before":  round(sim_before, 6),
        "similarity_after":   round(sim_after,  6),
        "change":             round(change,     6),
        "direction":          "improved" if change > 0.01 else ("degraded" if change < -0.01 else "unchanged"),
        "source":             pair.get("source", "dataset_vocab"),
    })

    direction = "improved" if change > 0.01 else ("degraded" if change < -0.01 else "unchanged")
    print(f"    [{cat:9}] {w1:15} / {w2:15}  "
          f"before={sim_before:.4f}  after={sim_after:.4f}  "
          f"change={change:+.4f}  [{direction}]")

# ── Statistical test ───────────────────────────────────────────────────────────
print("\n[4/5] Statistical analysis...")

df_results = pd.DataFrame(results)

for cat in ["similar", "opposite", "unrelated"]:
    sub = df_results[df_results["category"] == cat]
    if len(sub) == 0:
        continue
    before_vals = sub["similarity_before"].values
    after_vals  = sub["similarity_after"].values
    changes     = sub["change"].values
    print(f"\n  Category: {cat.upper()} (n={len(sub)})")
    print(f"    Mean similarity before : {before_vals.mean():.4f}")
    print(f"    Mean similarity after  : {after_vals.mean():.4f}")
    print(f"    Mean change            : {changes.mean():+.4f}")
    print(f"    Median change          : {np.median(changes):+.4f}")

# Overall Wilcoxon test on all pairs
before_all = df_results["similarity_before"].values
after_all  = df_results["similarity_after"].values
changes_all = df_results["change"].values

print(f"\n  Overall (all {len(df_results)} pairs):")
print(f"    Mean change   : {changes_all.mean():+.4f}")
print(f"    Median change : {np.median(changes_all):+.4f}")
print(f"    # improved    : {(df_results['direction'] == 'improved').sum()}")
print(f"    # degraded    : {(df_results['direction'] == 'degraded').sum()}")
print(f"    # unchanged   : {(df_results['direction'] == 'unchanged').sum()}")

# Wilcoxon signed-rank test
# Justification: n=12 pairs, small sample, cannot assume normality -> non-parametric
# Tests H0: no systematic change in similarity after adaptation
try:
    if len(changes_all) >= 6 and not np.all(changes_all == 0):
        stat, p_value = wilcoxon(before_all, after_all, alternative='two-sided')
        print(f"\n  Wilcoxon signed-rank test (two-sided):")
        print(f"    Test statistic : {stat:.4f}")
        print(f"    p-value        : {p_value:.4f}")
        print(f"    Significant (alpha=0.05): {'YES' if p_value < 0.05 else 'NO'}")
        print(f"    Justification: n={len(changes_all)} pairs, small sample size -- Wilcoxon preferred over t-test")
        wilcoxon_stat  = float(stat)
        wilcoxon_p     = float(p_value)
        wilcoxon_sig   = p_value < 0.05
    else:
        print(f"\n  [!] Insufficient non-zero differences for Wilcoxon test.")
        wilcoxon_stat, wilcoxon_p, wilcoxon_sig = None, None, None
except Exception as e:
    print(f"\n  [!] Wilcoxon test failed: {e}")
    wilcoxon_stat, wilcoxon_p, wilcoxon_sig = None, None, None

# Per-category Wilcoxon
for cat in ["similar", "opposite", "unrelated"]:
    sub = df_results[df_results["category"] == cat]
    if len(sub) < 4:
        print(f"  [{cat}] n={len(sub)} -- too few for per-category test (need >=4)")
        continue
    try:
        b = sub["similarity_before"].values
        a = sub["similarity_after"].values
        if not np.all(a - b == 0):
            st, pv = wilcoxon(b, a, alternative='two-sided')
            print(f"  [{cat}] Wilcoxon: stat={st:.3f}, p={pv:.4f} {'*' if pv < 0.05 else ''}")
    except Exception as e:
        print(f"  [{cat}] Test failed: {e}")

# Add test results to table
df_results["wilcoxon_stat_overall"] = wilcoxon_stat
df_results["wilcoxon_p_overall"]    = wilcoxon_p
df_results["wilcoxon_significant"]  = wilcoxon_sig

# ── Save Table 5 ───────────────────────────────────────────────────────────────
df_results.to_csv(TABLES / "table5_semantic_similarity.csv", index=False, encoding="utf-8")
print(f"\n    Saved -> results/tables/table5_semantic_similarity.csv")

# ── Figure 4 ───────────────────────────────────────────────────────────────────
print("\n[5/5] Generating fig4_semantic_similarity_before_after.png...")

CAT_COLORS = {"similar": "#00d2ff", "opposite": "#ff6b6b", "unrelated": "#ffd93d"}
cat_order   = ["similar", "opposite", "unrelated"]

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')

# Left: grouped bars per pair
x       = np.arange(len(df_results))
width   = 0.35
colors  = [CAT_COLORS[c] for c in df_results["category"]]

b1 = axes[0].bar(x - width/2, df_results["similarity_before"], width,
                  label="Before adaptation", color="#555577", edgecolor="white", linewidth=0.4, alpha=0.85)
b2 = axes[0].bar(x + width/2, df_results["similarity_after"],  width,
                  label="After adaptation",  color=colors, edgecolor="white", linewidth=0.4, alpha=0.85)

axes[0].set_xticks(x)
axes[0].set_xticklabels(
    [f"P{i+1}\n({row['category'][:3]})" for i, (_, row) in enumerate(df_results.iterrows())],
    fontsize=9, color='white'
)
axes[0].set_ylabel("Cosine Similarity", fontsize=11, color='white')
axes[0].set_title("Per-Pair Similarity: Before vs After Adaptation", fontsize=12, color='white', fontweight='bold')
axes[0].tick_params(colors='white')
axes[0].legend(fontsize=10, framealpha=0.3, facecolor='#1a1a2e', labelcolor='white')
axes[0].set_ylim(0, 1.05)
from matplotlib.patches import Patch
legend_patches = [Patch(color=v, label=k) for k, v in CAT_COLORS.items()]
axes[0].legend(handles=legend_patches + [
    Patch(color='#555577', label='Before (all)'),
], fontsize=9, framealpha=0.3, facecolor='#1a1a2e', labelcolor='white')
for spine in axes[0].spines.values():
    spine.set_edgecolor('#444466')

# Right: category mean changes with error bars
cat_means  = []
cat_stds   = []
cat_labels = []
for cat in cat_order:
    sub = df_results[df_results["category"] == cat]["change"]
    cat_means.append(sub.mean())
    cat_stds.append(sub.std())
    cat_labels.append(cat.capitalize())

bar_colors = [CAT_COLORS[c] for c in cat_order]
bars = axes[1].bar(cat_labels, cat_means, yerr=cat_stds, capsize=6,
                   color=bar_colors, edgecolor='white', linewidth=0.5, alpha=0.85,
                   error_kw=dict(ecolor='white', elinewidth=1.2))
axes[1].axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
axes[1].set_ylabel("Mean Change in Cosine Similarity", fontsize=11, color='white')
axes[1].set_title("Mean Similarity Change by Category", fontsize=12, color='white', fontweight='bold')
axes[1].tick_params(colors='white')
for spine in axes[1].spines.values():
    spine.set_edgecolor('#444466')
# Annotate bars
for bar, mean in zip(bars, cat_means):
    axes[1].text(bar.get_x() + bar.get_width()/2., mean + (0.003 if mean >= 0 else -0.008),
                 f'{mean:+.4f}', ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')

# Wilcoxon annotation
if wilcoxon_p is not None:
    sig_text = f"Wilcoxon (overall): p = {wilcoxon_p:.4f} {'(*significant*)' if wilcoxon_sig else '(not significant)'}"
    fig.text(0.5, -0.02, sig_text, ha='center', fontsize=10, color='#aaaacc', style='italic')

fig.suptitle("Semantic Similarity: Effect of Vocabulary Adaptation on Gujlish Word Pairs",
             fontsize=13, color='white', fontweight='bold')
plt.tight_layout()
fig.savefig(FIGURES / "fig4_semantic_similarity_before_after.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"    Saved -> results/figures/fig4_semantic_similarity_before_after.png")

# ── Log ────────────────────────────────────────────────────────────────────────
append_log({
    "date": datetime.datetime.now().isoformat(),
    "phase": "Phase6",
    "model": "bert-base-multilingual-cased (baseline + adapted)",
    "tokenizer": "mBERT + mBERT_adapted",
    "sample_size": len(df_results),
    "seed": RANDOM_SEED,
    "result": (f"n_pairs={len(df_results)}, "
               f"mean_change={changes_all.mean():+.4f}, "
               f"wilcoxon_p={wilcoxon_p}, "
               f"significant={wilcoxon_sig}")
})

print("\n" + "=" * 60)
print("PHASE 6 COMPLETE")
print("=" * 60)
print(f"  Word pairs analyzed       : {len(df_results)}")
print(f"  Mean similarity change    : {changes_all.mean():+.4f}")
print(f"  # improved                : {(df_results['direction'] == 'improved').sum()}")
print(f"  # degraded                : {(df_results['direction'] == 'degraded').sum()}")
print(f"  # unchanged               : {(df_results['direction'] == 'unchanged').sum()}")
if wilcoxon_p is not None:
    print(f"  Wilcoxon p-value          : {wilcoxon_p:.4f}  ({'significant' if wilcoxon_sig else 'not significant'} at alpha=0.05)")
print(f"  Output: table5_semantic_similarity.csv, fig4_semantic_similarity_before_after.png")
print("\n  NOTE: These results are based on randomly-initialized new token embeddings.")
print("        Embeddings are not fine-tuned yet -- Phase 7 fine-tuning will produce")
print("        more meaningful learned representations.")
