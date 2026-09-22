# Gujlish Tokenization Fragmentation Study

**Paper**: *"Investigating the Impact of Subword Tokenization Fragmentation on Semantic Representation of Gujarati-English Code-Mixed Text"*

**Status**: ✅ All 9 phases complete

---

## 🚀 Quick Start for GPU Collaborators (Phase 7 only)

If someone shared this repo with you to run the final experiment on your GPU:

```bash
# 1. Clone
git clone https://github.com/jaypatel2465/gujlish-tokenization-study.git
cd gujlish-tokenization-study

# 2. Install PyTorch with CUDA (visit https://pytorch.org for the exact command for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Run — that's it!
python run_phase7.py
```

**Expected time**: ~5–15 min on a modern GPU (RTX 3060+), ~4–8 hours on CPU.

---

## Project Structure

```
gujlish-tokenization-study/
├── run_phase7.py                        # ← GPU collaborator runs THIS
├── requirements.txt
├── data/
│   └── processed/
│       ├── dev_subset.csv               # 3,000-row subset for Phase 7 ✅
│       ├── gujlish_vocab_additions.txt  # 66 Gujarati tokens added to mBERT ✅
│       └── human_audit_sample.csv       # 150-row manual audit sample
├── scripts/
│   ├── phase1_dataset.py                # Stage 1: Download + filter
│   ├── phase2_clean.py                  # Stage 1: Clean dataset
│   ├── phase3_fragmentation.py          # Stage 2: Fragmentation baseline
│   ├── phase4_vocab_selection.py        # Stage 2: Gujlish vocab selection
│   ├── phase5_vocab_adapt.py            # Stage 2: Adapt mBERT tokenizer
│   ├── phase5b_fragmentation.py         # Stage 2b: Post-adaptation fragmentation
│   ├── phase6_semantic.py               # Stage 3: Semantic similarity
│   ├── phase7_classification.py         # Stage 3: Sentiment classification (GPU)
│   ├── error_analysis.py                # Stage 3: Error & fold analysis
│   ├── phase8_ablation.py               # Stage 4: Ablation + effect size
│   └── phase9_embedding_analysis.py     # Stage 4: Contextual embedding analysis
├── models/
│   └── mbert_adapted/                   # Adapted tokenizer (config + tokenizer.json)
├── results/
│   ├── tables/                          # 13 CSV/JSON tables (all phases)
│   └── figures/                         # 9 PNG figures (300 dpi)
├── logs/
│   └── experiment_log.csv
├── RESULTS_SUMMARY.md                   # ← Complete paper-ready results
├── LIMITATIONS.md
└── Output/                              # Phase 7 GPU run outputs (raw)
```

---

## Full Pipeline — All 9 Phases

### Stage 1 — Data Foundation
```bash
python scripts/phase1_dataset.py    # Download + filter Gujlish data
python scripts/phase2_clean.py      # Dedup, spam removal
```
→ Tables 1-2 | 21,346 clean Gujlish sentences

### Stage 2 — Tokenization & Vocabulary
```bash
python scripts/phase3_fragmentation.py    # mBERT vs MuRIL baseline
python scripts/phase4_vocab_selection.py  # Top Gujarati tokens (freq ≥ 10)
python scripts/phase5_vocab_adapt.py      # Adapt mBERT (+66 tokens)
python scripts/phase5b_fragmentation.py   # Re-measure post-adaptation
```
→ Tables 2b-4, Figures 1-3 | mBERT: 79.3% fragmented → 79.2% after adaptation

### Stage 3 — Experiments
```bash
python scripts/phase6_semantic.py         # Semantic similarity (12 word pairs)
python scripts/phase7_classification.py   # 5-fold CV sentiment classification
python scripts/error_analysis.py          # Fold analysis + RESULTS_SUMMARY update
```
→ Tables 5-7, Figures 4-6 | Best model: mBERT adapted, F1 = 71.26%

### Stage 4 — Analysis & Ablation
```bash
python scripts/phase8_ablation.py           # Effect size + bootstrap CI
python scripts/phase9_embedding_analysis.py # t-SNE + sentence-level fragmentation
```
→ Tables 8-9, Figures 7-9 | Cohen's d = +0.106 (adapted vs baseline)

---

## Results At-a-Glance

| Table | Content | Key Number |
|---|---|---|
| Table 1 | Dataset statistics | 21,346 rows, 58.7% positive |
| Table 2 | Cleaning report | 13 rows removed (0.06%) |
| Table 2b | Fragmentation baseline | mBERT 79.3% vs MuRIL 70.0% |
| Table 3 | Top 20 fragmented words | 66 tokens selected |
| Table 4 | Before/after adaptation | −1.61% freq-weighted fragmentation |
| Table 5 | Semantic similarity | p=0.25, not significant (pre-fine-tuning) |
| Table 6 | Classification results | mBERT adapted **F1=71.26%** (best) |
| Table 7 | Error/fold analysis | Adapted wins 3/5 folds |
| Table 8 | Ablation study | Cohen's d=+0.106, Bootstrap CI=[−0.012, +0.015] |
| Table 9 | Embedding analysis | Drift cos-sim=0.9963, sep improves +0.0006 |

| Figure | Content |
|---|---|
| Fig 1 | Subwords-per-word distribution |
| Fig 2 | Top 20 fragmented Gujlish words |
| Fig 3 | Fragmentation before/after adaptation |
| Fig 4 | Semantic similarity before/after |
| Fig 5 | Model performance comparison (Phase 7) |
| Fig 6 | Per-fold F1 line chart + delta chart |
| Fig 7 | Ablation comparison (F1, bootstrap CI, Cohen's d) |
| Fig 8 | Sentence embedding space (PCA) — baseline vs adapted |
| Fig 9 | Sentence-level fragmentation distribution |

---

## Key Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Gujlish dominance filter | `(guj+eng) > 1.5 × hindi` | Ensures clear Gujlish signal |
| Frequency threshold (Phase 4) | ≥ 10 occurrences | Avoids rare-word noise |
| Vocabulary additions | 66 tokens (target: 75) | Pool exhausted at 66 |
| CV folds (Phase 7) | 5-fold stratified | Standard for robust evaluation |
| Epochs (Phase 7) | 3 (GPU) / 2 (CPU) | Auto-set by device |
| Batch size (Phase 7) | 8–64 (auto) | Scales with VRAM |
| FP16 | Enabled on CUDA | ~2× speed |
| Random seed | 42 | Fixed for reproducibility |
| Bootstrap samples (Phase 8) | 10,000 | Robust CI estimation |
| Embedding sample (Phase 9) | 300 (100/class) | CPU-feasible |

---

## Notes
- ~97.2% of sentiment labels are `model_predicted` (not human-annotated). Results are preliminary until `data/processed/human_audit_sample.csv` is filled in.
- `hf_cache/` (~1.6 GB) and `models/mbert_adapted/model.safetensors` (~680 MB) are git-ignored — auto-downloaded on first run.
- See `LIMITATIONS.md` for full methodological caveats.
- See `RESULTS_SUMMARY.md` for paper-ready numbers, interpretations, and all figure captions.
