# Vocabulary Adaptation and Continued MLM Pre-Training for Gujarati-English Code-Mixed Sentiment Analysis

Code, data and result artifacts for the paper

> D. Patel, J. Patel, R. Kachhia, N. Thakor. *Vocabulary Adaptation and Continued MLM Pre-Training for Gujarati-English Code-Mixed Sentiment Analysis.* G. H. Patel College of Engineering & Technology, The CVM University.

Multilingual models such as mBERT and MuRIL split Gujarati-English code-mixed ("Gujlish") text into many subword pieces. This project measures that fragmentation on a 51,161-sentence Gujarati/Gujlish sentiment corpus. It then tests whether reducing fragmentation helps three-class sentiment classification. The paper compares four models under identical 5-fold cross-validation:

| Model | Vocabulary | Macro-F1 (mean ± SD, 5 folds) | Accuracy |
|---|---|---|---|
| mBERT baseline (`bert-base-multilingual-cased`) | 119,547 | 0.7376 ± 0.0042 | 0.7596 |
| MuRIL baseline (`google/muril-base-cased`) | 197,258 | 0.7141 ± 0.0075 | 0.7372 |
| mBERT OBPE +498 (no MLM) | 120,045 | 0.7372 ± 0.0046 | 0.7585 |
| **mBERT 798-token + MLM (ours)** | 120,345 | **0.7456 ± 0.0041** | **0.7652** |

On the 50,827 sentences shared by all runs, the 798-token + MLM model beats the mBERT baseline: McNemar χ² = 11.89, p = 5.6×10⁻⁴, and the bootstrap 95% CI for the macro-F1 difference is [+0.005, +0.011]. Adding 498 tokens *without* continued pre-training gives no gain (p = 0.678).

---

## Pipeline

```
 Phase 1  Data collection      scripts/phase1_dataset.py, scripts/phase1b_expand_dataset.py
    │      HF dataset (21,729) + 12 YouTube channels (29,826)
 Phase 2  Cleaning              scripts/phase2_clean.py            → 21,346 + 29,826 → 51,161 merged
    │
 Phase 3  Fragmentation         scripts/phase3_fragmentation.py    → mBERT 79.33% / MuRIL 70.02% of word types split
    │
 Phase 4  Vocabulary selection  scripts/phase4_vocab_selection.py  score(w) = log2 f(w) × mBERT pieces(w)
 Phase 5  Vocabulary adaptation scripts/phase5_vocab_adapt.py      75-token pilot (fragmentation only)
    │                           training/build_obpe_ablation.py    +498 tokens          → "OBPE +498 (no MLM)"
    │                           training/build_obpe_base.py        +498 Gujarati tokens ┐
 Phase 6  MLM pre-training      training/mlm_pretrain.py           +300 Romanized tokens┘ + continued MLM → "798-token + MLM"
    │
 Phase 7  Semantic evaluation   scripts/phase6_semantic.py, scripts/phase9_embedding_analysis.py
    │
 Phase 8  Classification        training/train_classifier.py       5-fold CV, identical protocol for all 4 models
    │
 Phase 9  Statistics            evaluation/evaluate.py, evaluation/significance.py
```

The script names under `scripts/` keep their original phase numbers. The table under [Paper table → file](#paper-table--file) maps every paper table to the file that produces it.

## Repository layout

```
├── data/processed/
│   ├── filtered_clean.csv                 21,346 cleaned sentences (HF source, Phase 2 output)
│   ├── expanded_dev_subset_full.csv       YouTube expansion, corpus version v1 (→ 51,161 merged)
│   ├── expanded_dev_subset_full_v2.csv    YouTube expansion, corpus version v2 (→ 51,155 merged, MLM run)
│   ├── clean_sample_3000.csv              extra sample, part of the MLM pre-training corpus
│   ├── dev_subset.csv                     3,000-row stratified subset (Phase 2; used by Phase 7 embedding analysis)
│   ├── human_audit_sample.csv             150-sentence stratified label audit (50 per class)
│   ├── gujlish_vocab_additions.txt        75-token pilot vocabulary (Phase 4)
│   └── gujlish_vocab_additions_500.txt    498-token OBPE vocabulary (no-MLM ablation)
├── scripts/                               Phases 1–5 and 7: data, fragmentation, pilot adaptation, semantics
├── training/
│   ├── common.py                          shared 5-fold CV protocol (seed, splits, hyperparameters, outputs)
│   ├── train_classifier.py                Phase 8 entry point: --model {mbert, muril, obpe, mlm}
│   ├── build_obpe_ablation.py             builds models/mbert_obpe_ablation (OBPE +498, no MLM)
│   ├── build_obpe_base.py                 builds models/mbert_obpe_base (498 Gujarati-script tokens)
│   └── mlm_pretrain.py                    +300 Romanized tokens + continued MLM → models/mbert_798_mlm
├── evaluation/
│   ├── evaluate.py                        Tables 5, 9, 13 + figures from saved predictions
│   ├── significance.py                    Tables 10, 12, 13 (truncation), 14
│   └── predict.py                         inference with a fine-tuned classifier
├── models/                                tokenizer/config files only (weights are not in git)
│   ├── mbert_pilot75/                     75-token pilot (Phases 5 and 7)
│   ├── mbert_obpe_ablation/               OBPE +498 tokenizer
│   └── mbert_798_mlm/                     final 798-token tokenizer, token list, MLM metrics
├── results/
│   ├── predictions/<model>/               per-fold metrics + per-sentence predictions of the paper runs
│   ├── classification/                    outputs of evaluation/*.py
│   ├── tables/, figures/                  outputs of scripts/phase*.py
├── logs/experiment_log.csv                run log written by the phase scripts
├── requirements.txt                       main environment
└── requirements-mlm.txt                   environment for OBPE base building + MLM pre-training
```

## Setup

Python 3.10+ is required.

```bash
git clone https://github.com/jaypatel2465/gujlish-tokenization-study.git
cd gujlish-tokenization-study
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or: pip install torch (CPU)
pip install -r requirements.txt
```

`training/build_obpe_base.py` and `training/mlm_pretrain.py` require `transformers==4.57.6`. Run them in a separate environment built from `requirements-mlm.txt` (they were run on a Kaggle Tesla T4). Everything else uses `requirements.txt` (Transformers ≥ 5).

Hugging Face models are cached in `hf_cache/` inside the repository.

## Reproducing the reported numbers (no GPU, about 2 minutes)

The per-sentence predictions of the paper runs are included in `results/predictions/`, so every classification table can be recomputed on a CPU:

```bash
python evaluation/evaluate.py       # Tables 5, 9, 13 and figures → results/classification/
python evaluation/significance.py   # Tables 10, 12, 13 (truncation), 14
```

Expected output (abridged):

```
paired rows (present in all 3 runs): 50,827
McNemar mbert_798_mlm vs mbert_baseline: n10=3807 n01=3511 chi2=11.89 p=5.64e-04
McNemar mbert_baseline vs muril_baseline: n10=5105 n01=3970 chi2=141.70 p=1.13e-32
McNemar mbert_798_mlm vs muril_baseline: n10=5345 n01=3914 chi2=220.86 p=5.89e-50
Bootstrap mbert_798_mlm - mbert_baseline: mean=+0.0081 95% CI [+0.0047, +0.0114]
Fold t-test mbert_798_mlm vs mbert_baseline: t=4.94 p=0.0078 d=+2.21
```

The per-sentence predictions of the OBPE +498 (no MLM) run are not part of this repository. Its row in Tables 9, 11 and 12 is as reported in the paper. To recompute it, train it with `--model obpe` (below); `evaluate.py` and `significance.py` include it automatically once `results/predictions/mbert_obpe498_nomlm/` exists.

## Full pipeline

### 1. Data (Phases 1–2)

```bash
python scripts/phase1_dataset.py        # download ShrutiPatel3011/gujarati-english-codemixed-sentiment, filter, audit sample
python scripts/phase2_clean.py          # dedup + URL/spam filtering → filtered_clean.csv, dev_subset.csv
python scripts/phase1b_expand_dataset.py   # optional: YouTube Data API v3 collection (needs YOUTUBE_API_KEY)
```

The labelled YouTube expansion used in the paper is shipped as `data/processed/expanded_dev_subset_full*.csv`, so Phase 1b is only needed to collect new comments. Labels are LLM-generated; see the paper (Section 3.2) for the label audit.

**Corpus versions.** The classifier merges `filtered_clean.csv` with one YouTube file and de-duplicates on text:

| Version | YouTube file | Merged sentences | Used by |
|---|---|---|---|
| v1 | `expanded_dev_subset_full.csv` | 51,161 | mBERT, MuRIL, OBPE +498 |
| v2 | `expanded_dev_subset_full_v2.csv` | 51,155 | mBERT 798-token + MLM (pre-training and fine-tuning) |

`train_classifier.py` selects the right version for each model automatically. Paired comparisons use only the 50,827 sentences present in every run (paper Section 3.8).

### 2. Fragmentation and pilot vocabulary (Phases 3–5)

```bash
python scripts/phase3_fragmentation.py    # Table 4, Fig. 1; writes word_frequencies.csv / word_piece_counts.csv
python scripts/phase4_vocab_selection.py  # 75-token pilot selection (freq ≥ 10, ≥ 2 pieces, log2 f × pieces)
python scripts/phase5_vocab_adapt.py      # pilot model + before/after fragmentation (Table 6 pilot column, Fig. 3)
```

### 3. Adapted models (Phases 5–6)

```bash
# OBPE +498, no MLM (ablation model)
python training/build_obpe_ablation.py    # → models/mbert_obpe_ablation/

# 798-token + MLM (proposed model) — GPU, requirements-mlm.txt environment
python training/build_obpe_base.py        # → models/mbert_obpe_base/   (+498 Gujarati-script tokens)
python training/mlm_pretrain.py           # → models/mbert_798_mlm/     (+300 Romanized tokens, continued MLM)
```

New-token embeddings for the Romanized tokens are initialised as the mean of the token's original mBERT subword embeddings. Masking follows the standard BERT protocol (15%; 80/10/10). All other pre-training settings are in the `CFG` class of `training/mlm_pretrain.py`. The selected tokens are recorded in `models/mbert_798_mlm/token_expansion_info.json`.

### 4. Semantic evaluation (Phase 7)

```bash
python scripts/phase6_semantic.py            # Table 7, Fig. 4 (12 word pairs, Wilcoxon)
python scripts/phase9_embedding_analysis.py  # Table 8, Figs. 8–9 (300 sentences)
```

### 5. Sentiment classification (Phase 8, GPU)

```bash
python training/train_classifier.py --model mbert
python training/train_classifier.py --model muril
python training/train_classifier.py --model obpe
python training/train_classifier.py --model mlm
```

The protocol is identical for every model (`training/common.py`, paper Table 2):

| Setting | Value |
|---|---|
| Cross-validation | stratified 5-fold, `random_state=42` |
| Epochs / batch / lr | 5 / 32 / 2e-5, AdamW (weight decay 0.01), 10% warmup |
| Max sequence length | 96 |
| Loss | class-weighted cross-entropy (`compute_class_weight('balanced')`, per fold) |
| Precision / seed | fp16 on CUDA / 42 |

Each run writes `results/predictions/<model>/fold_metrics.csv` and `fold{1..5}_predictions.csv`. Runs are resumable, and completed folds are skipped. Each model can be trained on a different machine; copy its `results/predictions/<model>/` folder back before evaluating.

Useful flags:

- `--save-models` also saves each fold's classifier to `models/classifiers/<model>/fold<k>/` (needed for inference).
- `--smoke-test N` runs a quick check on N stratified sentences for 1 epoch. Results go to `results/predictions/smoke_<model>/`.
- `--model-path` points a variant at another checkpoint (local folder or Hugging Face id).

### 6. Statistics (Phase 9)

```bash
python evaluation/evaluate.py
python evaluation/significance.py
```

### 7. Inference

```bash
python training/train_classifier.py --model mlm --save-models
python evaluation/predict.py --model-dir models/classifiers/mbert_798_mlm/fold1 \
    --text "Ekdum saras song chhe, maja avi gai" "આ વિડિઓ ખૂબ જ સરસ છે"
python evaluation/predict.py --model-dir models/classifiers/mbert_798_mlm/fold1 \
    --input-csv comments.csv --output-csv predictions.csv
```

## Paper table → file

| Paper | Produced by | File |
|---|---|---|
| Table 1 | `scripts/phase1_dataset.py`, `scripts/phase2_clean.py` | `results/tables/table1_dataset_statistics.csv`, `table2_cleaning_report.csv` |
| Table 4 | `scripts/phase3_fragmentation.py` | `results/tables/table2_fragmentation_baseline.csv` |
| Table 5 | `evaluation/evaluate.py` | `results/classification/code_mixed_fragmentation_ttest.csv` |
| Table 6 | `scripts/phase5_vocab_adapt.py`, `evaluation/evaluate.py` | `results/tables/table4_before_after_fragmentation.csv`, Table 5 file |
| Table 7 | `scripts/phase6_semantic.py` | `results/tables/table5_semantic_similarity.csv` |
| Table 8 | `scripts/phase9_embedding_analysis.py` | `results/tables/table9_embedding_analysis.csv` |
| Table 9 | `evaluation/evaluate.py` | `results/classification/classification_summary.csv` |
| Table 10 | `evaluation/significance.py` | `results/classification/per_class_metrics.csv` |
| Table 11 | `training/train_classifier.py` | `results/predictions/<model>/fold_metrics.csv` |
| Table 12 | `evaluation/significance.py` | `results/classification/significance_tests.json` |
| Table 13 | `evaluation/evaluate.py`, `evaluation/significance.py` | `fragmentation_stratified_accuracy.csv`, `truncation_by_quartile.csv` |
| Table 14 | `evaluation/significance.py` | `results/classification/audit_subset_check.csv` |

## Reproducibility notes

- Each configuration was trained once with seed 42. GPU non-determinism means retrained models will not match the saved predictions bit-for-bit.
- Vocabulary selection and MLM pre-training use the text, but never the labels, of all corpus sentences. This is the transductive setting described in paper Section 3.1.
- Model weights are not stored in git. The paper lists the OBPE +498 checkpoint on the Hugging Face Hub as `rudrakachhia/mbert-obpe-adapted`; the MLM checkpoint is to be released there. Otherwise, rebuild them with the scripts above.
- `logs/experiment_log.csv` is the run log appended by the phase scripts.

## Data and ethics

The original 21,346-sentence dataset is derived from [ShrutiPatel3011/gujarati-english-codemixed-sentiment](https://huggingface.co/datasets/ShrutiPatel3011/gujarati-english-codemixed-sentiment). The YouTube comments were collected through the YouTube Data API v3 from public videos. Labels are LLM-generated, and a stratified sample of 150 was audited by the authors.

## Authors

Devarsh Patel, Jay Patel (corresponding, jaypatel240605@gmail.com), Rudra Kachhia, Nand Thakor. Computer Engineering Department, G. H. Patel College of Engineering & Technology, The CVM University, Vallabh Vidyanagar, Gujarat, India.
