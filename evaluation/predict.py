"""
predict.py — sentiment inference with a fine-tuned classifier
=============================================================
Uses a classifier saved by
    python training/train_classifier.py --model mlm --save-models
(one model per CV fold under models/classifiers/<run>/fold<k>/).

Examples:
    python evaluation/predict.py --model-dir models/classifiers/mbert_798_mlm/fold1 \
        --text "Ekdum saras song chhe, maja avi gai" "આ વિડિઓ ખૂબ જ સરસ છે"

    python evaluation/predict.py --model-dir models/classifiers/mbert_798_mlm/fold1 \
        --input-csv comments.csv --output-csv predictions.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_SEQ_LEN = 96  # same as training (training/common.py)


def predict(texts, model_dir, batch_size=32):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], truncation=True, max_length=MAX_SEQ_LEN,
                    padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits, dim=-1).cpu()
        conf, idx = probs.max(dim=-1)
        out += [(model.config.id2label[int(k)], float(c)) for k, c in zip(idx, conf)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", nargs="+")
    src.add_argument("--input-csv", help="CSV with a 'text' column")
    ap.add_argument("--output-csv")
    args = ap.parse_args()

    if not Path(args.model_dir).exists():
        sys.exit(f"{args.model_dir} not found — train with --save-models first.")
    texts = args.text or pd.read_csv(args.input_csv, encoding="utf-8-sig")["text"].astype(str).tolist()
    preds = predict(texts, args.model_dir)
    df = pd.DataFrame({"text": texts, "pred_label": [p for p, _ in preds],
                       "confidence": [round(c, 4) for _, c in preds]})
    if args.output_csv:
        df.to_csv(args.output_csv, index=False, encoding="utf-8")
        print(f"Wrote {len(df):,} predictions to {args.output_csv}")
    else:
        for row in df.itertuples():
            print(f"{row.pred_label:<9} {row.confidence:.3f}  {row.text}")


if __name__ == "__main__":
    main()
