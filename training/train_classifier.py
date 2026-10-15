"""
train_classifier.py — Phase 8: 3-class sentiment classification, 5-fold CV
================================================================================
One entry point for the four model variants compared in the paper. Every
variant uses the identical protocol from training/common.py (stratified
5-fold CV, seed 42, 5 epochs, lr 2e-5, max length 96, class-weighted loss).

    python training/train_classifier.py --model mbert   # mBERT baseline
    python training/train_classifier.py --model muril   # MuRIL baseline
    python training/train_classifier.py --model obpe    # mBERT OBPE +498, no MLM
    python training/train_classifier.py --model mlm     # mBERT 798-token + MLM

Each variant can run on its own machine / Kaggle T4 session. Runs are
resumable: completed folds in results/predictions/<model>/ are skipped.

OUTPUT (results/predictions/<checkpoint>/):
  label_map.json           — label2id / dataset info for this run
  fold_metrics.csv         — per-fold accuracy / macro-F1 / precision / recall
  fold{1..5}_predictions.csv — per-example text, true/pred label, correctness,
                             confidence, fragmentation_ratio, is_code_mixed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, run_training_pipeline  # noqa: E402

# name -> (display name, model path, predictions dir, corpus version)
MODELS = {
    "mbert": ("mBERT_baseline", "bert-base-multilingual-cased", "mbert_baseline", "v1"),
    "muril": ("MuRIL_baseline", "google/muril-base-cased", "muril_baseline", "v1"),
    # Built by training/build_obpe_ablation.py (published as
    # rudrakachhia/mbert-obpe-adapted).
    "obpe": ("mBERT_OBPE_498", str(ROOT / "models" / "mbert_obpe_ablation"), "mbert_obpe498_nomlm", "v1"),
    # Built by training/build_obpe_base.py + training/mlm_pretrain.py.
    "mlm": ("mBERT_798tok_MLM", str(ROOT / "models" / "mbert_798_mlm"), "mbert_798_mlm", "v2"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--model-path", help="override the checkpoint path / HF id for this variant")
    ap.add_argument("--save-models", action="store_true",
                    help="also save each fold's fine-tuned classifier to models/classifiers/")
    ap.add_argument("--smoke-test", type=int, metavar="N",
                    help="quick pipeline check on N stratified rows, 1 epoch "
                         "(writes to results/predictions/smoke_<model>/)")
    args = ap.parse_args()

    name, path, ckpt, data_version = MODELS[args.model]
    path = args.model_path or path
    if path.startswith(str(ROOT)) and not Path(path, "model.safetensors").exists() \
            and not Path(path, "pytorch_model.bin").exists():
        sys.exit(f"No model weights in {path}. Build them first (see README) or pass --model-path.")

    run_training_pipeline(
        model_name=name,
        model_path=path,
        checkpoint_dirname=f"smoke_{ckpt}" if args.smoke_test else ckpt,
        data_version=data_version,
        max_rows=args.smoke_test,
        n_epochs=1 if args.smoke_test else None,
        save_models=args.save_models,
    )


if __name__ == "__main__":
    main()
