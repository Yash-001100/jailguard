"""
Platt scaling calibration for JailGuard.

Fits a logistic regression on the model's raw logits using a held-out
calibration set, then saves the parameters so inference/predict.py can
apply them at runtime.

Run with:
    python benchmarks/calibrate.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm

SAVE_DIR       = os.path.join(os.path.dirname(__file__), "..", "saved_model")
CALIB_PATH     = os.path.join(os.path.dirname(__file__), "..", "inference", "calibration.json")
DATA_DIR       = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_LENGTH     = 256
CALIB_SAMPLES  = 10_000   # use first 10k of test set for calibration
BATCH_SIZE     = 256


def collect_logits(df, model, tokenizer, device):
    """Run model on df, return raw unsafe logits and true labels."""
    model.eval()
    all_logits, all_labels = [], []

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Collecting logits"):
        batch_texts  = df["text"].iloc[i:i+BATCH_SIZE].tolist()
        batch_labels = df["label"].iloc[i:i+BATCH_SIZE].tolist()

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            ).logits  # shape (batch, 2)

        # Raw unsafe logit (pre-softmax) — this is what Platt scaling fits on
        all_logits.extend(logits[:, 1].cpu().numpy().tolist())
        all_labels.extend(batch_labels)

    return np.array(all_logits), np.array(all_labels)


def fit_platt(logits, labels):
    """Fit logistic regression on raw logits -> calibrated probabilities."""
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(logits.reshape(-1, 1), labels)
    return lr


def evaluate_calibration(probs, labels, n_bins=10):
    """Show how well-calibrated the probabilities are."""
    fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins)
    print(f"\n  {'Predicted prob':<18} {'Actual freq':<14} {'Diff':>6}")
    print(f"  {'-'*40}")
    for pred, actual in zip(mean_pred, fraction_pos):
        diff = actual - pred
        bar  = "+" * int(abs(diff) * 20)
        print(f"  {pred:<18.3f} {actual:<14.3f} {diff:>+.3f}  {bar}")


def run():
    print("=" * 55)
    print("  PLATT SCALING CALIBRATION")
    print("=" * 55)

    # Load model
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(SAVE_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR).to(device)

    # Load test set — split into calibration and eval
    df       = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8")
    calib_df = df.iloc[:CALIB_SAMPLES]
    eval_df  = df.iloc[CALIB_SAMPLES:]
    print(f"Calibration set: {len(calib_df):,}  |  Eval set: {len(eval_df):,}\n")

    # Step 1: Collect logits on calibration set
    print("Step 1 — Collecting logits on calibration set...")
    calib_logits, calib_labels = collect_logits(calib_df, model, tokenizer, device)

    # Before calibration: raw softmax probabilities
    raw_probs = torch.softmax(
        torch.tensor(calib_logits).unsqueeze(1).repeat(1, 2) * torch.tensor([-1.0, 1.0]),
        dim=1,
    )[:, 1].numpy()
    # Simpler: just sigmoid of the raw logit
    raw_probs = 1 / (1 + np.exp(-calib_logits))

    print("\n  Before calibration (raw sigmoid probabilities):")
    evaluate_calibration(raw_probs, calib_labels)

    # Step 2: Fit Platt scaling
    print("\nStep 2 — Fitting logistic regression (Platt scaling)...")
    platt = fit_platt(calib_logits, calib_labels)
    calib_probs = platt.predict_proba(calib_logits.reshape(-1, 1))[:, 1]

    print("\n  After calibration (Platt-scaled probabilities):")
    evaluate_calibration(calib_probs, calib_labels)

    # Step 3: Evaluate on held-out eval set
    print("\nStep 3 — Evaluating on held-out eval set...")
    eval_logits, eval_labels = collect_logits(eval_df, model, tokenizer, device)
    eval_probs  = platt.predict_proba(eval_logits.reshape(-1, 1))[:, 1]
    eval_preds  = (eval_probs >= 0.5).astype(int)
    f1          = f1_score(eval_labels, eval_preds)

    tp  = sum(1 for p, l in zip(eval_preds, eval_labels) if p == 1 and l == 1)
    fn  = sum(1 for p, l in zip(eval_preds, eval_labels) if p == 0 and l == 1)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  F1 after calibration: {f1:.4f}")
    print(f"  FNR after calibration: {fnr:.4f}")

    # Step 4: Save calibration parameters
    params = {
        "coef":      float(platt.coef_[0][0]),
        "intercept": float(platt.intercept_[0]),
    }
    with open(CALIB_PATH, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\n  Calibration params saved to {CALIB_PATH}")
    print(f"  coef={params['coef']:.6f}  intercept={params['intercept']:.6f}")
    print("\n" + "=" * 55)
    print("  DONE — update inference/predict.py to load calibration.json")
    print("=" * 55)


if __name__ == "__main__":
    run()
