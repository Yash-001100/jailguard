"""
Train a lightweight attack type classifier on top of frozen DistilBERT embeddings.

Strategy:
  - Load the trained binary model (DistilBertForSequenceClassification)
  - Extract the [CLS] token embedding (last hidden layer) for each jailbreak
    example in train.csv
  - Fit a 6-class LogisticRegression on those embeddings
  - Evaluate on test.csv jailbreak examples
  - Save to inference/attack_head.joblib

This is fast (~5-8 min) because DistilBERT stays frozen — we only train
a 768->6 linear classifier on top.

Run with:
    python model/train_attack_head.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm

SAVE_DIR         = os.path.join(os.path.dirname(__file__), "..", "saved_model")
ATTACK_HEAD_PATH = os.path.join(os.path.dirname(__file__), "..", "inference", "attack_head.joblib")
DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_LENGTH       = 256
BATCH_SIZE       = 256

ATTACK_TYPES = [
    "prompt_injection",
    "role_play_manipulation",
    "encoding_obfuscation",
    "incremental_escalation",
    "system_manipulation",
    "data_extraction",
]


def extract_cls(df, model, tokenizer, device):
    """Run model with output_hidden_states=True, return [CLS] embeddings."""
    model.eval()
    all_cls = []

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Extracting embeddings"):
        batch = df["text"].iloc[i:i+BATCH_SIZE].tolist()
        enc   = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
                output_hidden_states=True,
            )
        # Last hidden layer, [CLS] token → shape (batch, 768)
        cls = out.hidden_states[-1][:, 0, :].cpu().numpy()
        all_cls.append(cls)

    return np.vstack(all_cls)


def run():
    print("=" * 57)
    print("  ATTACK TYPE HEAD TRAINING")
    print("=" * 57)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(SAVE_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR).to(device)

    # Load CSVs and filter to jailbreak examples with known attack types
    print("\nLoading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8")
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  encoding="utf-8")

    train_atk = train_df[
        (train_df["label"] == 1) & (train_df["attack_type"].isin(ATTACK_TYPES))
    ].reset_index(drop=True)
    test_atk  = test_df[
        (test_df["label"] == 1) & (test_df["attack_type"].isin(ATTACK_TYPES))
    ].reset_index(drop=True)

    print(f"  Train jailbreak examples: {len(train_atk):,}")
    print(f"  Test  jailbreak examples: {len(test_atk):,}")
    print("\n  Attack type distribution (train):")
    counts = train_atk["attack_type"].value_counts()
    for atype in ATTACK_TYPES:
        n = counts.get(atype, 0)
        bar = "#" * (n // 500)
        print(f"    {atype:<30} {n:>6}  {bar}")

    # Step 1: Extract embeddings
    print("\nStep 1 — Extracting train embeddings...")
    train_X = extract_cls(train_atk, model, tokenizer, device)
    train_y = train_atk["attack_type"].values

    print("\nStep 2 — Extracting test embeddings...")
    test_X = extract_cls(test_atk, model, tokenizer, device)
    test_y = test_atk["attack_type"].values

    # Step 2: Fit logistic regression with scaling + balanced class weights
    print("\nStep 3 — Fitting LogisticRegression (scaled, class_weight=balanced)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="saga",
            class_weight="balanced",
        )),
    ])
    pipeline.fit(train_X, train_y)
    print("  Done.")

    # Step 3: Evaluate
    print("\nStep 4 — Evaluating on test set...")
    preds    = pipeline.predict(test_X)
    macro    = f1_score(test_y, preds, average="macro",    zero_division=0)
    weighted = f1_score(test_y, preds, average="weighted", zero_division=0)
    present_types = sorted(set(test_y))
    print(classification_report(test_y, preds, labels=present_types, zero_division=0))
    print(f"  Macro F1:    {macro:.4f}")
    print(f"  Weighted F1: {weighted:.4f}")
    print()
    print("  Note: incremental_escalation / system_manipulation / data_extraction")
    print("  have very few training samples (<100). These classes will be less reliable.")

    # Step 4: Save
    payload = {"model": pipeline, "classes": ATTACK_TYPES}
    joblib.dump(payload, ATTACK_HEAD_PATH)
    size_kb = os.path.getsize(ATTACK_HEAD_PATH) / 1024
    print(f"\n  Attack head saved: {ATTACK_HEAD_PATH}  ({size_kb:.0f} KB)")
    print("\n" + "=" * 57)
    print("  DONE")
    print("=" * 57)


if __name__ == "__main__":
    run()
