import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import os
import time

# ── Hyperparameters (from paper Table 1) ─────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 128       # safe for 16GB VRAM; 4x faster than paper's 32
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_model")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class JailbreakDataset(Dataset):
    """Lazy tokenisation — tokenises each sample at access time, not upfront.
    This lets training start immediately instead of waiting several minutes."""

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, loader, device, label="Eval"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {label}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    f1 = f1_score(all_labels, all_preds)
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  [{label}] F1: {f1:.4f}  |  FNR: {fnr:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["safe", "jailbreak"], digits=4))
    return f1


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8")
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  encoding="utf-8")
    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}\n")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = JailbreakDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    test_dataset  = JailbreakDataset(test_df["text"].tolist(),  test_df["label"].tolist(),  tokenizer)

    # num_workers=0 required on Windows — multiprocessing causes hangs otherwise
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE * 2,            num_workers=0, pin_memory=True)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Steps per epoch: {len(train_loader):,}")
    print(f"Total steps:     {total_steps:,}")
    print(f"Starting training...\n")

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss   = total_loss / len(train_loader)
        epoch_mins = (time.time() - epoch_start) / 60
        print(f"\nEpoch {epoch+1} — avg loss: {avg_loss:.4f}  |  time: {epoch_mins:.1f} min")

        f1 = evaluate(model, test_loader, device, label=f"Epoch {epoch+1}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  ** Best F1: {best_f1:.4f} — saved to saved_model/\n")

    print(f"Training complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
