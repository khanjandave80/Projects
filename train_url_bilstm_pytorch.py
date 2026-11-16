"""
train_url_bilstm_pytorch.py
===========================
Mission 2/3 — Character-level BiLSTM URL classifier (PyTorch).

Reads the 3-class dataset produced by prepare_url_data.py:
  class 0 = Safe
  class 1 = Phishing
  class 2 = Suspicious

Trains a BiLSTM that reads URLs character-by-character and saves:
  models/url_bilstm/model.pt
  models/url_bilstm/char_index.json
  models/url_bilstm/config.json

At the end it prints a FULL EVALUATION REPORT:
  - Accuracy / Precision / Recall / F1 (macro + weighted)
  - Per-class classification report
  - Confusion matrix (raw + saved PNG)
  - Training / validation curves (saved PNG)
"""

import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless-safe (works in Colab / no display)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

# -------------------------------------------------------
# Paths (override to '/content' when running in Colab)
# -------------------------------------------------------
BASE_DIR = "/content" if os.path.exists("/content") else os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "models", "url_bilstm")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Config
# -------------------------------------------------------
MAX_LEN     = 200
EMBED_DIM   = 64
LSTM_UNITS  = 128
BATCH_SIZE  = 512
EPOCHS      = 8
LR          = 1e-3
RANDOM_STATE = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print(f"Device: {DEVICE}")
print(f"Using PyTorch {torch.__version__}")


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------
class URLDataset(Dataset):
    def __init__(self, urls, labels, char_index, max_len):
        self.max_len = max_len
        self.char_index = char_index
        self.urls = list(urls)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        url = str(self.urls[idx])[: self.max_len]
        unk = self.char_index.get("<UNK>", 1)
        seq = [self.char_index.get(c, unk) for c in url]
        seq = seq + [0] * (self.max_len - len(seq))  # pad (0 reserved)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# -------------------------------------------------------
# Model
# -------------------------------------------------------
class URLBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, num_classes, max_len, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop_emb = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            embed_dim,
            lstm_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.dropout = nn.Dropout(0.3)
        hidden = lstm_units * 2  # bidirectional
        self.fc1 = nn.Linear(hidden, 64)
        self.relu = nn.ReLU()
        self.drop_fc = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        emb = self.drop_emb(self.embedding(x))          # B x T x E
        out, _ = self.lstm(emb)                          # B x T x 2H
        out = out[:, -1, :]                              # take last timestep
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.drop_fc(out)
        logits = self.fc2(out)
        return logits


# -------------------------------------------------------
# Load data
# -------------------------------------------------------
print("\nLoading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "url_train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "url_val.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "url_test.csv"))

train_df = train_df.dropna(subset=["url", "label"])
val_df   = val_df.dropna(subset=["url", "label"])
test_df  = test_df.dropna(subset=["url", "label"])

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Build char vocab from training URLs
all_chars = sorted(set("".join(train_df["url"].astype(str).tolist())))
char_index = {c: i + 1 for i, c in enumerate(all_chars)}  # 0 reserved for padding
char_index["<UNK>"] = len(char_index) + 1
VOCAB_SIZE = len(char_index) + 1
print(f"Vocab size: {VOCAB_SIZE}")

train_ds = URLDataset(train_df["url"], train_df["label"], char_index, MAX_LEN)
val_ds   = URLDataset(val_df["url"],   val_df["label"],   char_index, MAX_LEN)
test_ds  = URLDataset(test_df["url"],  test_df["label"],  char_index, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Class weights
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_df["label"].values,
)
class_weights = torch.tensor(class_weights_arr, dtype=torch.float32).to(DEVICE)
print("Class weights:", class_weights_arr)

# -------------------------------------------------------
# Model + optimizer
# -------------------------------------------------------
model = URLBiLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    lstm_units=LSTM_UNITS,
    num_classes=3,
    max_len=MAX_LEN,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(weight=class_weights)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
)


# -------------------------------------------------------
# Training helpers
# -------------------------------------------------------
def run_epoch(loader, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


# -------------------------------------------------------
# Train
# -------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(EPOCHS):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | Val loss {val_loss:.4f} acc {val_acc*100:.2f}%")

# -------------------------------------------------------
# Evaluate on test
# -------------------------------------------------------
print("\nEvaluating on test set...")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

CLASS_NAMES = ["Safe", "Phishing", "Suspicious"]

# -------------------------------------------------------
# FULL EVALUATION REPORT
# -------------------------------------------------------
acc      = accuracy_score(all_labels, all_preds)
mac_prec = precision_score(all_labels, all_preds, average="macro")
mac_rec  = recall_score(all_labels, all_preds, average="macro")
mac_f1   = f1_score(all_labels, all_preds, average="macro")
wei_prec = precision_score(all_labels, all_preds, average="weighted")
wei_rec  = recall_score(all_labels, all_preds, average="weighted")
wei_f1   = f1_score(all_labels, all_preds, average="weighted")

print("\n" + "=" * 60)
print("=== FULL EVALUATION REPORT ===")
print("=" * 60)
print(f"\nAccuracy            : {acc*100:.2f}%")
print(f"Macro Precision     : {mac_prec*100:.2f}%")
print(f"Macro Recall        : {mac_rec*100:.2f}%")
print(f"Macro F1            : {mac_f1*100:.2f}%")
print(f"Weighted Precision  : {wei_prec*100:.2f}%")
print(f"Weighted Recall     : {wei_rec*100:.2f}%")
print(f"Weighted F1         : {wei_f1*100:.2f}%")
print(f"Misclassified       : {(all_preds != all_labels).sum():,} / {len(all_labels):,}")

print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)
print("\n--- Confusion Matrix (raw) ---")
print(cm)

print("\n--- Per-Class Metrics ---")
class_prec = precision_score(all_labels, all_preds, average=None)
class_rec  = recall_score(all_labels, all_preds, average=None)
class_f1   = f1_score(all_labels, all_preds, average=None)
header = f"{'Class':<12}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Support':>10}"
print(header)
print("-" * len(header))
for i, name in enumerate(CLASS_NAMES):
    support = int((all_labels == i).sum())
    print(f"{name:<12}{class_prec[i]*100:>11.2f}%{class_rec[i]*100:>9.2f}%{class_f1[i]*100:>9.2f}%{support:>10,}")

# -------------------------------------------------------
# Confusion matrix plot (saved as PNG)
# -------------------------------------------------------
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — BiLSTM URL Classifier\n(Test Set)")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
print(f"\nConfusion matrix saved: {cm_path}")

# -------------------------------------------------------
# Training curves (saved as PNG)
# -------------------------------------------------------
epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
axes[0].plot(epochs, history["val_loss"],   "g-o", label="Val Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Training / Validation Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc")
axes[1].plot(epochs, [a * 100 for a in history["val_acc"]],   "g-o", label="Val Acc")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Training / Validation Accuracy")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(OUT_DIR, "training_curves.png")
plt.savefig(curves_path, dpi=200)
print(f"Training curves saved: {curves_path}")

# -------------------------------------------------------
# Save artifacts
# -------------------------------------------------------
torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))

with open(os.path.join(OUT_DIR, "char_index.json"), "w", encoding="utf-8") as f:
    json.dump(char_index, f, indent=2)

with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "max_len": MAX_LEN,
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
            "lstm_units": LSTM_UNITS,
            "num_classes": 3,
            "class_names": ["Safe", "Phishing", "Suspicious"],
        },
        f,
        indent=2,
    )

print("\nSaved files in:", OUT_DIR)
print("  model.pt")
print("  char_index.json")
print("  config.json")
print("  confusion_matrix.png")
print("  training_curves.png")
print("\nDone!")

