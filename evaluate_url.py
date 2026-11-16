"""
evaluate_url.py
===============
Tests the saved BiLSTM URL model (Mission 2/3) on the held-out test set.

- Loads the trained model from models/url_bilstm/ (model.pt + char_index.json + config.json)
- Evaluates on data/url_test.csv (unseen URLs)
- Prints classification report + confusion matrix
- Runs manual inference checks on known phishing / safe / suspicious URLs

Run:  python evaluate_url.py
"""

import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models", "url_bilstm")
TEST_DATA    = os.path.join(BASE_DIR, "data", "url_test.csv")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 512
CLASS_NAMES  = ["Safe", "Phishing", "Suspicious"]


# -------------------------------------------------------
# Model (must match train_url_bilstm_pytorch.py)
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
        emb = self.drop_emb(self.embedding(x))
        out, _ = self.lstm(emb)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.drop_fc(out)
        return self.fc2(out)


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
        seq = seq + [0] * (self.max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# -------------------------------------------------------
# Load model + vocab
# -------------------------------------------------------
print("Loading model...")
with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)
with open(os.path.join(MODEL_DIR, "char_index.json"), "r", encoding="utf-8") as f:
    char_index = json.load(f)

MAX_LEN = cfg["max_len"]

model = URLBiLSTM(
    vocab_size=cfg["vocab_size"],
    embed_dim=cfg["embed_dim"],
    lstm_units=cfg["lstm_units"],
    num_classes=cfg["num_classes"],
    max_len=MAX_LEN,
).to(DEVICE)

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location=DEVICE))
model.eval()
print(f"Model loaded on {DEVICE} | vocab={cfg['vocab_size']} | max_len={MAX_LEN} | classes={cfg['num_classes']}")


# -------------------------------------------------------
# Evaluate on held-out test set
# -------------------------------------------------------
print("\nLoading test data...")
df = pd.read_csv(TEST_DATA)
df = df.dropna(subset=["url", "label"])
df["url"] = df["url"].astype(str)

dataset = URLDataset(df["url"].values, df["label"].values, char_index, MAX_LEN)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro")
rec  = recall_score(all_labels, all_preds, average="macro")
f1   = f1_score(all_labels, all_preds, average="macro")

print("\n" + "=" * 60)
print("UNSEEN TEST SET RESULTS (url_test.csv)")
print("=" * 60)
print(f"Accuracy  : {acc * 100:.2f}%")
print(f"Precision : {prec * 100:.2f}%")
print(f"Recall    : {rec * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"Misclassified: {(all_preds != all_labels).sum():,} / {len(all_labels):,}")
print()
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — BiLSTM URL Classifier\nHeld-out Test Set")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "data", "url_confusion_matrix.png"), dpi=200)
plt.show()


# -------------------------------------------------------
# Manual inference checks
# -------------------------------------------------------
MANUAL_URLS = [
    # Phishing
    ("http://secure-bankofamerica-login.xyz/verify", 1),
    ("http://paypal-account-restore.tk/update", 1),
    ("http://sbi-kyc-update.co/verify", 1),
    ("http://amazon-rewards-claim.ml/win?id=29471", 1),
    ("http://netflix-billing-update.top/renew", 1),
    ("http://192.168.1.1/amazon-verify", 1),
    ("http://google-account-login.cf/auth", 1),
    # Suspicious
    ("http://menu-download.tk/menu.apk", 2),
    ("http://free-tools-download.cf/tool.exe", 2),
    ("http://cashback-rewards-app.gq/app.zip", 2),
    # Safe
    ("https://www.amazon.com/your-orders", 0),
    ("https://www.hdfcbank.com/netbanking", 0),
    ("https://accounts.google.com/signin", 0),
    ("https://github.com/octocat/Hello-World", 0),
    ("https://www.zomato.com/orders", 0),
    ("upi://pay?pa=merchant@okhdfcbank&pn=TrustedShop&am=250", 0),
]

print("\n" + "=" * 60)
print("MANUAL URL INFERENCE TEST")
print("=" * 60)

correct = 0
for url, actual in MANUAL_URLS:
    unk = char_index.get("<UNK>", 1)
    seq = [char_index.get(c, unk) for c in url[:MAX_LEN]]
    seq = seq + [0] * (MAX_LEN - len(seq))
    x = torch.tensor([seq], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    confidence = round(float(probs[pred]) * 100, 1)
    result = "✓" if pred == actual else "✗"
    label_str = CLASS_NAMES[pred]
    actual_str = CLASS_NAMES[actual]
    print(f"{result} Predicted: {label_str:10} ({confidence}%) | Actual: {actual_str:10} | {url[:60]}")
    if pred == actual:
        correct += 1

print(f"\nManual test accuracy: {correct}/{len(MANUAL_URLS)} ({correct/len(MANUAL_URLS)*100:.0f}%)")
print("\nConfusion matrix saved: data/url_confusion_matrix.png")

