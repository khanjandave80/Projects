# train_model.py
# Fine-tunes DistilBERT for phishing email detection
# Input : subject + body + sender + urls (combined text)
# Output: phishing (1) or safe (0)

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# Config
# -------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "train_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MAX_LEN     = 256
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# -------------------------------------------------------
# Dataset
# -------------------------------------------------------
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids'      : self.encodings['input_ids'][idx],
            'attention_mask' : self.encodings['attention_mask'][idx],
            'labels'         : self.labels[idx]
        }

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str).str[:1024]  # cap length before tokenization

X_temp, X_test, y_temp, y_test = train_test_split(
    df['text'], df['label'], test_size=0.15, random_state=42, stratify=df['label']
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# -------------------------------------------------------
# Tokenizer + Datasets
# -------------------------------------------------------
print("Tokenizing...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_dataset = EmailDataset(X_train.values, y_train.values, tokenizer)
val_dataset   = EmailDataset(X_val.values,   y_val.values,   tokenizer)
test_dataset  = EmailDataset(X_test.values,  y_test.values,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# -------------------------------------------------------
# Model
# -------------------------------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# -------------------------------------------------------
# Train
# -------------------------------------------------------
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(DEVICE)
            mask  = batch['attention_mask'].to(DEVICE)
            labs  = batch['labels'].to(DEVICE)
            out   = model(input_ids=ids, attention_mask=mask)
            preds = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

history = {'train_f1': [], 'val_f1': [], 'val_recall': []}

print("\n" + "="*60)
print("TRAINING")
print("="*60)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        ids   = batch['input_ids'].to(DEVICE)
        mask  = batch['attention_mask'].to(DEVICE)
        labs  = batch['labels'].to(DEVICE)
        out   = model(input_ids=ids, attention_mask=mask, labels=labs)
        loss  = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        if (i+1) % 200 == 0:
            print(f"  Epoch {epoch+1} | Step {i+1}/{len(train_loader)} | Loss: {total_loss/(i+1):.4f}")

    tr_preds, tr_labels = evaluate(train_loader)
    vl_preds, vl_labels = evaluate(val_loader)
    tr_f1  = f1_score(tr_labels, tr_preds)
    vl_f1  = f1_score(vl_labels, vl_preds)
    vl_rec = recall_score(vl_labels, vl_preds)
    history['train_f1'].append(tr_f1)
    history['val_f1'].append(vl_f1)
    history['val_recall'].append(vl_rec)

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train F1: {tr_f1*100:.2f}% | Val F1: {vl_f1*100:.2f}% | Val Recall: {vl_rec*100:.2f}%")
    print("-"*60)

# -------------------------------------------------------
# Test set evaluation
# -------------------------------------------------------
print("\nEvaluating on held-out test set...")
te_preds, te_labels = evaluate(test_loader)

acc  = accuracy_score(te_labels,  te_preds)
prec = precision_score(te_labels, te_preds)
rec  = recall_score(te_labels,    te_preds)
f1   = f1_score(te_labels,        te_preds)

print("\n" + "="*60)
print("FINAL TEST RESULTS")
print("="*60)
print(f"Accuracy  : {acc *100:.2f}%")
print(f"Precision : {prec*100:.2f}%")
print(f"Recall    : {rec *100:.2f}%")
print(f"F1-Score  : {f1  *100:.2f}%")
print(f"Misclassified: {(te_preds!=te_labels).sum():,} / {len(te_labels):,}")
print()
print(classification_report(te_labels, te_preds, target_names=["Safe","Phishing"]))

# Confusion matrix
cm = confusion_matrix(te_labels, te_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe','Phishing'],
            yticklabels=['Safe','Phishing'])
plt.title("Confusion Matrix — DistilBERT\nPhishing Email Detection")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "data", "confusion_matrix.png"))
plt.show()

# Training curve
plt.figure(figsize=(8,4))
plt.plot(range(1, EPOCHS+1), history['train_f1'], 'b-o', label='Train F1')
plt.plot(range(1, EPOCHS+1), history['val_f1'],   'g-o', label='Val F1')
plt.plot(range(1, EPOCHS+1), history['val_recall'],'r-o', label='Val Recall')
plt.xlabel("Epoch"); plt.ylabel("Score")
plt.title("Training Curve — DistilBERT")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "data", "training_curve.png"))
plt.show()

# -------------------------------------------------------
# Save model + tokenizer
# -------------------------------------------------------
model.save_pretrained(os.path.join(MODELS_DIR, "distilbert_phishing"))
tokenizer.save_pretrained(os.path.join(MODELS_DIR, "distilbert_phishing"))
print(f"\nModel saved: models/distilbert_phishing/")
