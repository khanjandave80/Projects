# evaluate.py
# Tests the saved DistilBERT model on fully unseen data (Ling + TREC-07)
# Also tests on manually written fake bank emails

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert_phishing")
DATA_PATH  = os.path.join(BASE_DIR, "data", "unseen_test.csv")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 256
BATCH_SIZE = 32

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids'     : self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels'        : self.labels[idx]
        }

# -------------------------------------------------------
# Load model
# -------------------------------------------------------
print("Loading model...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# -------------------------------------------------------
# Evaluate on unseen dataset
# -------------------------------------------------------
print("Loading unseen test data (Ling + TREC-07)...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text','label'])
df['text'] = df['text'].astype(str).str[:1024]

dataset = EmailDataset(df['text'].values, df['label'].values, tokenizer)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        out  = model(input_ids=ids, attention_mask=mask)
        preds = torch.argmax(out.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

print("\n" + "="*60)
print("UNSEEN DATA RESULTS (Ling + TREC-07)")
print("="*60)
print(f"Accuracy  : {accuracy_score(all_labels, all_preds)*100:.2f}%")
print(f"Precision : {precision_score(all_labels, all_preds)*100:.2f}%")
print(f"Recall    : {recall_score(all_labels, all_preds)*100:.2f}%")
print(f"F1-Score  : {f1_score(all_labels, all_preds)*100:.2f}%")
print(f"Misclassified: {(all_preds!=all_labels).sum():,} / {len(all_labels):,}")
print()
print(classification_report(all_labels, all_preds, target_names=["Safe","Phishing"]))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Safe','Phishing'],
            yticklabels=['Safe','Phishing'])
plt.title("Confusion Matrix — Unseen Test (Ling + TREC-07)")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "data", "unseen_confusion_matrix.png"))
plt.show()

# -------------------------------------------------------
# Manual fake bank emails (robustness check)
# -------------------------------------------------------
MANUAL_EMAILS = [
    # Phishing
    ("Subject: Urgent: Your account has been suspended Body: Dear customer, your bank account has been suspended due to suspicious activity. Click here to verify: http://secure-bank-login.xyz/verify From: support@bankofamerica-secure.net URLs: http://secure-bank-login.xyz/verify", 1),
    ("Subject: Action Required: Update your KYC Body: Your account will be blocked in 24 hours. Please submit your OTP and Aadhaar details at http://kyc-update-bank.tk From: noreply@sbi-kyc-update.com URLs: http://kyc-update-bank.tk", 1),
    ("Subject: You have won a reward Body: Congratulations! Click http://paypal-reward-claim.ml/win to claim your $500 reward now. From: rewards@paypal-offers.ga URLs: http://paypal-reward-claim.ml/win", 1),
    ("Subject: Verify your identity Body: We noticed unusual login. Confirm your password at http://192.168.1.1/login From: security@amazon-alert.cf URLs: http://192.168.1.1/login", 1),
    ("Subject: Your Netflix subscription expired Body: Renew now to avoid losing access http://netflix-billing-update.top/renew From: billing@netflix-support.xyz URLs: http://netflix-billing-update.top/renew", 1),
    # Safe
    ("Subject: Your monthly statement is ready Body: Dear customer, your statement for June 2025 is ready. Login at https://www.hdfcbank.com From: statements@hdfcbank.com URLs: https://www.hdfcbank.com", 0),
    ("Subject: Meeting scheduled for tomorrow Body: Hi team, the project meeting is at 10am. From: manager@company.com URLs: ", 0),
    ("Subject: Your order has been shipped Body: Your Amazon order #12345 has been shipped. Track at https://www.amazon.com/orders From: shipment-tracking@amazon.com URLs: https://www.amazon.com/orders", 0),
    ("Subject: Password changed successfully Body: Your password was changed. If this was not you contact support at https://support.google.com From: no-reply@google.com URLs: https://support.google.com", 0),
    ("Subject: Welcome to GitHub Body: Your account is ready. Get started at https://github.com From: noreply@github.com URLs: https://github.com", 0),
]

texts  = [e[0] for e in MANUAL_EMAILS]
labels = [e[1] for e in MANUAL_EMAILS]

enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors='pt')
with torch.no_grad():
    out   = model(input_ids=enc['input_ids'].to(DEVICE), attention_mask=enc['attention_mask'].to(DEVICE))
    preds = torch.argmax(out.logits, dim=1).cpu().numpy()
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()

print("\n" + "="*60)
print("MANUAL FAKE BANK EMAIL TEST")
print("="*60)
correct = 0
for i, (text, actual) in enumerate(MANUAL_EMAILS):
    pred       = preds[i]
    confidence = round(probs[i][pred] * 100, 1)
    result     = "✓" if pred == actual else "✗"
    label_str  = "Phishing" if pred == 1 else "Safe"
    actual_str = "Phishing" if actual == 1 else "Safe"
    print(f"{result} Predicted: {label_str:8} ({confidence}%) | Actual: {actual_str:8} | {text[:60]}...")
    if pred == actual: correct += 1

print(f"\nManual test accuracy: {correct}/{len(MANUAL_EMAILS)} ({correct/len(MANUAL_EMAILS)*100:.0f}%)")
