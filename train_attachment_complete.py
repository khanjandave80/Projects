"""CUDA attachment classifier with checkpointing and evaluation plots.

Run:
    py -3.12 train_attachment_complete.py
"""

import json
import os
import random
from contextlib import nullcontext

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cic_attachment_text.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "attachment_distilbert")
MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42
CHUNK_SIZE = 50000


class AttachmentDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.texts = frame["model_text"].tolist()
        self.labels = frame["attachment_label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_attachment_data():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Attachment dataset not found: {DATA_PATH}")
    columns = ["original_filename", "file_format", "attachment_label", "extracted_text", "extraction_status", "sha256"]
    pieces = []
    print("Reading attachment CSV in chunks...")
    for chunk_number, chunk in enumerate(pd.read_csv(DATA_PATH, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False), 1):
        chunk = chunk[chunk["attachment_label"].isin([0, 1])].copy()
        if chunk.empty:
            continue
        chunk["sha256"] = chunk["sha256"].fillna(chunk["original_filename"])
        chunk["original_filename"] = chunk["original_filename"].fillna("unknown_filename").astype(str)
        chunk["file_format"] = chunk["file_format"].fillna("unknown_format").astype(str)
        chunk["extraction_status"] = chunk["extraction_status"].fillna("unknown_status").astype(str)
        chunk["extracted_text"] = chunk["extracted_text"].fillna("").astype(str).str.slice(0, 12000)
        chunk["model_text"] = (
            "File format: " + chunk["file_format"] + "\nFilename: " + chunk["original_filename"]
            + "\nExtraction: " + chunk["extraction_status"] + "\nContent: " + chunk["extracted_text"]
        )
        pieces.append(chunk[["sha256", "model_text", "attachment_label"]])
        if chunk_number % 10 == 0:
            print(f"  Read {chunk_number * CHUNK_SIZE:,} rows...")
    if not pieces:
        raise ValueError("No attachment rows with attachment_label 0 or 1 were found.")
    frame = pd.concat(pieces, ignore_index=True)
    before = len(frame)
    frame = frame.drop_duplicates("sha256").drop(columns="sha256")
    print(f"Usable rows: {len(frame):,} (removed {before - len(frame):,} duplicates)")
    if frame["attachment_label"].nunique() != 2:
        raise ValueError("Attachment data must contain both benign (0) and malicious (1) labels.")
    return frame


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate(model, loader, device, use_amp):
    model.eval()
    total_loss = 0.0
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_context:
                output = model(**batch)
            total_loss += output.loss.item() * batch["labels"].size(0)
            predictions.extend(output.logits.argmax(dim=1).cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    return total_loss / len(loader.dataset), accuracy_score(labels, predictions), f1_score(labels, predictions, zero_division=0), labels, predictions


def save_plots(history, metrics, labels, predictions):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(9, 5))
    for key, label in [("train_loss", "Train loss"), ("val_loss", "Validation loss"), ("train_accuracy", "Train accuracy"), ("val_accuracy", "Validation accuracy")]:
        plt.plot(epochs, history[key], "o-", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Attachment training history")
    plt.xticks(list(epochs))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=160)
    plt.close()

    names = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics[name.lower()] for name in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=["#2563eb", "#0f766e", "#d97706", "#dc2626"])
    plt.ylim(0, 1.08)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    plt.ylabel("Score")
    plt.title("Attachment test metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_metrics.png"), dpi=160)
    plt.close()

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Oranges", xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Attachment confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=160)
    plt.close()


def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    if use_amp:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    frame = load_attachment_data()
    train, temporary = train_test_split(frame, test_size=0.30, random_state=SEED, stratify=frame["attachment_label"])
    validation, test = train_test_split(temporary, test_size=0.50, random_state=SEED, stratify=temporary["attachment_label"])
    print(f"Train: {len(train):,} | Validation: {len(validation):,} | Test: {len(test):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    loader_options = {"batch_size": BATCH_SIZE, "num_workers": 0, "pin_memory": use_amp}
    train_loader = DataLoader(AttachmentDataset(train, tokenizer), shuffle=True, **loader_options)
    validation_loader = DataLoader(AttachmentDataset(validation, tokenizer), shuffle=False, **loader_options)
    test_loader = DataLoader(AttachmentDataset(test, tokenizer), shuffle=False, **loader_options)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    counts = np.bincount(train["attachment_label"].to_numpy(), minlength=2)
    class_weights = torch.tensor(len(train) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model.config.id2label = {0: "Benign", 1: "Malicious"}
    model.config.label2id = {"Benign": 0, "Malicious": 1}
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [], "val_f1": []}
    best_f1 = -1.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for step, batch in enumerate(train_loader, 1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_context:
                output = model(**batch)
                loss = torch.nn.functional.cross_entropy(output.logits, batch["labels"], weight=class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            size = batch["labels"].size(0)
            running_loss += loss.item() * size
            correct += (output.logits.argmax(dim=1) == batch["labels"]).sum().item()
            seen += size
            if step % 200 == 0 or step == len(train_loader):
                print(f"Epoch {epoch + 1}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss {running_loss / seen:.4f}")
        train_loss = running_loss / seen
        train_accuracy = correct / seen
        val_loss, val_accuracy, val_f1, _, _ = evaluate(model, validation_loader, device, use_amp)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)
        print(f"Epoch {epoch + 1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | train_acc={train_accuracy:.4f} | val_acc={val_accuracy:.4f} | val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"Saved best model to {MODEL_DIR}")

    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    _, _, _, labels, predictions = evaluate(best_model, test_loader, device, use_amp)
    report = classification_report(labels, predictions, target_names=["Benign", "Malicious"], output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
    }
    with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    save_plots(history, metrics, labels, predictions)
    print("\nFinal attachment test results")
    print(json.dumps(metrics, indent=2))
    print(f"All outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
