"""Fast CUDA training and evaluation for the prepared message dataset.

Run from this folder:
    py train_complete.py

Install dependencies first if needed:
    py -m pip install torch transformers pandas scikit-learn matplotlib seaborn
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "message_distilbert")
MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42

TRAIN_FILE = os.path.join(DATA_DIR, "message_train.csv")
VAL_FILE = os.path.join(DATA_DIR, "message_validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "message_test.csv")


class MessageDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["message_label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frame(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, usecols=["text", "message_label"])
    frame = frame.dropna(subset=["text", "message_label"]).copy()
    frame["message_label"] = frame["message_label"].astype(int)
    invalid = sorted(set(frame["message_label"]) - {0, 1})
    if invalid:
        raise ValueError(f"message_label must contain only 0 and 1; found {invalid}")
    return frame


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate(model, loader, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_context:
                output = model(**batch)
            total_loss += output.loss.item() * batch["labels"].size(0)
            all_predictions.extend(output.logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    labels = np.asarray(all_labels)
    predictions = np.asarray(all_predictions)
    return (
        total_loss / len(loader.dataset),
        accuracy_score(labels, predictions),
        f1_score(labels, predictions, zero_division=0),
        labels,
        predictions,
    )


def save_training_plots(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], "o-", label="Train loss")
    plt.plot(epochs, history["val_loss"], "o-", label="Validation loss")
    plt.plot(epochs, history["train_accuracy"], "o-", label="Train accuracy")
    plt.plot(epochs, history["val_accuracy"], "o-", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and validation history")
    plt.xticks(list(epochs))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=160)
    plt.close()


def save_evaluation_plots(metrics, labels, predictions):
    names = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics[name.lower()] for name in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=["#2563eb", "#0f766e", "#d97706", "#7c3aed"])
    plt.ylim(0, 1.08)
    plt.ylabel("Score")
    plt.title("Test-set evaluation metrics")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_metrics.png"), dpi=160)
    plt.close()

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Safe", "Spam/Phishing"],
        yticklabels=["Safe", "Spam/Phishing"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=160)
    plt.close()


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    if use_amp:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("CUDA is unavailable; training will use CPU.")

    train_frame = load_frame(TRAIN_FILE)
    val_frame = load_frame(VAL_FILE)
    test_frame = load_frame(TEST_FILE)
    print(f"Train: {len(train_frame):,} | Validation: {len(val_frame):,} | Test: {len(test_frame):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data = MessageDataset(train_frame, tokenizer)
    val_data = MessageDataset(val_frame, tokenizer)
    test_data = MessageDataset(test_frame, tokenizer)
    loader_options = {
        "batch_size": BATCH_SIZE if use_amp else 8,
        "num_workers": 0,
        "pin_memory": use_amp,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    counts = np.bincount(train_frame["message_label"].to_numpy(), minlength=2)
    weights = torch.tensor(len(train_frame) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model.config.id2label = {0: "Safe", 1: "Spam/Phishing"}
    model.config.label2id = {"Safe": 0, "Spam/Phishing": 1}

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
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_context:
                output = model(**batch)
                weighted_loss = torch.nn.functional.cross_entropy(output.logits, batch["labels"], weight=weights)
            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size = batch["labels"].size(0)
            running_loss += weighted_loss.item() * batch_size
            correct += (output.logits.argmax(dim=1) == batch["labels"]).sum().item()
            seen += batch_size
            if step % 200 == 0 or step == len(train_loader):
                print(f"Epoch {epoch + 1}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss {running_loss / seen:.4f}")

        train_loss = running_loss / seen
        train_accuracy = correct / seen
        val_loss, val_accuracy, val_f1, _, _ = evaluate(model, val_loader, device, use_amp)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| train_acc={train_accuracy:.4f} | val_acc={val_accuracy:.4f} | val_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"Saved best model to {MODEL_DIR}")

    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    save_training_plots(history)

    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    _, _, _, labels, predictions = evaluate(best_model, test_loader, device, use_amp)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "classification_report": classification_report(
            labels, predictions, target_names=["Safe", "Spam/Phishing"], output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    save_evaluation_plots(metrics, labels, predictions)

    print("\nFinal test results")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(f"Model and reports saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
