"""
prepare_url_data.py
===================
Mission 2/3 dataset preparation.

Merges the two raw URL datasets:
  - data/malicious_phish.csv   (url,type  -> benign/phishing/defacement/malware)
  - data/URL dataset.csv       (url,type  -> legitimate/phishing)

Maps the raw labels into the 3-class scheme used by the BiLSTM:
  class 0 = Safe       (benign + legitimate)
  class 1 = Phishing   (phishing)
  class 2 = Suspicious (malware + defacement)

Writes a balanced, cleaned CSV and train/val/test splits:
  data/url_train.csv
  data/url_val.csv
  data/url_test.csv
"""

import os
import random

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["url", "type"]].dropna().copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df = df[df["url"] != ""]
    df = df[df["url"] != "nan"]
    df = df.drop_duplicates(subset=["url"])
    return df


def map_label(t: str) -> int:
    if t in ("benign", "legitimate"):
        return 0          # Safe
    if t == "phishing":
        return 1          # Phishing
    if t in ("malware", "defacement"):
        return 2          # Suspicious
    raise ValueError(f"Unknown type: {t}")


def main() -> None:
    print("=" * 60)
    print("PREPARE URL DATA  (3-class)")
    print("=" * 60)

    # -------------------------------------------------------
    # Load both CSVs
    # -------------------------------------------------------
    df1 = load_and_clean(os.path.join(DATA_DIR, "malicious_phish.csv"))
    df2 = load_and_clean(os.path.join(DATA_DIR, "URL dataset.csv"))

    print(f"\nmalicious_phish.csv : {len(df1):,} rows")
    print(df1["type"].value_counts().to_string())
    print(f"\nURL dataset.csv     : {len(df2):,} rows")
    print(df2["type"].value_counts().to_string())

    # -------------------------------------------------------
    # Merge + label
    # -------------------------------------------------------
    merged = pd.concat([df1, df2], ignore_index=True)
    merged = merged.drop_duplicates(subset=["url"]).reset_index(drop=True)
    merged["label"] = merged["type"].map(map_label)

    print(f"\nMerged unique rows  : {len(merged):,}")
    print("\n3-class distribution (before balancing):")
    print(merged["label"].value_counts().sort_index().to_string())

    # -------------------------------------------------------
    # Balance to the minority class count (cap for speed)
    # -------------------------------------------------------
    counts = merged["label"].value_counts().sort_index()
    min_count = int(counts.min())
    # Cap each class so the final dataset stays manageable
    cap = min(min_count, 120_000)
    print(f"\nBalancing each class to {cap:,} samples...")

    balanced_parts = []
    for label in range(3):
        cls = merged[merged["label"] == label]
        if len(cls) > cap:
            cls = cls.sample(n=cap, random_state=RANDOM_STATE)
        balanced_parts.append(cls)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nFinal balanced distribution:")
    print(balanced["label"].value_counts().sort_index().to_string())

    # -------------------------------------------------------
    # Save merged + balanced master
    # -------------------------------------------------------
    master_path = os.path.join(DATA_DIR, "url_dataset_3class.csv")
    balanced[["url", "type", "label"]].to_csv(master_path, index=False)
    print(f"\nSaved master        : {master_path}")

    # -------------------------------------------------------
    # Train / val / test split (stratified)
    # -------------------------------------------------------
    from sklearn.model_selection import train_test_split

    X = balanced["url"]
    y = balanced["label"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=RANDOM_STATE, stratify=y_temp
    )

    def save_split(urls, labels, name):
        out = pd.DataFrame({"url": urls, "label": labels})
        path = os.path.join(DATA_DIR, name)
        out.to_csv(path, index=False)
        print(f"Saved {name:18s}: {len(out):,} rows  |  {out['label'].value_counts().sort_index().to_dict()}")

    save_split(X_train, y_train, "url_train.csv")
    save_split(X_val,   y_val,   "url_val.csv")
    save_split(X_test,  y_test,  "url_test.csv")

    print("\n" + "=" * 60)
    print("DONE — ready to train with train_url_bilstm_pytorch.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

