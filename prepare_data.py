# prepare_data.py
# Merges all 7 datasets, cleans labels, combines text fields
# Train set : Assassin, CEAS-08, Enron, TREC-05, TREC-06
# Unseen test: Ling, TREC-07

import pandas as pd
import os

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

def load(filename):
    return pd.read_csv(
        os.path.join(DATASET_DIR, filename),
        engine='python', on_bad_lines='skip', encoding='latin1'
    )

def clean_label(series):
    """Normalize labels to 0/1 int. Rows with corrupt labels are dropped."""
    s = series.astype(str).str.strip()
    s = s.map({'0': 0, '1': 1, '0.0': 0, '1.0': 1})   # NaN for anything else
    return s

def build_text(row):
    parts = []
    for col, prefix in [('subject','Subject:'), ('body','Body:'),
                        ('sender','From:'), ('urls','URLs:')]:
        if col in row.index:
            val = str(row[col]).strip()
            if val and val.lower() not in ('nan', 'none', ''):
                parts.append(f"{prefix} {val}")
    return " ".join(parts)

def process(df):
    df = df.copy()
    df['label'] = clean_label(df['label'])
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df['text']  = df.apply(build_text, axis=1)
    df = df[df['text'].str.strip() != '']
    df = df[['text', 'label']].drop_duplicates(subset=['text'])
    return df

# -------------------------------------------------------
# Load and process
# -------------------------------------------------------
print("Loading datasets...")

train_files  = ['Assassin.csv', 'CEAS-08.csv', 'Enron.csv', 'TREC-05.csv', 'TREC-06.csv']
unseen_files = ['Ling.csv', 'TREC-07.csv']

train_dfs  = [process(load(f)) for f in train_files]
unseen_dfs = [process(load(f)) for f in unseen_files]

train_df  = pd.concat(train_dfs,  ignore_index=True).drop_duplicates(subset=['text'])
unseen_df = pd.concat(unseen_dfs, ignore_index=True).drop_duplicates(subset=['text'])

# Remove any unseen samples that leaked into train
overlap = set(train_df['text']) & set(unseen_df['text'])
train_df  = train_df[~train_df['text'].isin(overlap)]
unseen_df = unseen_df[~unseen_df['text'].isin(overlap)]

# -------------------------------------------------------
# Save
# -------------------------------------------------------
train_df.to_csv(os.path.join(OUT_DIR, "train_data.csv"),  index=False)
unseen_df.to_csv(os.path.join(OUT_DIR, "unseen_test.csv"), index=False)

print(f"\nTrain set  : {len(train_df):,} emails")
print(f"  Phishing : {(train_df['label']==1).sum():,}")
print(f"  Safe     : {(train_df['label']==0).sum():,}")
print(f"\nUnseen test: {len(unseen_df):,} emails")
print(f"  Phishing : {(unseen_df['label']==1).sum():,}")
print(f"  Safe     : {(unseen_df['label']==0).sum():,}")
print(f"\nSaved: data/train_data.csv")
print(f"Saved: data/unseen_test.csv")
