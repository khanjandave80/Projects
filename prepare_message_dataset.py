import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "dataset", "combined_sms_mail_attachment.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(OUTPUT_DIR, "message_train.csv")
VAL_CSV = os.path.join(OUTPUT_DIR, "message_validation.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "message_test.csv")


def main():
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, usecols=["text", "message_label", "channel", "source_dataset"])
    print(f"Total rows: {len(df):,}")

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    print(f"After removing empty text: {len(df):,}")

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"Removed {before_dedup - len(df):,} duplicate texts")
    print(f"Unique texts: {len(df):,}")

    safe_count = (df["message_label"] == 0).sum()
    spam_count = (df["message_label"] == 1).sum()
    sms_count = (df["channel"] == "sms").sum()
    email_count = (df["channel"] == "email").sum()
    print(f"Safe (0): {safe_count:,}")
    print(f"Spam/Phishing (1): {spam_count:,}")
    print(f"SMS: {sms_count:,}")
    print(f"Email: {email_count:,}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["message_label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["message_label"]
    )

    print(f"Train: {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test: {len(test_df):,}")

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"Saved: {TRAIN_CSV}")
    print(f"Saved: {VAL_CSV}")
    print(f"Saved: {TEST_CSV}")


if __name__ == "__main__":
    main()