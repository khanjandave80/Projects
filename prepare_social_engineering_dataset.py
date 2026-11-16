"""Create the binary dataset for the social-engineering mission.

Source: data/URL dataset.csv only.
Output: data/social_engineering_200k.csv
"""

import os

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SOURCE_PATH = os.path.join(DATA_DIR, "URL dataset.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "social_engineering_200k.csv")
TARGET_ROWS = 200_000
RANDOM_STATE = 42


def main() -> None:
    source = pd.read_csv(SOURCE_PATH, usecols=["url", "type"])
    source["url"] = source["url"].astype(str).str.strip()
    source["type"] = source["type"].astype(str).str.strip().str.lower()
    source = source[source["url"].ne("") & source["url"].ne("nan")]
    source = source[source["type"].isin(["legitimate", "phishing"])]
    source = source.drop_duplicates(subset=["url"])

    scam = source[source["type"].eq("phishing")]
    safe = source[source["type"].eq("legitimate")]
    scam_count = min(len(scam), TARGET_ROWS // 2)
    safe_count = TARGET_ROWS - scam_count
    if len(safe) < safe_count:
        raise ValueError("The source does not contain enough safe records.")

    selected = pd.concat(
        [
            scam.sample(n=scam_count, random_state=RANDOM_STATE),
            safe.sample(n=safe_count, random_state=RANDOM_STATE),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    output = pd.DataFrame(
        {
            "id": range(1, len(selected) + 1),
            "url": selected["url"],
            "category": selected["type"].map(
                {"phishing": "scam", "legitimate": "safe"}
            ),
            "label": selected["type"].map({"phishing": 1, "legitimate": 0}),
        }
    )
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(output):,} records to {OUTPUT_PATH}")
    print(output["category"].value_counts().to_string())


if __name__ == "__main__":
    main()