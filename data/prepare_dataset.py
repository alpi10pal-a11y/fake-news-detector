"""
Sub-Task 1 — Dataset Preparation
Merges Fake.csv and True.csv into a single dataset.csv with columns: text, label.
Run from the project root:  python data/prepare_dataset.py
"""

import os
import sys
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")
OUT_PATH = os.path.join(DATA_DIR, "dataset.csv")
RANDOM_SEED = 42


def check_source_files():
    missing = []
    if not os.path.exists(FAKE_PATH):
        missing.append(FAKE_PATH)
    if not os.path.exists(TRUE_PATH):
        missing.append(TRUE_PATH)
    if missing:
        print("ERROR: The following source files are missing:")
        for p in missing:
            print(f"  {p}")
        print("\nDownload them from:")
        print("  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("Then place Fake.csv and True.csv inside the data/ folder and re-run this script.")
        sys.exit(1)


def load_and_label(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\nLoaded {path}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Rows    : {len(df)}")

    # Build a combined text column from title + body when both exist,
    # otherwise fall back to whichever column is present.
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "title" in df.columns:
        df["text"] = df["title"].fillna("")
    elif "text" in df.columns:
        df["text"] = df["text"].fillna("")
    else:
        print(f"  ERROR: neither 'title' nor 'text' column found in {path}")
        sys.exit(1)

    df["label"] = label
    return df[["text", "label"]]


def main():
    check_source_files()

    fake_df = load_and_label(FAKE_PATH, "FAKE")
    true_df = load_and_label(TRUE_PATH, "REAL")

    combined = pd.concat([fake_df, true_df], ignore_index=True)

    # Strip whitespace from text
    combined["text"] = combined["text"].str.strip()

    # Remove rows where text is empty after stripping
    before = len(combined)
    combined = combined[combined["text"].str.len() > 0]
    removed = before - len(combined)
    if removed:
        print(f"\nRemoved {removed} row(s) with empty text.")

    # Shuffle with fixed seed for reproducibility
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Save
    combined.to_csv(OUT_PATH, index=False)

    # Report
    print(f"\n{'='*45}")
    print(f"  dataset.csv written to: {OUT_PATH}")
    print(f"{'='*45}")
    print(f"  Total rows : {len(combined)}")
    counts = combined["label"].value_counts()
    for label, count in counts.items():
        pct = count / len(combined) * 100
        print(f"  {label:<6} : {count}  ({pct:.1f}%)")
    print(f"{'='*45}")
    print("\nSub-Task 1 complete.")


if __name__ == "__main__":
    main()
