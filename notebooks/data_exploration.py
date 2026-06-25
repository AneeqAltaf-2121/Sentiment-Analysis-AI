"""Exploratory analysis of the IMDb dataset (no cleaning or modeling)."""

import os

import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "IMDB Dataset.csv")

pd.set_option("display.max_colwidth", 100)

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("FIRST FEW ROWS")
print("=" * 80)
print(df.head())

print("\n" + "=" * 80)
print("DATASET DIMENSIONS")
print("=" * 80)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n" + "=" * 80)
print("COLUMN NAMES")
print("=" * 80)
print(df.columns.tolist())

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)
print(df.describe(include="all"))

print("\n" + "=" * 80)
print("SUMMARY INFO")
print("=" * 80)
df.info()

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
print(df.isnull().sum())

print("\n" + "=" * 80)
print("DUPLICATE RECORDS")
print("=" * 80)
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\n" + "=" * 80)
print("CLASS BALANCE (sentiment)")
print("=" * 80)
print(df["sentiment"].value_counts())
print(df["sentiment"].value_counts(normalize=True))

print("\n" + "=" * 80)
print("REVIEW LENGTH ANALYSIS")
print("=" * 80)

review_character_lengths = df["review"].str.len()
print(review_lengths.describe())

print("\n" + "=" * 80)
print("SAMPLE POSITIVE REVIEWS")
print("=" * 80)
for review in df.loc[df["sentiment"] == "positive", "review"].head(3):
    print(review[:300], "...\n")

print("\n" + "=" * 80)
print("SAMPLE NEGATIVE REVIEWS")
print("=" * 80)
for review in df.loc[df["sentiment"] == "negative", "review"].head(3):
    print(review[:300], "...\n")