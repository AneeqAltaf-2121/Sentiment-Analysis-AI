"""Text preprocessing pipeline for the IMDb reviews dataset.

Cleans the `review` column into a new `clean_review` column and saves the
result for the feature-extraction / modeling phase. The original `review`
column and the source CSV are left untouched.
"""

import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "IMDB Dataset.csv")
CLEAN_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "imdb_clean.csv")

pd.set_option("display.max_colwidth", 100)


def _ensure_nltk_data():
    required = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for resource_path, package in required.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
EXTRA_WHITESPACE_RE = re.compile(r"\s+")


def clean_review(text):
    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = EXTRA_WHITESPACE_RE.sub(" ", text).strip()

    tokens = [word for word in text.split() if word not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]

    return " ".join(tokens)


df = pd.read_csv(RAW_DATA_PATH)
df["clean_review"] = df["review"].apply(clean_review)

print("=" * 80)
print("ORIGINAL VS CLEANED REVIEWS (SAMPLE)")
print("=" * 80)
for original, cleaned in df[["review", "clean_review"]].head(5).itertuples(index=False):
    print("ORIGINAL:", original[:300])
    print("CLEANED :", cleaned[:300])
    print("-" * 80)

df["review_word_count"] = df["review"].str.split().str.len()
df["clean_review_word_count"] = df["clean_review"].str.split().str.len()

print("\n" + "=" * 80)
print("REVIEW LENGTH COMPARISON (WORD COUNT)")
print("=" * 80)
print(df[["review_word_count", "clean_review_word_count"]].describe())

avg_original = df["review_word_count"].mean()
avg_cleaned = df["clean_review_word_count"].mean()
print(f"\nAverage word count - original: {avg_original:.1f}, cleaned: {avg_cleaned:.1f}")
print(f"Average reduction: {(1 - avg_cleaned / avg_original) * 100:.1f}%")

print("\n" + "=" * 80)
print("MISSING VALUE CHECK")
print("=" * 80)
print(df[["review", "clean_review"]].isnull().sum())
print(f"Empty clean_review strings: {(df['clean_review'].str.len() == 0).sum()}")

df.to_csv(CLEAN_DATA_PATH, index=False, columns=["review", "sentiment", "clean_review"])
print(f"\nSaved cleaned dataset to {CLEAN_DATA_PATH}")
