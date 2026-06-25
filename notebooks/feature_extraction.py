"""TF-IDF feature extraction for the cleaned IMDb reviews.

Loads the preprocessed dataset, encodes sentiment labels, vectorizes
`clean_review` with TF-IDF, and saves the fitted vectorizer and feature
matrix for the model-training phase.
"""

import os

import joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

CLEAN_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "imdb_clean.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "tfidf_features.npz")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.joblib")

os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(CLEAN_DATA_PATH)

reviews = df["clean_review"]
sentiments = df["sentiment"]

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(sentiments)

print("=" * 80)
print("LABEL ENCODING")
print("=" * 80)
for cls, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{cls} -> {code}")

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(reviews)

print("\n" + "=" * 80)
print("TF-IDF MATRIX")
print("=" * 80)
print(f"Shape: {tfidf_matrix.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

feature_names = vectorizer.get_feature_names_out()
print(f"Sample feature names: {feature_names[:20].tolist()}")

print("\n" + "=" * 80)
print("SAVING ARTIFACTS")
print("=" * 80)
joblib.dump(vectorizer, VECTORIZER_PATH)
save_npz(FEATURES_PATH, tfidf_matrix)
joblib.dump(labels, LABELS_PATH)
print(f"Saved vectorizer to {VECTORIZER_PATH}")
print(f"Saved feature matrix to {FEATURES_PATH}")
print(f"Saved encoded labels to {LABELS_PATH}")
