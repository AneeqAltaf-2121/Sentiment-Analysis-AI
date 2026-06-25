"""Compare Logistic Regression and Multinomial Naive Bayes for sentiment
classification, then save the best-performing model as the project's final
classifier.

Uses the same TF-IDF features, train/test split, and random state as
`train_model.py` so the comparison is fair and reproducible.
"""

import os

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURES_PATH = os.path.join(MODELS_DIR, "tfidf_features.npz")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.joblib")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_classifier.joblib")

X = load_npz(FEATURES_PATH)
y = joblib.load(LABELS_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
}

metrics = {}
predictions = {}

for name, model in candidates.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    predictions[name] = y_pred

comparison_df = pd.DataFrame(metrics).T

print("=" * 80)
print("SIDE-BY-SIDE METRIC COMPARISON")
print("=" * 80)
print(comparison_df.round(4))

for name in candidates:
    print("\n" + "=" * 80)
    print(f"CLASSIFICATION REPORT - {name}")
    print("=" * 80)
    print(classification_report(y_test, predictions[name], target_names=["negative", "positive"]))

    print(f"CONFUSION MATRIX - {name} (rows=actual, cols=predicted, order=[negative, positive])")
    print(confusion_matrix(y_test, predictions[name]))

best_model_name = comparison_df["f1_score"].idxmax()
best_model = candidates[best_model_name]
best_row = comparison_df.loc[best_model_name]

print("\n" + "=" * 80)
print("FINAL MODEL SELECTION")
print("=" * 80)
print(f"Selected model: {best_model_name}")
print(
    f"Reason: highest F1-score ({best_row['f1_score']:.4f}) among candidates, "
    f"balancing precision ({best_row['precision']:.4f}) and recall ({best_row['recall']:.4f})."
)
print("\nFull comparison:")
print(comparison_df.round(4))

joblib.dump(best_model, FINAL_MODEL_PATH)
print(f"\nSaved final classifier ({best_model_name}) to {FINAL_MODEL_PATH}")
