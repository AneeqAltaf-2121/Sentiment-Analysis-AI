"""Train and evaluate a Logistic Regression sentiment classifier.

Loads the TF-IDF feature matrix and encoded labels produced in the
feature-extraction phase, trains on a train/test split, evaluates the
model, and saves the fitted classifier for later use by the app.
"""

import os

import joblib
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

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURES_PATH = os.path.join(MODELS_DIR, "tfidf_features.npz")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression_model.joblib")

X = load_npz(FEATURES_PATH)
y = joblib.load(LABELS_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=" * 80)
print("EVALUATION METRICS")
print("=" * 80)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

print("\n" + "=" * 80)
print("CONFUSION MATRIX (rows=actual, cols=predicted, order=[negative, positive])")
print("=" * 80)
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, MODEL_PATH)
print(f"\nSaved trained model to {MODEL_PATH}")
