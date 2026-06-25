"""Backend prediction logic for the sentiment analysis app.

Loads the saved TF-IDF vectorizer and final classifier once at import time,
cleans incoming review text with the same pipeline used during training,
and exposes `predict_sentiment` for use by `app.py` or any other interface.
"""

import os
import re

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_classifier.joblib")


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
    """Apply the same cleaning pipeline used to prepare training data."""
    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = EXTRA_WHITESPACE_RE.sub(" ", text).strip()

    tokens = [word for word in text.split() if word not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]

    return " ".join(tokens)


_VECTORIZER = None
_MODEL = None
_LOAD_ERROR = None

try:
    _VECTORIZER = joblib.load(VECTORIZER_PATH)
    _MODEL = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    _LOAD_ERROR = f"Model artifact not found: {exc.filename}"
except Exception as exc:  # pragma: no cover - unexpected load failure
    _LOAD_ERROR = f"Failed to load sentiment model: {exc}"


def predict_sentiment(text):
    """Predict sentiment for a raw review string.

    Returns a dict with keys "label", "confidence", and "error". On
    success "error" is None; on failure "label"/"confidence" are None and
    "error" describes what went wrong, so callers never need to catch
    exceptions to stay usable.
    """
    if _LOAD_ERROR is not None:
        return {"label": None, "confidence": None, "error": _LOAD_ERROR}

    if not isinstance(text, str) or not text.strip():
        return {"label": None, "confidence": None, "error": "Review text must be a non-empty string."}

    try:
        cleaned = clean_review(text)
        if not cleaned:
            return {
                "label": None,
                "confidence": None,
                "error": "Review contains no usable text after cleaning.",
            }

        features = _VECTORIZER.transform([cleaned])
        prediction = _MODEL.predict(features)[0]
        label = "Positive" if prediction == 1 else "Negative"

        confidence = None
        if hasattr(_MODEL, "predict_proba"):
            confidence = float(_MODEL.predict_proba(features)[0][prediction])

        return {"label": label, "confidence": confidence, "error": None}
    except Exception as exc:
        return {"label": None, "confidence": None, "error": f"Prediction failed: {exc}"}
