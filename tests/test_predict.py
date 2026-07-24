"""Unit tests for src.predict.

Uses a tiny in-memory model trained on synthetic data so the tests run
fully offline and don't depend on the (network-fetched) full dataset.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.predict import predict
from src.preprocessing import clean_corpus

TRAIN_TEXTS = [
    "win a free prize call now",
    "urgent claim your free cash reward",
    "free entry to win cash now click here",
    "are we still meeting for lunch",
    "let's catch up this weekend",
    "can you send me the report today",
]
TRAIN_LABELS = ["spam", "spam", "spam", "ham", "ham", "ham"]


def _build_model():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(clean_corpus(TRAIN_TEXTS))
    model = MultinomialNB()
    model.fit(X, TRAIN_LABELS)
    return model, vectorizer


def test_predict_returns_expected_keys():
    model, vectorizer = _build_model()
    result = predict("free cash prize", model, vectorizer)
    assert set(result.keys()) == {"label", "spam_probability"}
    assert result["label"] in ("spam", "ham")


def test_predict_spam_probability_in_range():
    model, vectorizer = _build_model()
    result = predict("call now to win a free reward", model, vectorizer)
    assert 0.0 <= result["spam_probability"] <= 1.0


def test_predict_classifies_obvious_ham():
    model, vectorizer = _build_model()
    result = predict("let's catch up this weekend", model, vectorizer)
    assert result["label"] == "ham"


def test_predict_classifies_obvious_spam():
    model, vectorizer = _build_model()
    result = predict("free cash prize win now", model, vectorizer)
    assert result["label"] == "spam"
