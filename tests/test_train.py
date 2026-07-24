"""Unit tests for src.train (model building and evaluation logic).

Uses a small synthetic dataset so tests run offline, without downloading
the full SMS Spam Collection dataset.
"""
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import clean_corpus
from src.train import build_models, evaluate

TEXTS = [
    "win a free prize call now",
    "urgent claim your free cash reward",
    "free entry to win cash now click here",
    "congratulations you have won a lottery",
    "are we still meeting for lunch",
    "let's catch up this weekend",
    "can you send me the report today",
    "see you at the office tomorrow",
]
LABELS = ["spam", "spam", "spam", "spam", "ham", "ham", "ham", "ham"]


def test_build_models_returns_two_candidates():
    models = build_models()
    assert set(models.keys()) == {"naive_bayes", "logistic_regression"}


def test_evaluate_returns_expected_metric_keys():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(clean_corpus(TEXTS))

    models = build_models()
    model = models["naive_bayes"]
    model.fit(X, LABELS)

    metrics = evaluate(model, X, LABELS)
    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
        "classification_report",
    }
    assert expected_keys == set(metrics.keys())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
