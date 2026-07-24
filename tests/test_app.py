"""Smoke tests for the Streamlit web UI (app.py), using Streamlit's
official headless AppTest framework -- no real browser needed.

A tiny synthetic model is swapped in via monkeypatched paths so this test
stays offline and fast, consistent with the rest of the suite.
"""
import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from streamlit.testing.v1 import AppTest

import src.train as train_module
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


@pytest.fixture
def fake_trained_model(tmp_path, monkeypatch):
    """Point src.train's saved-model paths at a tiny synthetic model."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(clean_corpus(TRAIN_TEXTS))
    model = MultinomialNB()
    model.fit(X, TRAIN_LABELS)

    model_path = tmp_path / "spam_classifier.joblib"
    vectorizer_path = tmp_path / "tfidf_vectorizer.joblib"
    metrics_path = tmp_path / "metrics.json"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    metrics_path.write_text(
        json.dumps(
            {
                "best_model": "naive_bayes",
                "results": {
                    "naive_bayes": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0}
                },
            }
        )
    )

    monkeypatch.setattr(train_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(train_module, "VECTORIZER_PATH", vectorizer_path)
    monkeypatch.setattr(train_module, "METRICS_PATH", metrics_path)


def test_app_loads_without_error(fake_trained_model):
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value == "📧 Spam Email Classifier"


def test_app_classifies_spam_example(fake_trained_model):
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    buttons = {b.label: b for b in at.button}
    buttons["Spam example"].click().run()
    buttons["Classify"].click().run()
    assert not at.exception
    assert any("SPAM" in e.value for e in at.error)


def test_app_classifies_ham_example(fake_trained_model):
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    buttons = {b.label: b for b in at.button}
    buttons["Ham example"].click().run()
    buttons["Classify"].click().run()
    assert not at.exception
    assert any("HAM" in s.value for s in at.success)


def test_app_warns_on_empty_input(fake_trained_model):
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    buttons = {b.label: b for b in at.button}
    buttons["Classify"].click().run()
    assert not at.exception
    assert len(at.warning) == 1
