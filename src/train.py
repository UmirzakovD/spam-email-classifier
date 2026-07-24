"""Train and evaluate spam classification models.

Trains two candidate models (Multinomial Naive Bayes and Logistic
Regression) on TF-IDF features, evaluates both on a held-out test split,
and persists the best-performing model (by F1-score) plus the fitted
vectorizer to the ``models/`` directory.
"""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
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

from src.data_loader import load_dataset
from src.preprocessing import clean_corpus

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
POS_LABEL = "spam"


def build_models() -> dict:
    """Return the registry of candidate models to train and compare."""
    return {
        "naive_bayes": MultinomialNB(alpha=0.3),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    """Compute standard classification metrics for a fitted model."""
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, pos_label=POS_LABEL),
        "recall": recall_score(y_test, preds, pos_label=POS_LABEL),
        "f1_score": f1_score(y_test, preds, pos_label=POS_LABEL),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=["ham", "spam"]).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }


def train(test_size: float = 0.2, random_state: int = RANDOM_STATE) -> dict:
    """Train all candidate models, pick the best by F1-score, and persist it.

    Returns:
        A dict with the winning model's name and the full metrics for every
        candidate model that was trained.
    """
    df = load_dataset()
    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = clean_corpus(df["text"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}
    fitted_models = {}
    for name, model in build_models().items():
        model.fit(X_train_vec, y_train)
        metrics = evaluate(model, X_test_vec, y_test)
        results[name] = metrics
        fitted_models[name] = model
        print(f"[{name}] accuracy={metrics['accuracy']:.4f}  f1={metrics['f1_score']:.4f}")

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} (F1-score={results[best_name]['f1_score']:.4f})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    return {"best_model": best_name, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Train the spam email/SMS classifier.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data used for testing.")
    args = parser.parse_args()
    train(test_size=args.test_size)


if __name__ == "__main__":
    main()
