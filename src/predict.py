"""Load the trained spam classifier and predict on new text.

Can be used as a library (``predict(text)``) or run directly as a CLI:

    python -m src.predict "Congratulations, you won a free prize!"
    python -m src.predict          # interactive mode
"""
import argparse
import sys

import joblib

from src.data_loader import DatasetDownloadError
from src.preprocessing import clean_text
from src.train import MODEL_PATH, VECTORIZER_PATH, train


def _load_artifacts():
    """Load the persisted model and vectorizer, training them if missing."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        print("No trained model found. Training one now (this only happens once)...\n")
        train()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str, model=None, vectorizer=None) -> dict:
    """Predict whether a message is spam.

    Args:
        text: Raw message text to classify.
        model: A fitted classifier. Loaded from disk if not provided.
        vectorizer: A fitted TF-IDF vectorizer. Loaded from disk if not provided.

    Returns:
        ``{"label": "spam" | "ham", "spam_probability": float | None}``
    """
    if model is None or vectorizer is None:
        model, vectorizer = _load_artifacts()

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    label = model.predict(vector)[0]

    spam_probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        spam_probability = float(model.predict_proba(vector)[0][classes.index("spam")])

    return {"label": label, "spam_probability": spam_probability}


def _print_result(message: str, result: dict) -> None:
    label = result["label"].upper()
    prob = result["spam_probability"]
    prob_str = f" (spam probability: {prob:.2%})" if prob is not None else ""
    print(f"[{label}]{prob_str}  {message!r}")


def main():
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham.")
    parser.add_argument("text", nargs="*", help="Message text to classify. Omit for interactive mode.")
    args = parser.parse_args()

    try:
        model, vectorizer = _load_artifacts()
    except DatasetDownloadError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.text:
        message = " ".join(args.text)
        _print_result(message, predict(message, model, vectorizer))
        return

    print("Spam Classifier CLI - type a message and press Enter (Ctrl+C to quit).\n")
    try:
        while True:
            message = input(">> ").strip()
            if not message:
                continue
            _print_result(message, predict(message, model, vectorizer))
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
