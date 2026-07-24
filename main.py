"""Single-command entry point for the Spam Email Classifier project.

Running ``python main.py`` will:
  1. Download the dataset if it isn't cached yet (see data/README.md).
  2. Train and compare Naive Bayes vs. Logistic Regression (or reuse an
     already-trained model found in models/).
  3. Run a couple of sample predictions and drop into an interactive
     prompt where you can type your own messages.
"""
from src.predict import _print_result, predict
from src.train import MODEL_PATH, VECTORIZER_PATH, train

SAMPLE_MESSAGES = [
    "Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
]


def main():
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        print("Found an existing trained model, skipping training.")
        print("(Delete the files in 'models/' to retrain from scratch.)\n")
    else:
        print("No trained model found - training now...\n")
        train()

    print("\n" + "=" * 60)
    print("Spam Email / SMS Classifier - demo")
    print("=" * 60)
    for text in SAMPLE_MESSAGES:
        _print_result(text, predict(text))

    print("\nType your own message below (Ctrl+C to exit).")
    try:
        while True:
            text = input(">> ").strip()
            if not text:
                continue
            _print_result(text, predict(text))
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
