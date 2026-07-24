# Spam Email Classifier

A production-style, end-to-end machine learning project that classifies SMS/email
messages as **spam** or **ham** (legitimate). Built with scikit-learn, TF-IDF
features, and a clean, testable, modular codebase.

## Features

- Automatic dataset download — no manual steps, no Kaggle account needed
- Text preprocessing pipeline: lowercasing, URL/number stripping, punctuation
  removal, tokenization, stopword filtering
- TF-IDF feature extraction (unigrams + bigrams)
- Two models trained and compared: **Multinomial Naive Bayes** vs.
  **Logistic Regression**; the best one (by F1-score) is saved automatically
- Full evaluation: accuracy, precision, recall, F1-score, confusion matrix
- CLI prediction tool + interactive demo, both runnable with a single command
- Unit tests covering preprocessing, training, and prediction
- EDA notebook for exploring the dataset

## Project Structure

```
spam-email-classifier/
├── data/                 # Dataset folder (auto-populated on first run)
│   └── README.md         # Dataset source & citation
├── notebooks/
│   └── eda.ipynb          # Exploratory data analysis
├── src/
│   ├── data_loader.py     # Downloads & caches the dataset
│   ├── preprocessing.py   # Text cleaning / tokenization
│   ├── train.py           # Trains, evaluates, and saves the best model
│   └── predict.py         # Loads the model and classifies new text (CLI)
├── models/                # Saved model + vectorizer (auto-populated)
├── tests/                 # Unit tests (pytest)
├── main.py                # Single-command entry point
├── requirements.txt
└── .gitignore
```

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
from the UCI Machine Learning Repository — 5,574 real, labeled SMS messages
(86.6% ham / 13.4% spam). See [`data/README.md`](data/README.md) for the full
citation. The dataset is downloaded and cached automatically the first time
you run the project — no manual download required.

## Quick Start

```bash
git clone <your-repo-url> spam-email-classifier
cd spam-email-classifier
pip install -r requirements.txt
python main.py
```

That's it. `main.py` downloads the dataset, trains and compares both models
(or reuses a previously saved model), prints sample predictions, and drops
you into an interactive prompt to try your own messages.

## Installation

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Run everything with one command:**

```bash
python main.py
```

**Train only** (re-trains and overwrites the saved model):

```bash
python -m src.train
```

**Predict on a single message:**

```bash
python -m src.predict "Congratulations! You've WON a free prize, click now!"
# [SPAM] (spam probability: 98.1%)  '...'
```

**Predict interactively:**

```bash
python -m src.predict
```

**Use it as a library:**

```python
from src.predict import predict

result = predict("Hey, are we still on for lunch tomorrow?")
print(result)  # {'label': 'ham', 'spam_probability': 0.001}
```

## Results

Evaluated on a stratified 80/20 train/test split (test set: 1,115 messages).

| Model                 | Accuracy | Precision | Recall | F1-score |
|------------------------|:--------:|:---------:|:------:|:--------:|
| **Naive Bayes** (best) | 0.9803   | 0.9847    | 0.8658 | 0.9214   |
| Logistic Regression     | 0.9767   | 0.9073    | 0.9195 | 0.9133   |

The best model (highest F1-score) is selected and saved automatically by
`src/train.py`. Full metrics, including the confusion matrix and per-class
report, are written to `models/metrics.json` after each training run.

## Testing

```bash
pytest tests/ -v
```

Unit tests cover text preprocessing, model training/evaluation logic, and
prediction — using small synthetic data so they run fully offline.

## Screenshot

<!-- Add a screenshot of the CLI demo here, e.g.: -->
<!-- ![CLI demo](docs/screenshot.png) -->

## Tech Stack

Python, pandas, scikit-learn, joblib, matplotlib, pytest, Jupyter.

## License

MIT
