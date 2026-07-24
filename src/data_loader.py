"""Download and load the SMS Spam Collection dataset.

The dataset is fetched from the official UCI Machine Learning Repository on
first use and cached locally as a CSV file, so no manual download step is
required (see ``data/README.md`` for source details and citation).
"""
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV_PATH = DATA_DIR / "sms_spam.csv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_MEMBER = "SMSSpamCollection"


def download_dataset(force: bool = False) -> Path:
    """Download the SMS Spam Collection dataset and cache it as a CSV file.

    Args:
        force: If True, re-download even if a cached CSV already exists.

    Returns:
        Path to the cached CSV file (columns: ``label``, ``text``).
    """
    if RAW_CSV_PATH.exists() and not force:
        return RAW_CSV_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATASET_URL} ...")
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "spam-classifier/1.0"})
    with urllib.request.urlopen(request) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        with archive.open(DATASET_MEMBER) as member:
            # quoting=QUOTE_NONE: the file is plain label\ttext, not CSV-quoted.
            # Without this, stray `"` characters inside some messages make
            # pandas merge adjacent lines, silently dropping real rows.
            df = pd.read_csv(
                member,
                sep="\t",
                header=None,
                names=["label", "text"],
                encoding="utf-8",
                quoting=csv.QUOTE_NONE,
            )

    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"Dataset cached at {RAW_CSV_PATH} ({len(df)} rows).")
    return RAW_CSV_PATH


def load_dataset() -> pd.DataFrame:
    """Return the dataset as a DataFrame, downloading it first if needed."""
    path = download_dataset()
    return pd.read_csv(path)
