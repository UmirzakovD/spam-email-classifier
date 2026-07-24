# Data

This project uses the **SMS Spam Collection** dataset from the UCI Machine
Learning Repository.

- Dataset page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Direct download (used by the code): https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
- Citation: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository.

The dataset contains 5,574 real SMS messages in English, each labeled
`ham` (legitimate) or `spam`.

## You don't need to download anything manually

Running `python main.py` (or `python -m src.train`) automatically:

1. Downloads `sms+spam+collection.zip` from the URL above.
2. Extracts the `SMSSpamCollection` file (tab-separated: `label\ttext`).
3. Caches it here as `sms_spam.csv`.

The cached CSV (and any raw archive) is git-ignored — this README is the
only file committed to `data/`, so the folder starts empty in a fresh
clone and fills in automatically the first time you run the project.
