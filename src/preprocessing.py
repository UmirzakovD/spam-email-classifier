"""Text cleaning and preprocessing utilities for the spam classifier.

The pipeline is intentionally dependency-free (regex tokenizer + a built-in
English stopword list) so preprocessing never needs to download extra
corpora at runtime -- it works offline right after ``pip install``.
"""
import re
import string

# A compact, standard English stopword list used to filter tokens.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll
    he's her here here's hers herself him himself his how how's i i'd i'll
    i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our
    ours ourselves out over own same shan't she she'd she'll she's should
    shouldn't so some such than that that's the their theirs them
    themselves then there there's these they they'd they'll they're
    they've this those through to too under until up very was wasn't we
    we'd we'll we're we've were weren't what what's when when's where
    where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves
    """.split()
)

_TOKEN_PATTERN = re.compile(r"[a-z]+")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_NUMBER_PATTERN = re.compile(r"\d+")
_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def clean_text(text: str) -> str:
    """Normalize a raw message into a cleaned, space-joined token string.

    Steps: lowercase -> strip URLs -> strip numbers -> strip punctuation ->
    tokenize -> drop stopwords and single-character tokens.
    """
    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _NUMBER_PATTERN.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    tokens = _TOKEN_PATTERN.findall(text)
    tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
    return " ".join(tokens)


def clean_corpus(texts) -> list:
    """Apply :func:`clean_text` to every item of an iterable of raw texts."""
    return [clean_text(str(t)) for t in texts]
