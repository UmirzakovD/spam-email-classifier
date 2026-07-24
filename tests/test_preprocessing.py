"""Unit tests for src.preprocessing."""
from src.preprocessing import clean_corpus, clean_text


def test_lowercases_text():
    assert clean_text("HELLO World") == "hello world"


def test_removes_punctuation():
    assert "!" not in clean_text("Free!!! Win now!!!")
    assert "," not in clean_text("Hello, world, again")


def test_removes_stopwords():
    cleaned = clean_text("this is a test of the system").split()
    for word in ("this", "is", "a", "of", "the"):
        assert word not in cleaned


def test_removes_urls_and_numbers():
    cleaned = clean_text("Visit http://example.com and call 12345 now")
    assert "http" not in cleaned
    assert "12345" not in cleaned


def test_keeps_meaningful_words():
    cleaned = clean_text("WIN a FREE prize now, click here!!!").split()
    assert "win" in cleaned
    assert "free" in cleaned
    assert "prize" in cleaned


def test_clean_corpus_returns_list_of_same_length():
    texts = ["Hello there!", "WIN a FREE prize now"]
    result = clean_corpus(texts)
    assert len(result) == len(texts)
    assert all(isinstance(t, str) for t in result)


def test_empty_string_returns_empty():
    assert clean_text("") == ""
