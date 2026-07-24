"""Unit tests for src.data_loader's retry / error-handling behavior.

Network calls are mocked so these tests run offline and don't depend on
the real UCI server being reachable.
"""
import urllib.error
from unittest.mock import patch

import pytest

from src.data_loader import DatasetDownloadError, _fetch_zip_bytes


def test_fetch_retries_then_succeeds():
    good_response = b"fake-zip-bytes"

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return good_response

    calls = {"count": 0}

    def flaky_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.URLError("temporary failure")
        return _FakeResponse()

    with patch("src.data_loader.urllib.request.urlopen", side_effect=flaky_urlopen), patch(
        "src.data_loader.time.sleep"
    ):
        result = _fetch_zip_bytes()

    assert result == good_response
    assert calls["count"] == 3


def test_fetch_raises_dataset_download_error_after_exhausting_retries():
    def always_fails(request, timeout):
        raise urllib.error.URLError("still down")

    with patch("src.data_loader.urllib.request.urlopen", side_effect=always_fails), patch(
        "src.data_loader.time.sleep"
    ):
        with pytest.raises(DatasetDownloadError):
            _fetch_zip_bytes()
