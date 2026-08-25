"""App import without external services or embedding-model download."""

import sys

from fastapi import FastAPI

from app.main import app


def test_app_import() -> None:
    assert isinstance(app, FastAPI)
    assert app.title
    assert app.version == "0.1.0"


def test_import_does_not_load_sentence_transformers() -> None:
    from app.main import app as again

    assert again is app
    assert "sentence_transformers" not in sys.modules
