import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# Skip heavy model download/load during unit tests.
os.environ.setdefault("SKIP_MODEL_LOAD", "1")


@pytest.fixture
def mock_whisper_model():
    segment = SimpleNamespace(start=0.0, end=1.5, text=" hello world ")
    info = SimpleNamespace(language="en", duration=12.34)

    model = MagicMock()
    model.transcribe.return_value = ([segment], info)
    return model


@pytest.fixture
def api_client(mock_whisper_model, monkeypatch):
    from fastapi.testclient import TestClient

    import app as app_module

    monkeypatch.setattr(app_module, "whisper_model", mock_whisper_model)
    monkeypatch.setattr(app_module, "diarization_pipeline", None)

    with TestClient(app_module.app) as client:
        yield client
