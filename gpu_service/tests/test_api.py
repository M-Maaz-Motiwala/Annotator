import io
import os
import wave

import pytest
import requests


def _make_wav_bytes(duration_seconds: float = 0.2, sample_rate: int = 16000) -> bytes:
    frame_count = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frame_count)
    return buf.getvalue()


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["diarization_enabled"] is False


def test_transcribe_returns_structured_json(api_client):
    wav_bytes = _make_wav_bytes()
    files = {"file": ("sample.wav", wav_bytes, "audio/wav")}
    data = {"call_id": "test_call_001", "language": "en"}

    resp = api_client.post("/transcribe", files=files, data=data)
    assert resp.status_code == 200

    body = resp.json()
    assert body["call_id"] == "test_call_001"
    assert body["language"] == "en"
    assert body["full_text"] == "hello world"
    assert len(body["segments"]) == 1
    assert body["segments"][0]["speaker"] == "UNKNOWN"
    assert body["diarization_enabled"] is False


def test_transcribe_json_endpoint_rejected(api_client):
    resp = api_client.post("/transcribe/json", json={"call_id": "x"})
    assert resp.status_code == 405


@pytest.mark.integration
def test_live_health_endpoint():
    api_url = os.getenv("GPU_API_URL")
    if not api_url:
        pytest.skip("Set GPU_API_URL to run integration tests, e.g. http://localhost:8000")

    resp = requests.get(f"{api_url}/health", timeout=30)
    resp.raise_for_status()
    body = resp.json()
    assert body["status"] == "ok"
