import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from process_batch import (
    call_api,
    load_manifest,
    load_processed_state,
    materialize_audio,
    save_processed_state,
    write_call_output,
)


def test_load_and_save_processed_state(tmp_path: Path):
    state_path = tmp_path / "progress.json"
    assert load_processed_state(state_path) == {}

    save_processed_state(state_path, {"call_a": "done"})
    assert load_processed_state(state_path) == {"call_a": "done"}


def test_load_manifest(tmp_path: Path, sample_manifest_items):
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in sample_manifest_items:
            f.write(json.dumps(item) + "\n")

    loaded = load_manifest(manifest_path)
    assert len(loaded) == len(sample_manifest_items)


def test_materialize_audio_file(sample_manifest_items):
    file_item = next(i for i in sample_manifest_items if i["source_type"] == "file")
    path = materialize_audio(file_item)
    assert path.exists()


def test_materialize_audio_zip_member(sample_manifest_items):
    zip_item = next(i for i in sample_manifest_items if i["source_type"] == "zip_member")
    path = materialize_audio(zip_item)
    assert path.exists()
    assert path.name == "call_two.mp3"


def test_write_call_output(tmp_path: Path, sample_manifest_items):
    item = sample_manifest_items[0]
    response = {"call_id": item["call_id"], "full_text": "hello", "segments": []}
    out_root = tmp_path / "transcripts"

    write_call_output(out_root, item, response)

    call_dir = out_root / item["call_id"]
    assert (call_dir / "transcript.json").exists()
    assert (call_dir / "source.json").exists()

    transcript = json.loads((call_dir / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["full_text"] == "hello"


def test_call_api_posts_multipart(monkeypatch, sample_manifest_items, tmp_path: Path):
    file_item = next(i for i in sample_manifest_items if i["source_type"] == "file")
    audio_path = materialize_audio(file_item)

    mock_response = MagicMock()
    mock_response.json.return_value = {"call_id": file_item["call_id"], "full_text": "ok"}
    mock_response.raise_for_status = MagicMock()

    captured = {}

    def fake_post(url, files=None, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["timeout"] = timeout
        return mock_response

    import requests

    monkeypatch.setattr(requests, "post", fake_post)

    result = call_api("http://gpu.local:8000", audio_path, file_item["call_id"], "en", 30)
    assert result["full_text"] == "ok"
    assert captured["url"] == "http://gpu.local:8000/transcribe"
    assert captured["data"]["call_id"] == file_item["call_id"]


@pytest.mark.integration
def test_live_gpu_health():
    api_url = os.getenv("GPU_API_URL")
    if not api_url:
        pytest.skip("Set GPU_API_URL to run integration tests, e.g. http://localhost:8000")

    import requests

    resp = requests.get(f"{api_url}/health", timeout=30)
    resp.raise_for_status()
    assert resp.json()["status"] == "ok"
