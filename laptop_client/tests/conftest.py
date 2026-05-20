import json
import wave
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_audio_tree(tmp_path: Path) -> Path:
    root = tmp_path / "drive"
    nested = root / "team_a" / "2024"
    nested.mkdir(parents=True)

    wav_path = nested / "call_one.wav"
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 800)

    mp3_path = nested / "notes.txt"
    mp3_path.write_text("not audio", encoding="utf-8")

    zip_path = root / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("inner/call_two.mp3", b"fake-mp3-content")
        zf.writestr("inner/readme.txt", b"ignore")

    return root


@pytest.fixture
def sample_manifest_items(sample_audio_tree: Path) -> list[dict]:
    from build_manifest import build_manifest

    return build_manifest(sample_audio_tree)
