"""Check laptop scripts and (optionally) GPU API.

Usage:
  python verify_setup.py
  python verify_setup.py --api-url http://<GPU_IP>:8000
"""

import argparse
import json
import sys
import tempfile
import wave
import zipfile
from pathlib import Path

import requests

from build_manifest import build_manifest


def _write_sample_drive(root: Path) -> None:
    nested = root / "sample" / "calls"
    nested.mkdir(parents=True)
    wav = nested / "test_call.wav"
    with wave.open(str(wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 800)

    zpath = root / "archive.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("nested/test_call.mp3", b"fake")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify laptop client setup.")
    parser.add_argument("--api-url", help="If set, also checks GPU /health")
    args = parser.parse_args()

    print("1) Manifest scan (temp sample folder)")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_sample_drive(root)
        items = build_manifest(root)
    if len(items) < 2:
        raise RuntimeError(f"expected at least 2 manifest items, got {len(items)}")
    print("   OK: found", len(items), "items")
    print("   sample:", json.dumps(items[0], ensure_ascii=True))

    if args.api_url:
        base = args.api_url.rstrip("/")
        print("2) GET /health")
        health = requests.get(f"{base}/health", timeout=30)
        health.raise_for_status()
        print("   OK:", health.json())
    else:
        print("2) GPU API skipped (pass --api-url to test connection)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAILED:", exc, file=sys.stderr)
        sys.exit(1)
