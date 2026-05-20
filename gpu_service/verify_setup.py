"""Check the GPU API is up. Run after docker compose up.

Usage:
  python verify_setup.py --api-url http://localhost:8000
"""

import argparse
import io
import sys
import wave

import requests


def _tiny_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GPU transcription API.")
    parser.add_argument("--api-url", required=True, help="e.g. http://localhost:8000")
    args = parser.parse_args()
    base = args.api_url.rstrip("/")

    print("1) GET /health")
    health = requests.get(f"{base}/health", timeout=30)
    health.raise_for_status()
    print("   OK:", health.json())

    print("2) POST /transcribe (small test clip)")
    files = {"file": ("verify.wav", _tiny_wav(), "audio/wav")}
    data = {"call_id": "verify_setup", "language": "en"}
    resp = requests.post(f"{base}/transcribe", files=files, data=data, timeout=600)
    resp.raise_for_status()
    body = resp.json()
    print("   OK: call_id =", body.get("call_id"), "| segments =", len(body.get("segments", [])))

    print("\nAll checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAILED:", exc, file=sys.stderr)
        sys.exit(1)
