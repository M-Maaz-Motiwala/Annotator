import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


def load_manifest(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_processed_state(state_path: Path) -> Dict[str, str]:
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_processed_state(state_path: Path, state: Dict[str, str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=True)


def materialize_audio(item: Dict) -> Path:
    if item["source_type"] == "file":
        return Path(item["source_path"])

    if item["source_type"] == "zip_member":
        zippath = Path(item["zip_path"])
        member = item["member_path"]
        tmp_dir = Path(tempfile.mkdtemp(prefix="call_extract_"))
        with zipfile.ZipFile(zippath) as zf:
            zf.extract(member, path=tmp_dir)
        return tmp_dir / member

    raise ValueError(f"Unknown source type: {item['source_type']}")


def call_api(api_url: str, audio_path: Path, call_id: str, language: str, timeout: int) -> Dict:
    with audio_path.open("rb") as f:
        files = {"file": (audio_path.name, f, "application/octet-stream")}
        data = {"call_id": call_id, "language": language}
        resp = requests.post(f"{api_url}/transcribe", files=files, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def write_call_output(out_root: Path, item: Dict, response: Dict) -> None:
    call_dir = out_root / item["call_id"]
    call_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = call_dir / "transcript.json"
    source_path = call_dir / "source.json"

    with transcript_path.open("w", encoding="utf-8") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)

    with source_path.open("w", encoding="utf-8") as f:
        json.dump(item, f, indent=2, ensure_ascii=True)


def cleanup_extracted(audio_path: Path, item: Dict) -> None:
    if item["source_type"] != "zip_member":
        return
    # Extracted path is inside a temp folder created by materialize_audio.
    root = audio_path
    while root.parent != root and "call_extract_" not in root.name:
        root = root.parent
    if "call_extract_" in root.name:
        for sub in sorted(root.rglob("*"), reverse=True):
            if sub.is_file():
                sub.unlink(missing_ok=True)
            else:
                sub.rmdir()
        root.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a batch of calls via GPU transcription API.")
    parser.add_argument("--manifest", default="manifest.jsonl", help="Input JSONL manifest path")
    parser.add_argument("--state", default="progress.json", help="Progress state JSON path")
    parser.add_argument("--out-root", default="transcripts", help="Output root folder")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of new calls to process")
    parser.add_argument("--api-url", required=True, help="GPU API base URL")
    parser.add_argument("--language", default="en", help="Whisper language code")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout seconds per file")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    state_path = Path(args.state).resolve()
    out_root = Path(args.out_root).resolve()

    all_items = load_manifest(manifest_path)
    state = load_processed_state(state_path)

    pending = [item for item in all_items if item["call_id"] not in state]
    batch = pending[: args.batch_size]

    print(f"Total in manifest: {len(all_items)}")
    print(f"Already processed: {len(state)}")
    print(f"Processing now: {len(batch)}")

    for item in tqdm(batch, desc="Transcribing"):
        call_id = item["call_id"]
        audio_path: Optional[Path] = None
        try:
            audio_path = materialize_audio(item)
            response = call_api(args.api_url, audio_path, call_id, args.language, args.timeout)
            write_call_output(out_root, item, response)
            state[call_id] = "done"
        except Exception as exc:
            state[call_id] = f"error: {exc}"
        finally:
            if audio_path is not None:
                cleanup_extracted(audio_path, item)
            save_processed_state(state_path, state)

    print("Batch complete.")


if __name__ == "__main__":
    main()
