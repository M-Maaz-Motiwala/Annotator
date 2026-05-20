import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List


AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".wma",
    ".opus",
    ".mp4",
    ".mov",
}


def make_call_id(source: str) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    safe_name = Path(source).name.replace(" ", "_")
    return f"{safe_name}_{digest}"


def iter_audio_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def iter_audio_members_in_zip(zip_path: Path) -> Iterable[str]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if Path(member).suffix.lower() in AUDIO_EXTENSIONS:
                    yield member
    except zipfile.BadZipFile:
        return


def build_manifest(root: Path) -> List[Dict]:
    items: List[Dict] = []
    root = root.resolve()

    for file_path in iter_audio_files(root):
        source = str(file_path)
        items.append(
            {
                "call_id": make_call_id(source),
                "source_type": "file",
                "source_path": source,
            }
        )

    for zip_path in root.rglob("*.zip"):
        if not zip_path.is_file():
            continue
        for member in iter_audio_members_in_zip(zip_path):
            source = f"{zip_path}::{member}"
            items.append(
                {
                    "call_id": make_call_id(source),
                    "source_type": "zip_member",
                    "zip_path": str(zip_path),
                    "member_path": member,
                }
            )

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Create transcription manifest from nested dirs/zips.")
    parser.add_argument("--root", required=True, help="Root path to scan (e.g., E:\\Google Drive)")
    parser.add_argument("--out", default="manifest.jsonl", help="Output JSONL manifest path")
    args = parser.parse_args()

    root = Path(args.root)
    records = build_manifest(root)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    print(f"Manifest written: {out_path}")
    print(f"Total call candidates: {len(records)}")


if __name__ == "__main__":
    main()
