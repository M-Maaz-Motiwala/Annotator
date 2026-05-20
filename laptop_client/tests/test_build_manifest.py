import json
from pathlib import Path

from build_manifest import build_manifest, make_call_id


def test_make_call_id_is_stable():
    source = r"E:\Google Drive\team\call.wav"
    first = make_call_id(source)
    second = make_call_id(source)
    assert first == second
    assert first.endswith("_" + first.split("_")[-1])


def test_build_manifest_finds_files_and_zip_members(sample_audio_tree: Path):
    items = build_manifest(sample_audio_tree)

    source_types = {item["source_type"] for item in items}
    assert "file" in source_types
    assert "zip_member" in source_types

    file_items = [i for i in items if i["source_type"] == "file"]
    zip_items = [i for i in items if i["source_type"] == "zip_member"]

    assert len(file_items) == 1
    assert file_items[0]["source_path"].endswith("call_one.wav")

    assert len(zip_items) == 1
    assert zip_items[0]["member_path"] == "inner/call_two.mp3"
    assert zip_items[0]["zip_path"].endswith("archive.zip")


def test_build_manifest_call_ids_are_unique(sample_manifest_items):
    call_ids = [item["call_id"] for item in sample_manifest_items]
    assert len(call_ids) == len(set(call_ids))


def test_manifest_jsonl_roundtrip(tmp_path: Path, sample_manifest_items):
    out_path = tmp_path / "manifest.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in sample_manifest_items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    loaded = []
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            loaded.append(json.loads(line))

    assert len(loaded) == len(sample_manifest_items)
